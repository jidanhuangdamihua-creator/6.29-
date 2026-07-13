#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
readonly PYTHON="${PROJECT_ROOT}/.venv/bin/python"
readonly UNIFIED_RUNNER="${PROJECT_ROOT}/scripts/run_unified_d1_d6.py"
readonly TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
readonly RESULT_ROOT="${PROJECT_ROOT}/outputs/runs/${TIMESTAMP}"
readonly LOG_ROOT="${PROJECT_ROOT}/outputs/parallel_mode_runs/${TIMESTAMP}"
readonly PID_FILE="${LOG_ROOT}/pids.tsv"
readonly RUNNER_LOG="${LOG_ROOT}/runner.log"
readonly MONITOR_INTERVAL_SECONDS=5
readonly TERMINATION_GRACE_SECONDS=10
readonly PID_RESOLUTION_ATTEMPTS=50
readonly PID_RESOLUTION_INTERVAL_SECONDS=0.1
readonly MAX_JOBS="${MAX_JOBS:-10}"
readonly DRY_RUN="${DRY_RUN:-0}"
readonly TASK_IDS=(
    d5_without
    d5_with
    d1_without
    d1_with
    d2_without
    d2_with
    d3_without
    d3_with
    d4_without
    d6_without
    d4_with
    d6_with
)
readonly DATASET_IDS=(d1 d2 d3 d4 d5 d6)

SETSID_MODE=""
CLEANUP_STARTED=0
VALIDATION_ERROR=""

PIDS=()
PGIDS=()
SUPERVISOR_PIDS=()
START_TIMES=()
DURATIONS=()
STATUSES=()
EXIT_CODES=()
FAILURE_REASONS=()
LOG_FILES=()
OUTPUT_DIRS=()
LAUNCHED_INDICES=()

usage() {
    printf 'Usage: DRY_RUN=1 MAX_JOBS=10 bash scripts/parallel_mode_runner.sh\n'
}

fail_usage() {
    printf 'ERROR: %s\n' "$1" >&2
    usage >&2
    exit 2
}

if (($# > 0)); then
    fail_usage "parallel_mode_runner.sh accepts environment variables only"
fi
if [[ ! "${MAX_JOBS}" =~ ^[0-9]+$ ]] || ((MAX_JOBS < 1)); then
    fail_usage "MAX_JOBS must be a positive integer"
fi
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
    fail_usage "DRY_RUN must be 0 or 1"
fi
if [[ "$(pwd -P)" != "${PROJECT_ROOT}" ]]; then
    printf 'ERROR: run this script from the project root: %s\n' "${PROJECT_ROOT}" >&2
    exit 2
fi

task_dataset() {
    local task_id="$1"
    printf '%s\n' "${task_id%%_*}"
}

task_mode() {
    local task_id="$1"
    printf '%s\n' "${task_id##*_}"
}

task_index() {
    local task_id="$1"
    local index
    for ((index = 0; index < ${#TASK_IDS[@]}; index++)); do
        if [[ "${TASK_IDS[${index}]}" == "${task_id}" ]]; then
            printf '%s\n' "${index}"
            return 0
        fi
    done
    return 1
}

iso_timestamp() {
    date '+%Y-%m-%dT%H:%M:%S%z'
}

format_task_command() {
    local task_id="$1"
    local dataset
    local mode
    dataset="$(task_dataset "${task_id}")"
    mode="$(task_mode "${task_id}")"
    printf '%s %s --only %s --info-sharing %s --output-dir %s/%s' \
        "${PYTHON}" "${UNIFIED_RUNNER}" "${dataset}" "${mode}" "${RESULT_ROOT}" "${task_id}"
}

if ((DRY_RUN == 1)); then
    printf '[DRY-RUN] run_id=%s\n' "${TIMESTAMP}"
    printf '[DRY-RUN] result root: %s\n' "${RESULT_ROOT}"
    printf '[DRY-RUN] log root: %s\n' "${LOG_ROOT}"
    printf '[DRY-RUN] MAX_JOBS=%s\n' "${MAX_JOBS}"
    for task_id in "${TASK_IDS[@]}"; do
        printf '[DRY-RUN] %s: %s log=%s/%s.log\n' \
            "${task_id}" "$(format_task_command "${task_id}")" "${LOG_ROOT}" "${task_id}"
    done
    exit 0
fi

for required_command in nohup setsid pgrep ps awk sed cp; do
    if ! command -v "${required_command}" >/dev/null 2>&1; then
        printf 'ERROR: required Linux command not found: %s\n' "${required_command}" >&2
        exit 2
    fi
done

setsid_help="$(LC_ALL=C setsid --help 2>&1 || true)"
if [[ "${setsid_help}" != *"--wait"* ]]; then
    printf 'ERROR: setsid must support --wait so experiment exit status remains observable\n' >&2
    exit 2
fi
if [[ "${setsid_help}" == *"--fork"* ]]; then
    SETSID_MODE="fork-wait"
else
    SETSID_MODE="wait"
fi

if [[ ! -x "${PYTHON}" ]]; then
    printf 'ERROR: Python executable not found: %s\n' "${PYTHON}" >&2
    exit 2
fi
if [[ ! -f "${UNIFIED_RUNNER}" ]]; then
    printf 'ERROR: unified runner not found: %s\n' "${UNIFIED_RUNNER}" >&2
    exit 2
fi
if [[ -e "${RESULT_ROOT}" || -e "${LOG_ROOT}" ]]; then
    printf 'ERROR: timestamp collision; refusing to reuse existing run paths: %s\n' "${TIMESTAMP}" >&2
    exit 2
fi

mkdir -p "${PROJECT_ROOT}/outputs/runs" "${PROJECT_ROOT}/outputs/parallel_mode_runs"
mkdir "${RESULT_ROOT}" "${LOG_ROOT}"
: >"${RUNNER_LOG}"
printf 'task\tpid\tpgid\tstatus\tevent_time\telapsed_seconds\texit_code\tlog_file\toutput_dir\n' >"${PID_FILE}"

log_message() {
    local message="$1"
    printf '%s\n' "${message}"
    printf '%s\n' "${message}" >>"${RUNNER_LOG}"
}

pid_matches_experiment() {
    local pid="$1"
    local task_id="$2"
    local output_dir="$3"
    local dataset
    local mode
    local command_line
    dataset="$(task_dataset "${task_id}")"
    mode="$(task_mode "${task_id}")"

    if ! kill -0 "${pid}" 2>/dev/null; then
        return 1
    fi
    command_line="$(LC_ALL=C ps -ww -o args= -p "${pid}" 2>/dev/null)" || return 1
    [[ "${command_line}" == *"${UNIFIED_RUNNER}"* \
        && "${command_line}" == *"--only ${dataset}"* \
        && "${command_line}" == *"--info-sharing ${mode}"* \
        && "${command_line}" == *"--output-dir ${output_dir}"* ]]
}

resolve_experiment_pid() {
    local supervisor_pid="$1"
    local task_id="$2"
    local output_dir="$3"
    local attempt
    local candidate_pid

    for ((attempt = 0; attempt < PID_RESOLUTION_ATTEMPTS; attempt++)); do
        while IFS= read -r candidate_pid; do
            if [[ "${candidate_pid}" =~ ^[0-9]+$ ]] \
                && pid_matches_experiment "${candidate_pid}" "${task_id}" "${output_dir}"; then
                printf '%s\n' "${candidate_pid}"
                return 0
            fi
        done < <(pgrep -P "${supervisor_pid}" 2>/dev/null || true)

        if [[ "${SETSID_MODE}" == "wait" ]] \
            && pid_matches_experiment "${supervisor_pid}" "${task_id}" "${output_dir}"; then
            printf '%s\n' "${supervisor_pid}"
            return 0
        fi
        if ! kill -0 "${supervisor_pid}" 2>/dev/null; then
            break
        fi
        sleep "${PID_RESOLUTION_INTERVAL_SECONDS}"
    done
    return 1
}

resolve_process_group() {
    local pid="$1"
    local pgid

    pgid="$(LC_ALL=C ps -o pgid= -p "${pid}" 2>/dev/null \
        | awk 'NR == 1 {gsub(/[[:space:]]/, "", $0); print; exit}')"
    if [[ ! "${pgid}" =~ ^[0-9]+$ ]] || [[ "${pgid}" != "${pid}" ]]; then
        return 1
    fi
    printf '%s\n' "${pgid}"
}

process_group_alive() {
    local pgid="$1"
    kill -0 -- "-${pgid}" 2>/dev/null
}

append_pid_event() {
    local task_id="$1"
    local status="$2"
    local elapsed="${3:--}"
    local exit_code="${4:--}"
    local index
    index="$(task_index "${task_id}")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${task_id}" "${PIDS[${index}]:--}" "${PGIDS[${index}]:--}" "${status}" \
        "$(iso_timestamp)" "${elapsed}" "${exit_code}" "${LOG_FILES[${index}]}" \
        "${OUTPUT_DIRS[${index}]}" >>"${PID_FILE}"
}

expected_csv() {
    local task_id="$1"
    local dataset
    local dataset_number
    local index
    dataset="$(task_dataset "${task_id}")"
    dataset_number="${dataset#d}"
    index="$(task_index "${task_id}")"
    printf '%s/results/dataset%s_%s_results.csv\n' \
        "${OUTPUT_DIRS[${index}]}" "${dataset_number}" "$(task_mode "${task_id}")"
}

validate_task_output() {
    local task_id="$1"
    local csv_path
    VALIDATION_ERROR=""
    csv_path="$(expected_csv "${task_id}")"
    if [[ ! -f "${csv_path}" ]]; then
        VALIDATION_ERROR="expected result CSV not found: ${csv_path}"
        return 1
    fi
    return 0
}

terminate_task_index() {
    local index="$1"
    local pgid="${PGIDS[${index}]:-}"
    local pid="${PIDS[${index}]:-}"
    local supervisor_pid="${SUPERVISOR_PIDS[${index}]:-}"

    if [[ "${pgid}" =~ ^[0-9]+$ ]] && process_group_alive "${pgid}"; then
        kill -TERM -- "-${pgid}" 2>/dev/null || true
        sleep 1
        if process_group_alive "${pgid}"; then
            kill -KILL -- "-${pgid}" 2>/dev/null || true
        fi
    elif [[ "${pid}" =~ ^[0-9]+$ ]]; then
        kill -TERM "${pid}" 2>/dev/null || true
        sleep 1
        kill -KILL "${pid}" 2>/dev/null || true
    fi
    if [[ "${supervisor_pid}" =~ ^[0-9]+$ ]]; then
        kill -TERM "${supervisor_pid}" 2>/dev/null || true
        kill -KILL "${supervisor_pid}" 2>/dev/null || true
        wait "${supervisor_pid}" 2>/dev/null || true
    fi
}

cleanup_after_signal() {
    local reason="$1"
    local index
    local deadline
    local any_alive
    local now

    if ((CLEANUP_STARTED == 1)); then
        return
    fi
    CLEANUP_STARTED=1
    trap - INT TERM
    set +e

    log_message "[ABORT] ${reason}"
    for index in "${!TASK_IDS[@]}"; do
        if [[ "${STATUSES[${index}]:-not_started}" != "running" ]]; then
            continue
        fi
        if process_group_alive "${PGIDS[${index}]}"; then
            log_message "[TERM] ${TASK_IDS[${index}]} pgid=${PGIDS[${index}]}"
            kill -TERM -- "-${PGIDS[${index}]}" 2>/dev/null
        fi
    done

    deadline=$((SECONDS + TERMINATION_GRACE_SECONDS))
    while ((SECONDS < deadline)); do
        any_alive=0
        for index in "${!TASK_IDS[@]}"; do
            if [[ "${STATUSES[${index}]:-not_started}" == "running" ]] \
                && process_group_alive "${PGIDS[${index}]}"; then
                any_alive=1
                break
            fi
        done
        if ((any_alive == 0)); then
            break
        fi
        sleep 1
    done

    now="$(date +%s)"
    for index in "${!TASK_IDS[@]}"; do
        if [[ "${STATUSES[${index}]:-not_started}" != "running" ]]; then
            continue
        fi
        if process_group_alive "${PGIDS[${index}]}"; then
            log_message "[KILL] ${TASK_IDS[${index}]} pgid=${PGIDS[${index}]}"
            kill -KILL -- "-${PGIDS[${index}]}" 2>/dev/null
        fi
        wait "${SUPERVISOR_PIDS[${index}]}" 2>/dev/null || true
        DURATIONS["${index}"]=$((now - START_TIMES[${index}]))
        STATUSES["${index}"]="interrupted"
        EXIT_CODES["${index}"]="-"
        append_pid_event "${TASK_IDS[${index}]}" "interrupted" "${DURATIONS[${index}]}" "-"
    done
}

handle_signal() {
    local signal_name="$1"
    local exit_code="$2"
    cleanup_after_signal "received ${signal_name}"
    exit "${exit_code}"
}

launch_task() {
    local index="$1"
    local task_id="${TASK_IDS[${index}]}"
    local log_file="${LOG_ROOT}/${task_id}.log"
    local output_dir="${RESULT_ROOT}/${task_id}"
    local dataset
    local mode
    dataset="$(task_dataset "${task_id}")"
    mode="$(task_mode "${task_id}")"

    LOG_FILES["${index}"]="${log_file}"
    OUTPUT_DIRS["${index}"]="${output_dir}"
    START_TIMES["${index}"]="$(date +%s)"
    DURATIONS["${index}"]="-"
    STATUSES["${index}"]="starting"
    EXIT_CODES["${index}"]="-"
    FAILURE_REASONS["${index}"]=""

    if [[ "${SETSID_MODE}" == "fork-wait" ]]; then
        nohup setsid --fork --wait "${PYTHON}" "${UNIFIED_RUNNER}" \
            --only "${dataset}" \
            --info-sharing "${mode}" \
            --output-dir "${output_dir}" \
            >"${log_file}" 2>&1 </dev/null &
    else
        nohup setsid --wait "${PYTHON}" "${UNIFIED_RUNNER}" \
            --only "${dataset}" \
            --info-sharing "${mode}" \
            --output-dir "${output_dir}" \
            >"${log_file}" 2>&1 </dev/null &
    fi
    SUPERVISOR_PIDS["${index}"]=$!

    if ! PIDS["${index}"]="$(resolve_experiment_pid \
        "${SUPERVISOR_PIDS[${index}]}" "${task_id}" "${output_dir}")"; then
        FAILURE_REASONS["${index}"]="failed to resolve Python PID"
        STATUSES["${index}"]="failed"
        EXIT_CODES["${index}"]="2"
        terminate_task_index "${index}"
        append_pid_event "${task_id}" "failed" 0 "2"
        log_message "[FAILED] ${task_id} elapsed=0s exit=2 reason=${FAILURE_REASONS[${index}]} log=${log_file}"
        return 1
    fi
    if ! PGIDS["${index}"]="$(resolve_process_group "${PIDS[${index}]}")"; then
        FAILURE_REASONS["${index}"]="invalid Python process group for pid=${PIDS[${index}]}"
        STATUSES["${index}"]="failed"
        EXIT_CODES["${index}"]="2"
        terminate_task_index "${index}"
        append_pid_event "${task_id}" "failed" 0 "2"
        log_message "[FAILED] ${task_id} elapsed=0s exit=2 reason=${FAILURE_REASONS[${index}]} log=${log_file}"
        return 1
    fi

    STATUSES["${index}"]="running"
    LAUNCHED_INDICES+=("${index}")
    append_pid_event "${task_id}" "started"
    log_message "[LAUNCHED] ${task_id} pid=${PIDS[${index}]} pgid=${PGIDS[${index}]} supervisor_pid=${SUPERVISOR_PIDS[${index}]} log=${log_file}"
    return 0
}

collect_results() {
    local dataset
    local mode
    local task_id
    local index
    local canonical_dir
    local src_csv
    local dst_csv
    local without_csv
    local with_csv
    local combined_csv

    for dataset in "${DATASET_IDS[@]}"; do
        canonical_dir="${RESULT_ROOT}/${dataset}/results"
        mkdir -p "${canonical_dir}"
        for mode in without with; do
            task_id="${dataset}_${mode}"
            index="$(task_index "${task_id}")"
            src_csv="$(expected_csv "${task_id}")"
            dst_csv="${canonical_dir}/$(basename "${src_csv}")"
            if [[ "${STATUSES[${index}]}" == "succeeded" && -f "${src_csv}" ]]; then
                cp "${src_csv}" "${dst_csv}"
                log_message "[COLLECT] ${task_id} -> ${dst_csv}"
            else
                log_message "[COLLECT] missing ${task_id} status=${STATUSES[${index}]:-not_started} source=${src_csv}"
            fi
        done

        without_csv="${canonical_dir}/dataset${dataset#d}_without_results.csv"
        with_csv="${canonical_dir}/dataset${dataset#d}_with_results.csv"
        combined_csv="${canonical_dir}/dataset${dataset#d}_results.csv"
        if [[ -f "${without_csv}" && -f "${with_csv}" ]]; then
            awk 'FNR == 1 && NR != 1 {next} {print}' "${without_csv}" "${with_csv}" >"${combined_csv}"
            log_message "[COLLECT] combined ${dataset} -> ${combined_csv}"
        else
            log_message "[COLLECT] skipped combined ${dataset}; one or more mode CSVs missing"
        fi
    done
}

trap 'handle_signal SIGINT 130' INT
trap 'handle_signal SIGTERM 143' TERM

log_message "[START] result root: ${RESULT_ROOT}"
log_message "[START] log root: ${LOG_ROOT}"
log_message "[START] max jobs: ${MAX_JOBS}"
log_message "[START] setsid mode: ${SETSID_MODE}"
for task_id in "${TASK_IDS[@]}"; do
    log_message "[QUEUED] ${task_id}"
done

next_index=0
running_count=0
completed_count=0
failed_count=0
success_count=0

while ((completed_count < ${#TASK_IDS[@]})); do
    while ((running_count < MAX_JOBS && next_index < ${#TASK_IDS[@]})); do
        if launch_task "${next_index}"; then
            running_count=$((running_count + 1))
        else
            completed_count=$((completed_count + 1))
            failed_count=$((failed_count + 1))
        fi
        next_index=$((next_index + 1))
    done

    made_progress=0
    for index in "${!TASK_IDS[@]}"; do
        task_id="${TASK_IDS[${index}]}"
        if [[ "${STATUSES[${index}]:-not_started}" != "running" ]]; then
            continue
        fi
        if kill -0 "${PIDS[${index}]}" 2>/dev/null; then
            continue
        fi

        if wait "${SUPERVISOR_PIDS[${index}]}"; then
            process_exit_code=0
        else
            process_exit_code=$?
        fi
        finished_at="$(date +%s)"
        DURATIONS["${index}"]=$((finished_at - START_TIMES[${index}]))
        EXIT_CODES["${index}"]="${process_exit_code}"

        if ((process_exit_code != 0)); then
            STATUSES["${index}"]="failed"
            FAILURE_REASONS["${index}"]="process exited with code ${process_exit_code}"
        elif ! validate_task_output "${task_id}"; then
            STATUSES["${index}"]="failed"
            FAILURE_REASONS["${index}"]="${VALIDATION_ERROR}"
        else
            STATUSES["${index}"]="succeeded"
        fi

        append_pid_event "${task_id}" "${STATUSES[${index}]}" "${DURATIONS[${index}]}" "${EXIT_CODES[${index}]}"
        if [[ "${STATUSES[${index}]}" == "succeeded" ]]; then
            log_message "[SUCCEEDED] ${task_id} elapsed=${DURATIONS[${index}]}s"
            success_count=$((success_count + 1))
        else
            log_message "[FAILED] ${task_id} elapsed=${DURATIONS[${index}]}s exit=${EXIT_CODES[${index}]} reason=${FAILURE_REASONS[${index}]} log=${LOG_FILES[${index}]}"
            failed_count=$((failed_count + 1))
        fi
        completed_count=$((completed_count + 1))
        running_count=$((running_count - 1))
        made_progress=1
    done

    if ((completed_count < ${#TASK_IDS[@]} && made_progress == 0)); then
        sleep "${MONITOR_INTERVAL_SECONDS}"
    fi
done

collect_results

log_message "[SUMMARY] succeeded=${success_count} failed=${failed_count}"
log_message "Result root: ${RESULT_ROOT}"
log_message "Log root: ${LOG_ROOT}"

if ((failed_count > 0)); then
    exit 1
fi
exit 0
