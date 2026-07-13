#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
readonly PYTHON="${PROJECT_ROOT}/.venv/bin/python"
readonly UNIFIED_RUNNER="${PROJECT_ROOT}/scripts/run_unified_d1_d6.py"
readonly TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
readonly RESULT_ROOT="${PROJECT_ROOT}/outputs/runs/${TIMESTAMP}"
readonly LOG_ROOT="${PROJECT_ROOT}/outputs/parallel_runs/${TIMESTAMP}"
readonly PID_FILE="${LOG_ROOT}/pids.tsv"
readonly RESOURCE_LOG="${LOG_ROOT}/resources.log"
readonly RUNNER_LOG="${LOG_ROOT}/runner.log"
readonly MONITOR_INTERVAL_SECONDS=30
readonly TERMINATION_GRACE_SECONDS=10
readonly PID_RESOLUTION_ATTEMPTS=50
readonly PID_RESOLUTION_INTERVAL_SECONDS=0.1
readonly MEMORY_WARNING_BYTES=4294967296
readonly DISABLE_COMPAT_RESULTS_COPY_ENV="RFE_DISABLE_COMPAT_RESULTS_COPY"
readonly ALL_DATASETS=(d1 d2 d3 d4 d5 d6)

DRY_RUN=0
SETSID_MODE=""
ONLY_VALUES=()
DATASETS=()

usage() {
    printf 'Usage: bash scripts/parallel_runner.sh [--only d1[,d2...]]... [--dry-run]\n'
}

fail_usage() {
    printf 'ERROR: %s\n' "$1" >&2
    usage >&2
    exit 2
}

while (($# > 0)); do
    case "$1" in
        --only)
            if (($# < 2)); then
                fail_usage "--only requires a dataset value"
            fi
            ONLY_VALUES+=("$2")
            shift
            ;;
        --only=*)
            ONLY_VALUES+=("${1#--only=}")
            ;;
        --dry-run)
            DRY_RUN=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail_usage "unknown argument: $1"
            ;;
    esac
    shift
done

select_datasets() {
    local option_index
    local token_index
    local requested_index
    local candidate_index
    local raw_value
    local token
    local normalized
    local candidate
    local requested=()
    local tokens=()

    if ((${#ONLY_VALUES[@]} == 0)); then
        DATASETS=("${ALL_DATASETS[@]}")
        return
    fi

    for ((option_index = 0; option_index < ${#ONLY_VALUES[@]}; option_index++)); do
        raw_value="${ONLY_VALUES[${option_index}]}"
        IFS=',' read -r -a tokens <<<"${raw_value}"
        for ((token_index = 0; token_index < ${#tokens[@]}; token_index++)); do
            token="${tokens[${token_index}]}"
            token="${token#"${token%%[![:space:]]*}"}"
            token="${token%"${token##*[![:space:]]}"}"
            if [[ -z "${token}" ]]; then
                continue
            fi
            case "${token}" in
                d1|D1) normalized="d1" ;;
                d2|D2) normalized="d2" ;;
                d3|D3) normalized="d3" ;;
                d4|D4) normalized="d4" ;;
                d5|D5) normalized="d5" ;;
                d6|D6) normalized="d6" ;;
                *) fail_usage "unknown dataset id: ${token}" ;;
            esac
            requested+=("${normalized}")
        done
    done

    if ((${#requested[@]} == 0)); then
        DATASETS=("${ALL_DATASETS[@]}")
        return
    fi

    for ((candidate_index = 0; candidate_index < ${#ALL_DATASETS[@]}; candidate_index++)); do
        candidate="${ALL_DATASETS[${candidate_index}]}"
        for ((requested_index = 0; requested_index < ${#requested[@]}; requested_index++)); do
            if [[ "${requested[${requested_index}]}" == "${candidate}" ]]; then
                DATASETS+=("${candidate}")
                break
            fi
        done
    done
}

select_datasets

if [[ "$(pwd -P)" != "${PROJECT_ROOT}" ]]; then
    printf 'ERROR: run this script from the project root: %s\n' "${PROJECT_ROOT}" >&2
    exit 2
fi

print_dataset_command() {
    local dataset="$1"
    printf '[%s] ./.venv/bin/python scripts/run_unified_d1_d6.py --only %s --output-dir %s/%s\n' \
        "${dataset}" "${dataset}" "${RESULT_ROOT}" "${dataset}"
}

if ((DRY_RUN == 1)); then
    printf '[DRY-RUN] result root: %s\n' "${RESULT_ROOT}"
    printf '[DRY-RUN] log root: %s\n' "${LOG_ROOT}"
    printf '[DRY-RUN] PID file: %s\n' "${PID_FILE}"
    for dataset in "${DATASETS[@]}"; do
        print_dataset_command "${dataset}"
    done
    exit 0
fi

for required_command in nohup setsid pgrep ps top free awk sed; do
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

mkdir -p "${PROJECT_ROOT}/outputs/runs" "${PROJECT_ROOT}/outputs/parallel_runs"
mkdir "${RESULT_ROOT}" "${LOG_ROOT}"

: >"${RUNNER_LOG}"
: >"${RESOURCE_LOG}"
printf 'dataset\tpid\tpgid\tstatus\tevent_time\telapsed_seconds\texit_code\tlog_file\toutput_dir\n' >"${PID_FILE}"

export "${DISABLE_COMPAT_RESULTS_COPY_ENV}=1"

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

CLEANUP_STARTED=0
VALIDATION_ERROR=""

iso_timestamp() {
    date '+%Y-%m-%dT%H:%M:%S%z'
}

dataset_index() {
    local dataset="$1"
    local index
    for ((index = 0; index < ${#DATASETS[@]}; index++)); do
        if [[ "${DATASETS[${index}]}" == "${dataset}" ]]; then
            printf '%s\n' "${index}"
            return 0
        fi
    done
    return 1
}

log_message() {
    local message="$1"
    printf '%s\n' "${message}"
    printf '%s\n' "${message}" >>"${RUNNER_LOG}"
}

pid_matches_experiment() {
    local pid="$1"
    local dataset="$2"
    local output_dir="$3"
    local command_line

    if ! kill -0 "${pid}" 2>/dev/null; then
        return 1
    fi
    command_line="$(LC_ALL=C ps -ww -o args= -p "${pid}" 2>/dev/null)" || return 1
    [[ "${command_line}" == *"${UNIFIED_RUNNER}"* \
        && "${command_line}" == *"--only ${dataset}"* \
        && "${command_line}" == *"--output-dir ${output_dir}"* ]]
}

resolve_experiment_pid() {
    local supervisor_pid="$1"
    local dataset="$2"
    local output_dir="$3"
    local attempt
    local candidate_pid

    for ((attempt = 0; attempt < PID_RESOLUTION_ATTEMPTS; attempt++)); do
        while IFS= read -r candidate_pid; do
            if [[ "${candidate_pid}" =~ ^[0-9]+$ ]] \
                && pid_matches_experiment "${candidate_pid}" "${dataset}" "${output_dir}"; then
                printf '%s\n' "${candidate_pid}"
                return 0
            fi
        done < <(pgrep -P "${supervisor_pid}" 2>/dev/null || true)

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

terminate_failed_launch() {
    local supervisor_pid="$1"
    local dataset="$2"
    local output_dir="$3"
    local candidate_pid
    local candidate_pgid
    local candidates=()

    while IFS= read -r candidate_pid; do
        if [[ ! "${candidate_pid}" =~ ^[0-9]+$ ]]; then
            continue
        fi
        candidates+=("${candidate_pid}")
        candidate_pgid="$(LC_ALL=C ps -o pgid= -p "${candidate_pid}" 2>/dev/null \
            | awk 'NR == 1 {gsub(/[[:space:]]/, "", $0); print; exit}')" \
            || candidate_pgid=""
        if [[ "${candidate_pgid}" =~ ^[0-9]+$ ]] && [[ "${candidate_pgid}" == "${candidate_pid}" ]]; then
            kill -TERM -- "-${candidate_pgid}" 2>/dev/null || true
        else
            kill -TERM "${candidate_pid}" 2>/dev/null || true
        fi
    done < <(pgrep -P "${supervisor_pid}" 2>/dev/null || true)

    if [[ "${SETSID_MODE}" == "wait" ]] \
        && pid_matches_experiment "${supervisor_pid}" "${dataset}" "${output_dir}"; then
        candidates+=("${supervisor_pid}")
        kill -TERM -- "-${supervisor_pid}" 2>/dev/null || true
    fi

    sleep 1
    for candidate_pid in "${candidates[@]}"; do
        if kill -0 "${candidate_pid}" 2>/dev/null; then
            kill -KILL -- "-${candidate_pid}" 2>/dev/null \
                || kill -KILL "${candidate_pid}" 2>/dev/null \
                || true
        fi
    done
    kill -TERM "${supervisor_pid}" 2>/dev/null || true
    kill -KILL "${supervisor_pid}" 2>/dev/null || true
    wait "${supervisor_pid}" 2>/dev/null || true
}

append_pid_event() {
    local dataset="$1"
    local status="$2"
    local elapsed="${3:--}"
    local exit_code="${4:--}"
    local index
    index="$(dataset_index "${dataset}")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${dataset}" "${PIDS[${index}]}" "${PGIDS[${index}]}" "${status}" \
        "$(iso_timestamp)" "${elapsed}" "${exit_code}" "${LOG_FILES[${index}]}" \
        "${OUTPUT_DIRS[${index}]}" >>"${PID_FILE}"
}

expected_csvs() {
    local dataset="$1"
    local dataset_number="${dataset#d}"
    local index
    index="$(dataset_index "${dataset}")"
    if ((dataset_number <= 3)); then
        printf '%s/results/dataset%s_results.csv\n' "${OUTPUT_DIRS[${index}]}" "${dataset_number}"
    else
        printf '%s/results/dataset%s_without_results.csv\n' "${OUTPUT_DIRS[${index}]}" "${dataset_number}"
        printf '%s/results/dataset%s_with_results.csv\n' "${OUTPUT_DIRS[${index}]}" "${dataset_number}"
    fi
}

sample_resources() {
    local sample_time
    local free_h
    local available_h
    local available_bytes
    local warning

    sample_time="$(iso_timestamp)"
    free_h="$(LC_ALL=C free -h)"
    available_h="$(awk '/^Mem:/ {print $7; exit}' <<<"${free_h}")"
    available_bytes="$(LC_ALL=C free -b | awk '/^Mem:/ {print $7; exit}')"

    {
        printf '\n===== %s =====\n' "${sample_time}"
        LC_ALL=C top -bn1 | sed -n '1,5p'
        printf '%s\n' "${free_h}"
    } >>"${RESOURCE_LOG}"

    if [[ "${available_bytes}" =~ ^[0-9]+$ ]] && ((available_bytes < MEMORY_WARNING_BYTES)); then
        warning="[WARNING] Available memory below 4GB: ${available_h:-unknown}"
        printf '%s\n' "${warning}" >>"${RESOURCE_LOG}"
        log_message "${warning}"
    fi
}

validate_summary_log() {
    local dataset="$1"
    local expected_rows=1
    local awk_status
    local index
    index="$(dataset_index "${dataset}")"
    if [[ "${dataset}" =~ ^d[4-6]$ ]]; then
        expected_rows=2
    fi

    if awk -F '|' -v expected="${expected_rows}" '
        function trim(value) {
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
            return value
        }
        /^D[1-6][[:space:]]*\|/ {
            count += 1
            if (tolower(trim($3)) == "missing" || tolower(trim($4)) == "missing") {
                found_missing = 1
            }
        }
        END {
            if (count != expected) {
                exit 2
            }
            if (found_missing) {
                exit 3
            }
        }
    ' "${LOG_FILES[${index}]}"; then
        return 0
    else
        awk_status=$?
    fi

    if ((awk_status == 2)); then
        VALIDATION_ERROR="summary row count mismatch (expected ${expected_rows})"
    else
        VALIDATION_ERROR="summary contains missing result path or row count"
    fi
    return 1
}

validate_result_csv() {
    local csv_path="$1"
    local validation_output

    if [[ ! -f "${csv_path}" ]]; then
        VALIDATION_ERROR="expected result CSV not found: ${csv_path}"
        return 1
    fi

    if validation_output="$("${PYTHON}" - "${csv_path}" 2>&1 <<'PY'
import csv
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    if not reader.fieldnames or "error" not in reader.fieldnames:
        raise SystemExit(f"result CSV is missing required error column: {path}")
    failed_rows = [
        str(index)
        for index, row in enumerate(reader, start=2)
        if str(row.get("error") or "").strip()
    ]
if failed_rows:
    raise SystemExit(
        f"result CSV contains non-empty error values at rows {','.join(failed_rows)}: {path}"
    )
PY
    )"; then
        return 0
    fi

    VALIDATION_ERROR="${validation_output//$'\n'/; }"
    return 1
}

validate_dataset_outputs() {
    local dataset="$1"
    local csv_path

    VALIDATION_ERROR=""
    if ! validate_summary_log "${dataset}"; then
        return 1
    fi

    while IFS= read -r csv_path; do
        if ! validate_result_csv "${csv_path}"; then
            return 1
        fi
    done < <(expected_csvs "${dataset}")
    return 0
}

process_group_alive() {
    local pgid="$1"
    kill -0 -- "-${pgid}" 2>/dev/null
}

mark_dataset_results_interrupted() {
    local dataset="$1"
    local csv_path
    local interrupted_path

    while IFS= read -r csv_path; do
        if [[ ! -e "${csv_path}" ]]; then
            continue
        fi
        interrupted_path="${csv_path}.INTERRUPTED"
        if [[ -e "${interrupted_path}" ]]; then
            log_message "[WARNING] interrupted marker already exists; leaving source unchanged: ${interrupted_path}"
            continue
        fi
        mv -- "${csv_path}" "${interrupted_path}"
        log_message "[INTERRUPTED] ${csv_path} -> ${interrupted_path}"
    done < <(expected_csvs "${dataset}")
}

cleanup_after_launch_failure() {
    local reason="$1"
    local dataset
    local index
    local launched_position
    local deadline
    local any_alive
    local now

    if ((CLEANUP_STARTED == 1)); then
        return
    fi
    CLEANUP_STARTED=1
    trap - INT TERM
    set +e

    log_message "[LAUNCH-ABORT] ${reason}"
    for ((launched_position = 0; launched_position < ${#LAUNCHED_INDICES[@]}; launched_position++)); do
        index="${LAUNCHED_INDICES[${launched_position}]}"
        dataset="${DATASETS[${index}]}"
        if process_group_alive "${PGIDS[${index}]}"; then
            log_message "[TERM] ${dataset} pgid=${PGIDS[${index}]}"
            kill -TERM -- "-${PGIDS[${index}]}" 2>/dev/null
        fi
    done

    deadline=$((SECONDS + TERMINATION_GRACE_SECONDS))
    while ((SECONDS < deadline)); do
        any_alive=0
        for ((launched_position = 0; launched_position < ${#LAUNCHED_INDICES[@]}; launched_position++)); do
            index="${LAUNCHED_INDICES[${launched_position}]}"
            if process_group_alive "${PGIDS[${index}]}"; then
                any_alive=1
                break
            fi
        done
        if ((any_alive == 0)); then
            break
        fi
        sleep 1
    done

    for ((launched_position = 0; launched_position < ${#LAUNCHED_INDICES[@]}; launched_position++)); do
        index="${LAUNCHED_INDICES[${launched_position}]}"
        dataset="${DATASETS[${index}]}"
        if process_group_alive "${PGIDS[${index}]}"; then
            log_message "[KILL] ${dataset} pgid=${PGIDS[${index}]}"
            kill -KILL -- "-${PGIDS[${index}]}" 2>/dev/null
        fi
        wait "${SUPERVISOR_PIDS[${index}]}" 2>/dev/null
        now="$(date +%s)"
        DURATIONS["${index}"]=$((now - START_TIMES[${index}]))
        STATUSES["${index}"]="interrupted"
        EXIT_CODES["${index}"]="-"
        append_pid_event "${dataset}" "interrupted" "${DURATIONS[${index}]}" "-"
        mark_dataset_results_interrupted "${dataset}"
    done
}

cleanup_after_signal() {
    local reason="$1"
    local dataset
    local index
    local launched_position
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
    for index in "${!DATASETS[@]}"; do
        dataset="${DATASETS[${index}]}"
        if [[ "${STATUSES[${index}]:-not_started}" != "running" ]]; then
            continue
        fi
        if process_group_alive "${PGIDS[${index}]}"; then
            log_message "[TERM] ${dataset} pgid=${PGIDS[${index}]}"
            kill -TERM -- "-${PGIDS[${index}]}" 2>/dev/null
        fi
    done

    deadline=$((SECONDS + TERMINATION_GRACE_SECONDS))
    while ((SECONDS < deadline)); do
        any_alive=0
        for index in "${!DATASETS[@]}"; do
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

    for index in "${!DATASETS[@]}"; do
        dataset="${DATASETS[${index}]}"
        if [[ "${STATUSES[${index}]:-not_started}" != "running" ]]; then
            continue
        fi
        if process_group_alive "${PGIDS[${index}]}"; then
            log_message "[KILL] ${dataset} pgid=${PGIDS[${index}]}"
            kill -KILL -- "-${PGIDS[${index}]}" 2>/dev/null
        fi
        wait "${SUPERVISOR_PIDS[${index}]}" 2>/dev/null
        now="$(date +%s)"
        DURATIONS["${index}"]=$((now - START_TIMES[${index}]))
        STATUSES["${index}"]="interrupted"
        EXIT_CODES["${index}"]="-"
        append_pid_event "${dataset}" "interrupted" "${DURATIONS[${index}]}" "-"
    done

    for ((launched_position = 0; launched_position < ${#LAUNCHED_INDICES[@]}; launched_position++)); do
        index="${LAUNCHED_INDICES[${launched_position}]}"
        if [[ "${STATUSES[${index}]}" == "succeeded" ]]; then
            continue
        fi
        mark_dataset_results_interrupted "${DATASETS[${index}]}"
    done
}

handle_signal() {
    local signal_name="$1"
    local exit_code="$2"
    cleanup_after_signal "received ${signal_name}"
    exit "${exit_code}"
}

trap 'handle_signal SIGINT 130' INT
trap 'handle_signal SIGTERM 143' TERM

log_message "[START] result root: ${RESULT_ROOT}"
log_message "[START] log root: ${LOG_ROOT}"
log_message "[START] setsid mode: ${SETSID_MODE}"

for index in "${!DATASETS[@]}"; do
    dataset="${DATASETS[${index}]}"
    LOG_FILES["${index}"]="${LOG_ROOT}/${dataset}.log"
    OUTPUT_DIRS["${index}"]="${RESULT_ROOT}/${dataset}"
    START_TIMES["${index}"]="$(date +%s)"
    DURATIONS["${index}"]="-"
    STATUSES["${index}"]="starting"
    EXIT_CODES["${index}"]="-"
    FAILURE_REASONS["${index}"]=""

    if [[ "${SETSID_MODE}" == "fork-wait" ]]; then
        nohup setsid --fork --wait "${PYTHON}" "${UNIFIED_RUNNER}" \
            --only "${dataset}" \
            --output-dir "${OUTPUT_DIRS[${index}]}" \
            >"${LOG_FILES[${index}]}" 2>&1 </dev/null &
    else
        nohup setsid --wait "${PYTHON}" "${UNIFIED_RUNNER}" \
            --only "${dataset}" \
            --output-dir "${OUTPUT_DIRS[${index}]}" \
            >"${LOG_FILES[${index}]}" 2>&1 </dev/null &
    fi
    SUPERVISOR_PIDS["${index}"]=$!

    if ! PIDS["${index}"]="$(resolve_experiment_pid \
        "${SUPERVISOR_PIDS[${index}]}" "${dataset}" "${OUTPUT_DIRS[${index}]}")"; then
        terminate_failed_launch \
            "${SUPERVISOR_PIDS[${index}]}" "${dataset}" "${OUTPUT_DIRS[${index}]}"
        cleanup_after_launch_failure "failed to resolve Python PID for ${dataset}"
        exit 2
    fi
    if ! PGIDS["${index}"]="$(resolve_process_group "${PIDS[${index}]}")"; then
        terminate_failed_launch \
            "${SUPERVISOR_PIDS[${index}]}" "${dataset}" "${OUTPUT_DIRS[${index}]}"
        cleanup_after_launch_failure "invalid Python process group for ${dataset}: pid=${PIDS[${index}]}"
        exit 2
    fi

    STATUSES["${index}"]="running"
    LAUNCHED_INDICES+=("${index}")
    append_pid_event "${dataset}" "started"
    log_message "[LAUNCHED] ${dataset} pid=${PIDS[${index}]} pgid=${PGIDS[${index}]} supervisor_pid=${SUPERVISOR_PIDS[${index}]} log=${LOG_FILES[${index}]}"
done

sample_resources

completed_count=0
while ((completed_count < ${#DATASETS[@]})); do
    for index in "${!DATASETS[@]}"; do
        dataset="${DATASETS[${index}]}"
        if [[ "${STATUSES[${index}]}" != "running" ]]; then
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
        elif ! validate_dataset_outputs "${dataset}"; then
            STATUSES["${index}"]="failed"
            FAILURE_REASONS["${index}"]="${VALIDATION_ERROR}"
        else
            STATUSES["${index}"]="succeeded"
        fi

        append_pid_event \
            "${dataset}" "${STATUSES[${index}]}" \
            "${DURATIONS[${index}]}" "${EXIT_CODES[${index}]}"
        if [[ "${STATUSES[${index}]}" == "succeeded" ]]; then
            log_message "[SUCCEEDED] ${dataset} elapsed=${DURATIONS[${index}]}s"
        else
            log_message "[WARNING] ${dataset} failed: ${FAILURE_REASONS[${index}]}; remaining datasets will continue"
        fi
        completed_count=$((completed_count + 1))
    done

    if ((completed_count < ${#DATASETS[@]})); then
        sleep "${MONITOR_INTERVAL_SECONDS}"
        sample_resources
    fi
done

success_count=0
failure_count=0
log_message ""
log_message "===== D1-D6 parallel run summary ====="
log_message "Successful datasets:"
for index in "${!DATASETS[@]}"; do
    dataset="${DATASETS[${index}]}"
    if [[ "${STATUSES[${index}]}" == "succeeded" ]]; then
        log_message "  ${dataset}: elapsed=${DURATIONS[${index}]}s output=${OUTPUT_DIRS[${index}]}"
        success_count=$((success_count + 1))
    fi
done
if ((success_count == 0)); then
    log_message "  (none)"
fi

log_message "Failed datasets:"
for index in "${!DATASETS[@]}"; do
    dataset="${DATASETS[${index}]}"
    if [[ "${STATUSES[${index}]}" == "failed" ]]; then
        log_message "  ${dataset}: exit_code=${EXIT_CODES[${index}]} reason=${FAILURE_REASONS[${index}]} log=${LOG_FILES[${index}]}"
        failure_count=$((failure_count + 1))
    fi
done
if ((failure_count == 0)); then
    log_message "  (none)"
fi

log_message "Result root: ${RESULT_ROOT}"
log_message "Log root: ${LOG_ROOT}"

if ((failure_count > 0)); then
    exit 1
fi
exit 0
