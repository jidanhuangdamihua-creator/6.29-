#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
readonly DEFAULT_UNIFIED_RUNNER="${PROJECT_ROOT}/scripts/run_unified_d1_d6.py"
readonly MAX_JOBS="${MAX_JOBS-6}"
readonly RESUME="${RESUME-0}"
readonly DRY_RUN="${DRY_RUN-0}"
readonly PROBE="${PROBE-0}"
readonly PUBLISH_GLOBAL="${PUBLISH_GLOBAL-1}"
readonly DATASETS="${DATASETS-d1,d2,d3,d4,d5,d6}"
readonly MODES="${MODES-without,with}"
readonly HORIZONS="${HORIZONS-1,2,3,4,5}"
readonly SEEDS="${SEEDS-42,43,44,45,46}"
readonly TERMINATION_GRACE_SECONDS="${TERMINATION_GRACE_SECONDS-10}"
readonly RUN_STAMP="$(date '+%Y%m%d_%H%M%S')"

fail_usage() {
    printf 'ERROR: %s\n' "$1" >&2
    exit 2
}

validate_bool() {
    local name="$1"
    local value="$2"
    if [[ "${value}" != "0" && "${value}" != "1" ]]; then
        fail_usage "${name} must be 0 or 1"
    fi
}

if (($# > 0)); then
    fail_usage "parallel_mode_runner.sh accepts configuration through environment variables"
fi
if [[ ! "${MAX_JOBS}" =~ ^([1-9]|1[0-2])$ ]]; then
    fail_usage "MAX_JOBS must be an integer from 1 through 12"
fi
if [[ ! "${TERMINATION_GRACE_SECONDS}" =~ ^[0-9]+$ ]]; then
    fail_usage "TERMINATION_GRACE_SECONDS must be a non-negative integer"
fi
validate_bool RESUME "${RESUME}"
validate_bool DRY_RUN "${DRY_RUN}"
validate_bool PROBE "${PROBE}"
validate_bool PUBLISH_GLOBAL "${PUBLISH_GLOBAL}"

validate_csv_selection() {
    local name="$1"
    local allowed="$2"
    shift 2
    local values=("$@")
    local item
    local existing
    local candidate
    ((${#values[@]} > 0)) || fail_usage "${name} must not be empty"
    for item in "${values[@]}"; do
        [[ -n "${item}" && ",${allowed}," == *",${item},"* ]] \
            || fail_usage "${name} contains invalid value: ${item:-<empty>}"
    done
    for item in "${values[@]}"; do
        existing=0
        for candidate in "${values[@]}"; do
            if [[ "${candidate}" == "${item}" ]]; then
                existing=$((existing + 1))
            fi
        done
        ((existing == 1)) || fail_usage "${name} contains duplicate value: ${item}"
    done
}

SELECTED_DATASETS=()
SELECTED_MODES=()
SELECTED_HORIZONS=()
SELECTED_SEEDS=()
IFS=',' read -r -a SELECTED_DATASETS <<<"${DATASETS}"
IFS=',' read -r -a SELECTED_MODES <<<"${MODES}"
IFS=',' read -r -a SELECTED_HORIZONS <<<"${HORIZONS}"
IFS=',' read -r -a SELECTED_SEEDS <<<"${SEEDS}"
validate_csv_selection DATASETS "d1,d2,d3,d4,d5,d6" "${SELECTED_DATASETS[@]}"
validate_csv_selection MODES "without,with" "${SELECTED_MODES[@]}"
validate_csv_selection HORIZONS "1,2,3,4,5" "${SELECTED_HORIZONS[@]}"
validate_csv_selection SEEDS "42,43,44,45,46" "${SELECTED_SEEDS[@]}"

if [[ -n "${RUN_ROOT+x}" ]]; then
    [[ -n "${RUN_ROOT}" ]] || fail_usage "RUN_ROOT must not be empty"
    if [[ "${RUN_ROOT}" = /* ]]; then
        FORMAL_RUN_ROOT="${RUN_ROOT}"
    else
        FORMAL_RUN_ROOT="${PROJECT_ROOT}/${RUN_ROOT}"
    fi
else
    FORMAL_RUN_ROOT="${PROJECT_ROOT}/outputs/runs/${RUN_STAMP}_formal"
fi
readonly FORMAL_RUN_ROOT

ALL_TASKS=(
    d5_without d5_with
    d6_without d6_with
    d4_without d4_with
    d3_without d3_with
    d2_without d2_with
    d1_without d1_with
)
if [[ "${PROBE}" == "1" ]]; then
    [[ "${MAX_JOBS}" == "4" ]] || fail_usage "PROBE=1 requires MAX_JOBS=4"
    [[ "${PUBLISH_GLOBAL}" == "0" ]] || fail_usage "PROBE=1 requires PUBLISH_GLOBAL=0"
    TASKS=(d1_without d1_with d2_without d2_with)
    SELECTED_DATASETS=(d1 d2)
    SELECTED_MODES=(without with)
    SELECTED_HORIZONS=(1 2 3 4 5)
    SELECTED_SEEDS=(42 43 44 45 46)
else
    [[ "${PUBLISH_GLOBAL}" == "1" ]] || fail_usage "full execution requires PUBLISH_GLOBAL=1"
    TASKS=()
    for task in "${ALL_TASKS[@]}"; do
        dataset="${task%%_*}"
        mode="${task#*_}"
        if [[ " ${SELECTED_DATASETS[*]} " == *" ${dataset} "* \
            && " ${SELECTED_MODES[*]} " == *" ${mode} "* ]]; then
            TASKS+=("${task}")
        fi
    done
fi
readonly TASK_COUNT="${#TASKS[@]}"
readonly CELL_COUNT="$((TASK_COUNT * ${#SELECTED_HORIZONS[@]} * ${#SELECTED_SEEDS[@]}))"

task_dataset() {
    printf '%s\n' "${1%%_*}"
}

task_mode() {
    printf '%s\n' "${1#*_}"
}

task_output_dir() {
    printf '%s/%s\n' "${FORMAL_RUN_ROOT}" "$1"
}

print_mode_command() {
    local task="$1"
    local dataset
    local mode
    dataset="$(task_dataset "${task}")"
    mode="$(task_mode "${task}")"
    printf '[MODE] %s %s %s --operation mode-worker --only %s --info-sharing %s --output-dir %s' \
        "${task}" "${PYTHON_BIN-${PROJECT_ROOT}/.venv/bin/python}" \
        "${UNIFIED_RUNNER_BIN-${DEFAULT_UNIFIED_RUNNER}}" \
        "${dataset}" "${mode}" "$(task_output_dir "${task}")"
    if [[ "${RESUME}" == "1" ]]; then
        printf ' --resume'
    fi
    printf '\n'
}

if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY-RUN] run_root=%s\n' "${FORMAL_RUN_ROOT}"
    printf '[CAPS] MAX_JOBS=%s publish_global=%s\n' \
        "${MAX_JOBS}" "${PUBLISH_GLOBAL}"
    printf '[SELECTION] datasets=%s modes=%s horizons=%s seeds=%s\n' \
        "${SELECTED_DATASETS[*]}" "${SELECTED_MODES[*]}" \
        "${SELECTED_HORIZONS[*]}" "${SELECTED_SEEDS[*]}"
    for task in "${TASKS[@]}"; do
        print_mode_command "${task}"
    done
    printf '[FORMAL PLAN] cells=%s unique=%s\n' "${CELL_COUNT}" "${CELL_COUNT}"
    exit 0
fi

resolve_python_bin() {
    local candidate
    if [[ -n "${PYTHON_BIN+x}" ]]; then
        [[ -n "${PYTHON_BIN}" ]] || return 1
        if candidate="$(command -v "${PYTHON_BIN}" 2>/dev/null)"; then
            printf '%s\n' "${candidate}"
            return 0
        fi
        return 1
    fi
    for candidate in "${PROJECT_ROOT}/.venv/bin/python" python3 python; do
        if [[ -x "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
        if candidate="$(command -v "${candidate}" 2>/dev/null)"; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    return 1
}

if ! PYTHON="$(resolve_python_bin)"; then
    fail_usage "no Python interpreter found; set PYTHON_BIN"
fi
readonly PYTHON
readonly UNIFIED_RUNNER="${UNIFIED_RUNNER_BIN-${DEFAULT_UNIFIED_RUNNER}}"
[[ -f "${UNIFIED_RUNNER}" ]] || fail_usage "unified runner not found: ${UNIFIED_RUNNER}"

if [[ -n "${SETSID_BIN+x}" ]]; then
    SETSID="$(command -v "${SETSID_BIN}" 2>/dev/null)" \
        || fail_usage "SETSID_BIN is not executable: ${SETSID_BIN}"
else
    SETSID="$(command -v setsid 2>/dev/null)" \
        || fail_usage "Linux setsid is required for formal supervision"
fi
readonly SETSID
if [[ "$("${SETSID}" --help 2>&1 || true)" != *"--wait"* ]]; then
    fail_usage "setsid must support --wait"
fi
for required in pgrep ps awk nohup; do
    command -v "${required}" >/dev/null 2>&1 \
        || fail_usage "required command not found: ${required}"
done
if [[ "$(pwd -P)" != "${PROJECT_ROOT}" ]]; then
    fail_usage "run this script from the project root: ${PROJECT_ROOT}"
fi

PREPARE_COMMAND=(
    "${PYTHON}" "${UNIFIED_RUNNER}"
    --operation prepare
    --output-dir "${FORMAL_RUN_ROOT}"
)
for dataset in "${SELECTED_DATASETS[@]}"; do
    PREPARE_COMMAND+=(--only "${dataset}")
done
if ((${#SELECTED_MODES[@]} == 1)); then
    PREPARE_COMMAND+=(--info-sharing "${SELECTED_MODES[0]}")
fi
for horizon in "${SELECTED_HORIZONS[@]}"; do
    PREPARE_COMMAND+=(--horizon "${horizon}")
done
for seed in "${SELECTED_SEEDS[@]}"; do
    PREPARE_COMMAND+=(--seed "${seed}")
done
if [[ "${RESUME}" == "1" ]]; then
    PREPARE_COMMAND+=(--resume)
fi
"${PREPARE_COMMAND[@]}"

readonly ATTEMPT_ID="${RUN_STAMP}_$$"
readonly LOG_ROOT="${FORMAL_RUN_ROOT}/supervisor/${ATTEMPT_ID}"
readonly PID_FILE="${LOG_ROOT}/pids.tsv"
readonly RUNNER_LOG="${LOG_ROOT}/runner.log"
mkdir -p "${LOG_ROOT}"
printf 'task\tlauncher_pid\tpid\tpgid\tstatus\tevent_time\telapsed_seconds\texit_code\tlog_file\toutput_dir\n' >"${PID_FILE}"
: >"${RUNNER_LOG}"

STATUSES=()
LAUNCHER_PIDS=()
WORKER_PIDS=()
PGIDS=()
START_TIMES=()
LOG_FILES=()
OUTPUT_DIRS=()
for index in "${!TASKS[@]}"; do
    STATUSES[${index}]="queued"
    LAUNCHER_PIDS[${index}]="-"
    WORKER_PIDS[${index}]="-"
    PGIDS[${index}]="-"
    START_TIMES[${index}]="-"
    LOG_FILES[${index}]="${LOG_ROOT}/${TASKS[${index}]}.log"
    OUTPUT_DIRS[${index}]="$(task_output_dir "${TASKS[${index}]}")"
done

CLEANUP_STARTED=0
FAILURE_CODE=0
RUNNING_COUNT=0
COMPLETED_COUNT=0
readonly SUPERVISOR_PGID="$(ps -o pgid= -p $$ | awk '{gsub(/[[:space:]]/, "", $0); print}')"

timestamp() {
    date '+%Y-%m-%dT%H:%M:%S%z'
}

log_message() {
    printf '%s\n' "$1"
    printf '%s\n' "$1" >>"${RUNNER_LOG}"
}

append_event() {
    local index="$1"
    local state="$2"
    local elapsed="${3--}"
    local exit_code="${4--}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${TASKS[${index}]}" "${LAUNCHER_PIDS[${index}]}" \
        "${WORKER_PIDS[${index}]}" "${PGIDS[${index}]}" "${state}" \
        "$(timestamp)" "${elapsed}" "${exit_code}" \
        "${LOG_FILES[${index}]}" "${OUTPUT_DIRS[${index}]}" >>"${PID_FILE}"
}

resolve_worker_pid() {
    local launcher_pid="$1"
    local attempt
    local candidate
    local candidate_args
    local candidates
    local args
    local unified_runner_name="${UNIFIED_RUNNER##*/}"
    local launcher_pgid
    for ((attempt = 0; attempt < 50; attempt++)); do
        candidates="$(pgrep -P "${launcher_pid}" 2>/dev/null || true)"
        while IFS= read -r candidate; do
            [[ "${candidate}" =~ ^[0-9]+$ ]] || continue
            candidate_args="$(ps -ww -o args= -p "${candidate}" 2>/dev/null || true)"
            if [[ "${candidate_args}" == *"${unified_runner_name}"* ]]; then
                printf '%s\n' "${candidate}"
                return 0
            fi
        done <<<"${candidates}"
        args="$(ps -ww -o args= -p "${launcher_pid}" 2>/dev/null || true)"
        if [[ "${args}" == *"${unified_runner_name}"* ]]; then
            launcher_pgid="$(ps -o pgid= -p "${launcher_pid}" 2>/dev/null \
                | awk '{gsub(/[[:space:]]/, "", $0); print}')"
            if [[ "${launcher_pgid}" =~ ^[0-9]+$ \
                && "${launcher_pgid}" != "${SUPERVISOR_PGID}" ]]; then
                printf '%s\n' "${launcher_pid}"
                return 0
            fi
        fi
        kill -0 "${launcher_pid}" 2>/dev/null || break
        sleep 0.1
    done
    return 1
}

resolve_worker_pgid() {
    local worker_pid="$1"
    local pgid
    pgid="$(ps -o pgid= -p "${worker_pid}" 2>/dev/null \
        | awk '{gsub(/[[:space:]]/, "", $0); print}')"
    if [[ ! "${pgid}" =~ ^[0-9]+$ || "${pgid}" == "${SUPERVISOR_PGID}" ]]; then
        return 1
    fi
    printf '%s\n' "${pgid}"
}

launch_task() {
    local index="$1"
    local task="${TASKS[${index}]}"
    local dataset
    local mode
    local launcher_pid
    local worker_pid
    local pgid
    local command
    dataset="$(task_dataset "${task}")"
    mode="$(task_mode "${task}")"
    command=(
        "${PYTHON}" "${UNIFIED_RUNNER}"
        --operation mode-worker
        --only "${dataset}"
        --info-sharing "${mode}"
        --output-dir "${OUTPUT_DIRS[${index}]}"
    )
    if [[ "${RESUME}" == "1" ]]; then
        command+=(--resume)
    fi

    START_TIMES[${index}]="$(date +%s)"
    STATUSES[${index}]="starting"
    nohup "${SETSID}" --wait "${command[@]}" \
        >"${LOG_FILES[${index}]}" 2>&1 </dev/null &
    launcher_pid=$!
    LAUNCHER_PIDS[${index}]="${launcher_pid}"
    if ! worker_pid="$(resolve_worker_pid "${launcher_pid}")"; then
        kill -TERM "${launcher_pid}" 2>/dev/null || true
        wait "${launcher_pid}" 2>/dev/null || true
        return 1
    fi
    if ! pgid="$(resolve_worker_pgid "${worker_pid}")"; then
        kill -TERM "${launcher_pid}" 2>/dev/null || true
        wait "${launcher_pid}" 2>/dev/null || true
        return 1
    fi
    WORKER_PIDS[${index}]="${worker_pid}"
    PGIDS[${index}]="${pgid}"
    STATUSES[${index}]="running"
    RUNNING_COUNT=$((RUNNING_COUNT + 1))
    append_event "${index}" started
    log_message "[LAUNCHED] ${task} pid=${worker_pid} pgid=${pgid} log=${LOG_FILES[${index}]}"
}

group_alive() {
    kill -0 -- "-$1" 2>/dev/null
}

cleanup_active() {
    local reason="$1"
    local index
    local deadline
    local alive
    local elapsed
    local worker_pid
    local pgid
    local launcher_pid
    if ((CLEANUP_STARTED == 1)); then
        return
    fi
    CLEANUP_STARTED=1
    trap - INT TERM
    set +e
    log_message "[ABORT] ${reason}"
    for index in "${!TASKS[@]}"; do
        worker_pid=""
        if [[ "${STATUSES[${index}]}" != "running" \
            && "${STATUSES[${index}]}" != "starting" ]]; then
            continue
        fi
        pgid="${PGIDS[${index}]}"
        launcher_pid="${LAUNCHER_PIDS[${index}]}"
        if [[ ! "${pgid}" =~ ^[0-9]+$ ]] \
            && [[ "${launcher_pid}" =~ ^[0-9]+$ ]]; then
            worker_pid="$(pgrep -P "${launcher_pid}" 2>/dev/null | head -n 1 || true)"
            if [[ ! "${worker_pid}" =~ ^[0-9]+$ ]]; then
                worker_pid="${launcher_pid}"
            fi
            pgid="$(resolve_worker_pgid "${worker_pid}" 2>/dev/null || true)"
        fi
        if [[ "${pgid}" =~ ^[0-9]+$ ]] && group_alive "${pgid}"; then
            PGIDS[${index}]="${pgid}"
            if [[ "${worker_pid-}" =~ ^[0-9]+$ ]]; then
                WORKER_PIDS[${index}]="${worker_pid}"
            fi
            kill -TERM -- "-${pgid}" 2>/dev/null
        elif [[ "${launcher_pid}" =~ ^[0-9]+$ ]]; then
            kill -TERM "${launcher_pid}" 2>/dev/null
        fi
    done
    deadline=$((SECONDS + TERMINATION_GRACE_SECONDS))
    while ((SECONDS < deadline)); do
        alive=0
        for index in "${!TASKS[@]}"; do
            if [[ "${STATUSES[${index}]}" != "running" \
                && "${STATUSES[${index}]}" != "starting" ]]; then
                continue
            fi
            if [[ "${PGIDS[${index}]}" =~ ^[0-9]+$ ]] \
                && group_alive "${PGIDS[${index}]}"; then
                alive=1
                break
            fi
        done
        ((alive == 0)) && break
        sleep 0.1
    done
    for index in "${!TASKS[@]}"; do
        if [[ "${STATUSES[${index}]}" != "running" \
            && "${STATUSES[${index}]}" != "starting" ]]; then
            continue
        fi
        if [[ "${PGIDS[${index}]}" =~ ^[0-9]+$ ]] \
            && group_alive "${PGIDS[${index}]}"; then
            kill -KILL -- "-${PGIDS[${index}]}" 2>/dev/null
        fi
        if [[ "${LAUNCHER_PIDS[${index}]}" =~ ^[0-9]+$ ]]; then
            wait "${LAUNCHER_PIDS[${index}]}" 2>/dev/null
        fi
        elapsed=$(($(date +%s) - START_TIMES[${index}]))
        STATUSES[${index}]="interrupted"
        append_event "${index}" interrupted "${elapsed}" -
    done
    set -e
}

handle_signal() {
    local name="$1"
    local code="$2"
    cleanup_active "received ${name}"
    exit "${code}"
}
trap 'handle_signal SIGINT 130' INT
trap 'handle_signal SIGTERM 143' TERM

reap_finished() {
    local index
    local code
    local elapsed
    for index in "${!TASKS[@]}"; do
        [[ "${STATUSES[${index}]}" == "running" ]] || continue
        if kill -0 "${LAUNCHER_PIDS[${index}]}" 2>/dev/null; then
            continue
        fi
        set +e
        wait "${LAUNCHER_PIDS[${index}]}"
        code=$?
        set -e
        elapsed=$(($(date +%s) - START_TIMES[${index}]))
        RUNNING_COUNT=$((RUNNING_COUNT - 1))
        COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
        if ((code == 0)); then
            STATUSES[${index}]="succeeded"
            append_event "${index}" succeeded "${elapsed}" 0
            log_message "[SUCCEEDED] ${TASKS[${index}]} elapsed=${elapsed}s"
        else
            STATUSES[${index}]="failed"
            append_event "${index}" failed "${elapsed}" "${code}"
            FAILURE_CODE="${code}"
            cleanup_active "${TASKS[${index}]} exited with code ${code}"
            return 1
        fi
    done
    return 0
}

log_message "[START] run_root=${FORMAL_RUN_ROOT} MAX_JOBS=${MAX_JOBS} cells=${CELL_COUNT}"
while ((COMPLETED_COUNT < TASK_COUNT)); do
    while ((RUNNING_COUNT < MAX_JOBS)); do
        candidate=-1
        for index in "${!TASKS[@]}"; do
            [[ "${STATUSES[${index}]}" == "queued" ]] || continue
            candidate="${index}"
            break
        done
        ((candidate >= 0)) || break
        if ! launch_task "${candidate}"; then
            FAILURE_CODE=2
            cleanup_active "failed to establish a safe PID/PGID for ${TASKS[${candidate}]}"
            exit 2
        fi
    done

    if ((RUNNING_COUNT == 0 && COMPLETED_COUNT < TASK_COUNT)); then
        cleanup_active "scheduler has queued work but no runnable worker"
        exit 2
    fi
    sleep 0.1
    if ! reap_finished; then
        exit "${FAILURE_CODE}"
    fi
done

if [[ "${PUBLISH_GLOBAL}" == "1" ]]; then
    "${PYTHON}" "${UNIFIED_RUNNER}" \
        --operation aggregate \
        --output-dir "${FORMAL_RUN_ROOT}"
fi
log_message "[COMPLETE] modes=${TASK_COUNT} run_root=${FORMAL_RUN_ROOT}"
