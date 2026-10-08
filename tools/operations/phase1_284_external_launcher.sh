#!/usr/bin/env bash
set -Eeuo pipefail

# External supervisor for the controlled D1-D6 Phase 1 matrix.
# It uses the existing prepare/mode-worker lifecycle and deliberately never
# invokes the global aggregate operation.

readonly PROJECT_ROOT="${PROJECT_ROOT-/home/jidanhuangdamihua/project}"
readonly EXPECTED_BRANCH="${EXPECTED_BRANCH-codex/改}"
readonly EXPECTED_HEAD="${EXPECTED_HEAD-630ae4ef33a271187ed7225437d0609d8a13e941}"
readonly MAX_JOBS="${MAX_JOBS-12}"
readonly D5_CONCURRENCY_TARGET="${D5_CONCURRENCY_TARGET-4}"
readonly DRY_RUN="${DRY_RUN-0}"
readonly RUN_TAG="${RUN_TAG-$(date '+%Y%m%d_%H%M%S')_phase1_284}"
readonly TERMINATION_GRACE_SECONDS="${TERMINATION_GRACE_SECONDS-15}"
readonly PYTHON="${PYTHON_BIN-${PROJECT_ROOT}/.venv/bin/python}"
readonly UNIFIED_RUNNER="${UNIFIED_RUNNER_BIN-${PROJECT_ROOT}/scripts/run_unified_d1_d6.py}"
readonly PHASE_ROOT="${PROJECT_ROOT}/outputs/runs/${RUN_TAG}"
readonly SUPERVISOR_ROOT="${PHASE_ROOT}/supervisor/phase1"
readonly PID_FILE="${SUPERVISOR_ROOT}/pids.tsv"
readonly RUNNER_LOG="${SUPERVISOR_ROOT}/runner.log"
readonly EXIT_FILE="${PHASE_ROOT}.EXIT_CODE"

fail() {
    printf 'PHASE1_LAUNCH_BLOCKED=%s\n' "$1" >&2
    exit 2
}

[[ "${MAX_JOBS}" == "12" ]] || fail "MAX_JOBS must be exactly 12"
[[ "${D5_CONCURRENCY_TARGET}" == "4" ]] \
    || fail "D5_CONCURRENCY_TARGET must be exactly 4"
[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] \
    || fail "DRY_RUN must be 0 or 1"
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9._-]*phase1_284[A-Za-z0-9._-]*$ ]] \
    || fail "RUN_TAG must be safe and contain phase1_284"
[[ -d "${PROJECT_ROOT}" ]] || fail "project root is missing: ${PROJECT_ROOT}"
[[ -f "${UNIFIED_RUNNER}" ]] || fail "unified runner is missing: ${UNIFIED_RUNNER}"

SUBRUN_NAMES=(
    d1_d4_d6_full_250
    d5_with_h345_s42_3
    d5_with_h12_s42_2
    d5_with_h3_s43_46_4
    d5_without_s42_5
    d5_without_s43_5
    d5_without_s44_5
    d5_without_s45_5
    d5_without_s46_5
)
SUBRUN_DATASETS=(
    d1,d2,d3,d4,d6
    d5 d5 d5 d5 d5 d5 d5 d5
)
SUBRUN_MODES=(
    both
    with with with without without without without without
)
SUBRUN_HORIZONS=(
    1,2,3,4,5
    3,4,5 1,2 3 1,2,3,4,5 1,2,3,4,5 1,2,3,4,5 1,2,3,4,5 1,2,3,4,5
)
SUBRUN_SEEDS=(
    42,43,44,45,46
    42 42 43,44,45,46 42 43 44 45 46
)
SUBRUN_CELLS=(250 3 2 4 5 5 5 5 5)

# Queue order makes the first batch exactly 4 D5 workers plus
# D4=2, D6=2, D1=2, D2=1, D3=1. Later D5-without chunks refill only
# vacated D5 slots, so the D5 cap remains four.
TASK_NAMES=(
    d5_with_h345_s42
    d5_with_h12_s42
    d5_with_h3_s43_46
    d5_without_s42
    d5_without_s43
    d5_without_s44
    d5_without_s45
    d5_without_s46
    d6_with
    d4_with
    d4_without
    d6_without
    d1_with
    d1_without
    d2_with
    d3_with
    d2_without
    d3_without
)
TASK_SUBRUNS=(
    d5_with_h345_s42_3
    d5_with_h12_s42_2
    d5_with_h3_s43_46_4
    d5_without_s42_5
    d5_without_s43_5
    d5_without_s44_5
    d5_without_s45_5
    d5_without_s46_5
    d1_d4_d6_full_250 d1_d4_d6_full_250 d1_d4_d6_full_250
    d1_d4_d6_full_250 d1_d4_d6_full_250 d1_d4_d6_full_250
    d1_d4_d6_full_250 d1_d4_d6_full_250 d1_d4_d6_full_250
    d1_d4_d6_full_250
)
TASK_DATASETS=(
    d5 d5 d5 d5 d5 d5 d5 d5
    d6 d4 d4 d6 d1 d1 d2 d3 d2 d3
)
TASK_MODES=(
    with with with without without without without without
    with with without without with without with with without without
)

CELL_ROWS=()
add_cell() {
    CELL_ROWS+=("$1"$'\t'"$2"$'\t'"$3"$'\t'"$4"$'\t'"$5")
}

for dataset in d1 d2 d3 d4 d6; do
    for mode in without with; do
        for horizon in 1 2 3 4 5; do
            for seed in 42 43 44 45 46; do
                add_cell "${dataset}" "${mode}" "${horizon}" "${seed}" d1_d4_d6_full_250
            done
        done
    done
done
for horizon in 1 2 3 4 5; do
    for seed in 42 43 44 45 46; do
        add_cell d5 without "${horizon}" "${seed}" "d5_without_s${seed}_5"
    done
done
for horizon in 1 2 3 4 5; do
    case "${horizon}" in
        1|2) subrun=d5_with_h12_s42_2 ;;
        3|4|5) subrun=d5_with_h345_s42_3 ;;
    esac
    add_cell d5 with "${horizon}" 42 "${subrun}"
done
for seed in 43 44 45 46; do
    add_cell d5 with 3 "${seed}" d5_with_h3_s43_46_4
done

readonly EXPECTED_CELLS=284
unique_cells="$(printf '%s\n' "${CELL_ROWS[@]}" | LC_ALL=C sort -u | wc -l | tr -d ' ')"
[[ "${#CELL_ROWS[@]}" == "${EXPECTED_CELLS}" ]] \
    || fail "generated cell count is ${#CELL_ROWS[@]}, expected ${EXPECTED_CELLS}"
[[ "${unique_cells}" == "${EXPECTED_CELLS}" ]] \
    || fail "generated unique count is ${unique_cells}, expected ${EXPECTED_CELLS}"

print_plan_summary() {
    printf 'PHASE1_EXPECTED_CELLS=284\n'
    printf 'PHASE1_UNIQUE_CELLS=%s\n' "${unique_cells}"
    printf 'D1=50\nD2=50\nD3=50\nD4=50\n'
    printf 'D5_WITHOUT=25\nD5_WITH=9\nD6=50\n'
    printf 'MAX_JOBS=%s\n' "${MAX_JOBS}"
    printf 'D5_CONCURRENCY_TARGET=%s\n' "${D5_CONCURRENCY_TARGET}"
    printf 'PUBLISH_GLOBAL=0\n'
    printf 'AGGREGATE_PUBLICATION=DISABLED_PHASE1_PARTIAL\n'
    printf 'RUN_TAG=%s\nRUN_ROOT=%s\n' "${RUN_TAG}" "${PHASE_ROOT}"
}

print_first_batch() {
    local total=0
    local d5=0
    local index
    printf 'FIRST_BATCH_BEGIN\n'
    for index in "${!TASK_NAMES[@]}"; do
        ((total >= MAX_JOBS)) && break
        if [[ "${TASK_DATASETS[${index}]}" == "d5" ]]; then
            ((d5 >= D5_CONCURRENCY_TARGET)) && continue
            d5=$((d5 + 1))
        fi
        printf '%02d\t%s\t%s\t%s\n' \
            "$((total + 1))" "${TASK_NAMES[${index}]}" \
            "${TASK_DATASETS[${index}]}" "${TASK_MODES[${index}]}"
        total=$((total + 1))
    done
    printf 'FIRST_BATCH_TOTAL=%s\nFIRST_BATCH_D5=%s\n' "${total}" "${d5}"
    printf 'FIRST_BATCH_END\n'
    [[ "${total}" == "12" && "${d5}" == "4" ]] \
        || fail "first batch does not satisfy 12 total / 4 D5"
}

print_plan_summary
print_first_batch
printf 'SUBRUNS_BEGIN\n'
for index in "${!SUBRUN_NAMES[@]}"; do
    printf '%s\tdatasets=%s\tmodes=%s\thorizons=%s\tseeds=%s\tcells=%s\n' \
        "${SUBRUN_NAMES[${index}]}" "${SUBRUN_DATASETS[${index}]}" \
        "${SUBRUN_MODES[${index}]}" "${SUBRUN_HORIZONS[${index}]}" \
        "${SUBRUN_SEEDS[${index}]}" "${SUBRUN_CELLS[${index}]}"
done
printf 'SUBRUNS_END\n'

if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'FORMAL_RUN_STARTED=NO\nDRY_RUN_ONLY=YES\n'
    exit 0
fi

[[ -x "${PYTHON}" ]] || fail "Python is not executable: ${PYTHON}"
for required in git setsid pgrep ps awk nohup sort find grep; do
    command -v "${required}" >/dev/null 2>&1 || fail "required command missing: ${required}"
done

cd "${PROJECT_ROOT}"
actual_branch="$(git branch --show-current)"
actual_head="$(git rev-parse HEAD)"
[[ "${actual_branch}" == "${EXPECTED_BRANCH}" ]] \
    || fail "BRANCH_MISMATCH actual=${actual_branch} expected=${EXPECTED_BRANCH}"
[[ "${actual_head}" == "${EXPECTED_HEAD}" ]] \
    || fail "HEAD_MISMATCH actual=${actual_head} expected=${EXPECTED_HEAD}"
[[ -z "$(git status --porcelain)" ]] || fail "formal project worktree is dirty"
[[ ! -e "${PHASE_ROOT}" ]] || fail "run root already exists: ${PHASE_ROOT}"
[[ ! -e "${EXIT_FILE}" ]] || fail "exit file already exists: ${EXIT_FILE}"

mkdir -p "${PHASE_ROOT}/subruns" "${SUPERVISOR_ROOT}"
printf 'dataset\tmode\thorizon\tseed\tsubrun\n' >"${PHASE_ROOT}/phase1_cells.tsv"
printf '%s\n' "${CELL_ROWS[@]}" >>"${PHASE_ROOT}/phase1_cells.tsv"
{
    printf 'BRANCH=%s\nHEAD=%s\nWORKTREE_CLEAN=YES\n' "${actual_branch}" "${actual_head}"
    printf 'RUN_TAG=%s\nRUN_ROOT=%s\n' "${RUN_TAG}" "${PHASE_ROOT}"
    printf 'TMUX_SESSION=%s\n' "${TMUX_SESSION_NAME-unknown}"
    printf 'EXPECTED_CELLS=284\nUNIQUE_CELLS=284\n'
    printf 'D1_CELLS=50\nD2_CELLS=50\nD3_CELLS=50\nD4_CELLS=50\n'
    printf 'D5_WITHOUT_CELLS=25\nD5_WITH_CELLS=9\nD6_CELLS=50\n'
    printf 'MAX_JOBS=12\nD5_CONCURRENCY_TARGET=4\nPUBLISH_GLOBAL=0\n'
    printf 'FORMAL_RUN_STARTED=NO\nSTART_EPOCH=%s\n' "$(date +%s)"
} >"${PHASE_ROOT}/phase1_summary.env"

log_message() {
    printf '%s\n' "$1"
    printf '%s\n' "$1" >>"${RUNNER_LOG}"
}
: >"${RUNNER_LOG}"

log_message "[PREPARE_START] subruns=${#SUBRUN_NAMES[@]}"
for index in "${!SUBRUN_NAMES[@]}"; do
    subrun_root="${PHASE_ROOT}/subruns/${SUBRUN_NAMES[${index}]}"
    prepare_log="${SUPERVISOR_ROOT}/prepare_${SUBRUN_NAMES[${index}]}.log"
    command=(
        "${PYTHON}" "${UNIFIED_RUNNER}"
        --operation prepare
        --output-dir "${subrun_root}"
    )
    IFS=',' read -r -a selected_datasets <<<"${SUBRUN_DATASETS[${index}]}"
    for dataset in "${selected_datasets[@]}"; do
        command+=(--only "${dataset}")
    done
    if [[ "${SUBRUN_MODES[${index}]}" != "both" ]]; then
        command+=(--info-sharing "${SUBRUN_MODES[${index}]}" )
    fi
    IFS=',' read -r -a selected_horizons <<<"${SUBRUN_HORIZONS[${index}]}"
    for horizon in "${selected_horizons[@]}"; do
        command+=(--horizon "${horizon}")
    done
    IFS=',' read -r -a selected_seeds <<<"${SUBRUN_SEEDS[${index}]}"
    for seed in "${selected_seeds[@]}"; do
        command+=(--seed "${seed}")
    done
    log_message "[PREPARE] ${SUBRUN_NAMES[${index}]} cells=${SUBRUN_CELLS[${index}]}"
    set +e
    "${command[@]}" >"${prepare_log}" 2>&1
    prepare_code=$?
    set -e
    if ((prepare_code != 0)); then
        printf '%s\n' "${prepare_code}" >"${EXIT_FILE}"
        log_message "[PREPARE_FAILED] ${SUBRUN_NAMES[${index}]} exit=${prepare_code} log=${prepare_log}"
        exit "${prepare_code}"
    fi
done
log_message '[PREPARE_COMPLETE] plans=9 cells=284 unique=284'
sed -i 's/^FORMAL_RUN_STARTED=NO$/FORMAL_RUN_STARTED=YES/' \
    "${PHASE_ROOT}/phase1_summary.env"

printf 'task\tdataset\tmode\tpid\tpgid\tstatus\tevent_time\telapsed_seconds\texit_code\tlog_file\toutput_dir\n' \
    >"${PID_FILE}"
STATUSES=()
PIDS=()
PGIDS=()
START_TIMES=()
LOG_FILES=()
OUTPUT_DIRS=()
for index in "${!TASK_NAMES[@]}"; do
    STATUSES[${index}]=queued
    PIDS[${index}]=-
    PGIDS[${index}]=-
    START_TIMES[${index}]=-
    LOG_FILES[${index}]="${SUPERVISOR_ROOT}/${TASK_NAMES[${index}]}.log"
    OUTPUT_DIRS[${index}]="${PHASE_ROOT}/subruns/${TASK_SUBRUNS[${index}]}/${TASK_DATASETS[${index}]}_${TASK_MODES[${index}]}"
done

RUNNING_COUNT=0
D5_RUNNING_COUNT=0
COMPLETED_COUNT=0
CLEANUP_STARTED=0

timestamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }

append_event() {
    local index="$1"
    local state="$2"
    local elapsed="${3--}"
    local exit_code="${4--}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${TASK_NAMES[${index}]}" "${TASK_DATASETS[${index}]}" \
        "${TASK_MODES[${index}]}" "${PIDS[${index}]}" "${PGIDS[${index}]}" \
        "${state}" "$(timestamp)" "${elapsed}" "${exit_code}" \
        "${LOG_FILES[${index}]}" "${OUTPUT_DIRS[${index}]}" >>"${PID_FILE}"
}

cleanup_active() {
    local reason="$1"
    local index
    local deadline
    ((CLEANUP_STARTED == 0)) || return
    CLEANUP_STARTED=1
    trap - INT TERM
    log_message "[ABORT] ${reason}"
    for index in "${!TASK_NAMES[@]}"; do
        [[ "${STATUSES[${index}]}" == "running" ]] || continue
        if [[ "${PGIDS[${index}]}" =~ ^[0-9]+$ ]]; then
            kill -TERM -- "-${PGIDS[${index}]}" 2>/dev/null || true
        fi
    done
    deadline=$((SECONDS + TERMINATION_GRACE_SECONDS))
    while ((SECONDS < deadline)); do
        alive=0
        for index in "${!TASK_NAMES[@]}"; do
            [[ "${STATUSES[${index}]}" == "running" ]] || continue
            if kill -0 "${PIDS[${index}]}" 2>/dev/null; then
                alive=1
                break
            fi
        done
        ((alive == 0)) && break
        sleep 0.2
    done
    for index in "${!TASK_NAMES[@]}"; do
        [[ "${STATUSES[${index}]}" == "running" ]] || continue
        if kill -0 "${PIDS[${index}]}" 2>/dev/null; then
            kill -KILL -- "-${PGIDS[${index}]}" 2>/dev/null || true
        fi
        wait "${PIDS[${index}]}" 2>/dev/null || true
        elapsed=$(($(date +%s) - START_TIMES[${index}]))
        STATUSES[${index}]=interrupted
        append_event "${index}" interrupted "${elapsed}" -
    done
}

handle_signal() {
    cleanup_active "received $1"
    printf '%s\n' "$2" >"${EXIT_FILE}"
    exit "$2"
}
trap 'handle_signal SIGINT 130' INT
trap 'handle_signal SIGTERM 143' TERM

launch_task() {
    local index="$1"
    local pid
    local pgid=''
    local attempt
    START_TIMES[${index}]="$(date +%s)"
    nohup setsid \
        "${PYTHON}" "${UNIFIED_RUNNER}" \
        --operation mode-worker \
        --only "${TASK_DATASETS[${index}]}" \
        --info-sharing "${TASK_MODES[${index}]}" \
        --output-dir "${OUTPUT_DIRS[${index}]}" \
        >"${LOG_FILES[${index}]}" 2>&1 </dev/null &
    pid=$!
    for ((attempt = 0; attempt < 50; attempt++)); do
        pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | awk '{gsub(/[[:space:]]/, "", $0); print}')"
        [[ "${pgid}" =~ ^[0-9]+$ && "${pgid}" == "${pid}" ]] && break
        kill -0 "${pid}" 2>/dev/null || break
        sleep 0.1
    done
    if [[ ! "${pgid}" =~ ^[0-9]+$ || "${pgid}" != "${pid}" ]]; then
        kill -TERM "${pid}" 2>/dev/null || true
        wait "${pid}" 2>/dev/null || true
        return 1
    fi
    PIDS[${index}]="${pid}"
    PGIDS[${index}]="${pgid}"
    STATUSES[${index}]=running
    RUNNING_COUNT=$((RUNNING_COUNT + 1))
    if [[ "${TASK_DATASETS[${index}]}" == "d5" ]]; then
        D5_RUNNING_COUNT=$((D5_RUNNING_COUNT + 1))
    fi
    append_event "${index}" started
    log_message "[LAUNCHED] ${TASK_NAMES[${index}]} pid=${pid} log=${LOG_FILES[${index}]}"
}

reap_finished() {
    local index
    local code
    local elapsed
    for index in "${!TASK_NAMES[@]}"; do
        [[ "${STATUSES[${index}]}" == "running" ]] || continue
        kill -0 "${PIDS[${index}]}" 2>/dev/null && continue
        set +e
        wait "${PIDS[${index}]}"
        code=$?
        set -e
        elapsed=$(($(date +%s) - START_TIMES[${index}]))
        RUNNING_COUNT=$((RUNNING_COUNT - 1))
        if [[ "${TASK_DATASETS[${index}]}" == "d5" ]]; then
            D5_RUNNING_COUNT=$((D5_RUNNING_COUNT - 1))
        fi
        if ((code == 0)); then
            STATUSES[${index}]=succeeded
            COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
            append_event "${index}" succeeded "${elapsed}" 0
            log_message "[SUCCEEDED] ${TASK_NAMES[${index}]} elapsed=${elapsed}s"
        else
            STATUSES[${index}]=failed
            append_event "${index}" failed "${elapsed}" "${code}"
            printf '%s\n' "${code}" >"${EXIT_FILE}"
            cleanup_active "${TASK_NAMES[${index}]} exited with code ${code}"
            exit "${code}"
        fi
    done
}

log_message "[START] run_root=${PHASE_ROOT} MAX_JOBS=12 D5_TARGET=4 workers=${#TASK_NAMES[@]} cells=284"
while ((COMPLETED_COUNT < ${#TASK_NAMES[@]})); do
    while ((RUNNING_COUNT < MAX_JOBS)); do
        candidate=-1
        for index in "${!TASK_NAMES[@]}"; do
            [[ "${STATUSES[${index}]}" == "queued" ]] || continue
            if [[ "${TASK_DATASETS[${index}]}" == "d5" \
                && "${D5_RUNNING_COUNT}" -ge "${D5_CONCURRENCY_TARGET}" ]]; then
                continue
            fi
            candidate="${index}"
            break
        done
        ((candidate >= 0)) || break
        if ! launch_task "${candidate}"; then
            printf '2\n' >"${EXIT_FILE}"
            cleanup_active "failed to establish isolated process group for ${TASK_NAMES[${candidate}]}"
            exit 2
        fi
    done
    ((RUNNING_COUNT > 0)) || fail "scheduler has queued work but no runnable worker"
    sleep 1
    reap_finished
done

completed_csvs="$(find "${PHASE_ROOT}/subruns" -type f \
    -path '*/cells/h*_s*/results/*.csv' -size +0c | wc -l | tr -d ' ')"
accepted_cells="$(find "${PHASE_ROOT}/subruns" -type f \
    -path '*/cells/h*_s*/results/*.acceptance.json' \
    -exec grep -lE '"passed"[[:space:]]*:[[:space:]]*true' {} + \
    | wc -l | tr -d ' ')"
if [[ "${completed_csvs}" != "284" || "${accepted_cells}" != "284" ]]; then
    printf '2\n' >"${EXIT_FILE}"
    fail "post-run acceptance count mismatch csv=${completed_csvs} accepted=${accepted_cells}"
fi

printf '0\n' >"${EXIT_FILE}"
log_message '[PHASE1_COMPLETE] accepted_cells=284 aggregate=disabled remaining_d5_with=16'
printf 'FORMAL_RUN_STARTED=YES\nPHASE1_COMPLETE=YES\nACCEPTED_CELLS=284\n'
printf 'FULL_300_COMPLETE=NO\nFORMAL_CLOSURE_COMPLETE=NO\n'
