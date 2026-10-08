#!/usr/bin/env bash
set -Eeuo pipefail

readonly PHASE_ROOT="${1:?usage: phase1_284_monitor.sh /absolute/phase1/run/root}"
readonly SUMMARY="${PHASE_ROOT}/phase1_summary.env"
readonly PID_FILE="${PHASE_ROOT}/supervisor/phase1/pids.tsv"
readonly EXIT_FILE="${PHASE_ROOT}.EXIT_CODE"

printf 'TIME=%s\n' "$(date -Is)"
if [[ -f "${SUMMARY}" ]]; then
    # The file contains only launcher-owned KEY=VALUE records.
    while IFS='=' read -r key value; do
        case "${key}" in
            RUN_TAG|RUN_ROOT|TMUX_SESSION|EXPECTED_CELLS|MAX_JOBS|D5_CONCURRENCY_TARGET|PUBLISH_GLOBAL|FORMAL_RUN_STARTED|START_EPOCH)
                printf '%s=%s\n' "${key}" "${value}"
                [[ "${key}" == "START_EPOCH" ]] && start_epoch="${value}"
                ;;
        esac
    done <"${SUMMARY}"
else
    printf 'SUMMARY=MISSING\n'
fi

tmux_session="$(awk -F= '$1=="TMUX_SESSION" {print $2}' "${SUMMARY}" 2>/dev/null || true)"
if [[ -n "${tmux_session}" ]] && tmux has-session -t "${tmux_session}" 2>/dev/null; then
    printf 'TMUX_ACTIVE=YES\n'
else
    printf 'TMUX_ACTIVE=NO\n'
fi

if [[ "${start_epoch-}" =~ ^[0-9]+$ ]]; then
    printf 'ELAPSED_SECONDS=%s\n' "$(($(date +%s) - start_epoch))"
fi

printf 'LOAD_AVERAGE='
awk '{print $1, $2, $3}' /proc/loadavg
printf 'CPU_USAGE='
top -bn1 | awk -F',' '/Cpu\(s\)/ && !seen {gsub(/^[[:space:]]+/, "", $1); print $1; seen=1}'

latest_status_file="$(mktemp)"
trap 'rm -f "${latest_status_file}"' EXIT
if [[ -f "${PID_FILE}" ]]; then
    awk -F'\t' 'NR>1 {line[$1]=$0} END {for (task in line) print line[task]}' \
        "${PID_FILE}" >"${latest_status_file}"
fi

active_workers="$(awk -F'\t' '$6=="started" {count++} END {print count+0}' "${latest_status_file}")"
d5_workers="$(awk -F'\t' '$2=="d5" && $6=="started" {count++} END {print count+0}' "${latest_status_file}")"
succeeded_workers="$(awk -F'\t' '$6=="succeeded" {count++} END {print count+0}' "${latest_status_file}")"
failed_workers="$(awk -F'\t' '$6=="failed" || $6=="interrupted" {count++} END {print count+0}' "${latest_status_file}")"
printf 'ACTIVE_WORKERS=%s\nD5_ACTIVE_WORKERS=%s\n' "${active_workers}" "${d5_workers}"
printf 'SUCCEEDED_MODE_WORKERS=%s\nFAILED_MODE_WORKERS=%s\n' "${succeeded_workers}" "${failed_workers}"

printf 'RUNNING_CELLS_BEGIN\n'
while IFS=$'\t' read -r task dataset mode pid pgid status event_time elapsed exit_code log_file output_dir; do
    [[ "${status}" == "started" ]] || continue
    current_cell="$(grep -E '^\[D[1-6]-(with|without)-h[1-5]-s4[2-6]\]' "${log_file}" 2>/dev/null | tail -n 1 | sed -E 's/^\[([^]]+)\].*/\1/' || true)"
    printf '%s\tdataset=%s\tmode=%s\tpid=%s\tcell=%s\n' \
        "${task}" "${dataset}" "${mode}" "${pid}" "${current_cell:-preflight_or_mode_publication}"
done <"${latest_status_file}"
printf 'RUNNING_CELLS_END\n'

completed_total=0
for dataset in d1 d2 d3 d4 d5 d6; do
    for mode in without with; do
        expected=25
        [[ "${dataset}_${mode}" == "d5_with" ]] && expected=9
        completed="$(find "${PHASE_ROOT}/subruns" -type f \
            -path "*/${dataset}_${mode}/cells/h*_s*/results/*.csv" \
            -size +0c 2>/dev/null | wc -l | tr -d ' ')"
        running="$(awk -F'\t' -v dataset="${dataset}" -v mode="${mode}" \
            '$2==dataset && $3==mode && $6=="started" {count++} END {print count+0}' \
            "${latest_status_file}")"
        pending=$((expected - completed - running))
        ((pending < 0)) && pending=0
        completed_total=$((completed_total + completed))
        printf '%s_%s completed=%s running=%s pending=%s expected=%s\n' \
            "${dataset}" "${mode}" "${completed}" "${running}" "${pending}" "${expected}"
    done
done
printf 'COMPLETED_CELLS=%s/284\n' "${completed_total}"
printf 'SUCCESS_CELLS=%s\nFAILED_WORKERS=%s\n' "${completed_total}" "${failed_workers}"
if [[ -f "${EXIT_FILE}" ]]; then
    printf 'EXIT_CODE=%s\n' "$(sed -n '1p' "${EXIT_FILE}")"
else
    printf 'EXIT_CODE=RUNNING_OR_NOT_STARTED\n'
fi
printf 'AGGREGATE_PUBLICATION=DISABLED_PHASE1_PARTIAL\n'
