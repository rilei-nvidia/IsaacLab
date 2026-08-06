#!/usr/bin/env bash
# Sweep the runtime benchmark over env counts and tasks.
#
# Run from the Isaac Lab repo root:
#     scripts/benchmarks/sweep_envs.sh
#
# Results land in $OUT_ROOT/<task_slug>/envs_<n>/, one benchmark JSON plus a run.log
# per combination. A failure (OOM at high env counts is the usual one) is recorded and
# the sweep continues; failures are listed at the end and the exit status is non-zero.
#
# Env overrides:
#     DRY_RUN=1        print the commands without running them
#     ENV_COUNTS="512 1024"          override the env-count sweep
#     OUT_ROOT=benchmarks/foo        override the output root
#     SEED / NUM_STEPS / WARMUP_STEPS / PYTHON

set -uo pipefail   # deliberately not -e: one failed run must not abort the sweep

PYTHON=${PYTHON:-.venv/bin/python}
SCRIPT=${SCRIPT:-scripts/benchmarks/runtime.py}
OUT_ROOT=${OUT_ROOT:-benchmarks/sweep}
SEED=${SEED:-42}
NUM_STEPS=${NUM_STEPS:-150}
WARMUP_STEPS=${WARMUP_STEPS:-50}
DRY_RUN=${DRY_RUN:-0}

read -r -a env_counts <<<"${ENV_COUNTS:-512 1024 2048 4096}"

# "<task>|<presets>" -- the preset vocabulary is per-task, so it cannot be shared:
# lift exposes ovrtx_renderer/rgb128/single_camera, while the shadow hand and cartpole
# camera tasks expose ovrtx/rgb. Drop a line here to skip a task.
tasks=(
    "Isaac-Lift-KukaAllegro-Camera|newton_mjwarp,ovrtx_renderer,rgb128,single_camera"
    "Isaac-Reorient-Cube-Shadow-Camera-Direct|newton_mjwarp,ovrtx,rgb"
    "Isaac-Cartpole-Camera-Direct|newton_mjwarp,ovrtx,rgb"
)

if [[ ! -x "$PYTHON" ]]; then
    echo "error: interpreter not found at '$PYTHON' (run from the Isaac Lab repo root, or set PYTHON)" >&2
    exit 2
fi
if [[ ! -f "$SCRIPT" ]]; then
    echo "error: '$SCRIPT' not found (run from the Isaac Lab repo root, or set SCRIPT)" >&2
    exit 2
fi

failures=()
total=0
sweep_start=$SECONDS

for entry in "${tasks[@]}"; do
    task="${entry%%|*}"
    presets="${entry##*|}"

    # Isaac-Lift-KukaAllegro-Camera -> lift_kukaallegro_camera
    slug="${task}"

    for envs in "${env_counts[@]}"; do
        out="$OUT_ROOT/$slug/envs_$envs"
        log="$out/run.log"
        total=$((total + 1))

        cmd=(
            "$PYTHON" "$SCRIPT"
            --task "$task" "presets=$presets"
            --seed "$SEED"
            --num_envs "$envs"
            --num_steps "$NUM_STEPS"
            --warmup_steps "$WARMUP_STEPS"
            --benchmark_formatter json
            --output_path "$out"
        )

        if [[ "$DRY_RUN" != "0" ]]; then
            printf '[sweep] (%d) ' "$total"
            printf '%q ' "${cmd[@]}"
            printf '>%q 2>&1\n' "$log"
            continue
        fi

        mkdir -p "$out"
        echo "[sweep] ($total) $task @ $envs envs -> $out"
        run_start=$SECONDS
        "${cmd[@]}" >"$log" 2>&1
        rc=$?
        elapsed=$((SECONDS - run_start))

        if [[ $rc -eq 0 ]]; then
            echo "[sweep]     ok in ${elapsed}s"
        else
            echo "[sweep]     FAILED (exit $rc) after ${elapsed}s -- tail of $log:" >&2
            tail -n 5 "$log" | sed 's/^/[sweep]     | /' >&2
            failures+=("$task @ $envs envs (exit $rc)")
        fi
    done
done

echo
if [[ "$DRY_RUN" != "0" ]]; then
    echo "[sweep] dry run: $total combinations, nothing executed"
    exit 0
fi
echo "[sweep] $((total - ${#failures[@]}))/$total runs succeeded in $((SECONDS - sweep_start))s"
if ((${#failures[@]})); then
    echo "[sweep] failures:" >&2
    printf '[sweep]   %s\n' "${failures[@]}" >&2
    exit 1
fi
