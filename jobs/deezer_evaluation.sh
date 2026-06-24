#!/bin/bash
# Deezer off-policy evaluation: estimator accuracy across target policies,
# logging temperatures, and slate dimensions. Run for both deterministic
# and realized reward signals.

METRICS=(CarouselExpStreams CarouselAnyStream)
ESTIMATORS=(OnPolicy IPS IPS_SN PI PI_SN DM_tree)

for realized in 0 1; do
    REALIZED_FLAG=""
    [ "$realized" = "1" ] && REALIZED_FLAG="--realized 1"

    # Estimator grid: fixed slate (m=100, l=12), uniform logging, vary target policy & estimator
    TARGETS=(Optimal Popularity Segment Random LearnedLogistic)
    for metric in "${METRICS[@]}"; do
        for target in "${TARGETS[@]}"; do
            for approach in "${ESTIMATORS[@]}"; do
                ZFLAG=""
                [[ "$approach" == DM* ]] && ZFLAG="-z 50000"
                python Parallel.py -d Deezer -m 100 -l 12 -v "$metric" -a "$approach" \
                    --target "$target" -t 0.0 -s 1000000 -u 1000 -n 387 \
                    -o ./results/deezer/estimator_grid/ $ZFLAG $REALIZED_FLAG --start 1 --stop 51
            done
        done
    done

    # Non-uniform logging: fixed slate (m=100, l=12), vary logging temperature (alpha)
    for temp in 0.5 1.0; do
        for metric in "${METRICS[@]}"; do
            for approach in "${ESTIMATORS[@]}"; do
                ZFLAG=""
                [[ "$approach" == DM* ]] && ZFLAG="-z 50000"
                python Parallel.py -d Deezer -m 100 -l 12 -v "$metric" -a "$approach" \
                    --target Optimal -t "$temp" -s 1000000 -u 1000 -n 387 \
                    -o ./results/deezer/non_uniform_logging/ $ZFLAG $REALIZED_FLAG --start 1 --stop 51
            done
        done
    done

    # Slate sweep: uniform logging, vary slate dimensions (m, l)
    ML_PAIRS=("20 12" "50 12" "100 12" "100 3" "100 5")
    SWEEP_ESTIMATORS=(OnPolicy IPS PI)
    for pair in "${ML_PAIRS[@]}"; do
        read -r m l <<< "$pair"
        for metric in "${METRICS[@]}"; do
            for approach in "${SWEEP_ESTIMATORS[@]}"; do
                python Parallel.py -d Deezer -m "$m" -l "$l" -v "$metric" -a "$approach" \
                    --target Optimal -t 0.0 -s 1000000 -u 1000 -n 387 \
                    -o ./results/deezer/slate_sweep/ $REALIZED_FLAG --start 1 --stop 51
            done
        done
    done
done
