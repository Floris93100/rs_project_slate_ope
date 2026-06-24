#!/bin/bash
# Deezer fairness evaluation: group-exposure metrics across estimators and
# target policies, under uniform logging.

METRICS=(SlateGroupExposure SlateAWRF SlateNDKL)
APPROACHES=(OnPolicy IPS IPS_SN PI PI_SN)
TARGETS=(Popularity Random)

for metric in "${METRICS[@]}"; do
    for approach in "${APPROACHES[@]}"; do
        for target in "${TARGETS[@]}"; do
            python Parallel.py -d Deezer -m 100 -l 12 -t 0.0 -v "$metric" -a "$approach" \
                --target "$target" --fair_groups 2 -s 1000000 -u 1000 \
                -o ./results/deezer_fairness/ --start 1 --stop 26
        done
    done
done
