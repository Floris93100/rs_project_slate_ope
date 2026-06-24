#!/bin/bash
# Deezer PI-based ranker optimization across reward metrics and random seeds.

METRICS=(CarouselExpStreams CarouselAnyStream)
SEEDS=(34 1887 14 42 5)

for metric in "${METRICS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        python optimization_deezer.py -v "$metric" -r ridge -m 100 -l 12 \
            -s 1000000 -n "$seed" --cache ./data/Deezer/deezer_50k.npz \
            -o ./results/deezer_optimization/pi_opt/
    done
done
