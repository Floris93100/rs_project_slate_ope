#!/bin/bash
# MSLR ranker optimization (LambdaMART-style ensemble), per fold and per
# value metric used to score the learned ranking.

for fold in 1 2 3 4 5; do
    for metric in ERR NDCG; do
        python Optimization.py --value_metric "$metric" --ensemble 1000 --leaves 70 \
            --length_ranking 3 --fold "$fold"
    done
done
