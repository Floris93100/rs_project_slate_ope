#!/bin/bash
# MSLR off-policy evaluation: estimator accuracy across slate sizes, logging
# policies, ranking metrics, target rankers, and (for DM estimators) the
# size of the regression training set.

PI_APPROACHES=(OnPolicy IPS IPS_SN PI PI_SN)
DM_APPROACHES=(DM_tree DM_lasso DMc_lasso)
DM_TRAIN_SIZES=(1000 3000 10000 30000 100000 300000 1000000 3000000)

# Logging conditions: "M L LOG_RANKER ALPHA" (ALPHA=0.0 means uniform logging)
CONDITIONS=(
    "10  5  none  0.0"   # small slate, uniform logging
    "10  5  tree  1.0"
    "10  5  tree  2.0"
    "10  5  lasso 1.0"
    "10  5  lasso 2.0"
    "100 10 none  0.0"   # large slate, uniform logging
    "100 10 tree  0.5"
    "100 10 tree  1.0"
    "100 10 lasso 0.5"
    "100 10 lasso 1.0"
)

for metric in NDCG ERR; do
    for target in tree lasso; do
        for cond in "${CONDITIONS[@]}"; do
            read -r M L F ALPHA <<< "$cond"
            LOG_FLAGS="-t 0.0"
            [ "$ALPHA" != "0.0" ] && LOG_FLAGS="-f $F -t $ALPHA"

            for approach in "${PI_APPROACHES[@]}"; do
                python Parallel.py -d MSLR -m "$M" -l "$L" -v "$metric" $LOG_FLAGS \
                    -e "$target" -a "$approach" -z -1 --start 0 --stop 10
            done

            for approach in "${DM_APPROACHES[@]}"; do
                for train_size in "${DM_TRAIN_SIZES[@]}"; do
                    python Parallel.py -d MSLR -m "$M" -l "$L" -v "$metric" $LOG_FLAGS \
                        -e "$target" -a "$approach" -z "$train_size" --start 0 --stop 10
                done
            done
        done
    done
done

# RBP metric: small slate only, PI-family estimators only
RBP_CONDITIONS=(
    "lasso 0.0"
    "lasso 1.0"
    "lasso 2.0"
    "tree  1.0"
    "tree  2.0"
)
for target in tree lasso; do
    for cond in "${RBP_CONDITIONS[@]}"; do
        read -r F ALPHA <<< "$cond"
        for approach in "${PI_APPROACHES[@]}"; do
            python Parallel.py -d MSLR -m 10 -l 5 -v RBP -f "$F" -t "$ALPHA" \
                -e "$target" -a "$approach" -z -1 --start 0 --stop 10
        done
    done
done
