#!/bin/bash
# Preprocess raw MSLR text files into the project's binary dataset format,
# then consolidate the per-fold MSLR datasets into combined files.

# MSLR-WEB10K: 5 folds x {train, vali, test}
for fold in 1 2 3 4 5; do
    for split in train vali test; do
        python -c "
import Datasets
d = Datasets.Datasets()
d.loadTxt('./data/MSLR-WEB10K/Fold${fold}/${split}.txt', 'MSLR10k-${fold}-${split}')
del d
"
    done
done

# MSLR-WEB30K: 5 folds, train split only
for fold in 1 2 3 4 5; do
    python -c "import Datasets; d = Datasets.Datasets(); d.loadTxt('./data/MSLR-WEB30K/Fold${fold}/train.txt', 'MSLRWEB30k')"
done

# Deezer
python -c "import Datasets; d = Datasets.Datasets(); d.loadDeezer(user_features_path='./data/Deezer/user_features.csv', playlist_features_path='./data/Deezer/playlist_features.csv', cache_path='./data/Deezer/deezer_50k.npz'); del d"

# Merge per-fold MSLR datasets into consolidated files
python consolidate_mslr.py
