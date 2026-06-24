# Off-policy evaluation for slate recommendation: A reproducibility study
This repository is used for our Recommender Systems project on off-policy evaluation for slate recommendation. It is based on the original `slates_semisynth_expts` codebase.

*by Floris de Kam, Sem IJsendijk, Wes André, Finley Helms, Valentijn Oldenburg*



## Introduction
Off-policy evaluation for slate recommendation is difficult because full-slate importance weighting requires overlap between logged and target slates, which becomes rare in large slate spaces.
We reproduce the main experiments of Swaminathan et al. and test whether the pseudoinverse estimator (PI) generalizes to a Deezer music carousel setting.
Our reproduction confirms the main estimator trend: PI and self-normalized PI outperform full-slate IPS when the reward is close to additive.
On Deezer, PI works well for additive expected-stream rewards, but shows a bias floor for nonlinear any-stream rewards.
However, PI can still use logged slate rewards to train strong carousel policies, nearly matching supervised learning from true per-item labels.
We also extend the OPE setup to rank-aware exposure metrics and find that PI is reliable for additive exposure targets, while nonlinear exposure metrics introduce structural bias.
Overall, PI is a strong alternative to full-slate IPS, but works best when rewards are close to additive and logged slates overlap sufficiently with the target policy.

## Repository layout

- `Datasets.py`, `Settings.py`: dataset loading (MSLR, Deezer) and shared paths/config.
- `Policy.py`, `GammaDP.py`: logging/target policies and the Gamma matrices used by the pseudoinverse (PI) estimator.
- `Estimators.py`: the off-policy value estimators (OnPolicy, IPS, PI, DM, ...).
- `Metrics.py`: reward and fairness metrics (NDCG, ERR, RBP, Carousel rewards, Slate fairness metrics).
- `Parallel.py`: entry point for running a single off-policy evaluation condition (MSLR or Deezer).
- `Optimization.py`: entry point for MSLR ranker optimization experiments.
- `optimization_deezer.py`: entry point for Deezer PI-based carousel ranker optimization.
- `consolidate_mslr.py`: pre-processing of MSLR dataset.
- `aggregate_and_plot.py`: post-processing: dataset consolidation and result aggregation/plotting.
- `jobs/`: shell scripts giving an overview of every experiment configuration we ran, grouped by category (preprocessing, MSLR evaluation/optimization, Deezer evaluation/optimization/fairness).
- `data/`, `results/`, `plots/`, `logs/`: local data, raw result files, generated figures, and run logs.

## Setup

Clone the repository:

```powershell
git clone https://github.com/Floris93100/rs_project_slate_ope.git
cd rs_project_slate_ope
```

Create and activate the environment:

```powershell
conda create -n slate-ope python=3.10 -y
conda activate slate-ope
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Data

Create a local data folder:

```powershell
mkdir data
```

Place the required MSLR/Deezer ranking dataset files in this folder:

- **MSLR** (MSLR-WEB10K / MSLR-WEB30K): https://www.microsoft.com/en-us/research/project/mslr/
- **Deezer** (carousel bandits dataset): https://zenodo.org/records/4048678#.X22w4pMza3J

After downloading, run the preprocessing jobs in [`jobs/preprocessing.sh`](jobs/preprocessing.sh) to convert the raw files into the `.npz` format the rest of the pipeline expects.

## Running experiments

Every experiment is launched through one of three entry points: `Parallel.py` (off-policy evaluation, MSLR or Deezer), `Optimization.py` (MSLR ranker optimization), and `optimization_deezer.py` (Deezer PI-based ranker optimization). The most relevant `Parallel.py` arguments are:

| Argument | Short | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dataset` | `-d` | `"MSLR"` | Which dataset to load (`MSLR`, `MSLR10k`, `MQ2008`, `Deezer`, ...). |
| `--approach` | `-a` | `"IPS"` | OPE estimator: `OnPolicy`, `IPS`, `IPS_SN`, `PI`, `PI_SN`, `DM_tree`, `DM_lasso`, `DMc_lasso`, ... |
| `--value_metric` | `-v` | `"NDCG"` | Evaluation metric (`NDCG`, `ERR`, `RBP`, `CarouselExpStreams`, `CarouselAnyStream`, `SlateGroupExposure`, ...). |
| `--max_docs` | `-m` | `100` | Candidate-set size per query (slate candidates), `m`. |
| `--length_ranking` | `-l` | `10` | Slate size, `l`. |
| `--temperature` | `-t` | `0.0` | Logging-policy temperature (`0.0` = uniform). |
| `--logging_ranker` | `-f` | `"lasso"` | Model used for the logging (behavior) policy (`tree`/`lasso`). |
| `--evaluation_ranker` | `-e` | `"lasso"` | Model used for the evaluation (target) policy (`tree`/`lasso`). |
| `--target` | `-P` | `"Optimal"` | Deezer target policy pi (ignored for MSLR/MQ) |
| `--trainingSize` | `-z` | `-1` | Logs used to train DM-family estimators (required for `DM_*`). |
| `--start` / `--stop` | | `1` / `1` | Range of seed iterations to run. |
| `--output_dir` | `-o` | `./data/` | Destination folder for result files. |

Each run writes a `.z` file (joblib-serialized) containing the logging-step checkpoints, MSE-vs-checkpoint, predicted values, and the true target-policy value; aggregate and plot them with `aggregate_and_plot.py`.

### Example: evaluation (MSLR)

```bash
python Parallel.py -d MSLR -m 10 -l 5 -v NDCG -f lasso -t 1.0 -e tree -a PI --start 0 --stop 10 -o ./results/
```

### Example: optimization (MSLR)

```bash
python Optimization.py --value_metric NDCG --ensemble 1000 --leaves 70 --length_ranking 3 --fold 1
```

### Example: evaluation (Deezer)

```bash
python Parallel.py -d Deezer -m 100 -l 12 -v CarouselExpStreams -a PI --target Optimal -t 0.0 -s 1000000 -n 387 -o ./results/deezer
```

### Full experiment suite

The complete set of experiments behind our results is laid out in [`jobs/`](jobs/):

- [`jobs/preprocessing.sh`](jobs/preprocessing.sh)
- [`jobs/evaluation_mslr.sh`](jobs/evaluation_mslr.sh)
- [`jobs/optimization_mslr.sh`](jobs/optimization_mslr.sh)
- [`jobs/deezer_evaluation.sh`](jobs/deezer_evaluation.sh)
- [`jobs/deezer_optimization.sh`](jobs/deezer_optimization.sh)
- [`jobs/deezer_fairness.sh`](jobs/deezer_fairness.sh)

## Acknowledgements

This project builds directly on:

- Swaminathan et al., *Off-policy Evaluation for Slate Recommendation*
    - paper: https://arxiv.org/abs/1605.04812
    - code: https://github.com/adith387/slates_semisynth_expts
- Deezer's carousel bandits work 
    - paper: https://arxiv.org/abs/2009.06546
    - code: https://github.com/deezer/carousel_bandits



---

# Original README

Everything below this point is from the original `slates_semisynth_expts` repository.

---



# slates_semisynth_expts
Semi-synthetic experiments to test several approaches for off-policy evaluation and optimization of slate recommenders.

Contact: Adith Swaminathan (adswamin@microsoft.com)

These python scripts and classes run semi-synthetic experiments on the MSLR and MQ datasets
to study off-policy estimators for the slate bandit problem (combinatorial contextual bandits).

For Evaluation experiments:
Usage: python Parallel.py
Refer Parallel.py::main for examples on how to set up other variants of experiments
The make_parallel_eval.sh bash script creates the entire suite of experiments reported in [1] as parallel cluster jobs.

Data:
MSLR-30K has 31K queries, each with up to 1251 judged documents on relevance scale of {0, 1, 2, 3, 4}
MSLR-10K has 10K queries, each with up to 908 judged documents on relevance scale of {0, 1, 2, 3, 4}
Both MSLR datasets have <query, document> features of dimension 136

MQ2007 has 1692 queries, each with between 6 and 147 documents judged on relevance scale of {0, 1, 2}
MQ2008 has 784 queries, each with between 5 and 121 documents judged on relevance scale of {0, 1, 2}
Both datasets have <query, document> feature vectors of dimension 46

Refer Datasets.py::main for how to read in these datasets
    Download and uncompress the dataset files in the ../../Data/ folder
    MSLR: https://www.microsoft.com/en-us/research/project/mslr/
    MQ: https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/
    After reading in the uncompressed datasets once, 
    the Datasets script creates *.npz files in the ../../Data/ folder as pre-processed numpy arrays for faster consumption by other scripts
    
    
For Optimization experiments:
Usage: python Optimization.py
Refer Optimization.py::main for examples on how to set up other variants of experiments
The make_parallel_opt.sh bash script creates the entire suite of experiments reported in [1] as parallel cluster jobs.

[1] Off policy evaluation for slate recommendation, https://arxiv.org/abs/1605.04812 ; https://nips.cc/Conferences/2017/Schedule?showEvent=9146