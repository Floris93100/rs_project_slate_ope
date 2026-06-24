"""Train SUP and PI-OPT carousel rankers on Deezer and compare to Optimal/Uniform."""

if __name__ == "__main__":
    import os
    import argparse
    import numpy

    import Datasets
    import Settings
    import Policy
    import Metrics

    from sklearn.linear_model import Ridge
    from sklearn.ensemble import GradientBoostingRegressor

    parser = argparse.ArgumentParser(description='Deezer PI-OPT optimization.')
    parser.add_argument('--length_ranking', '-l', type=int, default=12)
    parser.add_argument('--max_docs', '-m', type=int, default=100,
                        help='popularity-filter candidate set to top-m (<=0 = no filter)')
    parser.add_argument('--ranker', '-r', type=str, default='ridge', choices=['ridge', 'gbrt'])
    parser.add_argument('--value_metric', '-v', type=str, default='CarouselExpStreams',
                        choices=['CarouselExpStreams', 'CarouselAnyStream'])
    parser.add_argument('--logSize', '-s', type=int, default=1000000)
    parser.add_argument('--numpy_seed', '-n', type=int, default=387)
    parser.add_argument('--test_frac', type=float, default=0.3)
    parser.add_argument('--cache', type=str, default=Settings.DATA_DIR + 'Deezer/deezer_50k.npz')
    parser.add_argument('--alpha', type=float, default=1.0, help='ridge L2 strength')
    parser.add_argument('--ensemble', type=int, default=100, help='gbrt trees')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='gbrt lr')
    parser.add_argument('--subsample', type=float, default=0.5, help='gbrt subsample')
    parser.add_argument('--leaves', type=int, default=15, help='gbrt max_leaf_nodes')
    parser.add_argument('--output_dir', '-o', type=str, default=Settings.DATA_DIR,
                        help='directory for the results CSV (appended across runs)')
    args = parser.parse_args()

    l = args.length_ranking

    import scipy.sparse

    def subset_users(data, idx, suffix):
        """Build a Datasets view restricted to the given user indices."""
        sub = Datasets.Datasets()
        sub.name = data.name + suffix
        sub.relevances = data.relevances[idx]
        sub.docsPerQuery = numpy.full(len(idx), data.relevances.shape[1], dtype=numpy.int64)
        sub.mask = None
        sub.userFeatures = data.userFeatures[idx]
        sub.playlistFeatures = data.playlistFeatures            # shared across users
        sub.userSegment = None if data.userSegment is None else data.userSegment[idx]
        sub.features = [scipy.sparse.csr_matrix(data.playlistFeatures)] * len(idx)
        return sub

    def make_metric(name, dataset):
        """Instantiate the requested value metric on the given dataset."""
        if name == 'CarouselExpStreams':
            return Metrics.CarouselExpStreams(dataset, l)
        return Metrics.CarouselAnyStream(dataset, l)

    def pair_blocks(userF, plF, chunk=512):
        """Yield (start, count, X) blocks of user-by-playlist cross features."""
        n, d = userF.shape
        for s in range(0, n, chunk):
            blk = userF[s:s + chunk]
            X = (blk[:, None, :] * plF[None, :, :]).reshape(-1, d).astype(numpy.float32)
            yield s, blk.shape[0], X

    def fit_ranker(userF, plF, labels):
        """Fit a ridge or GBRT regressor on per-(user,playlist) labels."""
        Xs, ys = [], []
        for s, b, X in pair_blocks(userF, plF):
            Xs.append(X)
            ys.append(labels[s:s + b].reshape(-1))
        X = numpy.vstack(Xs)
        y = numpy.concatenate(ys).astype(numpy.float64)
        if args.ranker == 'ridge':
            model = Ridge(alpha=args.alpha)
        else:
            model = GradientBoostingRegressor(n_estimators=args.ensemble,
                                              learning_rate=args.learning_rate,
                                              subsample=args.subsample,
                                              max_leaf_nodes=args.leaves)
        model.fit(X, y)
        return model

    def score_matrix(model, userF, plF):
        """Predict the full (n_users, n_playlists) score matrix in chunks."""
        n = userF.shape[0]
        m = plF.shape[0]
        scores = numpy.empty((n, m), dtype=numpy.float64)
        for s, b, X in pair_blocks(userF, plF):
            scores[s:s + b] = model.predict(X).reshape(b, m)
        return scores

    def eval_scores(scores, metric):
        """Rank top-l by score per user and evaluate by the true reward."""
        n = scores.shape[0]
        order = numpy.argsort(-scores, axis=1)[:, :l]
        tot = 0.0
        for u in range(n):
            tot += metric.computeMetric(u, order[u])
        return tot / n

    def eval_uniform(dataset, metric, seed):
        """Mean true reward of random l-carousels: the logging-policy floor."""
        rng = numpy.random.RandomState(seed)
        n = dataset.relevances.shape[0]
        m = dataset.relevances.shape[1]
        tot = 0.0
        for u in range(n):
            tot += metric.computeMetric(u, rng.choice(m, size=l, replace=False))
        return tot / n

    def impute_pi_targets(trainData, loggingPolicy, trainMetric):
        """Impute per-item PI targets from logged uniform-slate rewards."""
        numQueries = len(trainData.docsPerQuery)
        m = trainData.relevances.shape[1]
        validDocs = min(m, l)
        Gamma = loggingPolicy.gammas[m]                          # (l*m, l*m)
        targets = [numpy.zeros((validDocs, m)) for _ in range(numQueries)]
        hist = numpy.zeros(numQueries, dtype=numpy.int64)
        numpy.random.seed(args.numpy_seed)
        seen = 0.0
        for j in range(args.logSize):
            u = numpy.random.randint(0, numQueries)
            slate = loggingPolicy.predict(u, l)
            val = trainMetric.computeMetric(u, slate)
            hist[u] += 1
            seen += (val - seen) / (j + 1)
            cols = [k * m + int(slate[k]) for k in range(validDocs)]    # gather Gamma columns
            phi = val * Gamma[:, cols].sum(axis=1)
            targets[u] += numpy.asarray(phi, dtype=numpy.float64).reshape(validDocs, m)
            if j % 100000 == 0:
                print(".", end="", flush=True)
        print("", flush=True)
        for u in range(numQueries):
            if hist[u] > 0:
                targets[u] /= hist[u]
        per_item = numpy.vstack([t.mean(axis=0) for t in targets])      # average over slots
        return per_item, seen

    data = Datasets.Datasets()
    data.loadDeezerNpz(args.cache)

    if args.max_docs >= 1:
        Policy.filter_deezer_candidates(data, args.max_docs)

    nUsers = data.relevances.shape[0]
    rng = numpy.random.RandomState(args.numpy_seed)
    perm = rng.permutation(nUsers)
    nTest = int(round(args.test_frac * nUsers))
    testIdx = perm[:nTest]
    trainIdx = perm[nTest:]

    trainData = subset_users(data, trainIdx, "-train")
    testData = subset_users(data, testIdx, "-test")
    print("OptDeezer [INFO] users train=%d test=%d  m=%d  l=%d  metric=%s  ranker=%s"
          % (len(trainIdx), len(testIdx), data.relevances.shape[1], l,
             args.value_metric, args.ranker), flush=True)

    loggingPolicy = Policy.UniformPolicy(trainData, False)
    loggingPolicy.setupGamma(l)

    trainMetric = make_metric(args.value_metric, trainData)
    testMetric = make_metric(args.value_metric, testData)

    print("OptDeezer [INFO] imputing PI targets ...", flush=True)
    piTargets, seenPerf = impute_pi_targets(trainData, loggingPolicy, trainMetric)
    print("OptDeezer [LOG] mean logged (uniform) reward seen on train: %.5f" % seenPerf, flush=True)

    print("OptDeezer [INFO] fitting SUP ...", flush=True)
    supModel = fit_ranker(trainData.userFeatures, trainData.playlistFeatures, trainData.relevances)
    print("OptDeezer [INFO] fitting PI-OPT ...", flush=True)
    piModel = fit_ranker(trainData.userFeatures, trainData.playlistFeatures, piTargets)

    sup_test = eval_scores(score_matrix(supModel, testData.userFeatures, testData.playlistFeatures), testMetric)
    pi_test = eval_scores(score_matrix(piModel, testData.userFeatures, testData.playlistFeatures), testMetric)
    opt_test = eval_scores(testData.relevances, testMetric)                       # skyline
    unif_test = eval_uniform(testData, testMetric, args.numpy_seed + 1)           # floor

    denom = (opt_test - unif_test)
    nan = float('nan')
    pi_recovery = nan if abs(denom) < 1e-12 else (pi_test - unif_test) / denom
    sup_recovery = nan if abs(denom) < 1e-12 else (sup_test - unif_test) / denom

    def pct(x):
        return "n/a" if x != x else "%.1f%%" % (100.0 * x)

    print("\n==== PI-OPT results (%s, m=%d, l=%d, %s) ===="
          % (args.value_metric, data.relevances.shape[1], l, args.ranker), flush=True)
    print("  Optimal (skyline) : %.5f" % opt_test, flush=True)
    print("  SUP (true labels) : %.5f" % sup_test, flush=True)
    print("  PI-OPT (page-only): %.5f" % pi_test, flush=True)
    print("  Uniform (floor)   : %.5f" % unif_test, flush=True)
    print("  PI-OPT recovery   : %s of (Optimal - Uniform)" % pct(pi_recovery), flush=True)
    print("  SUP recovery      : %s of (Optimal - Uniform)" % pct(sup_recovery), flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "optdeezer_results.csv")
    header = ("metric,m,l,ranker,logSize,seed,n_train,n_test,"
              "optimal,sup,pi_opt,uniform,pi_recovery,sup_recovery")
    row = "%s,%d,%d,%s,%d,%d,%d,%d,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g" % (
        args.value_metric, data.relevances.shape[1], l, args.ranker, args.logSize,
        args.numpy_seed, len(trainIdx), len(testIdx),
        opt_test, sup_test, pi_test, unif_test, pi_recovery, sup_recovery)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a") as fh:
        if write_header:
            fh.write(header + "\n")
        fh.write(row + "\n")
    print("OptDeezer [csv] appended -> %s" % csv_path, flush=True)
