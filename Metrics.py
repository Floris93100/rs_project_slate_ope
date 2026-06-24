"""Classes that define different metrics for semi-synthetic experiments."""
import numpy
import sys


class Metric:
    """Base class: subclasses supply computeMetric(query_id, ranking) -> value."""
    def __init__(self, dataset, ranking_size):
        self.rankingSize=ranking_size
        self.dataset=dataset
        self.name=None


class ConstantMetric(Metric):
    """Metric that always returns a fixed constant value."""
    def __init__(self, dataset, ranking_size, constant):
        Metric.__init__(self, dataset, ranking_size)
        self.constant=constant
        self.name='Constant'
        print("ConstantMetric:init [INFO] RankingSize", ranking_size, "\t Constant", constant, flush=True)

    def computeMetric(self, query_id, ranking):
        return self.constant


class DCG(Metric):
    """Discounted Cumulative Gain over the given ranking."""
    def __init__(self, dataset, ranking_size):
        Metric.__init__(self, dataset, ranking_size)
        self.discountParams=1.0+numpy.array(range(self.rankingSize), dtype=numpy.float64)
        self.discountParams[0]=2.0
        self.discountParams[1]=2.0
        self.discountParams=numpy.reciprocal(numpy.log2(self.discountParams))
        self.name='DCG'
        print("DCG:init [INFO] RankingSize", ranking_size, flush=True)

    def computeMetric(self, query_id, ranking):
        relevanceList=self.dataset.relevances[query_id][ranking]
        gain=numpy.exp2(relevanceList)-1.0
        dcg=numpy.dot(self.discountParams[0:numpy.shape(gain)[0]], gain)
        return dcg


class NDCG(Metric):
    """DCG normalized by the maximum achievable DCG per query."""
    def __init__(self, dataset, ranking_size, allow_repetitions):
        Metric.__init__(self, dataset, ranking_size)
        self.discountParams=1.0+numpy.array(range(self.rankingSize), dtype=numpy.float64)
        self.discountParams[0]=2.0
        self.discountParams[1]=2.0
        self.discountParams=numpy.reciprocal(numpy.log2(self.discountParams))
        self.name='NDCG'

        self.normalizers=[]
        numQueries=len(self.dataset.docsPerQuery)
        for currentQuery in range(numQueries):
            validDocs=min(self.dataset.docsPerQuery[currentQuery], ranking_size)
            currentRelevances=self.dataset.relevances[currentQuery]

            if self.dataset.mask is not None:
                currentRelevances=currentRelevances[self.dataset.mask[currentQuery]]   # filtered dataset

            maxRelevances=None
            if allow_repetitions:
                maxRelevances=numpy.repeat(currentRelevances.max(), validDocs)
            else:
                maxRelevances=-numpy.sort(-currentRelevances)[0:validDocs]

            maxGain=numpy.exp2(maxRelevances)-1.0
            maxDCG=numpy.dot(self.discountParams[0:validDocs], maxGain)

            self.normalizers.append(maxDCG)

            if currentQuery % 1000==0:
                print(".", end="", flush=True)

        print("", flush=True)
        print("NDCG:init [INFO] RankingSize", ranking_size, "\t AllowRepetitions?", allow_repetitions, flush=True)

    def computeMetric(self, query_id, ranking):
        normalizer=self.normalizers[query_id]
        if normalizer<=0.0:
            return 0.0
        else:
            relevanceList=self.dataset.relevances[query_id][ranking]
            gain=numpy.exp2(relevanceList)-1.0
            dcg=numpy.dot(self.discountParams[0:numpy.shape(gain)[0]], gain)
            return dcg*1.0/normalizer

    def getMax(self,ranking_size):
        return 1.0

    def getMin(self,ranking_size):
        return 0.0

class ERR(Metric):
    """Expected Reciprocal Rank, calibrated to the dataset's max relevance grade."""
    def __init__(self, dataset, ranking_size):
        Metric.__init__(self, dataset, ranking_size)
        self.name='ERR'

        self.maxrel=None
        if self.dataset.name.startswith('MSLR'):
            self.maxrel=numpy.exp2(4)   # MSLR max grade = 4
        elif self.dataset.name.startswith('MQ200'):
            self.maxrel=numpy.exp2(2)   # MQ200* max grade = 2
        else:
            print("ERR:init [ERR] Unknown dataset. Use MSLR/MQ200*", flush=True)
            sys.exit(0)

        print("ERR:init [INFO] RankingSize", ranking_size, flush=True)

    def computeMetric(self, query_id, ranking):
        relevanceList=self.dataset.relevances[query_id][ranking]
        gain=numpy.exp2(relevanceList)-1.0
        probs=gain*1.0/self.maxrel
        validDocs=numpy.shape(probs)[0]
        err=0.0
        p=1.0
        for i in range(validDocs):
            err+=p*probs[i]/(i+1)
            p=p*(1-probs[i])
        return err

    def getMax(self, ranking_size):
        probs=[(self.maxrel-1.0)/self.maxrel for i in range(ranking_size)]
        validDocs=numpy.shape(probs)[0]
        err=0.0
        p=1.0
        for i in range(validDocs):
            err+=p*probs[i]/(i+1)
            p=p*(1-probs[i])
        return err

    def getMin(self, ranking_size):
        return 0.0

class RBP(Metric):
    """Rank-Biased Precision with persistence parameter p."""
    def __init__(self, dataset, ranking_size, p=0.85):
        Metric.__init__(self, dataset, ranking_size)
        self.p = p
        self.name = 'RBP'
        self.weights = (1.0 - p) * numpy.array([p**k for k in range(ranking_size)], dtype=numpy.float64)
        if self.dataset.name.startswith('MSLR'):
            self.maxrel = 4.0
        elif self.dataset.name.startswith('MQ200'):
            self.maxrel = 2.0
        else:
            self.maxrel = 1.0
        print("RBP:init [INFO] RankingSize", ranking_size, "\t p", p, flush=True)

    def computeMetric(self, query_id, ranking):
        relevanceList = self.dataset.relevances[query_id][ranking]
        normalized = relevanceList / self.maxrel
        validDocs = numpy.shape(normalized)[0]
        return numpy.dot(self.weights[:validDocs], normalized)

    def getMax(self, ranking_size):
        return 1.0

    def getMin(self, ranking_size):
        return 0.0


class MaxRelevance(Metric):
    """Maximum relevance grade present in the ranking."""
    def __init__(self, dataset, ranking_size):
        Metric.__init__(self, dataset, ranking_size)
        self.name='MaxRelevance'
        print("MaxRelevance:init [INFO] RankingSize", ranking_size, flush=True)

    def computeMetric(self, query_id, ranking):
        relevanceList=self.dataset.relevances[query_id][ranking]
        maxRelevance=1.0*relevanceList.max()
        return maxRelevance


class SumRelevance(Metric):
    """Sum of relevance grades over the ranking."""
    def __init__(self, dataset, ranking_size):
        Metric.__init__(self, dataset, ranking_size)
        self.name='SumRelevance'
        print("SumRelevance:init [INFO] RankingSize", ranking_size, flush=True)

    def computeMetric(self, query_id, ranking):
        relevanceList=self.dataset.relevances[query_id][ranking]
        sumRelevance=relevanceList.sum(dtype=numpy.float64)
        return sumRelevance


class CarouselExpStreams(Metric):
    """Additive expected-streams reward; linear in slate, so PI is unbiased."""
    def __init__(self, dataset, ranking_size):
        Metric.__init__(self, dataset, ranking_size)
        self.name = 'CarouselExpStreams'
        print("CarouselExpStreams:init [INFO] RankingSize", ranking_size, flush=True)

    def computeMetric(self, query_id, ranking):
        probs = self.dataset.relevances[query_id][ranking]      # p(u, s_j) per slot
        return probs.sum(dtype=numpy.float64) / self.rankingSize

    def sampleMetric(self, query_id, ranking):
        probs = self.dataset.relevances[query_id][ranking]
        streams = numpy.random.random_sample(probs.shape[0]) < probs   # one Bernoulli per slot
        return streams.sum(dtype=numpy.float64) / self.rankingSize

    def getMax(self, ranking_size):
        return 1.0

    def getMin(self, ranking_size):
        return 0.0


class CarouselAnyStream(Metric):
    """Any-stream reward; the product term breaks linearity, so PI is biased."""
    def __init__(self, dataset, ranking_size):
        Metric.__init__(self, dataset, ranking_size)
        self.name = 'CarouselAnyStream'
        print("CarouselAnyStream:init [INFO] RankingSize", ranking_size, flush=True)

    def computeMetric(self, query_id, ranking):
        probs = self.dataset.relevances[query_id][ranking]      # p(u, s_j) per slot
        return 1.0 - numpy.prod(1.0 - probs, dtype=numpy.float64)

    def sampleMetric(self, query_id, ranking):
        probs = self.dataset.relevances[query_id][ranking]
        streams = numpy.random.random_sample(probs.shape[0]) < probs   # one Bernoulli per slot
        return 1.0 if streams.any() else 0.0

    def getMin(self, ranking_size):
        return 0.0

    def getMax(self, ranking_size):
        return 1.0


def assign_popularity_groups(dataset, n_groups):
    """Bucket candidates into n_groups by popularity; return (groups, desired shares)."""
    pop = numpy.asarray(dataset.relevances, dtype=numpy.float64).mean(axis=0)
    m = pop.shape[0]
    order = numpy.argsort(pop)          # ascending: tail first
    ranks = numpy.empty(m, dtype=numpy.int64)
    ranks[order] = numpy.arange(m)
    groups = (ranks * n_groups // m).astype(numpy.int64)    # group 0 = least popular
    counts = numpy.bincount(groups, minlength=n_groups).astype(numpy.float64)
    desired = counts / counts.sum()
    desired = numpy.maximum(desired, 1e-12)
    desired = desired / desired.sum()
    return groups, desired


def attention_weights(ranking_size):
    """DCG-like position attention weights w_j = 1/log2(j+2)."""
    j = numpy.arange(ranking_size, dtype=numpy.float64)
    return numpy.reciprocal(numpy.log2(j + 2.0))


class SlateGroupExposure(Metric):
    """Linear fairness target: attention-weighted exposure share of one group."""
    def __init__(self, dataset, ranking_size, n_groups=2, target_group=0):
        Metric.__init__(self, dataset, ranking_size)
        self.name = 'SlateGroupExposure'
        self.nGroups = int(n_groups)
        self.targetGroup = int(target_group)
        self.groups, self.desired = assign_popularity_groups(dataset, self.nGroups)
        self.attn = attention_weights(ranking_size)
        print("SlateGroupExposure:init [INFO] RankingSize", ranking_size,
              "\t Groups", self.nGroups, "\t TargetGroup", self.targetGroup,
              "\t Desired", numpy.round(self.desired, 4), flush=True)

    def computeMetric(self, query_id, ranking):
        ranking = numpy.asarray(ranking)
        L = ranking.shape[0]
        w = self.attn[:L]
        g = self.groups[ranking]
        num = numpy.dot(w, (g == self.targetGroup).astype(numpy.float64))
        return num / w.sum()

    def getMin(self, ranking_size):
        return 0.0

    def getMax(self, ranking_size):
        return 1.0


class SlateAWRF(Metric):
    """Mildly nonlinear fairness target: total-variation distance to desired group mix."""
    def __init__(self, dataset, ranking_size, n_groups=2, target_group=0):
        Metric.__init__(self, dataset, ranking_size)
        self.name = 'SlateAWRF'
        self.nGroups = int(n_groups)
        self.groups, self.desired = assign_popularity_groups(dataset, self.nGroups)
        self.attn = attention_weights(ranking_size)
        print("SlateAWRF:init [INFO] RankingSize", ranking_size,
              "\t Groups", self.nGroups, "\t Desired", numpy.round(self.desired, 4), flush=True)

    def computeMetric(self, query_id, ranking):
        ranking = numpy.asarray(ranking)
        L = ranking.shape[0]
        w = self.attn[:L]
        W = w.sum()
        g = self.groups[ranking]
        a = numpy.zeros(self.nGroups, dtype=numpy.float64)
        for grp in range(self.nGroups):
            a[grp] = numpy.dot(w, (g == grp).astype(numpy.float64))
        a /= W
        return 0.5 * numpy.abs(a - self.desired).sum()

    def getMin(self, ranking_size):
        return 0.0

    def getMax(self, ranking_size):
        return 1.0


class SlateNDKL(Metric):
    """Nonlinear, prefix-based fairness target: discounted KL to desired group mix."""
    def __init__(self, dataset, ranking_size, n_groups=2, target_group=0):
        Metric.__init__(self, dataset, ranking_size)
        self.name = 'SlateNDKL'
        self.nGroups = int(n_groups)
        self.groups, self.desired = assign_popularity_groups(dataset, self.nGroups)
        self.discount = numpy.reciprocal(numpy.log2(numpy.arange(ranking_size, dtype=numpy.float64) + 2.0))
        self.Z = self.discount.sum()
        self.logMax = float(numpy.log(1.0 / self.desired.min()))
        print("SlateNDKL:init [INFO] RankingSize", ranking_size,
              "\t Groups", self.nGroups, "\t Desired", numpy.round(self.desired, 4),
              "\t Max", round(self.logMax, 4), flush=True)

    def computeMetric(self, query_id, ranking):
        ranking = numpy.asarray(ranking)
        L = ranking.shape[0]
        g = self.groups[ranking]
        counts = numpy.zeros(self.nGroups, dtype=numpy.float64)
        Z = self.discount[:L].sum()
        ndkl = 0.0
        for k in range(L):
            counts[g[k]] += 1.0
            Dk = counts / (k + 1.0)
            mask = Dk > 0.0
            kl = numpy.sum(Dk[mask] * numpy.log(Dk[mask] / self.desired[mask]))
            ndkl += self.discount[k] * kl
        return ndkl / Z

    def getMin(self, ranking_size):
        return 0.0

    def getMax(self, ranking_size):
        return self.logMax


if __name__=="__main__":
    import Settings
    import Datasets

    deezerData = Datasets.Datasets()
    deezerData.loadDeezer(
        Settings.DATA_DIR+"Deezer/user_features.csv",
        Settings.DATA_DIR+"Deezer/playlist_features.csv",
        cache_path=Settings.DATA_DIR+"Deezer/deezer_50k.npz"
    )

    expstreams=CarouselExpStreams(deezerData, 4)
    print("CarouselExpStreams", expstreams.computeMetric(0, [0, 1, 2, 3]), flush=True)

    anystream=CarouselAnyStream(deezerData, 4)
    print("CarouselAnyStream", anystream.computeMetric(0, [0, 1, 2, 3]), flush=True)

    grpexp = SlateGroupExposure(deezerData, 4, n_groups=2, target_group=0)        # fairness sanity checks
    print("SlateGroupExposure", grpexp.computeMetric(0, [0, 1, 2, 3]), flush=True)
    awrf = SlateAWRF(deezerData, 4, n_groups=2)
    print("SlateAWRF", awrf.computeMetric(0, [0, 1, 2, 3]), flush=True)
    ndkl = SlateNDKL(deezerData, 4, n_groups=2)
    print("SlateNDKL", ndkl.computeMetric(0, [0, 1, 2, 3]),
          "(max", ndkl.getMax(4), ")", flush=True)
    del grpexp, awrf, ndkl

    mslrData = Datasets.Datasets()
    mslrData.loadNpz(Settings.DATA_DIR+"mslr/mslr")

    const=ConstantMetric(mslrData, 4, 5.0)
    print("Constant", const.computeMetric(0, [0, 1, 2, 3]), flush=True)
    del const

    dcg=DCG(mslrData, 4)
    print("DCG", dcg.computeMetric(0, [0, 1, 2, 3]), flush=True)
    del dcg

    ndcg=NDCG(mslrData, 4, False)
    print("NDCG NoRep", ndcg.computeMetric(0, [0, 1, 2, 3]), flush=True)
    del ndcg

    ndcg=NDCG(mslrData, 4, True)
    print("NDCG YesRep", ndcg.computeMetric(0, [0, 1, 2, 3]), flush=True)
    del ndcg

    err=ERR(mslrData, 4)
    print("ERR", err.computeMetric(0, [0, 1, 2, 3]), flush=True)
    del err

    maxrel=MaxRelevance(mslrData, 4)
    print("MaxRelevance", maxrel.computeMetric(0, [0, 1, 2, 3]), flush=True)
    del maxrel

    sumrel=SumRelevance(mslrData, 4)
    print("SumRelevance", sumrel.computeMetric(0, [0, 1, 2, 3]), flush=True)
    del sumrel
