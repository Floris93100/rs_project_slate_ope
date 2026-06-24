###Classes that pre-process datasets for semi-synthetic experiments
import numpy
import scipy.sparse
import os
import os.path


class Datasets:
    def __init__(self):
        #Must call loadTxt(...) / loadNpz(...) / loadDeezer(...) to set these members
        #before using Datasets objects elsewhere
        self.relevances = None
        self.features = None
        self.docsPerQuery = None
        self.queryMappings = None
        self.name = None
        #For filtered datasets, some docsPerQuery may be masked
        self.mask = None
        #Extra members populated only by loadDeezer(...) (None for MSLR/MQ datasets)
        self.userFeatures = None        # x_u    (context representation)
        self.playlistFeatures = None    # theta_a (action weight vectors)
        self.userSegment = None         # k-means segment id per user

    ###As a side-effect, loadTxt(...) stores a npz file for
    ###faster subsequent loading via loadNpz(...)
    #file_name: (str) Path to dataset file (.txt format)
    #name: (str) String to identify this Datasets object henceforth
    def loadTxt(self, file_name, name):
        #Internal: Counters to keep track of docID and qID
        previousQueryID = None
        docID = None
        qID = 0
        relevanceArray = None
        #QueryMappings: list[int],length=numQueries
        self.queryMappings = []
        self.name = name
        #DocsPerQuery: list[int],length=numQueries
        self.docsPerQuery = []
        #Relevances: list[Alpha],length=numQueries; Alpha:= numpy.array[int],length=docsForQuery
        self.relevances = []
        #Features: list[Alpha],length=numQueries;
        #Alpha:= scipy.sparse.coo_matrix[double],shape=(docsForQuery, numFeatures)
        featureRows = None
        featureCols = None
        featureVals = None
        self.features = []
        numFeatures = None

        #Now read in data
        with open(file_name, 'r') as f:
            outputFilename = file_name[:-4]
            outputFileDir = outputFilename + '_processed'
            if not os.path.exists(outputFileDir):
                os.makedirs(outputFileDir)

            for line in f:
                tokens = line.split(' ', 2)
                relevance = int(tokens[0])
                queryID = int(tokens[1].split(':', 1)[1])
                #Remove any trailing comments before extracting features
                remainder = tokens[2].split('#', 1)
                featureTokens = remainder[0].strip().split(' ')
                if numFeatures is None:
                    numFeatures = len(featureTokens) + 1

                if (previousQueryID is None) or (queryID != previousQueryID):
                    #Begin processing a new query's documents
                    docID = 0
                    if relevanceArray is not None:
                        #Previous query's data should be persisted to file/self.members
                        currentRelevances = numpy.asarray(relevanceArray,
                                                         dtype=numpy.int64)
                        self.relevances.append(currentRelevances)
                        numpy.savez_compressed(os.path.join(outputFileDir, str(qID) + '_rel'),
                                               relevances=currentRelevances)
                        maxDocs = len(relevanceArray)
                        self.docsPerQuery.append(maxDocs)
                        currentFeatures = scipy.sparse.coo_matrix((featureVals, (featureRows, featureCols)),
                                                                  shape=(maxDocs, numFeatures), dtype=numpy.float64)
                        currentFeatures = currentFeatures.tocsr()
                        self.features.append(currentFeatures)
                        scipy.sparse.save_npz(os.path.join(outputFileDir, str(qID) + '_feat'),
                                              currentFeatures)
                        qID += 1
                        self.queryMappings.append(previousQueryID)
                        if len(self.docsPerQuery) % 100 == 0:
                            print(".", end="", flush=True)

                    relevanceArray = []
                    featureRows = []
                    featureCols = []
                    featureVals = []
                    previousQueryID = queryID
                else:
                    docID += 1

                relevanceArray.append(relevance)
                #Add a feature for the the intercept
                featureRows.append(docID)
                featureCols.append(0)
                featureVals.append(0.01)
                for featureToken in featureTokens:
                    featureTokenSplit = featureToken.split(':', 1)
                    featureIndex = int(featureTokenSplit[0])
                    featureValue = float(featureTokenSplit[1])
                    featureRows.append(docID)
                    featureCols.append(featureIndex)
                    featureVals.append(featureValue)

            #Finish processing the final query's data
            currentRelevances = numpy.asarray(relevanceArray, dtype=numpy.int64)
            self.relevances.append(currentRelevances)
            numpy.savez_compressed(os.path.join(outputFileDir, str(qID) + '_rel'),
                                   relevances=currentRelevances)
            maxDocs = len(relevanceArray)
            self.docsPerQuery.append(maxDocs)
            currentFeatures = scipy.sparse.coo_matrix((featureVals, (featureRows, featureCols)),
                                                      shape=(maxDocs, numFeatures), dtype=numpy.float64)
            currentFeatures = currentFeatures.tocsr()
            self.features.append(currentFeatures)
            scipy.sparse.save_npz(os.path.join(outputFileDir, str(qID) + '_feat'),
                                  currentFeatures)
            self.queryMappings.append(previousQueryID)

        #Persist meta-data for the dataset for faster loading through loadNpz
        numpy.savez_compressed(outputFilename, docsPerQuery=self.docsPerQuery,
                               name=self.name, queryMappings=self.queryMappings)
        print("", flush=True)
        print("Datasets:loadTxt [INFO] Loaded", file_name,
              "\t NumQueries", len(self.docsPerQuery),
              "\t [Min/Max]DocsPerQuery", min(self.docsPerQuery),
              max(self.docsPerQuery), flush=True)

    #file_name: (str) Path to dataset file/directory
    def loadNpz(self, file_name):
        with numpy.load(file_name+'.npz') as npFile:
            self.docsPerQuery=npFile['docsPerQuery']
            self.name=str(npFile['name'])
            self.queryMappings=npFile['queryMappings']

        allRelFile  = file_name+'_all_rel.npy'
        allFeatFile = file_name+'_all_feat.npz'
        if os.path.exists(allRelFile) and os.path.exists(allFeatFile):
            # Fast path: single-file consolidated load
            allRel  = numpy.load(allRelFile)
            allFeat = scipy.sparse.load_npz(allFeatFile)
            self.relevances = []
            self.features   = []
            rowOffsets = numpy.concatenate([[0], numpy.cumsum(self.docsPerQuery)])
            for i, dpq in enumerate(self.docsPerQuery):
                self.relevances.append(allRel[rowOffsets[i]:rowOffsets[i]+dpq])
                self.features.append(allFeat[rowOffsets[i]:rowOffsets[i]+dpq])
            print("Datasets:loadNpz [INFO] Loaded", file_name, "\t NumQueries", len(self.docsPerQuery),
                        "\t [Min/Max]DocsPerQuery", min(self.docsPerQuery),
                        max(self.docsPerQuery), "\t [Sum] docsPerQuery", sum(self.docsPerQuery), flush=True)
            return

        fileDir = file_name+'_processed'
        if os.path.exists(fileDir):
            self.relevances=[]
            self.features=[]

            qID=0
            while os.path.exists(os.path.join(fileDir, str(qID)+'_rel.npz')):
                with numpy.load(os.path.join(fileDir, str(qID)+'_rel.npz')) as currRelFile:
                    self.relevances.append(currRelFile['relevances'])

                self.features.append(scipy.sparse.load_npz(os.path.join(fileDir, str(qID)+'_feat.npz')))

                qID+=1

                if qID%100==0:
                    print(".", end="", flush=True)

        print("", flush=True)
        print("Datasets:loadNpz [INFO] Loaded", file_name, "\t NumQueries", len(self.docsPerQuery),
                    "\t [Min/Max]DocsPerQuery", min(self.docsPerQuery),
                    max(self.docsPerQuery), "\t [Sum] docsPerQuery", sum(self.docsPerQuery), flush=True)
        

    def loadDeezer(self, user_features_path, playlist_features_path, name="Deezer",
                   n_users=50000, seed=387, dtype=numpy.float32, cache_path=None):
        #Fast path: reload a previously built cache
        if cache_path is not None and os.path.exists(cache_path):
            self.loadDeezerNpz(cache_path)
            return

        import pandas as pd                    # lazy: only loadDeezer needs pandas
        from scipy.special import expit        # numerically stable logistic sigma(.)

        rng = numpy.random.RandomState(seed)
        self.name = name
        self.mask = None                       # set later when filtering A(x) to top-m

        pl = pd.read_csv(playlist_features_path)
        for idcol in ("Unnamed: 0", "index", "id", "playlist", "playlist_id"):
            if idcol in pl.columns:
                pl = pl.drop(columns=[idcol])
        theta = pl.select_dtypes(include=[numpy.number]).to_numpy(dtype=numpy.float64)   # (m, d)
        theta = numpy.nan_to_num(theta, nan=0.0, posinf=0.0, neginf=0.0)
        m, d = theta.shape
        if d != 97:
            print("Datasets:loadDeezer [WARN] theta width %d (expected 97); "
                  "check playlist_features.csv columns." % d, flush=True)

        emb_cols = ["dim_%d" % i for i in range(96)]
        keep_key = keep_emb = keep_seg = keep_idx = None
        k = n_users
        seen = 0
        for chunk in pd.read_csv(user_features_path, chunksize=100000):
            c = len(chunk)
            emb = chunk[emb_cols].to_numpy(dtype=numpy.float64)                       # (c, 96)
            seg = (chunk["segment"].to_numpy() if "segment" in chunk.columns
                   else numpy.full(c, -1, dtype=numpy.int64))
            idx = numpy.arange(seen, seen + c, dtype=numpy.int64)
            key = rng.random_sample(c)
            if keep_key is None:
                keep_key, keep_emb, keep_seg, keep_idx = key, emb, seg, idx
            else:
                keep_key = numpy.concatenate([keep_key, key])
                keep_emb = numpy.concatenate([keep_emb, emb])
                keep_seg = numpy.concatenate([keep_seg, seg])
                keep_idx = numpy.concatenate([keep_idx, idx])
            if k is not None and len(keep_key) > k:        # prune to k smallest keys
                sel = numpy.argpartition(keep_key, k)[:k]
                keep_key, keep_emb = keep_key[sel], keep_emb[sel]
                keep_seg, keep_idx = keep_seg[sel], keep_idx[sel]
            seen += c

        order = numpy.argsort(keep_idx)                    # restore CSV order (cosmetic)
        emb = keep_emb[order]
        emb = numpy.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
        self.userSegment = keep_seg[order].astype(numpy.int64)
        self.queryMappings = keep_idx[order]               # original CSV row id per user
        U = emb.shape[0]

        #Append the bias term: x_u = [ embedding(96) , 1 ] in R^d (matches the README).
        X = numpy.concatenate([emb, numpy.ones((U, 1))], axis=1)
        assert X.shape[1] == theta.shape[1], \
            "x_u dim (%d) != theta_a dim (%d)" % (X.shape[1], theta.shape[1])

        #   relevances[u][ranking] is what every carousel Metric reads.
        self.relevances = expit(X.dot(theta.T))
        self.relevances = numpy.nan_to_num(self.relevances, nan=0.0, posinf=1.0, neginf=0.0).astype(dtype)

        self.docsPerQuery = numpy.full(U, m, dtype=numpy.int64)      # |A(x)| = m for all x
        self.userFeatures = X
        self.playlistFeatures = theta

        theta_sparse = scipy.sparse.csr_matrix(theta)
        self.features = [theta_sparse] * U

        print("Datasets:loadDeezer [INFO] name=%s  |U|=%d  m=%d  d=%d  "
              "mean p=%.4f  max p=%.4f" %
              (name, U, m, d, float(self.relevances.mean()), float(self.relevances.max())),
              flush=True)

        if cache_path is not None:
            self._saveDeezerNpz(cache_path)

    def _saveDeezerNpz(self, cache_path):
        numpy.savez_compressed(cache_path,
                               relevances=self.relevances,
                               docsPerQuery=self.docsPerQuery,
                               queryMappings=self.queryMappings,
                               userFeatures=self.userFeatures,
                               playlistFeatures=self.playlistFeatures,
                               userSegment=self.userSegment,
                               name=self.name)
        print("Datasets:loadDeezer [INFO] cached ->", cache_path, flush=True)

    def loadDeezerNpz(self, cache_path):
        with numpy.load(cache_path, allow_pickle=False) as f:
            self.relevances = f['relevances']
            self.docsPerQuery = f['docsPerQuery']
            self.queryMappings = f['queryMappings']
            self.userFeatures = f['userFeatures']
            self.playlistFeatures = f['playlistFeatures']
            self.userSegment = f['userSegment']
            self.name = str(f['name'])
        self.mask = None
        theta_sparse = scipy.sparse.csr_matrix(self.playlistFeatures)
        self.features = [theta_sparse] * self.relevances.shape[0]
        print("Datasets:loadDeezerNpz [INFO] Loaded", cache_path,
              "\t |U|", self.relevances.shape[0], "\t m", self.relevances.shape[1], flush=True)


if __name__ == "__main__":
    import Settings
    """  
    mq2008Data=Datasets()
    mq2008Data.loadTxt(Settings.DATA_DIR+'MQ2008.txt', 'MQ2008')
    mq2008Data.loadNpz(Settings.DATA_DIR+'MQ2008')
    del mq2008Data

    mq2007Data=Datasets()
    mq2007Data.loadTxt(Settings.DATA_DIR+'MQ2007.txt', 'MQ2007')
    mq2007Data.loadNpz(Settings.DATA_DIR+'MQ2007')
    del mq2007Data
    """    
    deezer = Datasets()
    deezer.loadDeezer(
        Settings.DATA_DIR + 'Deezer/user_features.csv',
        Settings.DATA_DIR + 'Deezer/playlist_features.csv',
        n_users = 50000,
        seed = 387,
        cache_path = Settings.DATA_DIR + 'Deezer/deezer_50k.npz'
    )
    del deezer

    mslrData = Datasets()
    mslrData.loadTxt(Settings.DATA_DIR + 'MSLR-WEB10K/mslr.txt', 'MSLR10k')
    del mslrData

    for foldID in range(1, 6):
        for fraction in ['train', 'vali', 'test']:
            mslrData = Datasets()
            mslrData.loadTxt(Settings.DATA_DIR + 'MSLR-WEB10K\\Fold' + str(foldID) + '\\' + fraction + '.txt', 'MSLR10k-' + str(foldID) + '-' + fraction)
            del mslrData

    mslrData = Datasets()
    mslrData.loadTxt(Settings.DATA_DIR + 'MSLR/mslr.txt', 'MSLR')
    del mslrData

    for foldID in range(1, 6):
        for fraction in ['train', 'vali', 'test']:
            mslrData = Datasets()
            mslrData.loadTxt(Settings.DATA_DIR + 'MSLR\\Fold' + str(foldID) + '\\' + fraction + '.txt', 'MSLR-' + str(foldID) + '-' + fraction)
            del mslrData
