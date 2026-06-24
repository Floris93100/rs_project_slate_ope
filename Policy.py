###Class that models a policy for exploration or evaluation
import numpy
import scipy.sparse
import sklearn.model_selection
import sklearn.tree
import sklearn.ensemble
import sklearn.linear_model
import joblib
import os
import sys
import Settings
import GammaDP
import scipy.linalg
import itertools

#UniformGamma(...) computes a Gamma_pinv matrix for uniform exploration
#num_candidates: (int) Number of candidates, m
#ranking_size: (int) Size of slate, l
#allow_repetitions: (bool) If True, repetitions were allowed in the ranking
def UniformGamma(num_candidates, ranking_size, allow_repetitions):
    validDocs=ranking_size
    if not allow_repetitions:
        validDocs=min(ranking_size, num_candidates)

    gamma=numpy.empty((num_candidates*validDocs, num_candidates*validDocs), dtype=numpy.float64)
    if num_candidates==1:
        gamma.fill(1.0)
    else:
        #First set all the off-diagonal blocks
        if allow_repetitions:
            gamma.fill(1.0/(num_candidates*num_candidates))
        else:
            gamma.fill(1.0/(num_candidates*(num_candidates-1)))
            #Correct the diagonal of each off-diagonal block: Pairwise=0
            for p in range(1,validDocs):
                diag=numpy.diagonal(gamma, offset=p*num_candidates)
                diag.setflags(write=True)
                diag.fill(0)

                diag=numpy.diagonal(gamma, offset=-p*num_candidates)
                diag.setflags(write=True)
                diag.fill(0)

        #Now correct the diagonal blocks: Diagonal matrix with marginals = 1/m
        for j in range(validDocs):
            currentStart=j*num_candidates
            currentEnd=(j+1)*num_candidates
            gamma[currentStart:currentEnd, currentStart:currentEnd]=0
            numpy.fill_diagonal(gamma, 1.0/num_candidates)

    gammaInv = scipy.linalg.pinv(gamma + numpy.eye(gamma.shape[0]) * numpy.finfo(float).eps)
    return (num_candidates, gammaInv)


#NonUniformGamma(...) computes a Gamma_pinv matrix for non-uniform exploration
#num_candidates: (int) Number of candidates, m
#decay: (double) Decay factor. Doc Selection Prob \propto exp2(-decay * floor[ log2(rank) ])
#ranking_size: (int) Size of slate, l
#allow_repetitions: (bool) If True, repetitions were allowed in the ranking
def NonUniformGamma(num_candidates, decay, ranking_size, allow_repetitions):
    validDocs=ranking_size
    if not allow_repetitions:
        validDocs=min(ranking_size, num_candidates)

    multinomial=numpy.arange(1, num_candidates+1, dtype=numpy.float64)
    multinomial=numpy.exp2((-decay)*numpy.floor(numpy.log2(multinomial)))

    for i in range(1,num_candidates):
        prevVal=multinomial[i-1]
        currVal=multinomial[i]
        if numpy.isclose(currVal, prevVal):
            multinomial[i]=prevVal

    gamma=None
    if num_candidates==1:
        gamma=numpy.ones((num_candidates*validDocs, num_candidates*validDocs), dtype=numpy.longdouble)
    else:
        if allow_repetitions:
            offDiagonal=numpy.outer(multinomial, multinomial)
            gamma=numpy.tile(offDiagonal, (validDocs, validDocs))
            for j in range(validDocs):
                currentStart=j*num_candidates
                currentEnd=(j+1)*num_candidates
                gamma[currentStart:currentEnd, currentStart:currentEnd]=numpy.diag(multinomial)
        else:
            gammaVals=GammaDP.GammaCalculator(multinomial.tolist(), validDocs)
            gamma=numpy.diag(numpy.ravel(gammaVals.unitMarginals))

            for p in range(validDocs):
                for q in range(p+1, validDocs):
                    pairMarginals=gammaVals.pairwiseMarginals[(p,q)]
                    currentRowStart=p*num_candidates
                    currentRowEnd=(p+1)*num_candidates
                    currentColumnStart=q*num_candidates
                    currentColumnEnd=(q+1)*num_candidates
                    gamma[currentRowStart:currentRowEnd, currentColumnStart:currentColumnEnd]=pairMarginals
                    gamma[currentColumnStart:currentColumnEnd, currentRowStart:currentRowEnd]=pairMarginals.T

    normalizer=numpy.sum(multinomial, dtype=numpy.longdouble)
    multinomial=multinomial/normalizer

    gammaInv=scipy.linalg.pinv(gamma)
    return (num_candidates, multinomial, gammaInv)


class RecursiveSlateEval:
    def __init__(self, scores):
        self.m=scores.shape[0]
        self.l=scores.shape[1]
        self.scores=scores
        self.sortedIndices=numpy.argsort(scores, axis=0)
        self.bestSoFar=None
        self.bestSlate=None
        self.counter=0
        self.upperPos=numpy.amax(scores, axis=0)
        self.eval_slate([], 0.0)
        print(self.m, self.counter, flush=True)

    def eval_slate(self, slate_prefix, prefix_value):
        currentPos=len(slate_prefix)
        if currentPos==self.l:
            self.counter+=1
            if self.bestSoFar is None or prefix_value > self.bestSoFar:
                self.bestSoFar=prefix_value
                self.bestSlate=slate_prefix
            return

        docSet=set(slate_prefix)
        bestFutureVal=0.0
        if currentPos < self.l:
            bestFutureVal=self.upperPos[currentPos:].sum()
        delta=prefix_value+bestFutureVal
        for i in range(self.m):
            currentDoc=self.sortedIndices[-1-i, currentPos]
            if currentDoc in docSet:
                continue
            currentVal=self.scores[currentDoc, currentPos]
            if self.bestSoFar is None or ((currentVal+delta) > self.bestSoFar):
                self.eval_slate(slate_prefix + [currentDoc], prefix_value+currentVal)
            else:
                break

class Policy:
    #dataset: (Datasets) Must be initialized using Datasets.loadTxt(...)/loadNpz(...)
    #allow_repetitions: (bool) If true, the policy predicts rankings with repeated documents
    def __init__(self, dataset, allow_repetitions):
        self.dataset=dataset
        self.allowRepetitions=allow_repetitions
        self.name=None
        ###All sub-classes of Policy should supply a predict method
        ###Requires: (int) query_id; (int) ranking_size.
        ###Returns: list[int],length=min(ranking_size,docsPerQuery[query_id]) ranking


class L2RPolicy(Policy):
    def __init__(self, dataset, ranking_size, model_type, greedy_select, cross_features):
        Policy.__init__(self, dataset, False)
        self.rankingSize=ranking_size
        self.numDocFeatures=dataset.features[0].shape[1]
        self.modelType=model_type
        self.crossFeatures=cross_features
        self.hyperParams=numpy.logspace(0,2,num=5,base=10).tolist()
        if self.modelType=='tree' or self.modelType=='gbrt':
            self.tree=None
        else:
            self.policyParams=None

        self.greedy=greedy_select

        self.numFeatures=self.numDocFeatures+self.rankingSize
        if self.crossFeatures:
            self.numFeatures+=self.numDocFeatures*self.rankingSize
        print("L2RPolicy:init [INFO] Dataset:", dataset.name, flush=True)

    def createFeature(self, docFeatures, position):
        currFeature=numpy.zeros(self.numFeatures, dtype=numpy.float64)
        currFeature[0:self.numDocFeatures]=docFeatures
        currFeature[self.numDocFeatures+position]=1
        if self.crossFeatures:
            currFeature[self.numDocFeatures+self.rankingSize+position*self.numDocFeatures: \
                        self.numDocFeatures+self.rankingSize+(position+1)*self.numDocFeatures]=docFeatures

        return currFeature.reshape(1,-1)

    def predict(self, query_id, ranking_size):
        allowedDocs=self.dataset.docsPerQuery[query_id]
        validDocs=min(allowedDocs, self.rankingSize)

        allScores=numpy.zeros((allowedDocs, validDocs), dtype=numpy.float64)
        allFeatures=self.dataset.features[query_id].toarray()

        for doc in range(allowedDocs):
            docID=doc
            if self.dataset.mask is not None:
                docID=self.dataset.mask[query_id][doc]
            for pos in range(validDocs):
                currFeature=self.createFeature(allFeatures[docID,:], pos)

                if self.modelType=='tree' or self.modelType=='gbrt':
                    allScores[doc, pos]=self.tree.predict(currFeature)
                else:
                    allScores[doc, pos]=currFeature.dot(self.policyParams)

        tieBreaker=1e-14*numpy.random.random((allowedDocs, validDocs))
        allScores+=tieBreaker
        upperBound=numpy.amax(allScores, axis=0)

        producedRanking=None
        if self.greedy:
                                        # Tinus: heb dit veranderd naar int64, was eerst int32
            producedRanking=numpy.empty(validDocs, dtype=numpy.int64)
            currentVal=0.0
            for i in range(validDocs):
                maxIndex=numpy.argmax(allScores)
                chosenDoc,chosenPos = numpy.unravel_index(maxIndex, allScores.shape)
                currentVal+=allScores[chosenDoc, chosenPos]
                if self.dataset.mask is None:
                    producedRanking[chosenPos]=chosenDoc
                else:
                    producedRanking[chosenPos]=self.dataset.mask[query_id][chosenDoc]

                allScores[chosenDoc,:] = float('-inf')
                allScores[:,chosenPos] = float('-inf')

            self.debug=upperBound.sum()-currentVal
        else:
            slateScorer=RecursiveSlateEval(allScores)
            if self.dataset.mask is None:
                producedRanking=numpy.array(slateScorer.bestSlate)
            else:
                producedRanking=self.dataset.mask[slateScorer.bestSlate]

            self.debug=upperBound.sum()-slateScorer.bestSoFar
            del slateScorer

        del allFeatures
        del allScores

        return producedRanking

    def train(self, dataset, targets, hyper_params):
        numQueries=len(dataset.docsPerQuery)
        validDocs=numpy.minimum(dataset.docsPerQuery, self.rankingSize)
        queryDocPosTriplets=numpy.dot(dataset.docsPerQuery, validDocs)
        designMatrix=numpy.zeros((queryDocPosTriplets, self.numFeatures), dtype=numpy.float32, order='F')
        regressionTargets=numpy.zeros(queryDocPosTriplets, dtype=numpy.float64, order='C')
        sampleWeights=numpy.zeros(queryDocPosTriplets, dtype=numpy.float32)
        currID=-1
        for i in range(numQueries):
            numAllowedDocs=dataset.docsPerQuery[i]
            currValidDocs=validDocs[i]
            allFeatures=dataset.features[i].toarray()

            for doc in range(numAllowedDocs):
                docID=doc
                if dataset.mask is not None:
                    docID=dataset.mask[i][doc]

                for j in range(currValidDocs):
                    currID+=1

                    designMatrix[currID,:]=self.createFeature(allFeatures[docID,:], j)
                    regressionTargets[currID]=targets[i][j,doc]
                    sampleWeights[currID]=1.0/(numAllowedDocs * currValidDocs)

        for i in targets:
            del i
        del targets

        print("L2RPolicy:train [LOG] Finished creating features and targets ",
                numpy.amin(regressionTargets), numpy.amax(regressionTargets), numpy.median(regressionTargets), flush=True)
        print("L2RPolicy:train [LOG] Histogram of targets ", numpy.histogram(regressionTargets), flush=True)

        if self.modelType == 'gbrt':
            tree=sklearn.ensemble.GradientBoostingRegressor(learning_rate=hyper_params['lr'],
                            n_estimators=hyper_params['ensemble'], subsample=hyper_params['subsample'], max_leaf_nodes=hyper_params['leaves'],
                            max_features=1.0)
            tree.fit(designMatrix, regressionTargets, sample_weight=sampleWeights)
            self.tree=tree
            print("L2RPolicy:train [INFO] %s" % self.modelType, flush=True)

        elif self.modelType == 'ridge':
            ridgeCV=sklearn.linear_model.RidgeCV(alphas=self.hyperParams, fit_intercept=False,
                                                            cv=3)
            ridgeCV.fit(designMatrix, regressionTargets, sample_weight=sampleWeights)
            self.policyParams=ridgeCV.coef_
            print("L2RPolicy:train [INFO] Done. ", flush=True)

        else:
            print("L2RPolicy:train [ERR] %s not supported." % self.modelType, flush = True)
            sys.exit(0)

        print("L2R:train [INFO] Created %s predictor using dataset %s." %
                (self.modelType, dataset.name), flush = True)


class DeterministicPolicy(Policy):
    #model_type: (str) Model class to use for scoring documents
    def __init__(self, dataset, model_type, regress_gains=False, weighted_ls=False, hyper_params=None):
        Policy.__init__(self, dataset, False)
        self.modelType=model_type
        self.hyperParams={'alpha': (numpy.logspace(-3,2,num=6,base=10)).tolist()}
        if hyper_params is not None:
            self.hyperParams=hyper_params

        self.regressGains=regress_gains
        self.weighted=weighted_ls

        self.treeDepths={'max_depth': list(range(3,21,3))}

        #Must call train(...) to set all these members
        #before using DeterministicPolicy objects elsewhere
        self.featureList=None
        if self.modelType=='tree':
            self.tree=None
        else:
            self.policyParams=None

        #These members are set by predictAll(...) method
        self.savedRankingsSize=None
        self.savedRankings=None

        print("DeterministicPolicy:init [INFO] Dataset", dataset.name, flush=True)

    #feature_list: list[int],length=unmaskedFeatures; List of features that should be used for training
    #name: (str) String to help identify this DeterministicPolicy object henceforth
    def train(self, feature_list, name):
        self.featureList=feature_list
        self.name=name+'-'+self.modelType
        modelFile=Settings.DATA_DIR+self.dataset.name+'_'+self.name
        if 'alpha' not in self.hyperParams:
            #Expecting hyper-params for GBRT; Add those hyper-params to the model file name
            modelFile=modelFile+'ensemble-'+str(self.hyperParams['ensemble'])+'_lr-'+str(self.hyperParams['lr'])+'_subsample-'+str(self.hyperParams['subsample'])+'_leaves-'+str(self.hyperParams['leaves'])

        if self.modelType=='tree' or self.modelType=='gbrt':
            modelFile+='.z'
        else:
            modelFile+='.npz'

        self.savedRankingsSize=None
        self.savedRankings=None

        if os.path.exists(modelFile):
            if self.modelType=='tree' or self.modelType=='gbrt':
                self.tree=joblib.load(modelFile)
                print("DeterministicPolicy:train [INFO] Using precomputed policy", modelFile, flush=True)
            else:
                with numpy.load(modelFile) as npFile:
                    self.policyParams=npFile['policyParams']
                print("DeterministicPolicy:train [INFO] Using precomputed policy", modelFile, flush=True)
                print("DeterministicPolicy:train [INFO] PolicyParams", self.policyParams,flush=True)
        else:
            numQueries=len(self.dataset.features)

            allFeatures=None
            allTargets=None
            print("DeterministicPolicy:train [INFO] Constructing features and targets", flush=True)

            if self.dataset.mask is None:
                allFeatures=scipy.sparse.vstack(self.dataset.features, format='csc')
                allTargets=numpy.hstack(self.dataset.relevances)
            else:
                temporaryFeatures=[]
                temporaryTargets=[]
                for currentQuery in range(numQueries):
                    temporaryFeatures.append(self.dataset.features[currentQuery][self.dataset.mask[currentQuery], :])
                    temporaryTargets.append(self.dataset.relevances[currentQuery][self.dataset.mask[currentQuery]])

                allFeatures=scipy.sparse.vstack(temporaryFeatures, format='csc')
                allTargets=numpy.hstack(temporaryTargets)

            if self.regressGains:
                allTargets=numpy.exp2(allTargets)-1.0

            allSampleWeights=None
            fitParams=None
            if self.weighted:
                allSampleWeights=numpy.array(self.dataset.docsPerQuery, dtype=numpy.float64)
                allSampleWeights=numpy.reciprocal(allSampleWeights)
                allSampleWeights=numpy.repeat(allSampleWeights, self.dataset.docsPerQuery)
                fitParams={'sample_weight': allSampleWeights}

            #Restrict features to only the unmasked features
            if self.featureList is not None:
                print("DeterministicPolicy:train [INFO] Masking unused features. Remaining feature size",
                    len(feature_list), flush=True)
                allFeatures = allFeatures[:, self.featureList]

            print("DeterministicPolicy:train [INFO] Beginning training", self.modelType, flush=True)
            if self.modelType=='tree':
                treeCV=sklearn.model_selection.GridSearchCV(sklearn.tree.DecisionTreeRegressor(criterion="squared_error",
                                                        splitter="random", min_samples_split=4,
                                                        min_samples_leaf=4),
                                param_grid=self.treeDepths,
                                scoring=None, n_jobs=1,
                                cv=5, refit=True, verbose=0, pre_dispatch="1*n_jobs",
                                error_score='raise', return_train_score=False)

                if fitParams is not None:
                    treeCV.fit(allFeatures, allTargets, **fitParams)
                else:
                    treeCV.fit(allFeatures, allTargets)
                self.tree=treeCV.best_estimator_
                print("DeterministicPolicy:train [INFO] Done. Best depth",
                            treeCV.best_params_['max_depth'], flush=True)
                joblib.dump(self.tree, modelFile, compress=9, protocol=-1)

            elif self.modelType=='lasso':
                lassoCV=sklearn.model_selection.GridSearchCV(sklearn.linear_model.Lasso(fit_intercept=False,
                                                        precompute=False, copy_X=False,
                                                        max_iter=3000, tol=1e-4, warm_start=False, positive=False,
                                                        random_state=None, selection='random'),
                                param_grid=self.hyperParams,
                                scoring=None, n_jobs=1,
                                cv=5, refit=True, verbose=0, pre_dispatch="1*n_jobs",
                                error_score='raise', return_train_score=False)

                if fitParams is not None:
                    lassoCV.fit(allFeatures, allTargets, **fitParams)
                else:
                    lassoCV.fit(allFeatures, allTargets)
                self.policyParams=lassoCV.best_estimator_.coef_
                print("DeterministicPolicy:train [INFO] Done. CVAlpha", lassoCV.best_params_['alpha'], flush=True)
                print("DeterministicPolicy:train [INFO] PolicyParams", self.policyParams,flush=True)
                numpy.savez_compressed(modelFile, policyParams=self.policyParams)

            elif self.modelType == 'ridge':
                ridgeCV=sklearn.model_selection.GridSearchCV(sklearn.linear_model.Ridge(fit_intercept=False,
                                                                                    copy_X=False,
                                                                                    max_iter=3000, tol=1e-4, random_state=None),
                                                         param_grid=self.hyperParams,
                                                         n_jobs=1,
                                                         cv=3, refit=True, verbose=0, pre_dispatch='1*n_jobs')
                if fitParams is not None:
                    ridgeCV.fit(allFeatures, allTargets, **fitParams)
                else:
                    ridgeCV.fit(allFeatures, allTargets)
                self.policyParams=ridgeCV.best_estimator_.coef_
                print("DeterministicPolicy:train [INFO] Done. CVAlpha", ridgeCV.best_params_['alpha'], flush=True)
            elif self.modelType=='gbrt':
                tree=sklearn.ensemble.GradientBoostingRegressor(learning_rate=self.hyperParams['lr'],
                            n_estimators=self.hyperParams['ensemble'], subsample=self.hyperParams['subsample'], max_leaf_nodes=self.hyperParams['leaves'],
                            max_features=1.0)
                tree.fit(allFeatures, allTargets, sample_weight=allSampleWeights)
                self.tree=tree
                print("DeterministicPolicy:train [INFO] Done.", flush=True)
                joblib.dump(self.tree, modelFile, compress=9, protocol=-1)

            else:
                print("DeterministicPolicy:train [ERR] %s not supported." % self.modelType, flush=True)
                sys.exit(0)

    #query_id: (int) Query ID in self.dataset
    #ranking_size: (int) Size of ranking. Returned ranking length is min(ranking_size,docsPerQuery[query_id])
    #                       Use ranking_size=-1 to rank all available documents for query_id
    def predict(self, query_id, ranking_size):
        if self.savedRankingsSize is not None and self.savedRankingsSize==ranking_size:
            return self.savedRankings[query_id]

        allowedDocs=self.dataset.docsPerQuery[query_id]
        validDocs=ranking_size
        if ranking_size <= 0 or validDocs > allowedDocs:
            validDocs=allowedDocs

        currentFeatures=None
        if self.dataset.mask is None:
            if self.featureList is not None:
                currentFeatures=self.dataset.features[query_id][:, self.featureList]
            else:
                currentFeatures=self.dataset.features[query_id]

        else:
            currentFeatures=self.dataset.features[query_id][self.dataset.mask[query_id], :]
            if self.featureList is not None:
                currentFeatures=currentFeatures[:, self.featureList]

        allDocScores=None
        if self.modelType=='tree':
            allDocScores=self.tree.predict(currentFeatures)
        elif self.modelType=='gbrt':
            allDocScores=self.tree.predict(currentFeatures.toarray())
        else:
            allDocScores=currentFeatures.dot(self.policyParams)

        tieBreaker=numpy.random.random(allDocScores.size)
        sortedDocScores=numpy.lexsort((tieBreaker,-allDocScores))[0:validDocs]
        if self.dataset.mask is None:
            return sortedDocScores
        else:
            return self.dataset.mask[query_id][sortedDocScores]

    #ranking_size: (int) Size of ranking. Returned ranking length is min(ranking_size,docsPerQuery[query_id])
    #                       Use ranking_size=-1 to rank all available documents for query_id
    def predictAll(self, ranking_size):
        if self.savedRankingsSize is not None and self.savedRankingsSize==ranking_size:
            return

        numQueries=len(self.dataset.features)
        predictedRankings=[]
        for i in range(numQueries):
            predictedRankings.append(self.predict(i, ranking_size))

            if i%100==0:
                print(".", end="", flush=True)

        self.savedRankingsSize=ranking_size
        self.savedRankings=predictedRankings
        print("", flush=True)
        print("DeterministicPolicy:predictAll [INFO] Generated all predictions for %s using policy: " %
                self.dataset.name, self.name, flush=True)

    #num_allowed_docs: (int) Filters the dataset where the max docs per query is num_allowed_docs.
    #                        Uses policyParams to rank and filter the original document set.
    def filterDataset(self, num_allowed_docs):
        self.savedRankingsSize=None
        self.savedRankings=None

        numQueries=len(self.dataset.docsPerQuery)

        self.dataset.name=self.dataset.name+'-filt('+self.name+'-'+str(num_allowed_docs)+')'

        newMask = []
        for i in range(numQueries):
            producedRanking=self.predict(i, num_allowed_docs)
            self.dataset.docsPerQuery[i]=numpy.shape(producedRanking)[0]
            newMask.append(producedRanking)
            if i%100==0:
                print(".", end="", flush=True)

        self.dataset.mask=newMask
        print("", flush=True)
        print("DeterministicPolicy:filteredDataset [INFO] New Name", self.dataset.name, "\t MaxNumDocs", num_allowed_docs, flush=True)


class UniformPolicy(Policy):
    def __init__(self, dataset, allow_repetitions):
        Policy.__init__(self, dataset, allow_repetitions)
        self.name='Unif-'
        if allow_repetitions:
            self.name+='Rep'
        else:
            self.name+='NoRep'

        #These members are set on-demand by setupGamma(...)
        self.gammas=None
        self.gammaRankingSize=None

        print("UniformPolicy:init [INFO] Dataset: %s AllowRepetitions:" % dataset.name,
                        allow_repetitions, flush=True)

    #ranking_size: (int) Size of ranking.
    def setupGamma(self, ranking_size):
        if self.gammaRankingSize is not None and self.gammaRankingSize==ranking_size:
            print("UniformPolicy:setupGamma [INFO] Gamma has been pre-computed for this ranking_size. Size of Gamma cache:", len(self.gammas), flush=True)
            return

        gammaFile=Settings.DATA_DIR+self.dataset.name+'_'+self.name+'_'+str(ranking_size)+'.z'
        if os.path.exists(gammaFile):
            self.gammas=joblib.load(gammaFile)
            self.gammaRankingSize=ranking_size
            print("UniformPolicy:setupGamma [INFO] Using precomputed gamma", gammaFile, flush=True)

        else:
            self.gammas={}
            self.gammaRankingSize=ranking_size

            candidateSet=set(self.dataset.docsPerQuery)

            responses=joblib.Parallel(n_jobs=4, verbose=50)(joblib.delayed(UniformGamma)(i, ranking_size, self.allowRepetitions) for i in candidateSet)

            for tup in responses:
                self.gammas[tup[0]]=tup[1]

            joblib.dump(self.gammas, gammaFile, compress=9, protocol=-1)
            print("", flush=True)
            print("UniformPolicy:setupGamma [INFO] Finished creating Gamma_pinv cache. Size", len(self.gammas), flush=True)

    def predict(self, query_id, ranking_size):
        allowedDocs=self.dataset.docsPerQuery[query_id]

        validDocs=ranking_size
        if ranking_size < 0 or ((not self.allowRepetitions) and (validDocs > allowedDocs)):
            validDocs=allowedDocs

        producedRanking=None
        if self.allowRepetitions:
            producedRanking=numpy.random.choice(allowedDocs, size=validDocs,
                                replace=True)
        else:
            producedRanking=numpy.random.choice(allowedDocs, size=validDocs,
                                replace=False)

        if self.dataset.mask is None:
            return producedRanking
        else:
            return self.dataset.mask[query_id][producedRanking]


class NonUniformPolicy(Policy):
    def __init__(self, deterministic_policy, dataset, allow_repetitions, decay):
        Policy.__init__(self, dataset, allow_repetitions)
        self.decay = decay
        self.policy = deterministic_policy
        self.name='NonUnif-'
        if allow_repetitions:
            self.name+='Rep'
        else:
            self.name+='NoRep'
        self.name += '(' + deterministic_policy.name + ';' + str(decay) + ')'

        #These members are set on-demand by setupGamma
        self.gammas=None
        self.multinomials=None
        self.gammaRankingSize=None

        print("NonUniformPolicy:init [INFO] Dataset: %s AllowRepetitions:" % dataset.name,
                        allow_repetitions, "\t Decay:", decay, flush=True)


    def setupGamma(self, ranking_size):
        if self.gammaRankingSize is not None and self.gammaRankingSize==ranking_size:
            print("NonUniformPolicy:setupGamma [INFO] Gamma has been pre-computed for this ranking_size. Size of Gamma cache:", len(self.gammas), flush=True)
            return

        gammaFile=Settings.DATA_DIR+self.dataset.name+'_'+self.name+'_'+str(ranking_size)+'.z'
        if os.path.exists(gammaFile):
            self.gammas, self.multinomials=joblib.load(gammaFile)
            self.gammaRankingSize=ranking_size
            print("NonUniformPolicy:setupGamma [INFO] Using precomputed gamma", gammaFile, flush=True)

        else:
            self.gammas={}
            self.multinomials={}
            self.gammaRankingSize=ranking_size

            candidateSet=set(self.dataset.docsPerQuery)
            responses=joblib.Parallel(n_jobs=4, verbose=50)(joblib.delayed(NonUniformGamma)(i, self.decay, ranking_size, self.allowRepetitions) for i in candidateSet)

            for tup in responses:
                self.gammas[tup[0]]=tup[2]
                self.multinomials[tup[0]]=tup[1]

            joblib.dump((self.gammas, self.multinomials), gammaFile, compress=9, protocol=-1)
            print("", flush=True)
            print("NonUniformPolicy:setupGamma [INFO] Finished creating Gamma_pinv cache. Size", len(self.gammas), flush=True)

        self.policy.predictAll(-1)

    def predict(self, query_id, ranking_size):
        allowedDocs=self.dataset.docsPerQuery[query_id]
        underlyingRanking=self.policy.predict(query_id, -1)

        validDocs=ranking_size
        if ranking_size < 0 or ((not self.allowRepetitions) and (validDocs > allowedDocs)):
            validDocs=allowedDocs
            
        currentDistribution=numpy.asarray(self.multinomials[allowedDocs], dtype=numpy.float64)
        # Floris: np.random.choice requires p to be float64
        p64 = numpy.asarray(currentDistribution, dtype=numpy.float64)
        p64 = p64 / p64.sum()
        producedRanking=None
        if self.allowRepetitions:
            producedRanking=numpy.random.choice(allowedDocs, size=validDocs,
                                replace=True, p=p64)
        else:
            producedRanking=numpy.random.choice(allowedDocs, size=validDocs,
                                replace=False, p=p64)
                                
        return underlyingRanking[producedRanking]
        
        
def _valid_docs(dataset, query_id, ranking_size):
    """Length of a slate for this context: min(l, |A(x)|), with l<0 meaning 'all'."""
    allowed = int(dataset.docsPerQuery[query_id])
    if ranking_size is None or ranking_size < 0:
        return allowed
    return min(int(ranking_size), allowed)


def uniform_slate_propensity(num_candidates, slate_len, replacement=False):
    """mu_uniform(s | x) for an ordered slate of distinct playlists."""
    m = int(num_candidates)
    l = int(slate_len)
    if replacement:
        return 1.0 / (float(m) ** l)
    denom = 1.0                         # m * (m-1) * ... * (m-l+1)
    for k in range(l):
        denom *= float(m - k)
    return 1.0 / denom

class _DeterministicTargetBase(object):
    """Shared machinery for deterministic top-l targets built from a score matrix."""

    def __init__(self, dataset, name):
        self.dataset = dataset
        self.name = name
        self.predictions = None         # frozen (|U|, l) int64 array
        self._predictionSize = None

    def _scores(self):
        raise NotImplementedError

    def train(self, *args, **kwargs):
        """Most Deezer targets need no fitting; DeezerLogisticPolicy overrides."""
        pass

    def predictAll(self, ranking_size):
        m = self.dataset.relevances.shape[1]
        l = m if (ranking_size is None or ranking_size < 0) else min(int(ranking_size), m)
        scores = self._scores()
        order = numpy.argsort(-scores, axis=1, kind="stable")     # stable: ties -> lower index
        self.predictions = numpy.ascontiguousarray(order[:, :l], dtype=numpy.int64)
        self._predictionSize = l
        return self.predictions

    def predict(self, query_id, ranking_size):
        l = _valid_docs(self.dataset, query_id, ranking_size)
        if self.predictions is None or self._predictionSize < l:
            self.predictAll(l if (ranking_size is None or ranking_size < 0) else ranking_size)
        return self.predictions[query_id, :l]


class DeezerOptimalPolicy(_DeterministicTargetBase):
    """Optimal target: top-l by the true p(u,a) = x_u^T theta_a (the skyline)."""

    def __init__(self, dataset):
        _DeterministicTargetBase.__init__(self, dataset, "DeezerOptimal")

    def _scores(self):
        return numpy.asarray(self.dataset.relevances, dtype=numpy.float64)


class DeezerPopularityPolicy(_DeterministicTargetBase):
    """Popularity target: top-l by global mean p(.,a) -- the same slate for every user."""

    def __init__(self, dataset):
        _DeterministicTargetBase.__init__(self, dataset, "DeezerPopularity")

    def _scores(self):
        rel = numpy.asarray(self.dataset.relevances, dtype=numpy.float64)
        pop = rel.mean(axis=0, dtype=numpy.float64)              # (m,)
        return numpy.broadcast_to(pop, rel.shape)                # (|U|, m) view


class DeezerSegmentPolicy(_DeterministicTargetBase):
    """Segment target: top-l by the mean probability within the user's k-means segment."""

    def __init__(self, dataset):
        _DeterministicTargetBase.__init__(self, dataset, "DeezerSegment")

    def _scores(self):
        rel = numpy.asarray(self.dataset.relevances, dtype=numpy.float64)
        U, m = rel.shape
        seg = getattr(self.dataset, "userSegment", None)
        if seg is None:
            seg = numpy.full(U, -1, dtype=numpy.int64)
        seg = numpy.asarray(seg, dtype=numpy.int64)

        globalPop = rel.mean(axis=0, dtype=numpy.float64)        # fallback for seg=-1

        scores = numpy.empty((U, m), dtype=numpy.float64)
        uniqueSegs = numpy.unique(seg)
        for s in uniqueSegs:
            rows = numpy.flatnonzero(seg == s)
            if s < 0:
                scores[rows, :] = globalPop
            else:
                scores[rows, :] = rel[rows].mean(axis=0, dtype=numpy.float64)
        return scores


class DeezerRandomPolicy(_DeterministicTargetBase):
    """Random target: a fixed (seeded) per-user random top-l, near the mu target."""

    def __init__(self, dataset, seed=387):
        _DeterministicTargetBase.__init__(self, dataset, "DeezerRandom")
        self._seed = int(seed)
        self._frozenScores = None

    def _scores(self):
        if self._frozenScores is None:
            rng = numpy.random.RandomState(self._seed)
            U, m = self.dataset.relevances.shape
            self._frozenScores = rng.random_sample((U, m)).astype(numpy.float64)
        return self._frozenScores


class DeezerLogisticPolicy(_DeterministicTargetBase):
    """Learned target: a logistic ranker fit to logged (user, playlist, stream) outcomes."""

    def __init__(self, dataset, C=1.0):
        _DeterministicTargetBase.__init__(self, dataset, "DeezerLogistic")
        self._C = float(C)
        self._weights = None            # (d,) learned coefficients
        self._bias = 0.0

    def _pairFeatures(self, user_ids, playlist_ids):
        X = numpy.asarray(self.dataset.userFeatures, dtype=numpy.float64)
        T = numpy.asarray(self.dataset.playlistFeatures, dtype=numpy.float64)
        return X[user_ids] * T[playlist_ids]          # (n, d) elementwise product

    def fitFromLoggedStreams(self, pairs):
        """Fit a logistic ranker on (user_id, playlist_id, stream_bit) triples."""
        import sklearn.linear_model
        pairs = list(pairs)
        if len(pairs) == 0:
            raise ValueError("DeezerLogisticPolicy.fitFromLoggedStreams: empty log.")
        users = numpy.fromiter((p[0] for p in pairs), dtype=numpy.int64, count=len(pairs))
        plays = numpy.fromiter((p[1] for p in pairs), dtype=numpy.int64, count=len(pairs))
        y = numpy.fromiter((int(p[2]) for p in pairs), dtype=numpy.int64, count=len(pairs))
        feats = self._pairFeatures(users, plays)

        if numpy.unique(y).size < 2:
            self._weights = feats.mean(axis=0)        # degenerate log: mean-feature fallback
            self._bias = 0.0
            print("DeezerLogisticPolicy:fit [WARN] Only one stream class in log; "
                  "using mean-feature fallback.", flush=True)
            return self

        clf = sklearn.linear_model.LogisticRegression(
            C=self._C, fit_intercept=True, max_iter=2000, solver="lbfgs")
        clf.fit(feats, y)
        self._weights = clf.coef_.ravel().astype(numpy.float64)
        self._bias = float(clf.intercept_.ravel()[0])
        print("DeezerLogisticPolicy:fit [INFO] Trained on %d logged streams "
              "(positive rate %.4f)." % (len(pairs), y.mean()), flush=True)
        return self

    def fitFromDataset(self, n_samples=200000, seed=387):
        """Emulate a stream log: sample (u, a) uniformly and draw Bernoulli(p(u,a))."""
        rng = numpy.random.RandomState(int(seed))
        U, m = self.dataset.relevances.shape
        users = rng.randint(0, U, size=int(n_samples))
        plays = rng.randint(0, m, size=int(n_samples))
        probs = numpy.asarray(self.dataset.relevances, dtype=numpy.float64)[users, plays]
        streams = (rng.random_sample(int(n_samples)) < probs).astype(numpy.int64)
        pairs = zip(users.tolist(), plays.tolist(), streams.tolist())
        return self.fitFromLoggedStreams(pairs)

    def train(self, *args, **kwargs):
        """Fit from the dataset if not already fit; called by Parallel.py."""
        if self._weights is None:
            self.fitFromDataset(kwargs.get("n_samples", 200000),
                                kwargs.get("seed", 387))

    def _scores(self):
        if self._weights is None:
            self.fitFromDataset()
        X = numpy.asarray(self.dataset.userFeatures, dtype=numpy.float64)      # (U, d)
        T = numpy.asarray(self.dataset.playlistFeatures, dtype=numpy.float64)  # (m, d)
        weightedX = X * self._weights[numpy.newaxis, :]
        return weightedX.dot(T.T) + self._bias                                 # (U, m)


def make_uniform_logger(dataset, ranking_size, allow_repetitions=False):
    """Build + Gamma-setup the primary uniform logger (closed-form Gamma_pinv)."""
    logger = UniformPolicy(dataset, allow_repetitions)
    logger.setupGamma(ranking_size)
    return logger


class SoftmaxLogger(object):
    """Auxiliary-model Plackett-Luce logger: sample slots without replacement
    with P(pick a | remaining) proportional to exp(alpha * score(u,a))."""

    def __init__(self, dataset, scores, alpha=1.0, replacement=False,
                 seed=387, name=None):
        self.dataset = dataset
        self.scores = numpy.asarray(scores, dtype=numpy.float64)     # (U, m)
        self.alpha = float(alpha)
        self.replacement = bool(replacement)
        self.allowRepetitions = bool(replacement)
        self._rng = numpy.random.RandomState(int(seed))
        self.name = name if name is not None else ("Softmax(a=%g)" % self.alpha)

    def _logits(self, query_id):
        return self.alpha * self.scores[query_id]                    # (m,)

    def predict(self, query_id, ranking_size):
        l = _valid_docs(self.dataset, query_id, ranking_size)
        logits = self._logits(query_id)
        m = logits.shape[0]
        w = numpy.exp(logits - logits.max())                # numerically stable softmax
        chosen = numpy.empty(l, dtype=numpy.int64)
        available = numpy.ones(m, dtype=bool)
        for j in range(l):
            probs = numpy.where(available, w, 0.0)
            total = probs.sum()
            probs = probs / total
            pick = self._rng.choice(m, p=probs)
            chosen[j] = pick
            if not self.replacement:
                available[pick] = False
        return chosen

    def slate_propensity(self, query_id, ranking):
        """mu(ranking | x) for the ordered slate under the Plackett-Luce model."""
        ranking = numpy.asarray(ranking, dtype=numpy.int64)
        logits = self._logits(query_id)
        w = numpy.exp(logits - logits.max())
        prob = 1.0
        remainingMask = numpy.ones(w.shape[0], dtype=bool)
        for a in ranking:
            denom = w[remainingMask].sum()
            prob *= w[a] / denom
            if not self.replacement:
                remainingMask[a] = False
        return float(prob)


class EpsUniformLogger(object):
    """eps-mixed logger: mu_eps = (1-eps) * base + eps * uniform, for absolute continuity."""

    def __init__(self, base_logger, epsilon=0.1, name=None):
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError("epsilon must be in [0, 1].")
        self.base = base_logger
        self.dataset = base_logger.dataset
        self.epsilon = float(epsilon)
        self.replacement = getattr(base_logger, "replacement", False)
        self.allowRepetitions = self.replacement
        self._rng = numpy.random.RandomState(
            getattr(base_logger, "_rng", numpy.random.RandomState(387)).randint(1 << 30))
        base_name = getattr(base_logger, "name", "base")
        self.name = name if name is not None else ("EpsUnif(%g)+%s" % (self.epsilon, base_name))

    def predict(self, query_id, ranking_size):
        l = _valid_docs(self.dataset, query_id, ranking_size)
        if self._rng.random_sample() < self.epsilon:
            m = int(self.dataset.docsPerQuery[query_id])
            return self._rng.choice(m, size=l, replace=self.replacement).astype(numpy.int64)
        return numpy.asarray(self.base.predict(query_id, ranking_size), dtype=numpy.int64)

    def slate_propensity(self, query_id, ranking):
        ranking = numpy.asarray(ranking, dtype=numpy.int64)
        m = int(self.dataset.docsPerQuery[query_id])
        pUnif = uniform_slate_propensity(m, ranking.size, self.replacement)
        pBase = self.base.slate_propensity(query_id, ranking)
        return (1.0 - self.epsilon) * pBase + self.epsilon * pUnif


def popularity_scores(dataset):
    """Global mean p(.,a) broadcast to every user -- a weak, policy-independent model."""
    rel = numpy.asarray(dataset.relevances, dtype=numpy.float64)
    return numpy.broadcast_to(rel.mean(axis=0, dtype=numpy.float64), rel.shape).copy()


def noisy_logistic_scores(dataset, n_samples=200000, seed=387):
    """A separate, weaker logistic model's logits -- a realistic auxiliary logger."""
    aux = DeezerLogisticPolicy(dataset)
    aux.fitFromDataset(n_samples=n_samples, seed=seed)
    return aux._scores()


def computePolicyValue(target_policy, metric, dataset, ranking_size, verbose=False):
    """Compute V(pi) = mean_u V(x_u, pi(x_u)), the gold standard every RMSE is graded on."""
    target_policy.predictAll(ranking_size)
    numQueries = len(dataset.docsPerQuery)
    value = 0.0
    for u in range(numQueries):
        slate = target_policy.predict(u, ranking_size)
        currentVal = metric.computeMetric(u, slate)
        value += (currentVal - value) / (u + 1)        # running mean
        if verbose and (u % 5000 == 0):
            print(".", end="", flush=True)
    if verbose:
        print("", flush=True)
    return float(value)


def _valueCacheKey(target_policy, metric, dataset, ranking_size):
    """Cache key for a (target, metric, m, l) policy-value lookup."""
    m = int(dataset.relevances.shape[1])
    return "Vpi_%s_%s_m%d_l%d" % (target_policy.name, metric.name, m, int(ranking_size))


def cachedPolicyValue(target_policy, metric, dataset, ranking_size,
                      cache_dir=None, recompute=False, verbose=False):
    """computePolicyValue, memoised to <cache_dir>/<key>.npz keyed by (target,reward,m,l)."""
    key = _valueCacheKey(target_policy, metric, dataset, ranking_size)
    path = None
    if cache_dir is not None:
        path = os.path.join(cache_dir, key + ".npz")
        if (not recompute) and os.path.exists(path):
            with numpy.load(path) as f:
                val = float(f["value"])
            print("DeezerPolicies:cachedPolicyValue [INFO] Loaded %s = %.6f" % (key, val),
                  flush=True)
            return val

    val = computePolicyValue(target_policy, metric, dataset, ranking_size, verbose=verbose)
    print("DeezerPolicies:cachedPolicyValue [LOG] %s = %.6f" % (key, val), flush=True)
    if path is not None:
        numpy.savez_compressed(path, value=numpy.float64(val), key=key)
        print("DeezerPolicies:cachedPolicyValue [INFO] cached -> %s" % path, flush=True)
    return val


TARGET_CHOICES = ("Optimal", "Segment", "LearnedLogistic", "Popularity", "Random")


def build_target_policy(name, dataset, seed=387, logistic_samples=200000):
    """Return an (untrained-where-needed) target policy by short name."""
    if name == "Optimal":
        return DeezerOptimalPolicy(dataset)
    if name == "Popularity":
        return DeezerPopularityPolicy(dataset)
    if name == "Segment":
        return DeezerSegmentPolicy(dataset)
    if name == "Random":
        return DeezerRandomPolicy(dataset, seed=seed)
    if name == "LearnedLogistic":
        pol = DeezerLogisticPolicy(dataset)
        pol.fitFromDataset(n_samples=logistic_samples, seed=seed)
        return pol
    raise ValueError("Unknown target policy '%s'. Choices: %s" % (name, TARGET_CHOICES))

def filter_deezer_candidates(dataset, max_docs):
        max_docs = int(max_docs)
        full_m = dataset.relevances.shape[1]

        if max_docs <= 0 or max_docs >= full_m:
            print("Parallel:filter_deezer_candidates [INFO] No Deezer filtering.", flush=True)
            return

        keep = numpy.argsort(-dataset.relevances.mean(axis=0))[:max_docs]
        keep = numpy.asarray(keep, dtype=numpy.int64)

        dataset.relevances = dataset.relevances[:, keep]
        dataset.docsPerQuery = numpy.full(dataset.relevances.shape[0], max_docs, dtype=numpy.int64)
        dataset.mask = None

        if dataset.playlistFeatures is not None:
            dataset.playlistFeatures = dataset.playlistFeatures[keep]
            import scipy.sparse
            theta_sparse = scipy.sparse.csr_matrix(dataset.playlistFeatures)
            dataset.features = [theta_sparse] * dataset.relevances.shape[0]

        dataset.name = dataset.name + "-popfilt(%d)" % max_docs

        print(
            "Parallel:filter_deezer_candidates [INFO] Filtered Deezer from m=%d to m=%d"
            % (full_m, max_docs),
            flush=True
        )

if __name__=="__main__":
    import Settings
    import Datasets
    import scipy.stats

    M=100
    L=10
    resetSeed=387

    mslrData=Datasets.Datasets()
    mslrData.loadNpz(Settings.DATA_DIR+'MSLR/mslr')

    anchorURLFeatures, bodyTitleDocFeatures=Settings.get_feature_sets("MSLR")

    numpy.random.seed(resetSeed)
    detLogger=DeterministicPolicy(mslrData, 'tree')
    detLogger.train(anchorURLFeatures, 'url')

    detLogger.filterDataset(M)

    filteredDataset=detLogger.dataset
    del mslrData
    del detLogger

    uniform=UniformPolicy(filteredDataset, False)
    uniform.setupGamma(L)
    del uniform

    numpy.random.seed(resetSeed)
    loggingPolicyTree=DeterministicPolicy(filteredDataset, 'tree')
    loggingPolicyTree.train(anchorURLFeatures, 'url')

    numpy.random.seed(resetSeed)
    targetPolicyTree=DeterministicPolicy(filteredDataset, 'tree')
    targetPolicyTree.train(bodyTitleDocFeatures, 'body')

    numpy.random.seed(resetSeed)
    loggingPolicyLinear=DeterministicPolicy(filteredDataset, 'lasso')
    loggingPolicyLinear.train(anchorURLFeatures, 'url')

    numpy.random.seed(resetSeed)
    targetPolicyLinear=DeterministicPolicy(filteredDataset, 'lasso')
    targetPolicyLinear.train(bodyTitleDocFeatures, 'body')

    numQueries=len(filteredDataset.docsPerQuery)

    TTtau=[]
    TToverlap=[]
    TLtau=[]
    TLoverlap=[]
    LTtau=[]
    LToverlap=[]
    LLtau=[]
    LLoverlap=[]
    LogLogtau=[]
    LogLogoverlap=[]
    TargetTargettau=[]
    TargetTargetoverlap=[]

    def computeTau(ranking1, ranking2):
        rank1set=set(ranking1)
        rank2set=set(ranking2)
        documents=rank1set | rank2set
        rankingSize=len(rank1set)

        newRanking1=numpy.zeros(len(documents), dtype=numpy.int64)
        newRanking2=numpy.zeros(len(documents), dtype=numpy.int64)

        for docID, doc in enumerate(documents):
            if doc not in rank1set:
                newRanking1[docID]=rankingSize + 1
                newRanking2[docID]=ranking2.index(doc)
            elif doc not in rank2set:
                newRanking2[docID]=rankingSize + 1
                newRanking1[docID]=ranking1.index(doc)
            else:
                newRanking1[docID]=ranking1.index(doc)
                newRanking2[docID]=ranking2.index(doc)

        return scipy.stats.kendalltau(newRanking1, newRanking2)[0], 1.0*len(rank1set&rank2set)/rankingSize

    numpy.random.seed(resetSeed)
    for currentQuery in range(numQueries):
        if filteredDataset.docsPerQuery[currentQuery]<4:
            continue

        logTreeRanking=loggingPolicyTree.predict(currentQuery, L).tolist()
        logLinearRanking=loggingPolicyLinear.predict(currentQuery, L).tolist()

        targetTreeRanking=targetPolicyTree.predict(currentQuery, L).tolist()
        targetLinearRanking=targetPolicyLinear.predict(currentQuery, L).tolist()

        tau, overlap=computeTau(logTreeRanking, targetTreeRanking)
        TTtau.append(tau)
        TToverlap.append(overlap)

        tau, overlap=computeTau(logTreeRanking, targetLinearRanking)
        TLtau.append(tau)
        TLoverlap.append(overlap)

        tau, overlap=computeTau(logLinearRanking, targetTreeRanking)
        LTtau.append(tau)
        LToverlap.append(overlap)

        tau, overlap=computeTau(logLinearRanking, targetLinearRanking)
        LLtau.append(tau)
        LLoverlap.append(overlap)

        tau, overlap=computeTau(logLinearRanking, logTreeRanking)
        LogLogtau.append(tau)
        LogLogoverlap.append(overlap)

        tau, overlap=computeTau(targetLinearRanking, targetTreeRanking)
        TargetTargettau.append(tau)
        TargetTargetoverlap.append(overlap)

        if len(TTtau) % 100 == 0:
            print(".", end="", flush=True)

    TTtau=numpy.array(TTtau)
    TLtau=numpy.array(TLtau)
    LTtau=numpy.array(LTtau)
    LLtau=numpy.array(LLtau)
    LogLogtau=numpy.array(LogLogtau)
    TargetTargettau=numpy.array(TargetTargettau)

    TToverlap=numpy.array(TToverlap)
    TLoverlap=numpy.array(TLoverlap)
    LToverlap=numpy.array(LToverlap)
    LLoverlap=numpy.array(LLoverlap)
    LogLogoverlap=numpy.array(LogLogoverlap)
    TargetTargetoverlap=numpy.array(TargetTargetoverlap)

    print("", flush=True)
    print("TTtau", numpy.amax(TTtau), numpy.amin(TTtau), numpy.mean(TTtau), numpy.std(TTtau), numpy.median(TTtau), len(numpy.where(TTtau > 0.99)[0]))
    print("TToverlap", numpy.amax(TToverlap), numpy.amin(TToverlap), numpy.mean(TToverlap), numpy.std(TToverlap), numpy.median(TToverlap), len(numpy.where(TToverlap > 0.99)[0]))
    print("TLtau", numpy.amax(TLtau), numpy.amin(TLtau), numpy.mean(TLtau), numpy.std(TLtau), numpy.median(TLtau), len(numpy.where(TLtau > 0.99)[0]))
    print("TLoverlap", numpy.amax(TLoverlap), numpy.amin(TLoverlap), numpy.mean(TLoverlap), numpy.std(TLoverlap), numpy.median(TLoverlap), len(numpy.where(TLoverlap > 0.99)[0]))
    print("LTtau", numpy.amax(LTtau), numpy.amin(LTtau), numpy.mean(LTtau), numpy.std(LTtau), numpy.median(LTtau), len(numpy.where(LTtau > 0.99)[0]))
    print("LToverlap", numpy.amax(LToverlap), numpy.amin(LToverlap), numpy.mean(LToverlap), numpy.std(LToverlap), numpy.median(LToverlap), len(numpy.where(LToverlap > 0.99)[0]))
    print("LLtau", numpy.amax(LLtau), numpy.amin(LLtau), numpy.mean(LLtau), numpy.std(LLtau), numpy.median(LLtau), len(numpy.where(LLtau > 0.99)[0]))
    print("LLoverlap", numpy.amax(LLoverlap), numpy.amin(LLoverlap), numpy.mean(LLoverlap), numpy.std(LLoverlap), numpy.median(LLoverlap), len(numpy.where(LLoverlap > 0.99)[0]))
    print("LogLogtau", numpy.amax(LogLogtau), numpy.amin(LogLogtau), numpy.mean(LogLogtau), numpy.std(LogLogtau), numpy.median(LogLogtau), len(numpy.where(LogLogtau > 0.99)[0]))
    print("LogLogoverlap", numpy.amax(LogLogoverlap), numpy.amin(LogLogoverlap), numpy.mean(LogLogoverlap), numpy.std(LogLogoverlap), numpy.median(LogLogoverlap), len(numpy.where(LogLogoverlap > 0.99)[0]))
    print("TargetTargettau", numpy.amax(TargetTargettau), numpy.amin(TargetTargettau), numpy.mean(TargetTargettau), numpy.std(TargetTargettau), numpy.median(TargetTargettau), len(numpy.where(TargetTargettau > 0.99)[0]))
    print("TargetTargetoverlap", numpy.amax(TargetTargetoverlap), numpy.amin(TargetTargetoverlap), numpy.mean(TargetTargetoverlap), numpy.std(TargetTargetoverlap), numpy.median(TargetTargetoverlap), len(numpy.where(TargetTargetoverlap > 0.99)[0]))
