# -*- coding: utf-8 -*-
"""
Created on Wed May 14 17:24:22 2014

@author: bhupinder.s
"""

#changes--------
#changes proprocessing module to preprocess
#commented out preprocessing in test (should be in an if block)
#changed feature.p in features.p (from.pickle)
#---------------

#Todo
#Passing different arguments to classifiers and vectorizers
#pmml support with augustus
#preprocessing
#feature selection
#http://sebastianraschka.com/Articles/2014_sequential_sel_algos.html
#https://github.com/mutantturkey/PyFeast
#http://mutantturkey.com/PyFeast/feast-module.html
#train file not there due to multiple os.chdir in subpackages
#changed mode = mysettings.modes['preprocess'] for preprocess_execute in run file

import os
#baseDir = 'D:/LatchiReddy/SampleProject/TextCategorization'#os.path.dirname(os.path.abspath(__file__)).replace('\', '/')
from lxml import etree

from io import StringIO
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import ShuffleSplit
from sklearn import cross_validation

import numpy as np
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import time
import collections

#import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.cluster import Ward
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import spectral_clustering
from sklearn.cluster import estimate_bandwidth
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn.mixture import GMM
from sklearn.neighbors import kneighbors_graph
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import DistanceMetric
from nltk.cluster.util import cosine_distance

from sklearn import preprocessing

import codecs
import pandas as pd
#import pylab as pl
import sys
import os
import re
import pickle
from sklearn.externals import joblib

'''
from featureselection.FeatureSelection import FeatureSelection
from preprocess.Preprocessing import Preprocessing
from preprocess.TablePreprocesser import TablePreprocesser
'''
from django.shortcuts import render

import pprint
import shutil

#import augustus #pmml consumer, producer
#import py2pmml #pmml converter (by zementis)

#Errors in feature selection
#Errors in confusion mat df

class SklCat:
    """
    Document classification based on scikit-learn
    """
    global CONCATENATION_SYMBOL;
    CONCATENATION_SYMBOL = "##__##"


    def __init__(self):
        pass
    
    def readDf(self, csvFile, fileFormat="csv"):
        """
        Reads an input csv/tab file and returns a pandas DataFrame object
        """
        if fileFormat == "csv":
            df= pd.read_csv(csvFile, sep=",")
        elif fileFormat == "tab":
            df= pd.read_table(csvFile, sep="\t")
        else:
            
            print "File IO Error: Unsupported file format!"
            
        #print '-------Reading DataFrame----------'
        print "\nTotal number of records: %s" % len(df.index)
        print "Columns imported: ", df.columns.tolist()
        #print "Preview of first few records:", df.head(5)
        #print df.head().to_string() #causes errors for some encodings
        #print '\n-------------------------------------------------------------------------------'
        return df

    def checkMissingValues(self, df):
        hasMissing = []
        for col in df.columns.tolist():
            if any(df[col].isnull()):
                hasMissing.append(col)
                print "Missing values found in column %s " % col
        return hasMissing
        
    def filterBySpeaker(self,df,speaker, colNames, lineByLevels):

        """Takes a pandas dataframe as input and returns only the records by 'speaker' (agent,customer)
        Discards all other lines not spoken by speaker
        If no speaker is mentioned (or speaker = None), the dataframe is returned as is.
        """
        if speaker == None:
            return df
        else:
            #print df[colNames['lineBy']].unique()
            df = df.ix[df[colNames['lineBy']]== lineByLevels[speaker],:]
            print "\nTotal Number of Records after filtering by speaker: ",df.shape[0]
            return df

    def testTrainSplit(self, X, y,sessionId, testSize=0.3, randomState = 1, splitType = 'stratified'):

        """Splits dataset into test and train groups
        The argument testSize (ranges from 0 and 1) determines the percentage of records that needs to go into test
        The function returns different train and test sets for different values of the argment 'randomState' (seed value)
        The train-test split is stratified (meaning the distribution of categories is the same in both test and train)
        In cases where stratified sampling is not possible
        (common causes - categories having very few records which cannot be divided into train and test according to the ratio specified)
        """
        #X = array(X)
        #y = array(y)

        sessionId = np.array(sessionId)
        if splitType == 'stratified':
            try:
                
                splitter = StratifiedShuffleSplit(y=y, n_iter=3, test_size= testSize, indices = True, random_state=randomState)
            except:
                print "\nStratified test-train splitter returned error! Switching to random sampling..."
                splitter = ShuffleSplit(n=len(X), n_iter=3, test_size=testSize, indices = True, random_state=randomState)
            
            for trainIndex, testIndex in splitter:
                XTrain, XTest = X[np.array(trainIndex)], X[np.array(testIndex)]
                #XTrain = XTrain[np.logical_not(np.isnan(XTrain))]
                y = np.asarray(y)
                #print type(trainIndex), type(testIndex)
                yTrain,yTest  = y[np.array(trainIndex)],y[np.array(testIndex)]
                sessionIdTrain, sessionIdTest = sessionId[np.array(trainIndex)],sessionId[np.array(testIndex)]

        elif splitType == 'random':
            
            splitter = ShuffleSplit(n=len(X), n_iter=3, test_size=testSize, indices = True, random_state=randomState) #(y=y, n_iter=3, testSize=testSize, indices = True, random_state=randomState)
            for trainIndex, testIndex in splitter:
                XTrain, XTest = X[trainIndex], X[testIndex]
                yTrain, yTest = y[trainIndex], y[testIndex]
                sessionIdTrain, sessionIdTest = sessionId[trainIndex], sessionId[testIndex]
        else:
            
            XTrain, XTest, yTrain, yTest, sessionIdTrain, sessionIdTest = train_test_split(X, y, sessionId, test_size=testSize, indices = True, random_state=randomState)

        print "\nPercentage of Train-Test Split %d:%d" %((1-testSize)*100, (testSize)*100)
        print "Length of Training Sample:", len(XTrain)
        print "Length of Validation Sample:", len(XTest)
        return XTrain, XTest, yTrain, yTest, sessionIdTrain, sessionIdTest



    def addLineNum(self, df, colNames):

        """Adds a column containing the line number of the particular chat"""

        Pp = Preprocessing()
        Pp.addLineNum(df, colNames)
        return df

    def getXyFromDf(self, df, Xcol = 'lineText', yCol = None):
        """
        Get X and y vectors from the dataset (X=chat text, y = tags/labels)
        """
        #print df
        X = df[Xcol].as_matrix().ravel() #tolist()
        y = df[yCol].tolist() #.as_matrix().ravel()
        return (X, y)
   
    def spellCorrectDf(self, df, colNames, minCharLen=5, maxDissimilarity=0.4, reportErrors = False, ignoreSpaces = True):
        """
            Spell corrector (uses EnchantSpellChecker in the backend)
        """
        Pp = Preprocessing()
        transformedDf = Pp.spellCorrectDf(df, colNames, minCharLen=minCharLen, maxDissimilarity=maxDissimilarity, reportErrors = reportErrors, ignoreSpaces = ignoreSpaces)
        return transformedDf

    def preprocess(self, df, colNames, lineByLevels,  topNLinesPrimary, folderName = 'Model', issueVocabList = None,  algo = None ):
        """
        Used to call algorithms for primary issue line selection - to restructure line-wise chat data 
        to a session level data (returns a dataframe)
        """
        Pp = Preprocessing()
        algoMap = {'concatenateLinesWindowBySpeaker': Pp.concatenateLinesWindowBySpeaker,
                   'getPrimaryLines': Pp.getPrimaryLines,
                   'getPrimaryLinesWeighted': Pp.getPrimaryLinesWeighted,
                   'getAllIssueLines': Pp.getAllIssueLines,
                   'getIssueSummary': Pp.getIssueSummary,
            }
        argsMap = {'concatenateLinesWindowBySpeaker': [[df, colNames], {'minCharLen': 5, 'maxDissimilarity': 0.4, 'reportErrors': True}],
                   'getPrimaryLines': Pp.getPrimaryLines,
                   'getPrimaryLinesWeighted': Pp.getPrimaryLinesWeighted,
                   'getAllIssueLines': Pp.getAllIssueLines,
                   'getIssueSummary': Pp.getIssueSummary,
            }   
        #masksDict = Pp.readRegexMasksDict(fileName = 'regex.csv')
        #df[colNames['lineText']].apply(lambda x: Pp.mask(x, masksDict, ignoreCase=True))
        #df[colNames['lineText']].apply(lambda x: Pp.mark(x, masksDict, ignoreCase=True))
        #transformedDf = Pp.concatenateLinesWindowBySpeaker(df, colNames, lineByLevels, fromLineNum=0, toLineNum=3, speaker=2, sep=' ###_END_### ', mode= mode)
        transformedDf = Pp.concatenateLinesWindowBySpeaker(df, colNames, lineByLevels, fromLineNum=0, toLineNum=3, speaker=2, sep=' ###_END_### ')
        #transformedDf = Pp.concatenateLinesWindowBySpeaker(df, colNames, lineByLevels, fromLineNum=0, toLineNum=5, speaker=2, sep=' || ')
        #transformedDf = Pp.getPrimaryLines(df, issueVocabList, colNames, lineByLevels)
        #transformedDf = Pp.getPrimaryLinesWeighted(df, issueVocabList, colNames, lineByLevels)
        #transformedDf = Pp.getAllIssueLines(df, issueVocabList, colNames, lineByLevels, sep = ' ###_END_### ')
        #transformedDf = Pp.getIssueSummary(df, colNames, lineByLevels, filterBy = lineByLevels['customer'], sep = ' . ')
        
        #transformedDf = Pp.getPrimaryLines(df, folderName, clfType, vecType, topNLinesPrimary, colNames, lineByLevels)
        #transformedDf.to_csv('Optus_01_15Apr_concat_UPDATED.csv', index=False)
        return transformedDf
        #return df


    def kBestSelection(self, Xa = None, ya = None, nFeatures = 50):
        """
        Feature selection using kBest (scikit-learn): selects top n features
        """
        fs = FeatureSelection()
        featureSelector, transformedX = fs.kBestSelection( X = Xa, y = ya, nFeatures =nFeatures)
        return featureSelector, transformedX

    def topPercentileSelection(self, Xa = None, ya = None, quantile=50):
        """
        Feature selection using topPercentile (scikit-learn): selects top n quantile features
        """
        fs = FeatureSelection()
        featureSelector, transformedX = fs.topPercentileSelection(X = Xa, y = ya, quantile=quantile)
        return featureSelector, transformedX        
    
    def lsa(self, X = None, nComponents = 3):
        """
        Latent semantic analysis uses SVD
        """
        fs = FeatureSelection()
        transformedX = fs.lsa( X = X, nComponents = nComponents)
        return transformedX
    
    def pca(self, X, nComponents = 3, whiten=False):
        """
        Principal component analysis
        """
        fs = FeatureSelection()
        pca, pca.explained_variance_ratio_, pca.components_ = fs.pca(X, nComponents = nComponents, whiten=whiten)
        return pca, pca.explained_variance_ratio_, pca.components_
        
    def vectorizeFit(self, corpus, vecType='hashVectorizer', args = None):
        """
        Vectorizer Fit function
        """
        #args1 = decode_error='ignore', stop_words='english', analyzer='word', ngram_range=(2, 7)
        #'decode_error':'ignore', 'stop_words':'english', 'lowercase': True, 'analyzer':'word',  'ngram_range':(1, 1), 'min_df':0.001
        #args2 = analyzer='word', ngram_range=(1, 5), min_df=1
        #args3 = n_features=10
        #decode_error = 'ignore', stop_words='english', lowercase = True, analyzer='char',  ngram_range=(3, 6), min_df = 0.001
        tfidfVec = TfidfVectorizer
        countVec = CountVectorizer
        hashVec = HashingVectorizer
        #hv.transform(corpus)
        
        vectorizerMap = {
            'tfidfVectorizer': tfidfVec,
            'binaryVectorizer': countVec,#CountVectorizer({'lowercase': True, 'analyzer':'word',  'ngram_range':(1, 3), 'binary': True}),
            'countVectorizer': countVec,
            'hashVectorizer': hashVec,
        }
        vec = vectorizerMap[vecType](**args)
        
        transformedX = vec.fit_transform(corpus)
        
        return vec, transformedX       
    """
    def testTrainSplit(self, X, y, testSize=0.3, randomState = 42):

        #Splits dataset into test and train groups

        X = [re.sub(r'{"looking.*}',"",str(i).lower()) for i in X]
        XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=testSize, random_state=randomState)
        print "lengths:", len(XTrain), len(XTest), len(yTrain), len(yTest)         
        return XTrain, XTest, yTrain, yTest
    """
    def modelSelect(self, classifier='BernoulliNB', args = None):
        """
        returns a classifier object for a classifier type specified by a string
        """
        #print 'modelSelect'
        classifierMap = { 
        #Classification Algoithms        
        'BernoulliNB': BernoulliNB,
        'MultinomialNB': MultinomialNB,
         #'SVM2': LinearSVC(probability=False),
         'SVM': LinearSVC, #may crash in certain circumstances
         'SVC': svm.SVC,
        'GaussianNB': GaussianNB,
        'DecisionTreeClassifier': DecisionTreeClassifier,
        'RandomForestClassifier': RandomForestClassifier,
        'OneVsRestClassifier': OneVsRestClassifier,
        'SGDClassifier': SGDClassifier,
        'LogisticRegression': LogisticRegression,
        #Clustering Algorithms
        'Ward': Ward,
        'KMeans': KMeans,
        'MiniBatchKMeans': MiniBatchKMeans,
        'AffinityPropagation': AffinityPropagation,
        'DBSCAN': DBSCAN,
        }
        print "modelSelect1",classifier
        return classifierMap[classifier](**args)
    
    def learn(self, classifier, X, y, modelType = 'classification'):
        """
        Function for Model training - returns a trained model
        """
        if modelType == 'classification':
            try:
                model = classifier.fit(X, y)
            except:
                model = classifier.fit(X.toarray(), y)
                
        elif modelType == 'clustering':
            try:
                model = classifier.fit(X)
            except:
                model = classifier.fit(X.toarray())
        else:
            print 'Failed to build model'
        #self.classifier.fit(transformed_trained_vector_x, y_train)
        return model
    
    def getConfusionMat(self, y_true, y_pred):
        """
        Calculates the confusion matrix (returns confusion matrix and labels)
        """
        labels = list(set(list(y_true)+list(y_pred)))
        labels.sort()

        cmat = confusion_matrix(y_true, y_pred,labels = labels)
        cmatDf = pd.DataFrame(cmat, columns = labels, index = labels) #, index = False
        #cmatDf = pd.DataFrame(cmat) #, index = False
        #return (cmat, labels)
        return cmatDf
    
    def validate(self, y_true, y_pred, avgType = 'micro',validationIndex = None):
        """
        Model validation scores
        """
        #multiClassDiagnostics
        #avgType = ['macro', 'micro', 'weighted', 'samples', None]
        #
        accuracy = metrics.accuracy_score(y_true, y_pred) 
        precision = metrics.precision_score(y_true, y_pred, average=avgType)
        recall = metrics.recall_score(y_true, y_pred, average=avgType)
        f1 = metrics.f1_score(y_true, y_pred, average=avgType)
        fbeta0_5 = metrics.fbeta_score(y_true, y_pred, average= avgType, beta=0.5)
        fbeta1 = metrics.fbeta_score(y_true, y_pred, average= avgType, beta=1)
        fbeta2 = metrics.fbeta_score(y_true, y_pred, average=avgType, beta=2)
        hamming_loss = metrics.hamming_loss(y_true, y_pred)
        jaccard = metrics.jaccard_similarity_score(y_true, y_pred)
        zeroOneLoss = metrics.zero_one_loss(y_true, y_pred, normalize=True)
        #
        diagnosticMetrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1,
                             'fbeta0_5': fbeta0_5, 'fbeta1': fbeta1, 'fbeta2': fbeta2,
                             'hamming loss': hamming_loss, 'jaccard': jaccard,
                             'zeroOneLoss': zeroOneLoss}
        cols = ['precision', 'recall', 'fbeta0_5', 'f1', 'fbeta2', 'accuracy', 'jaccard', 'hamming loss', 'zeroOneLoss' ]
        dfReport = pd.DataFrame(diagnosticMetrics, index = range(1), columns = cols)
        #return diagnosticMetrics
        return dfReport.T
        

    def multiClassPrecisionCurve(self, y_true, y_pred, avgType = 'micro', validationIndex = None ):
        """
        Gives the precision recall fscore support report
        """
        ##avgType = ['macro', 'micro', 'weighted', 'samples', None]
        #print Diagnostic metrics
        labels = [str(i) for i in list(set(list(y_true) + list(y_pred)))] #The array 'labels' is the list of categories (as strings)
        #labels = [str(i) for i in list(set(list(y_true)))] #The array 'labels' is the list of categories (as strings)
        labels.sort()
        
        #clfReport = classification_report(y_true, y_pred, target_names=labels) #If target_names is not set to labels, instead of category names, default numbering is used
        #print(clfReport)
        #clfReportDf = pd.DataFrame(clfReport)
        
        #Gives zero support = None, if avgType is not set to None - if avgType is None, it returns a series of metrics for each class label
        precision, recall, fScore, support = metrics.precision_recall_fscore_support(y_true, y_pred, beta=1,  labels=labels, average=None )
        #print precision, recall, fScore, support
        
        resultsDict = {'Category':labels, 'precision': self.roundToDecimal(precision), 'recall': self.roundToDecimal(recall),  'fScore': self.roundToDecimal(fScore), 'support': self.roundToDecimal(support)}
        
        dfReport = pd.DataFrame(resultsDict)#, columns = ['precision', 'recall', 'fScore', 'support'])
        
        return dfReport

    def compare(self, classifierList):
        """
        Used to compare multiple models
        """
        pass

    def crossValidate(self, clf, X, y, n_times = 5):
        """
        Cross-validation scores
        """
        #['accuracy', 'adjusted_rand_score', 'average_precision', 'f1',
        #'log_loss', 'mean_squared_error', 'precision', 'r2', 'recall', 'roc_auc']
        
        try:
            precision = cross_validation.cross_val_score(clf, X, y, cv=n_times,scoring='precision')
            recall = cross_validation.cross_val_score(clf, X, y, cv=n_times,scoring='recall')
            f1 = cross_validation.cross_val_score(clf, X, y, cv=n_times,scoring='f1')
            accuracy = cross_validation.cross_val_score(clf, X, y, cv=n_times,scoring='accuracy')
            #print "cross validaton scores are:", crossValScores
        except:
            precision = cross_validation.cross_val_score(clf, X.toarray(), y, cv=n_times,scoring='precision')
            recall = cross_validation.cross_val_score(clf, X.toarray(), y, cv=n_times,scoring='recall')
            f1 = cross_validation.cross_val_score(clf, X.toarray(), y, cv=n_times,scoring='f1')
            accuracy = cross_validation.cross_val_score(clf, X.toarray(), y, cv=n_times,scoring='accuracy')
            #print "cross validaton scores are:", crossValScores
        cvDf = pd.DataFrame({'precision' : precision, 'recall': recall, 'f1' : f1, 'accuracy': accuracy})
        #return crossValScores
        return cvDf

    def validateDistributions(self, y_true, y_pred):
        
        if len(y_true) != len(y_pred):
            print 'mismatch in vector lengths for %s and %s ' %(y_true, y_pred)
            sys.exit(0)
        
        y_true = pd.Series(y_true)
        y_pred = pd.Series(y_pred)
        y_true_counts = y_true.value_counts()
        y_pred_counts = y_pred.value_counts()
        
        labels = ['y_true_counts', 'y_pred_counts']
        dfReport = pd.concat({'y_true_counts': y_true_counts, 'y_pred_counts': y_pred_counts}, axis=1, join='outer') #, axis=1, join='outer'
        dfReport =  dfReport.fillna(0)
        
        dfReport['y_true_prec'] = dfReport['y_true_counts']/dfReport['y_true_counts'].sum()
        dfReport['y_pred_prec'] = dfReport['y_pred_counts']/dfReport['y_pred_counts'].sum()
        dfReport['percDiff'] = dfReport.apply(lambda row: (row['y_pred_counts'] - row['y_true_counts'])/row['y_true_counts'], axis = 1)
        return dfReport
        

    def validateForCoverage(self, y_true, y_pred, y_probs, coverage = 100):
        """
        Model validation at various coverage values
        """
        #probThreshold = np.percentile(y_probs,100-coverage)
        probThreshold = np.percentile(y_probs.unique(),100-coverage)
        y_true_reduced = [yt for yt, yp, pr in zip(y_true, y_pred, y_probs) if y_probs > probThreshold]
        y_pred_reduced = [yp for yt, yp, pr in zip(y_true, y_pred, y_probs) if y_probs > probThreshold]
        dfReport1 = self.validate( y_true_reduced, y_pred_reduced, avgType = 'micro',validationIndex = None)
        #dfReport2 = self.multiClassPrecisionCurve(y_true_reduced, y_pred_reduced, avgType = 'micro', validationIndex = None )
        
        dfReport2 = self.validateDistributions(self, y_true_reduced, y_pred_reduced)
        
        return dfReport1, dfReport2
    
    def printValidationResults(self,df,coverage):

        """ Print the precision, recall and f-score in a single line, each value rounded to two decimals
        This function assumes that df contains the columns 'y_prob_max', 'y_true' and 'y_pred'
        """
        try:
            probThreshold = np.percentile(df['y_prob_max'],100-coverage)
            dfProb = df[df['y_prob_max'] > probThreshold]
            print "Precision, Recall and F1 Score at "+str(coverage)+ "% coverage: ", self.roundToDecimal(metrics.precision_score(dfProb['y_true'].tolist(),dfProb['y_pred'].tolist())), self.roundToDecimal(metrics.recall_score(dfProb['y_true'].tolist(),dfProb['y_pred'].tolist())), self.roundToDecimal(metrics.f1_score(dfProb['y_true'].tolist(),dfProb['y_pred'].tolist()))
        except:
            prob = np.percentile(df['y_prob_max'].unique(),100-coverage)
            dfProb = df[df['y_prob_max'] >= prob]       
            print "Precision, Recall and F1 Score at "+str(coverage)+ "% coverage: ", self.roundToDecimal(metrics.precision_score(dfProb['y_true'].tolist(),dfProb['y_pred'].tolist())), self.roundToDecimal(metrics.recall_score(dfProb['y_true'].tolist(),dfProb['y_pred'].tolist())), self.roundToDecimal(metrics.f1_score(dfProb['y_true'].tolist(),dfProb['y_pred'].tolist()))
        #print "Recall at "+str(coverage)+ "% coverage: ", metrics.recall_score(dfProb['y_true'].tolist(),dfProb['y_pred'].tolist())
        #print "F1 score at "+str(coverage)+ "% coverage: ",metrics.f1_score(dfProb['y_true'].tolist(),dfProb['y_pred'].tolist())
        return 0

    def reduceToCoverage(self, df,score_column_name, coverage):

        """This function return only the top 'x' percentile of records based on the confidence score (score_column_name).
        The argument 'coverage' indicates the top percentile of records that should be returned.
        """

        probThreshold = np.percentile(df[score_column_name],100-coverage)
        dfProb = df[df[score_column_name]>probThreshold]
        return dfProb

    def vectorizeTransform(self, vec, corpus):
        """
        Transforms a corpus using a specified vectorizer (Used during model execution)
        """
        transformedX = vec.transform(corpus)
        return transformedX

    def predict(self, vec, clf, X, y = None):
        """
        Model scoring/prediction function (to be deprecated)
        """
        
        try:
            y_pred = clf.predict(X)
        except:
            y_pred = clf.predict(X.toarray())
        return y_pred

    def predict2(self, model, X, y = None, modelType = 'classification'):
        """
        Model scoring/prediction function
        """
        if modelType == 'classification':
            y_labels = model.classes_
        else:
            y_labels = None
        try:
            
            y_pred = model.predict(X)
        except Exception,e:
            print e
            #print 'model not able to predict'
            y_pred = model.predict(X.toarray())
            
        return y_pred, y_labels

    def predict3(self, model, X, y = None, modelType = 'classification', modelPickle = 'model.pkl', incremental = True,folderName = None):
        """
        Model scoring/prediction function
        """
        if modelType == 'classification':
            y_labels = model.classes_
        else:
            y_labels = None
        try:
            if incremental == True:
                model = model.partial_fit(X,['Login issue'])
                
                self.toPickle(model ,folderName+"/"+modelPickle,pickler = 'joblib' )
                y_pred = model.predict(X)
            else:
                y_pred = model.predict(X)
        except Exception,e:
            print e
            #print 'model not able to predict'
            y_pred = model.predict(X.toarray())
            
        return y_pred, y_labels

    def predictTop3(self, model, X, y = None):
	"""
	Returns top 3 predicted categories
	"""
        y_probs = self.probabilities2(model, X, y = None)
        sorted_indices = [i.argsort()[::-1] for i in y_probs]
        model_classes = model.classes_
        predicted_classes = [model_classes[i[0:3]] for i in sorted_indices]
        return predicted_classes
    
    def predict_top3(self,model,y_probs):
	"""
	Returns top 3 predicted categories (to be deprecated)

	"""
        model_classes = model.classes_
        sorted_indices = [i.argsort()[::-1] for i in y_probs]
        predicted_classes = [model_classes[i[0:3]] for i in sorted_indices]
        return predicted_classes

    def probabilities2(self, model, X, y = None):
        """
        Model scoring/prediction function
        """
        #probs = []
        y_labels = model.classes_
        try:
            try:
                y_probs = model.predict_proba(X)
            except:
                y_probs = model.predict_proba(X.toarray())
        except Exception,e:
            try:
                try:
                    y_probs = model.decision_function(X)
                except Exception,e:
                    y_probs = model.decision_function(X.array())
            except:
                print "Could not compute probabilities",e
        return y_probs, y_labels
    
    def probabilities(self, vec, clf, X, y = None):
        """
        Model scoring/prediction function (to be deprecated)
        """
        #probs = []
        try:
            try:
                probs = clf.predict_proba(X)
            except:
                probs = clf.predict_proba(X.toarray())
        except Exception,e:
            try:
                try:
                    probs = clf.decision_function(X)
                except Exception,e:
                    probs = clf.decision_function(X.array())
            except:
                print "Could not compute probabilities",e
        return probs
    """
    def toPickle(self, obj, fileName, pickler='joblib'):

        #Write object to a pickle file

        picklers = ['joblib', 'pickle']
        if pickler == 'joblib':
            joblib.dump(obj, fileName)
        else:
            fOut = open(fileName, 'wb')
            pickle.dump(obj, fOut)
            fOut.close()
        return 0

    def fromPickle(self, fileName, pickler='joblib'):

        #Load object from a pickle file


        picklers = ['joblib', 'pickle']
        if pickler == 'joblib':
            obj = joblib.load(fileName)
        else:
            fIn = open(fileName, 'rb')
            obj = pickle.load(fIn)
            fIn.close()
        return obj
    """
    def toPickle(self, obj, fileName, pickler = None):
        """
        Write object to a pickle file
        """
        pickle.dump(obj, open(fileName,'wb'))
        return 0

    def fromPickle(self, fileName, pickler):
        """
        Load object from a pickle file

        """
        obj = pickle.load(open(fileName,'r'))
        return obj

    def pipeline(self):
        pipeline = Pipeline([('tfidf', TfidfTransformer()),
                     ('chi2', SelectKBest(chi2, k=1000)),
                     ('nb', MultinomialNB())])
        return pipeline

    def toPmml(self):
        #http://augustus-docs.s3-website-us-east-1.amazonaws.com/Epydoc/augustus-0_5_3_0.uml/
        #https://code.google.com/p/augustus/downloads/detail?name=augustus-0.5.3.0.tar.gz
        #http://augustus-docs.s3-website-us-east-1.amazonaws.com/
        #https://code.google.com/p/augustus/downloads/list
        #http://augustus-docs.s3-website-us-east-1.amazonaws.com/Primer/html/
        pass

    def fromPmml(self):
        #Augustus --model ../models/produced_model.pmml \
            #--data ../../data/gaslog.xml  > ../results/output.xml
        pass

    def roundToDecimal(self,input,numberOfDecimals=2):

        """Returns the input with each element of it rounded down to a certain number of decimals as dictated by the argument 'numberOfDecimals'
        """
        
        if isinstance(input, collections.Iterable):
            # iterable
            result =  [round(i,numberOfDecimals)for i in input]
        elif input == None:
            result =  None
        else:
            # not iterable
            result = round(input,numberOfDecimals)
            
        return result

    """
    def IssuelineClassifier_prim(self, df, clf, colNames, lineByLevels,):

        df = df[colNames['lineBy']==lineByLevels['customer']]

        try:
            model = self.fromPickle("Results_Primary/PrimClassifier.pkl", pickler='joblib')
        except:
                print "cannot open model file"
        try:
            vec = self.fromPickle("Results_Primary/PrimVec.pkl", pickler='joblib')
        except:
            print "cannot open vectorizer file"

        transformedXTest = self.vectorizeTransform(vec, df[colNames['lineText']])
        y_pred = self.predict(vec, model, transformedXTest)
        y_probs = self.probabilities(vec, model, transformedXTest)

        df['prim_score'] = y_probs

        idx2 = df.groupby(colNames['sessionId'])['prim_score'].transform(lambda x: self.returnSignificantLines(x))

        df['SigLines'] = 0
        df['SigLines'][idx2 == 1] = 1

        concat_sentences = []
        Sess_Id = []

        for idx, df_segment in df.groupby(colNames['sessionId']):

            concat_sentences.append(" ##__## ".join(df_segment[colNames['lineText']].tolist()))
            Sess_Id.append(idx)

        transformed_data = pd.DataFrame.from_dict([Sess_Id, concat_sentences])
        transposed_data = pd.DataFrame.transpose(transformed_data)
        transposed_data.columns = [colNames['sessionId'],colNames['lineText']]

        return transposed_data
    """

    def returnSignificantLines(self, x, topNLinesPrimary):

        """ The objective of the function is to identify the lines (from the first few 'n' lines) which have the highest scores.
        Effectively the indices of such lines are returned
        x is an object of type 'series' and it consists of an array of numerical values (say 0.1, 2.3, 2.1, 3.5..).
        These numerical values are confidence scores of a particular line being an issue line (the higher the number, more is the probability that it is an issue line)
        We calculate the mean and median of the first 'n' values ('n' determined by the argument 'topNLinesPrimary')
        The function returns an array that is of the same length as x consisting of only 0s and 1s.
        If any element of x is greater than the mean and median (as calculated above) and falls within the first 'n' values, then it is a 1.
        All other corresponding elements are 0. This ensures that we return the high confidence lines and also limit those lines to be from the first few ones.`
        """

        lines = [0]*len(x)
        mean = x.head(topNLinesPrimary).mean()
        median = x.head(topNLinesPrimary).median()
        for i in range(1,min(len(x),topNLinesPrimary)+1): #min function in case the length of chat is less than 'topNLinesPrimary'
            if x.iloc[i-1] > mean and x.iloc[i-1] > median:
                lines[i-1] = 1
        return lines

    def issueLineClassifier(self, df, folderName, model, vec, topNLinesPrimary,colNames, lineByLevels, featureSelection, speaker):

        """Binary classifier - predicts whether or not a given line is an issue line and concatenates all issuelines in a particular sessionId.
        The function takes a pandas dataframe containing line-wise chat as input and returns another dataframe with one record for each sessionId.
        The 'lineText' columns of the returned dataframe containing a concatenation of lines deemed to be 'Issue Lines' based on prediction by the PrimaryLine Classifier model.
        Returns sessionId and LineText only.
        Not all sessionIds in the input are present in the output. Some sessionIds may have a single customer line or no customer line at all. Such sessionIds are discarded.
        """


        df = self.filterBySpeaker(df, speaker,colNames, lineByLevels) #Filtering out all other lines except lines by 'speaker' (argument)

        transformedXTest = self.vectorizeTransform(vec, df[colNames['lineText']]) # Vectorization of X (corpus)

	    #featureSelection
        if featureSelection == True:
                try:
                    featureSelector = self.fromPickle(folderName +"/PrimaryLineClassification/FeatureSelector.p", pickler='joblib')
                except:
                    print "cannot open feature selector file"

        transformedXTest = featureSelector.transform(transformedXTest)


        y_pred = self.predict(vec, model, transformedXTest)
        y_probs = self.probabilities(vec, model, transformedXTest) # returns the confidence score (for being an issue line) for each chat line.

        df['prim_score'] = [i[1] for i in y_probs] # Scores for '1' ('1' being issue line and '0' being non-issue line)

        sigLines = df.groupby(colNames['sessionId'])['prim_score'].transform(lambda x: self.returnSignificantLines(x, topNLinesPrimary)) #sigLines is a an array of 0s and 1s with the 1s representing the positions of possible issuelines
        df = df[sigLines==1] #Down-selecting to only primary issue lines. Discarding all other lines.

        concat_sentences= []
        Sess_Id = []

        #Concatenating significant lines for each Session Id

        for idx, df_segment in df.groupby(colNames['sessionId']):

            concat_sentences.append(CONCATENATION_SYMBOL.join(df_segment[colNames['lineText']].tolist())) # Concantenates issue lines separated by a concatenation symbol
            Sess_Id.append(idx)


        transformed_data = pd.DataFrame.from_dict([Sess_Id, concat_sentences])
        transposed_data = pd.DataFrame.transpose(transformed_data)
        transposed_data.columns = [colNames['sessionId'],colNames['lineText']]
        return transposed_data

    def clusterDiagnostics(self, y_true=None, y_pred=None, X=None):
        
        #http://scikit-learn.org/stable/modules/clustering.html
        if y_true != None and y_pred != None:                
            adjMutualInfoScore = metrics.adjusted_mutual_info_score(y_true, y_pred)
            normMutualInfoScore = metrics.normalized_mutual_info_score(y_true, y_pred)
            mutualInfoScore = metrics.mutual_info_score(y_true, y_pred)
            homogeneityScore = metrics.homogeneity_score(y_true, y_pred)
            completenessScore = metrics.completeness_score(y_true, y_pred) 
            vScore = metrics.v_measure_score(y_true, y_pred)
            #silhoutteScore = metrics.silhouette_score(X, labels, metric='euclidean')
            #print metrics.confusion_matrix(y_true, y_pred)
            results = {'MI': mutualInfoScore, 'AMI': adjMutualInfoScore, 
                       'NMI': normMutualInfoScore, 'homogeneity': homogeneityScore, 
                       'completeness': completenessScore, 'vScore': vScore,} 
                       #'silhoutteScore': silhoutteScore}
        elif y_pred != None and X !=None:
            silhoutteScore = metrics.silhouette_score(X, y_pred, metric='euclidean')
            results = {'silhoutteScore': silhoutteScore}
        dfResults = pd.DataFrame(results, index = range(1))
        return dfResults
    
    def distancesToClustCentres( self, X, centres, metric="euclidean", p=2 ):
        """ all distances X -> nearest centre, any metric
                euclidean2 (~ withinss) is more sensitive to outliers,
                cityblock (manhattan, L1) less sensitive
        """
        D = cdist( X, centres, metric=metric, p=p )  # |X| x |centres|
        return D.min(axis=1)  # all the distances

    def cluster(self, inFileName = 'clusteringInput.txt', outFileName = 'clusteringResults.csv', 
                colNames = None, lineByLevels = None, reduceDimension = True, nComponents = 5, Xcol = None, yCol = None, 
                vecType='tfidfVectorizer', sampleSize=3000, nClust=10, scaling=True, 
                argsVec = {'lowercase': True, 'analyzer':'word',  
                'ngram_range':(1, 3), 'min_df':5,}):

        df = self.readDf(inFileName, fileFormat="csv")
        df = df[0:sampleSize]
        df[colNames['lineText']] = df[colNames['lineText']].apply(lambda x: codecs.decode(str(x), "latin1",'ignore')) #throws error
        Xy = self.getXyFromDf(df, Xcol = Xcol, yCol = yCol)
        corpus = Xy[0]
        y = Xy[1]
        vec, transformedX = self.vectorizeFit(corpus, vecType = vecType, args = argsVec)
        #vec = kwClust.vectorize(corpus, vecType=vecType, args = argsVec)
        #stop_words=None, token_pattern=u'(?u)\b\w\w+\b',tokenizer=None, 
        #transformedX = vec.fit_transform(corpus).todense()
        transformedX = transformedX.todense()
        if reduceDimension == True:
            self.lsa(X = transformedX, nComponents = nComponents, normalize=True)
        Clust = Clustering247()
        y_pred = Clust.kmeans(transformedX, nClust, scaling=scaling, miniBatchKm = True)
        #y_pred2 = Clust.hclusters(dtm, nClust)
        #y_pred = Clust.kmeans(dfNew.as_matrix(), 3)
        #y_pred = Clust.hclusters(dfNew.ix.as_matrix(), 3)
        #y_pred = Clust.hclusters2(dfNew.as_matrix(), 3)
        df['cluster_label'] = y_pred
        #df['y_pred2'] = '' #y_pred2
        #y_true = df['y_true'].tolist()
        #print Clust.clusterDiagnostics(self, y_true=y_true, y_pred=y_pred)
        print Clust.clusterDiagnostics(y_pred=y_pred, X=dtm)
        #print Clust.clusterDiagnostics(y_pred=y_pred2, X=dtm)
        df.to_csv(outFileName, index = False)
        return df

    def run_infosec_preprocess(self):
        Pp = Preprocessing()
        try:
            masksDict = Pp.readRegexMasksDict(regExfileName = 'regex1.csv')
            masksDict1 = Pp.readRegexMasksDict(regExfileName = 'babynames.csv')
            masksDict2 = Pp.readRegexMasksDict(regExfileName = 'location_db.csv')
            masksDict.update(masksDict1)
            masksDict.update(masksDict2)
        except Exception,e:
            print 'Exception:',e
        df[colNames['lineText']] = df[colNames['lineText']].apply(lambda x: Pp.mask(x, masksDict, ignoreCase=True))

        df.to_csv(fileName, index=False)
        return df

    def run_text_normalize(self, df, colNames):
        """
        convert to lower text, strip punctuations, strip spaces & newlines, codec conversion
        """
        
        df[colNames['lineText']] = df[colNames['lineText']].apply(str).apply(lower)
        
        try:
            df[colNames['lineText']] = df[colNames['lineText']].apply(lambda x: codecs.decode(str(x), "ascii",'ignore'))
        except:
            try:
                df[colNames['lineText']] = df[colNames['lineText']].apply(lambda x: codecs.decode(str(x), "latin-1",'ignore'))
            except:
                try:
                    df[colNames['lineText']] = df[colNames['lineText']].apply(lambda x: codecs.decode(str(x), "utf-8",'ignore'))
                except:
                    print 'Decode Exception'

        return df

    def run_text_preprocess(self, df, colNames, lineByLevels, folderName= 'Preprocess', clfType = 'SVM', vecType= 'tfidf', topNLinesPrimary= 5, algo=None, spellCorrect = False, preprocessedFileName = 'preprocess.csv'):
        """
        Preprocess text file - primary line extraction, spell correction
        """
        #Data preprocessing-------
        self.addLineNum(df, colNames)
        try:
            df[colNames['issueLine']] = pd.notnull(df[colNames['L3']]).astype(int)
        except:
            pass
        df = self.preprocess(df, colNames, lineByLevels,  topNLinesPrimary = topNLinesPrimary, folderName = 'Model', issueVocabList = None,  algo = None )
        if spellCorrect == True:
            df = self.spellCorrectDf(df, colNames, minCharLen=5, maxDissimilarity=0.4, reportErrors = True, ignoreSpaces = True)
        #fileTokens = fileName.split('.')
        #preprocessedFileName = '%s_preprocessed.%s' % (fileTokens[0], fileTokens[1])
        df.to_csv(preprocessedFileName, index=False)
        
        return df

    def run_text_preprocess2(self, df, colNames, yCol):
        """
        Get X, y columns, indexes and column names
        """
        print colNames
        Xy = self.getXyFromDf(df, Xcol = colNames['lineText'], yCol = yCol)
        corpus = Xy[0]
        y = Xy[1]
        return corpus, y, df.index, colNames['lineText'], yCol

    def run_text_featureSelect(self, XTrain, yTrain, featureTopQuantile = 0.1, writeFeatureScores = False, folderName = 'Model', featureSelectorFile = 'FeatureSelector.pkl', pickler = None):
        """
        Feature Selection
        """
        #Transform features
        print 'A'
        print 'XTrain',XTrain
        featureSelector, transformedXTrain  = self.topPercentileSelection(Xa=XTrain, ya= yTrain, quantile=featureTopQuantile)
        #featureSelector, transformedXTrain  = self.kBestSelection(Xa=transformedXTrain,ya= yTrain, nFeatures=5000)
        print "Number of features after feature-selection: ", transformedXTrain.shape[1]
        
        #Save feature selector
        self.toPickle(featureSelector, folderName + '/' + featureSelectorFile, pickler=pickler)

        #writing feature scores--------
        #Category-wise feature selection
        if writeFeatureScores == True:
            scoreTypes = ['oddsRatio', 'chiSq', 'simplifiedChiSq', 'corrCoef',
                              'idfScore', 'mutualInfo', 'infoGain', 'lorScore', 'bnsScore', 'bnsIdfScore',
                              'nltk_chiSq', 'nltk_studentT', 'nltk_pmi', 'nltk_likelihoodRatio']
            scoreTypes = dict(zip(scoreTypes,scoreTypes))
            scoreType = scoreTypes['nltk_chiSq']
            fs = FeatureSelection()
            print "Calculating feature scores..."
            scoresDf = fs.featureScoresXY(XTrain.todense(), yTrain, xLabels = featureNames, scoreType = scoreType) #as_matrix()
            #topX = fs.getTopFeatures(X,y, scoreType = scoreType, topN = 1000)
            print "Writing feature score matrix..."
            scoresDf.to_csv(folderName+"/FeatureScore.csv")
                
        return featureSelector, transformedXTrain

    def run_text_vectorizeFit(self, XTrain, folderName = 'Model', vectorizerFile = 'Vectorizer.pkl', vecType = 'tfidf', argsVec = None, pickler = None):
        """
        Fit a vectorizer on training data
        """
        vec, XTrain = self.vectorizeFit(XTrain, vecType = vecType, args = argsVec) #transformedXTrain
        self.toPickle(vec, folderName+'/' + vectorizerFile, pickler=pickler)
        print 'XTrain',XTrain
        return vec, XTrain

    def run_text_vectorizeTransform(self, XTest, folderName = 'Model', vectorizerFile = 'Vectorizer.pkl', pickler = None):
        """
        Transform test data with a vectorizer fit on training data
        """
        #print '----->', XTest
        try:
            vectorizer = self.fromPickle(folderName +'/'+ vectorizerFile, pickler=pickler)
        except Exception as e:
            print "cannot open vectorizer file",e
        
        transformedXTest = vectorizer.transform(XTest)
        #print '#######',transformedXTest
        return transformedXTest

    
    def run_text_featureTransform(self, XTest, folderName = 'FeatureSelection', featureSelectorFile = 'FeatureSelector.pkl', pickler = None):
        """
        Load feature selector and transform features
        """
        try:
            featureSelector = self.fromPickle(folderName+'/'+ featureSelectorFile, pickler=pickler)
        except Exception,e:
            print e
            print "cannot open feature selector file"
        transformedXTest = featureSelector.transform(XTest)
        return transformedXTest
    
    def run_text_train(self, XTrain, yTrain, sessionIdTrain, clf, folderName  = 'Model', modelFileName = 'Classifier.pkl', modelType = 'classification'):
        """
        Train model on text data
        """        
        #model training
        model = self.learn(clf, XTrain, yTrain, modelType = modelType)
        self.toPickle(model, folderName + '/' + modelFileName, pickler='joblib')
        
        #self.toPickle(vec, folderName + vecFileName, pickler='joblib')
        
        #Write coeffs
        try:
            
            pd.DataFrame(model.coef_).to_csv(folderName+"/"+'Coefficients.csv')
            pd.DataFrame(model.intercept_).to_csv(folderName+"/"+'Intercept.csv')
        
            dfCoeffs = pd.DataFrame({'Variable': model.classes_, 'betaCoeffs': model.coef_.ravel(), 'intercept': model.intercept_})
            dfCoeffs['abs(beta)'] = dfCoeffs['betaCoeffs'].apply(lambda x: abs(x))
            dfCoeffs = dfCoeffs.sort(['abs(beta)'], ascending=[0])
            dfCoeffs.to_csv(folderName+"/"+'Coefficients1.csv', index=False)
       
        except Exception as e:
            print 'Error writing coefficients'
            
        return model   
        

    def run_text_test(self, XTest, yTest, sessionIdTest, folderName  = 'Model', modelFileName = 'Classifier.pkl', avgType = 'micro' , validationIndex = None, pickler=None, modelType = 'classification'):
        """
        Test/Validate model performance on test data
        """

        try:
            model = self.fromPickle(folderName+'/' + modelFileName, pickler=pickler)
        except:
            print "cannot open model file"
            
        #For structured data
        #y_pred = model.predict(XTest) #.ravel()
        #y_probs = None

        #For text data
        #y_pred = self.predict(vec, model, transformedXTest)
        #print XTest.shape
        y_true = yTest #.as_matrix().ravel()
        
        if modelType == 'classification':
            y_pred, y_labels = self.predict2(model, XTest, y = None, modelType = modelType)
            y_probs, y_labels = self.probabilities2(model, XTest)
            y_prob_max= [max(i) for i in y_probs]
               
            #Validation for classification model
              
            print 'Confusion matrix...'
            cmatDf = self.getConfusionMat(y_true, y_pred)
            print cmatDf
            cmatDf.to_csv(folderName +'/' + 'TestResults_CmatDf.csv')
    
            print 'Multi-class precision curve...'
            dfReport = self.multiClassPrecisionCurve(y_true, y_pred,  avgType = 'micro' , validationIndex = validationIndex)
            print dfReport
            dfReport.to_csv(folderName + '/' + 'TestResults_dfReport.csv')
    
            print 'Validation scores...'
            dfScores = self.validate(y_true, y_pred, avgType = avgType, validationIndex = None)
            print dfScores.T
            dfScores.T.to_csv(folderName + '/' + 'TestResults_scores.csv')

        else:
            #yprobs not calculated for clustering
            y_pred, y_labels = self.predict2(model, XTest, y = None, modelType = modelType)
            y_prob_max = np.zeros(len(y_pred))
            dfClust = self.clusterDiagnostics(y_true=y_true, y_pred=y_pred, X=None)
            print dfClust
            
        #For text data 
        dfProb1 = pd.DataFrame({'sessionId':sessionIdTest,  'y_true':yTest, 'y_pred':y_pred, 'y_prob_max':y_prob_max}) #'lineText':XTest,
        #dfProb2 = pd.DataFrame(y_probs, columns = y_labels) 
        #dfProb = pd.concat([dfProb1, dfProb2], axis=1)
        dfProb = dfProb1
        
        dfProb.to_csv(folderName + "/"+"TestResults_100coverage.csv",index = False)
        
        if modelType == 'classification':
            print '\n--------------Validating Distributions--------------\n'
            
            dfDist = self.validateDistributions(y_true, y_pred)
            print dfDist
            dfDist.to_csv(folderName + '/' + 'TestResults_Distribution_Check.csv', index = True)
            
            print "\n-------Precision, Recall and F1-Scores at different Coverages---------------\n"
        
            self.printValidationResults(dfProb,coverage=50)
            self.printValidationResults(dfProb,coverage=60)
            self.printValidationResults(dfProb,coverage=70)
            self.printValidationResults(dfProb,coverage=80)
            self.printValidationResults(dfProb,coverage=90)
            self.printValidationResults(dfProb,coverage=100)
        else:
            pass
        
        return dfProb, model

    def run_text_execute(self, XTest, folderName = 'Model', modelFile = 'Classifier.pkl', pickler=None, modelType = 'classification'):
        """
        Model Execution for text data
        """
        #Classification on the output of the previous step (Stage 2)
                
        try:
            model = self.fromPickle(folderName+'/'+ modelFile, pickler=pickler)
        except:
            print "cannot open model file"
        #try:
        #    vec = self.fromPickle(folderName+"/FeatureTransformer.p", pickler=pickler)
        #except:
        #    print "cannot open vectorizer file"

        #y_pred = model.predict(XTest)
        if modelType == 'classification':
            y_pred, y_labels = self.predict2(model, XTest, y = None, modelType = modelType)
            y_probs, y_labels = self.probabilities2(model, XTest)
            y_prob_max= [max(i) for i in y_probs]
        else:
            y_pred, y_labels = self.predict2(model, XTest, y = None, modelType = modelType)
            y_prob_max = np.zeros(len(y_pred))
        
        #y_true = yTest
        
        #not sure if the column order and the y_probs order is the same
        
        dfProb1 = pd.DataFrame({'y_pred':y_pred, 'y_prob_max': y_prob_max}) #'lineText':XTest,'y_true': None ,
        #dfProb2 = pd.DataFrame(y_probs, columns = y_labels) 
        #dfProb = pd.concat([dfProb1, dfProb2], axis=1)
        dfProb = dfProb1
        
        dfProb.to_csv(folderName + "/"+"Exec_Results_100coverage.csv",index = False)
        
        return dfProb,y_labels
        
        


    def run_text_execute_api_bulkDoc(self, XTest,folderName = 'Model', modelFile = 'Classifier.pkl', pickler=None, modelType = 'classification',usecase = None):
            """
            Model Execution for text data
            """
            #Classification on the output of the previous step (Stage 2)
                    
            try:
                model = self.fromPickle(folderName+'/'+ modelFile, pickler=pickler)
            except:
                print "cannot open model file"
            #try:
            #    vec = self.fromPickle(folderName+"/FeatureTransformer.p", pickler=pickler)
            #except:
            #    print "cannot open vectorizer file"
    
            #y_pred = model.predict(XTest)
            if modelType == 'classification':
                y_pred, y_labels = self.predict2(model, XTest, y = None, modelType = modelType)
                y_probs, y_labels = self.probabilities2(model, XTest)
                y_prob_max= [max(i) for i in y_probs]
            else:
                y_pred, y_labels = self.predict2(model, XTest, y = None, modelType = modelType)
                y_prob_max = np.zeros(len(y_pred))
            
            #y_true = yTest
            
            #not sure if the column order and the y_probs order is the same
            
            dfProb1 = pd.DataFrame({'y_pred':y_pred, 'y_prob_max': y_prob_max}) #'lineText':XTest,'y_true': None ,
            #dfProb2 = pd.DataFrame(y_probs, columns = y_labels) 
            #dfProb = pd.concat([dfProb1, dfProb2], axis=1)
            dfProb = dfProb1
            if usecase == 'ltv':
                #dfProb = pd.DataFrame({'y_pred':y_pred, 'y_prob_max': y_prob_max,'lineNum':lineNum})
                dfProb = dfProb.ix[dfProb['y_pred'] == 1,:]
                #print 'y_prob',dfProb
                #dfProb_dict = dfProb.to_dict(orient = 'records')
                #return {'category':y_pred,'pred_score':y_prob_max}
                
            #dfProb.to_csv(folderName + "/"+"Exec_Results_100coverage.csv",index = False)
            return dfProb
            
            #return y_pred,y_prob_max


    def run_text_execute_api_doc(self, XTest, folderName = 'Model', modelFile = 'Classifier.pkl', pickler=None, modelType = 'classification',y = None):
            """
            Model Execution for text data
            """
            #Classification on the output of the previous step (Stage 2)
                    
            try:
                model = self.fromPickle(folderName+'/'+ modelFile, pickler=pickler)
            except:
                print "cannot open model file"
            #try:
            #    vec = self.fromPickle(folderName+"/FeatureTransformer.p", pickler=pickler)
            #except:
            #    print "cannot open vectorizer file"
    
            #y_pred = model.predict(XTest)
            if modelType == 'classification':
                #y_pred, y_labels = self.predict2(model, XTest, y = None, modelType = modelType)
                y_pred, y_labels = self.predict3(model, XTest, y = None, modelType = modelType, incremental=True,folderName = folderName)
                y_probs, y_labels = self.probabilities2(model, XTest)
                y_prob_max= [max(i) for i in y_probs]
            else:
                y_pred, y_labels = self.predict2(model, XTest, y = None, modelType = modelType)
                y_prob_max = np.zeros(len(y_pred))
            
            #y_true = yTest
            
            #not sure if the column order and the y_probs order is the same
            '''
            dfProb1 = pd.DataFrame({'y_pred':y_pred, 'y_prob_max': y_prob_max}) #'lineText':XTest,'y_true': None ,
            #dfProb2 = pd.DataFrame(y_probs, columns = y_labels) 
            #dfProb = pd.concat([dfProb1, dfProb2], axis=1)
            dfProb = dfProb1
            dfProb.to_csv(folderName + "/"+"Exec_Results_100coverage.csv",index = False)
            
            return dfProb, y_labels
            '''
            return y_pred,y_prob_max



    def run_text(self, fileName = None, folderName = None, mode = 'preprocess', featureSelection=False, 
                 featureTopQuantile = 10, writeFeatureScores = False, colNames = None, lineByLevels = None,
                 filterBySpeaker = None, argsPrep = None, spellCorrect = False, clfType = 'SVM', argsClf = None, 
                 vecType = 'tfidfVectorizer', argsVec = None, testSize=0.3, randomState=1, splitType =None, 
                 coverage = 100, modelVariable = None, avgType = None, validationIndex =None, topNLinesPrimary = 5, 
                 primaryLineClassification = False, typePrep = 'classifier', preprocessFlag = False,
                 xCols = None, yCol = None, toDropCols = None, modelType = 'classification', scaling = False):
        """
        Training/Testing/Execution of Models for Text Classifications
        """
        #user inputs ----------------
        #text
        idCol = colNames['sessionId']
        yCol = colNames['target']
        
        #fsType = 'kBest'
        #modelFileName = '%s/%s.p' % (folderName, clfType)
        #vecFileName = '%s/%s.p' % (folderName,vecType)
        #featureSelectFileName = '%s/%s.p' % (folderName,fsType)
        #featureScoresFileName = str(folderName).join('/featureScores.csv')
        encodings = ["iso-8859-15", 'utf-8', 'ascii', 'latin1']
        Tp = TablePreprocesser()
        
        #Read data------------
        df = self.readDf(fileName, fileFormat='csv')
        
        print 'Total records: ', df.shape[0]
        allCols = df.columns
        
        folderName = folderName
        #Model pipeline-----------
        if 'train' in mode:
            sessionId = df[colNames['sessionId']].tolist()
            if preprocessFlag == True:
                #first preprocess block
                #, preprocessedFile = 'preprocess.csv'
                df = self.run_text_preprocess(df, colNames, lineByLevels, folderName= folderName, clfType = clfType, vecType= vecType,topNLinesPrimary= 5,algo=None, spellCorrect = spellCorrect)
            corpusX, y, dfIdx, XColumns, yCol =  self.run_text_preprocess2(df, colNames, yCol)
            #Sampling - Test-Train Split (Stratified/Random)
            #test train split
            corpusXTrain, corpusXTest, yTrain, yTest, sessionIdTrain, sessionIdTest = self.testTrainSplit(corpusX, y, sessionId, testSize=testSize, randomState = randomState, splitType = splitType)

            #second preprocess block for vectorization
            vec, XTrain = self.run_text_vectorizeFit(corpusXTrain, folderName = folderName, vectorizerFile = 'Vectorizer.pkl', vecType = 'tfidfVectorizer', argsVec = argsVec, pickler = None) #transformedXTrain
            #reload vectorizer from file
            XTest = self.run_text_vectorizeTransform(corpusXTest, folderName = folderName, vectorizerFile = 'Vectorizer.pkl', pickler = None)
            #XTest = vec.transform(XTest)
            if scaling == True:
            #colsToScale = xCols
            
                XTrain = Tp.scaling(XTrain, cols = None, scalerType = 'StandardScaler')
                XTest = Tp.scaling(XTest, cols = None, scalerType = 'StandardScaler')
            
            if featureSelection == True:
                featureSelector, XTrain = self.run_text_featureSelect( XTrain, yTrain, featureTopQuantile = featureTopQuantile, folderName = folderName, featureSelectorFile = 'FeatureSelector.pkl', pickler = None)
                XTest = self.run_text_featureTransform(XTest, folderName = folderName, featureSelectorFile = 'FeatureSelector.pkl', pickler = None)
            
                #Save features
                featureIndices = featureSelector.get_support(indices= True)
                allFeatures = vec.get_feature_names()
                featureNames = [allFeatures[i] for i in featureIndices]
                featureMap = dict(zip(featureIndices, featureNames))
                pd.DataFrame(featureMap.items(), columns = ['index', 'FeatureNames']).to_csv(folderName + '/' + 'FeatureNames.csv')
        
            #select model
            clf = self.modelSelect(classifier=clfType, args = argsClf)

            #train model
            #model, featureSelector = self.run_table_train(XTrain, yTrain, modelFileName, featureTopQuantile=featuresTopN, featureSelectFileName = featureSelectFileName )
            model = self.run_text_train( XTrain, yTrain, sessionIdTrain, clf, folderName  = folderName, modelFileName = 'Classifier.pkl')
            
            #validate model
            #y_pred1, y_true = self.run_table_test(XTest, yTest, validationFolder)        
            dfProb, model = self.run_text_test(XTest, yTest, sessionIdTest, folderName  = folderName, modelFileName = 'Classifier.pkl', avgType = 'micro' , validationIndex = None, pickler=None, modelType = modelType)
            print XTrain.todense().shape, XTest.shape
            print yTrain.shape, yTest.shape
            print type(np.array(XTrain)), type(yTrain)
            print np.row_stack((XTrain, XTest)).shape
            
            if modelType == 'classification':
                try:
                    cvResults = self.crossValidate(model, np.row_stack((np.array(XTrain.todense()), XTest)), np.append(yTrain, yTest), n_times=5)
                    print cvResults
                    cvResults.to_csv(folderName+"/"+'TestResults_CV_100Coverage.csv')
                except Exception,e:
                    print e
                    print 'Could not calculate cross-validation results'
                    
        if 'test' in mode:
            
            sessionId = df[colNames['sessionId']].tolist()
            sessionIdTest = sessionId
                           
            if preprocessFlag == True:
                #first preprocess block
                df = self.run_text_preprocess(df, colNames, lineByLevels, folderName= folderName, clfType = clfType, vecType= vecType,topNLinesPrimary= 5,algo=None, spellCorrect = spellCorrect)
            
            corpusXTest, yTest, dfIdx, XColumns, yCol =  self.run_text_preprocess2(df, colNames, yCol)
            #vectorize
            XTest = self.run_text_vectorizeTransform(corpusXTest, folderName = folderName, vectorizerFile = 'Vectorizer.pkl', pickler = None)
            if scaling == True:
                XTest = Tp.scaling( XTest, cols = None, scalerType = 'StandardScaler')            
            
            if featureSelection == True:
                XTest = self.run_text_featureTransform(XTest, folderName = folderName, featureSelectorFile = 'FeatureSelector.pkl', pickler = None)
            
            #test model
            #y_pred2, y_true = self.run_table_test(XTest, yTest, avgType = 'micro', folderName = 'Model')
            dfProb, model = self.run_text_test(XTest, yTest, sessionIdTest, folderName  = folderName, modelFileName = 'Classifier.pkl', avgType = 'micro' , validationIndex = None, pickler=None, modelType = modelType)

            print '\n--------------Cross Validation Scores--------------\n'

            cvResults = self.crossValidate(model, XTest, yTest, n_times=5)
            print cvResults
            cvResults.to_csv(folderName+"/"+'TestResults_CV_100Coverage.csv')
        
        if 'execute' in mode :
            
            sessionId = df[colNames['sessionId']].tolist()
            sessionIdTest = sessionId
            
            if preprocessFlag == True:
                #first preprocess block
                df = self.run_text_preprocess(df, colNames, lineByLevels, folderName= folderName, clfType = clfType, vecType= vecType,topNLinesPrimary= topNLinesPrimary,algo=None, spellCorrect = spellCorrect)
                #Assign the entire dataset to XTest, yTest (test-train split not required)
                
            XTest, yTest, dfIdx, XColumns, yCol =  self.run_text_preprocess2(df, colNames, yCol)
            #Vectorization
            XTest = self.run_text_vectorizeTransform(XTest, folderName = folderName, vectorizerFile = 'Vectorizer.pkl', pickler = None)

            if scaling == True:
                XTest = Tp.scaling( XTest, cols = None, scalerType = 'StandardScaler')  
                    
            if featureSelection == True:
                XTest = self.run_text_featureTransform(XTest, folderName = folderName, featureSelectorFile = 'FeatureSelector.pkl', pickler = None)
                
            #Execute model (model scoring/predictions)
            dfProb, y_labels = self.run_text_execute(XTest, sessionIdTest, colNames, folderName = folderName, modelFile = 'Classifier.pkl', pickler=None, modelType = modelType)
            #y_pred  = self.run_table_execute(XTest)
            
        dfMerged = pd.merge(df, dfProb, how='left',left_on=colNames['sessionId'], right_on=colNames['sessionId'], left_index=True, suffixes=('_x','_y'))
        dfMerged.to_csv(folderName + '/' + 'ModelResults.csv', index=False)
        #dfMerged, probThreshold = self.reduceToCoverage(dfMerged,'Prediction Score',coverage) #Reducing output to desired coverage before printing out
        
        return dfMerged
    
    
    def run_table_preprocess(self, df, xCols = None,toDropCols = None, numericCols = None, 
                             factorCols = None, toBinCols = None, toDummyCols = None, toDateTimeCols = None,
                             nBins = 10, binSize = -1, binCuts = [], handleMissingMethod= 'None', handleOutliersMethod = 'None' ):
        """
        Preprocessing structured data
        """
        Tp = TablePreprocesser()
        
        #drop columns
        if toDropCols != None:
            df = df.drop(toDropCols, axis=1)
            
        
        for col in toDropCols:
            if col in numericCols:
                numericCols.remove(col)
        for col in toDropCols:
            if col in toBinCols:
                toBinCols.remove(col)
        for col in toDropCols:
            if col in factorCols:
                factorCols.remove(col)
        for col in toDropCols:
            if col in toDateTimeCols:
                toDateTimeCols.remove(col)
        for col in toDropCols:
            if col in toDummyCols:
                toDummyCols.remove(col)
        for col in toDropCols:
            if col in xCols:
                xCols.remove(col)
        #Handle null/missing values
        df = Tp.handleMissingValues(df, handleMissing=handleMissingMethod)
        
        #outliers for numeric cols
        if numericCols != None:
            
            for col in numericCols:
                #df = Tp.handleOutliers(df, colName, ZscoreThreshold = 1.96, method = 'ignoreOutliers')
                df = Tp.handleOutliers(df, col, ZscoreThreshold = 1.96, method = handleOutliersMethod)
        #Perform binning after handling outliers
        #Binning numerical cols
        if toBinCols != None:
            for col in toBinCols:
                newCol = '%s_%s' % (col, '_binned')
                df = Tp.binCol2(df, col, newCol, cuts = binCuts, binSize=binSize, nBins = nBins)
                #add each col to factorCols
                factorCols.append(newCol)
       
        #Transform categorical columns
        if factorCols != None:
            transformations = Tp.transformCategoricalCols(df, factorCols)
            #remove -1 encoded entries for categorical variables
            #print 'df1',df
            print df.shape
            df = df[df>=0]
            print df.shape
            #print 'df2',df
            
        #remove -1 encoded entries for categorical variables
        #df = df[df>=0]        
        
        #Process datetime cols
        if toDateTimeCols != None:
            for dtCol in toDateTimeCols:
                df = self.dateTimeCol(df, dtCol)   
        #Dummy variables
        if toDummyCols != None:
            for d in toDummyCols:
                #dummies = pd.get_dummies(df[d], prefix='dummy_')
                df = Tp.createDummyVariables(df, d)
                
        return df
        
    def getXyFromTable(self, df, yCol, xCols = None ):
        
        if xCols != None:
            dfX = df.ix[:,xCols]
        else:
            dfX = df.drop(yCol, axis=1)
        
        X = dfX.as_matrix()
        y = df[yCol].as_matrix().ravel()  #.values()
        
        return X, y, df.index, dfX.columns, yCol

                
    def run_table(self, request,inFileName = None, folderName = 'Model',  xCols = None, yCol = None, 
                  toDropCols = None, numericCols = None,  factorCols = None, toBinCols = None, toDummyCols = None, toDateTimeCols = None, 
                  mode = 'preprocess', 
                  featureSelection=False, featureTopQuantile = 10, writeFeatureScores = False, argsTablePrep = None, 
                  clfType = 'SVC', argsClf = None, testSize=0.3, randomState=1, splitType =None, coverage = 100, avgType = 'micro', validationIndex = None,  preprocessFlag = False,
                  pipeline = None, colNames = None, scaling = True, modelType = 'classification', colsToScale = None):
                      
        """
        Training/Testing/Execution of Models for Structured Data Classification
        """
        #input variables----------------
        #xCols, yCol, toDropCols, numericCols, factorCols, toBinCols, toDummyCols, dateTimeCols
        featureSelectFileName = 'features.pkl'
        validationFolder = folderName
        folderName = folderName
        
        Tp = TablePreprocesser()     
     
        #encodings = ["iso-8859-15", 'utf-8', 'ascii', 'latin1']
        
        #Read data------------
        
        df = self.readDf(inFileName, fileFormat='csv')
        print 'Total records: ', df.shape[0]
        allCols = df.columns
        idCol = colNames['sessionId']
        hasMissing = self.checkMissingValues(df)
                
        
        #filter relevant cols to keep-----
        if xCols != None:
            df = df.ix[:, [idCol] + xCols + [yCol]]
            
        else:
            allCols = df.columns.tolist()
            xCols = allCols
            xCols.remove(colNames['sessionId'])
            xCols.remove(colNames['target'])
        if colsToScale == None:
            colsToScale = xCols
        
        #Model pipeline-----------
        if 'train' in mode:
            sessionId = df[colNames['sessionId']].tolist()
            
            if preprocessFlag == True:
                #Preprocess data
                df = self.run_table_preprocess(df, xCols = xCols,toDropCols = toDropCols, numericCols = numericCols, 
                             factorCols = factorCols, toBinCols = toBinCols, toDummyCols = toDummyCols, toDateTimeCols = toDateTimeCols,
                             nBins = 10, binSize = -1, binCuts = [], handleMissingMethod= 'None', handleOutliersMethod = 'None' )
                #check for missing values again
                hasMissing = self.checkMissingValues(df)
                
            if scaling == True:
                #colsToScale = xCols
                df = Tp.scalingDf( df, colNames = colsToScale, scalerType = 'StandardScaler')
                df.to_csv(folderName+"/scaledDf.csv", index = False)
            
            
            
            
            X, y, dfIdx, XColumns, yCol =  self.getXyFromTable(df, yCol, xCols = xCols )
            
            XTrain, XTest, yTrain, yTest, sessionIdTrain, sessionIdTest = self.testTrainSplit(X, y, sessionId, testSize=testSize, randomState = randomState, splitType = splitType)
            
            if featureSelection == True:
                
                
                
                featureSelector, XTrain = self.run_text_featureSelect(XTrain, yTrain, featureTopQuantile = featureTopQuantile, folderName = folderName, featureSelectorFile = 'FeatureSelector.pkl', pickler = None)
                XTest = self.run_text_featureTransform(XTest, folderName = folderName, featureSelectorFile = 'FeatureSelector.pkl', pickler = None)
                #Save features
                featureIndices = featureSelector.get_support(indices= True)
                featureNames = XColumns #vec.get_feature_names() #Needs to be checked
                featureMap = dict(zip(featureIndices, featureNames))
                pd.DataFrame({'Features': pd.Series(featureMap)}).to_csv(folderName+"/"+"FeatureNames.csv", index = True)
                
            #select model
            clf = self.modelSelect(classifier=clfType, args = argsClf)
            #train model
            #model, featureSelector = self.run_table_train(XTrain, yTrain, modelFileName, featureTopQuantile=featuresTopN, featureSelectFileName = featureSelectFileName )
            model = self.run_text_train( XTrain, yTrain, sessionIdTrain, clf, folderName  = folderName, modelFileName = 'Classifier.pkl', modelType = modelType)
            #validate model
            
            #y_pred1, y_true = self.run_table_test(XTest, yTest, validationFolder)
            
            dfProb, model = self.run_text_test(XTest, yTest, sessionIdTest, folderName  = folderName, modelFileName = 'Classifier.pkl', avgType = 'micro' , validationIndex = None, pickler=None, modelType = modelType)
        
            #print XTrain.shape, XTest.shape, yTrain.shape, yTest.shape
            #cvResults = self.crossValidate(model, np.row_stack((XTrain,XTest)), np.append(yTrain, yTest), n_times=5)
            #print cvResults
            #cvResults.to_csv(folderName+"/"+'TestResults_CV_100Coverage.csv')
            
        if 'test' in mode:
            
            sessionId = df[colNames['sessionId']].tolist()
            #sessionId = df[colNames['idCol']].tolist()
            sessionIdTest = sessionId
            
            if preprocessFlag  == True:
                #Preprocess data
                df = self.run_table_preprocess(df,  toDropCols = toDropCols, numericCols = numericCols, 
                             factorCols = factorCols, toBinCols = toBinCols, toDummyCols = toDummyCols, toDateTimeCols = toDateTimeCols,
                             nBins = 10, binSize = -1, binCuts = [], handleMissingMethod= 'None', handleOutliersMethod = 'None' )

            if scaling == True:
                #colsToScale = xCols
                df = Tp.scalingDf(df, colNames = colsToScale, scalerType = 'StandardScaler')
                df.to_csv(folderName+"/"+"scaledDf.csv", index = False)
                
            XTest, yTest, dfIdx, XColumns, yCol =  self.getXyFromTable(df, yCol, xCols = xCols )
            
            
            if featureSelection == True:
                XTest = self.run_text_featureTransform(XTest, folderName = folderName, featureSelectorFile = 'FeatureSelector.pkl', pickler = None)
            
                
            #test model
            #y_pred2, y_true = self.run_table_test(XTest, yTest, avgType = 'micro', folderName = 'Model')
            dfProb, model = self.run_text_test(XTest, yTest, sessionIdTest, folderName  = folderName, modelFileName = 'Classifier.pkl', avgType = 'micro' , validationIndex = None, pickler=None, modelType = modelType)

            if modelType == 'classification':
                cvResults = self.crossValidate(model, XTest, yTest, n_times=5)
                print cvResults
                cvResults.to_csv(folderName+"/"+'TestResults_CV_100Coverage.csv')
            
        if 'execute' in mode:
            
            sessionId = df[colNames['sessionId']].tolist()
            sessionIdTest = sessionId
            
            if preprocessFlag == True:
                #Preprocess data
                df = self.run_table_preprocess(df, toDropCols = toDropCols, numericCols = numericCols, 
                             factorCols = factorCols, toBinCols = toBinCols, toDummyCols = toDummyCols, toDateTimeCols = toDateTimeCols,
                             nBins = 10, binSize = -1, binCuts = [], handleMissingMethod= 'None', handleOutliersMethod = 'None' )

            if scaling == True:
                #colsToScale = xCols
                df = Tp.scalingDf( df, colNames = colsToScale, scalerType = 'StandardScaler')
                df.to_csv(folderName+"/"+'scaledDf.csv', index = False)
                
            XTest, yTest, dfIdx, XColumns, yCol =  self.getXyFromTable(df, yCol, xCols = xCols )
            
            if featureSelection == True:
                XTest = self.run_text_featureTransform(XTest, folderName = folderName, featureSelectorFile = 'FeatureSelector.pkl', pickler = None)
                
            #Execute model (model scoring/predictions)
            dfProb, y_labels = self.run_text_execute(XTest, sessionIdTest, colNames, folderName = folderName, modelFile = 'Classifier.pkl', pickler=None, modelType = modelType)
            #y_pred  = self.run_table_execute(XTest)
            
        dfMerged = pd.merge(df, dfProb, how='left',left_on=colNames['sessionId'], right_on=colNames['sessionId'], left_index=True, suffixes=('_x','_y'))
        dfMerged.to_csv(folderName + '/' + 'ModelResults.csv', index=False)
        #dfMerged, probThreshold = self.reduceToCoverage(dfMerged,'Prediction Score',coverage) #Reducing output to desired coverage before printing out
        
        return dfMerged
    def run(self, fileName = None, mode = 'preprocess', featureSelection=False, featureTopQuantile = 10, writeFeatureScores = False, colNames = None, lineByLevels = None,filterBySpeaker = None, argsPrep = None, spellCorrect = False, clfType = 'SVM', argsClf = None, vecType = 'tfidfVectorizer', argsVec = None, testSize=0.3, randomState=1, splitType =None, coverage = 100, modelVariable = None, avgType = None, validationIndex =None, folderName = None, topNLinesPrimary = 5, primaryLineClassification = False, typePrep = 'classifier', preprocessFlag = False):
            """
            Main function to run the classifier
            """
            #user inputs ----------------
            folderName = folderName
    
            #fsType = 'kBest'
            #modelFileName = '%s/%s.p' % (folderName, clfType)
            #vecFileName = '%s/%s.p' % (folderName,vecType)
            #featureSelectFileName = '%s/%s.p' % (folderName,fsType)
            #featureScoresFileName = str(folderName).join('/featureScores.csv')
            encodings = ["iso-8859-15", 'utf-8', 'ascii', 'latin1']
    
            print '\n----------Reading file \'%s\' -----------' % fileName
            df = self.readDf(fileName, fileFormat="csv")
            self.addLineNum(df, colNames)
            try:
                df[colNames['issueLine']] = pd.notnull(df[colNames['L3']]).astype(int)
            except:
                pass
            #issueLines.apply(lambda x: 0 if issueLines else 1)
            
            #Pp = Preprocessing()
            #try:
            #    masksDict = Pp.readRegexMasksDict(regExfileName = 'regex1.csv')
            #    masksDict1 = Pp.readRegexMasksDict(regExfileName = 'babynames.csv')
            #    masksDict2 = Pp.readRegexMasksDict(regExfileName = 'location_db.csv')
            #    masksDict.update(masksDict1)
            #    masksDict.update(masksDict2)
            #    #masksDict = masksDict1
            #except Exception,e:
            #    print 'Exception:',e
            #df[colNames['lineText']] = df[colNames['lineText']].apply(lambda x: Pp.mask(x, masksDict, ignoreCase=True))
            df.to_csv(fileName, index=False)
    
            #df = self.filterBySpeaker(df, filterBySpeaker, colNames, lineByLevels)
            #df['lineText'].apply(lambda x: x.replace(u'\u2019', '')).apply(lambda x: x.replace('0x92', ''))
            
            #df[colNames['lineText']] = df[colNames['lineText']].apply(lambda x: codecs.decode(str(x), "ascii",'ignore')) #throws error
            try:
                df[colNames['lineText']] = df[colNames['lineText']].apply(lambda x: codecs.decode(str(x), "ascii",'ignore'))
            except:
                try:
                    df[colNames['lineText']] = df[colNames['lineText']].apply(lambda x: codecs.decode(str(x), "latin-1",'ignore'))
                except:
                    try:
                        df[colNames['lineText']] = df[colNames['lineText']].apply(lambda x: codecs.decode(str(x), "utf-8",'ignore'))
                    except:
                        print 'Decode Exception'
                        
            #df[colNames['lineText']] = df[colNames['lineText']].apply(lambda x: codecs.decode(str(x), "ascii",'ignore')) #throws error
            #df['lineText'].encode('ascii', 'ignore')
            """"The following are a set of strings/regular expression which need to be removed since they are generic in nature and do not contribute to the classification vocabulary"""
    
            df['lineText'] = [re.sub(r'.*how can i help.*',"",str(i).lower()) for i in df[colNames['lineText']]]
            df['lineText'] = [re.sub(r'{"looking.*}',"",str(i).lower()) for i in df[colNames['lineText']]]
            df['lineText'] = [re.sub(r'context.*',"",str(i).lower()) for i in df[colNames['lineText']]]
            df['lineText'] = [re.sub(r"[!@#$%^&*().,?/;:']","",str(i)) for i in df[colNames['lineText']]]
            df['lineText'] = [re.sub(r'hi',"",str(i).lower()) for i in df[colNames['lineText']]]
            df['lineText'] = [re.sub(r'hey',"",str(i).lower()) for i in df[colNames['lineText']]]
            df['lineText'] = [re.sub(r'hello',"",str(i).lower()) for i in df[colNames['lineText']]]
            #df[colNames['lineText']] = [re.sub(r'.*how can i help.*',"",str(i).lower()) for i in df[colNames['lineText']]]
            #df[colNames['lineText']] = [re.sub(r'{"looking.*}',"",str(i).lower()) for i in df[colNames['lineText']]]
            #df[colNames['lineText']] = [re.sub(r'context.*',"",str(i).lower()) for i in df[colNames['lineText']]]
            #df[colNames['lineText']] = [re.sub(r"[!@#$%^&*().,?/;:']","",str(i)) for i in df[colNames['lineText']]]
            #df[colNames['lineText']] = [re.sub(r'hi',"",str(i).lower()) for i in df[colNames['lineText']]]
            #df[colNames['lineText']] = [re.sub(r'hey',"",str(i).lower()) for i in df[colNames['lineText']]]
            #df[colNames['lineText']] = [re.sub(r'hello',"",str(i).lower()) for i in df[colNames['lineText']]]
            
            #df = self.readDf(fileName, fileFormat="csv")
            #corpus = df['lineText'].as_matrix.ravel() #tolist()
            ##, Xcol = 'lineText', yCol = modelVariable
    
            #y = [str(each) for each in y]
    
            #The 'preprocess' mode is used to do any kind of preprocessing on the input text viz. stop-word removal, normalization etc.
    
            if mode == 'preprocess':
                df = self.preprocess(df, colNames, lineByLevels, folderName= folderName, clfType = clfType, vecType= vecType,topNLinesPrimary= topNLinesPrimary,algo=None)
                if spellCorrect:
                    df = self.spellCorrectDf(df, colNames, lineByLevels, minCharLen=5, maxDissimilarity=0.4, reportErrors = True, ignoreSpaces = True)
                fileTokens = fileName.split('.')
                preprocessedFileName = '%s_preprocessed.%s' % (fileTokens[0], fileTokens[1])
                df.to_csv(preprocessedFileName, index=False)
                print "Exiting after preprocessing file"
                return 0
    
            #The following section is used for any kind of classification (both primary line and L2/L2)
    
            elif mode == 'train' or mode == 'preprocess_train':
                #Model Learning
                #Data Prep
                if mode == 'preprocess_train':
                    folderName = folderName +"/PrimaryLineClassification"
    
                if not os.path.exists(folderName):
                    os.makedirs(folderName)
                #print 'preprocessFlag',preprocessFlag
                #-----------------
                #check preprocess flag
                if preprocessFlag == True:
                    df = self.preprocess(df, colNames, lineByLevels, folderName= folderName, clfType = clfType, vecType= vecType,topNLinesPrimary= topNLinesPrimary,algo=None)
                    if spellCorrect:
                        df = self.spellCorrectDf(df, colNames, lineByLevels, minCharLen=5, maxDissimilarity=0.4, reportErrors = True, ignoreSpaces = True)
                    fileTokens = fileName.split('.')
                    preprocessedFileName = '%s_preprocessed.%s' % (fileTokens[0], fileTokens[1])
                    df.to_csv(preprocessedFileName, index=False)
                    shutil.move(preprocessedFileName,folderName)
                
                #-----------------
    
                #If the classification is for identification of the primary issue line, then a subfolder 'PrimaryLineClassification' is created to store the output and log files
    
                Xy = self.getXyFromDf(df, Xcol = colNames['lineText'], yCol = modelVariable)
                corpus = Xy[0]
                y = Xy[1]
                print '\n----------------------------Model Training-----------------------------'
                print "\nModel variable is:", modelVariable
                XTrain, XTest, yTrain, yTest, sessionIdTrain, sessionIdTest = self.testTrainSplit(corpus, y, df[colNames['sessionId']],testSize=testSize, randomState = randomState)
                #Model Learning
    
                #XTrain = XTrain[1:10000]
                #yTrain = yTrain[1:10000]
                #argsClf = {'kernel': 'linear'}
                clf = self.modelSelect(classifier=clfType, args = argsClf)
                #argsVec = {'lowercase': True, 'analyzer':'word',  'ngram_range':(1, 1), 'min_df':1,}
                #print 'argsVec',argsVec
                vec, transformedXTrain = self.vectorizeFit(XTrain, vecType = vecType, args = argsVec)
    
                print "\nNumber of features before feature-selection: ",transformedXTrain.shape[1]
                #print np.asarray(y)
                #print metrics.silhouette_score(np.array([[0,0,0],[0,0,1],[0,1,0],[1,0,0],[0,1,0],[1,1,0]]),np.array([0,0,1,1,1]))
                #print metrics.silhouette_score(transformedXTrain.todense(), np.array(y), metric='cosine',sample_size = 5000)
    
                #feature selection
                if featureSelection == True:
                    
                    featureSelector, transformedXTrain  = self.topPercentileSelection(Xa=transformedXTrain, ya= yTrain, quantile=featureTopQuantile)
                    
                    #featureSelector, transformedXTrain  = self.kBestSelection(Xa=transformedXTrain,ya= yTrain, nFeatures=5000)
                    self.toPickle(featureSelector, folderName +"/FeatureSelector.p", pickler='joblib')
                    #get new feature labels??
                    """
                    featureNames_indices = featureSelector.get_support(indices= True)
                    print "After Feature Selection:"
                    featureNames = vec.get_feature_names()
                    #print featureNames
                    #print type(featureNames)
                    feature_list =  [featureNames[i] for i in featureNames_indices]
    
                    self.toPickle(feature_list,folderName+"/"+'features.p','')
                    #vec = featureSelector
                    """
                    featureNames_indices = featureSelector.get_support(indices= True)
                    featureNames = vec.get_feature_names()
                    feature_list =  [featureNames[i] for i in featureNames_indices]
                    pd.DataFrame(feature_list).to_csv("FeatureNames.csv")
    
                """
    
                dfAlt = pd.DataFrame(transformedXTrain)
                dfAlt['tag']= yTrain
    
                print dfAlt.shape
                print dfAlt
    
                keywords = []
    
                featureNames = vec.get_feature_names()
    
                for idx, df_segment in dfAlt.groupby('tag'):
                    print df_segment['lineText'].shape
                    print df_segment['tag'].shape
                    featureSelector, transformedXTrain  = self.topPercentileSelection(Xa=df_segment['lineText'],ya= df_segment['tag'], quantile=featureTopQuantile)
                    featureNames_indices = featureSelector.get_support(indices= True)
                    keywords.append([featureNames[i] for i in featureNames_indices])
    
                print keywords
    
                """
    
                print "Number of features after feature-selection: ",transformedXTrain.shape[1]
                #writing feature scores--------
                if writeFeatureScores == True:
                    scoreTypes = ['oddsRatio', 'chiSq', 'simplifiedChiSq', 'corrCoef',
                                  'idfScore', 'mutualInfo', 'infoGain', 'lorScore', 'bnsScore', 'bnsIdfScore',
                                  'nltk_chiSq', 'nltk_studentT', 'nltk_pmi', 'nltk_likelihoodRatio']
                    scoreTypes = dict(zip(scoreTypes,scoreTypes))
                    scoreType = scoreTypes['nltk_chiSq']
                    fs = FeatureSelection()
                    print "Calculating feature scores..."
                    scoresDf = fs.featureScoresXY(transformedXTrain.todense(), yTrain, xLabels = featureNames, scoreType = scoreType) #as_matrix()
                    #topX = fs.getTopFeatures(X,y, scoreType = scoreType, topN = 1000)
                    print "Writing feature score matrix..."
                    scoresDf.to_csv(folderName+"/FeatureScore.csv")
                #model = trained classifier
    
                #clf = OneVsRestClassifier(LinearSVC())
                #print yTrain
                model = self.learn(clf, transformedXTrain, yTrain)
                pd.DataFrame(model.coef_).to_csv('Coefficients.csv')
                pd.DataFrame(model.intercept_).to_csv('Intercept.csv')
                
                self.toPickle(model, folderName+"/Classifier.p", pickler='joblib')
                self.toPickle(vec, folderName+"/FeatureTransformer.p", pickler='joblib')
                """
    
                if primaryLineClassification == True:
                    self.toPickle(model, "Results_Primary/PrimClassifier.p", pickler='joblib')
                    self.toPickle(vec, "Results_Primary/PrimVec.p", pickler='joblib')
                """
                #Model Validation
                #model = self.fromPickle(modelFileName, pickler='joblib')
                if testSize == 0:
                    pass
                else:
                    print '\n-------------------------Model Validation---------------------------\n'
                    # do not validate sample if test sample size is 0
                    transformedXTest = self.vectorizeTransform(vec, XTest)
    
                    #featureSelection
                    if featureSelection == True:
                        #featureSelector, transformedXTest  = self.topPercentileSelection(transformedXTest, vec, quantile=10)
                        #featureSelector, transformedXTest  = self.kBestSelection(Xa=transformedXTest, ya=yTest, nFeatures=50)
                        transformedXTest = featureSelector.transform(transformedXTest)
    
                    y_pred = self.predict(vec, model, transformedXTest)
                    y_probs = self.probabilities(vec, model, transformedXTest)
                    y_prob_max= [max(i) for i in y_probs]
                    #y_probs is a two dimensional array with the columns in each row represting the confidence score of particular category
                    #The category that gets the highest score will be the 'predicted' category.
                    #Since we are concerned with the confidence score of the predicted category, we take the value of the maximum score in each row
                    y_true = yTest
                    dfProb = pd.DataFrame({'sessionId':sessionIdTest, 'lineText':XTest, 'y_true':yTest, 'y_pred':y_pred, 'y_prob_max':y_prob_max})
                    dfProb.to_csv(folderName+"/"+"Train_ValidationResults_100coverage.csv",index = False)
                    
    
                    print '\n--------------Cross Validation Scores--------------\n'
    
                    cvResults = self.crossValidate(model, transformedXTest, y_true, n_times=5)
                    print cvResults
                    cvResults.to_csv(folderName+"/"+'Train_Validation_cvResults_100Coverage.csv')
    
                    print "\n-------Precision, Recall and F1-Scores at different Coverages---------------\n"
    
                    self.printValidationResults(dfProb,coverage=50)
                    self.printValidationResults(dfProb,coverage=60)
                    self.printValidationResults(dfProb,coverage=70)
                    self.printValidationResults(dfProb,coverage=80)
                    self.printValidationResults(dfProb,coverage=90)
                    self.printValidationResults(dfProb,coverage=100)
                    
                    print "\n--------------------Category-wise Results at 100% coverage---------------------------------\n"
                    cmatDf = self.getConfusionMat(y_true, y_pred) #Confusion matrix
                    
                    cmatDf.to_csv(folderName+"/"+'Train_Validation_cmatDf_100Coverage.csv')
                    scores = self.validate(y_true, y_pred, avgType = avgType, validationIndex = validationIndex) #Cross validation results
                    scores.to_csv(folderName+"/"+'Train_Validation_scores_100Coverage.csv')
                    dfReport = self.multiClassPrecisionCurve(y_true, y_pred, avgType = avgType , validationIndex = validationIndex) #Precision,Recall and F1 scores.
                    dfReport.to_csv(folderName+"/"+'Train_Validation_dfReport_100Coverage.csv')
                    """
                    ##### Latchi######
                    
                    pd.set_option('display.max_colwidth', -1)
                    
                    correct_src = settings.STATIC_URL+"images"+"/"+"correct.png"
                    correct_html = "<img src='%s' height='25' width='25' ></img>" %correct_src
                     
                    wrong_src = settings.STATIC_URL+"images"+"/"+"wrong.png"
                    wrong_html = "<img src='%s' height='25' width='25' ></img>" %wrong_src
                    src = settings.STATIC_URL+"images"+"/"+"orange-minus-md.png"
                    broken_html = "<img src='%s' height='25' width='25'></img>" %src
                    dfReport['Acceptability']=dfReport['fScore'].apply(lambda x: correct_html if x>=0.7  else broken_html if  x >= 0.4 else wrong_html)
                    dfReport = dfReport.sort(columns=['fScore'],ascending=[False])
                    
                    
                    fScoreavg = self.roundToDecimal(sum([fscore*support for fscore,support in zip(dfReport['fScore'],dfReport['support'])])/sum(dfReport['support']))
                    precisionavg = self.roundToDecimal(sum([precision*support for precision,support in zip(dfReport['precision'],dfReport['support'])])/sum(dfReport['support']))
                    recallavg = self.roundToDecimal(sum([recall*support for recall,support in zip(dfReport['recall'],dfReport['support'])])/sum(dfReport['support']))
                    supportavg = self.roundToDecimal(sum(dfReport['support']))
                    
                    dfReport1 = pd.DataFrame({'Category':['avg/total'],'fScore':[fScoreavg],'precision':[precisionavg],'recall':[recallavg],'support':[supportavg],'Acceptability':['']})
                    dfReport = dfReport.append(dfReport1)
                    df_100 = dfReport.to_html(escape=False)"""
                    
                    #print df_100
                    #re.sub(r'&lt;','<',df_100)
                    #re.sub(r'&gt;','>',df_100)
                    #
                    #fScore = self.roundToDecimal(dfReport['fScore'])
                    #dfProb['y_prob_max'] = dfProb.apply(lambda x: max(x), axis = 1) #ignore session id
                    #dfProb['y_true'] = y_true
                    #dfProb['y_pred'] = y_pred
    
                    ############################# The following section prints validation results at the user desired coverage #####################
                    
                    dfProb = self.reduceToCoverage(dfProb,'y_prob_max', coverage) #Downselecting to user-desired coverage
                    
                    #The following conversion to list format is necessitated because the function that spit out validation metrics except inputs as lists not pandas series
    
                    XTest = dfProb['lineText'].tolist()
                    y_true = dfProb['y_true'].tolist()
                    y_pred = dfProb['y_pred'].tolist()
                    y_prob_max = dfProb['y_prob_max'].tolist()
    
                    print "\n"
                    pd.DataFrame({'XTest': XTest, 'y__true': y_true, 'y_pred': y_pred, 'y_prob_max':y_prob_max}).to_csv(folderName+"/"+"Train_ValidationResults_"+str(coverage)+"coverage.csv",index = False)
                    #pd.DataFrame([XTest, y_true, y_pred]).transpose().to_csv(folderName+"/Train_ValidationResults.csv",index = False, columns= ('lineText','y_true','y_pred'))
                    #cmat, labels = self.getConfusionMat(y_true, y_pred)
                    #print cmat, labels
    
                    print "--------------------Category-wise Results at "+str(coverage),"%coverage---------------------------------\n"
                    cmatDf = self.getConfusionMat(y_true, y_pred)
                    cmatDf.to_csv(folderName+"/"+'Train_Validation_'+str(coverage)+'cmatDf.csv')
                    scores = self.validate(y_true, y_pred, avgType = avgType, validationIndex = validationIndex)
                    scores.to_csv(folderName+"/"+'Train_Validation_'+str(coverage)+'scores.csv')
                    dfReport = self.multiClassPrecisionCurve(y_true, y_pred, avgType = avgType , validationIndex = validationIndex)
                    #print dfReport
                    dfReport.to_csv(folderName+"/"+'Train_Validation_'+str(coverage)+'dfReport.csv')
                    
                    
                    """
                    ###Latchi###
                    
                    dfReport['Acceptability']=dfReport['fScore'].apply(lambda x: correct_html if x>=0.7  else broken_html if  x >= 0.4 else wrong_html)
                    dfReport = dfReport.sort(columns=['fScore'],ascending=[False])
                    fScoreavg = self.roundToDecimal(sum([fscore*support for fscore,support in zip(dfReport['fScore'],dfReport['support'])])/sum(dfReport['support']))
                    precisionavg = self.roundToDecimal(sum([precision*support for precision,support in zip(dfReport['precision'],dfReport['support'])])/sum(dfReport['support']))
                    recallavg = self.roundToDecimal(sum([recall*support for recall,support in zip(dfReport['recall'],dfReport['support'])])/sum(dfReport['support']))
                    supportavg = self.roundToDecimal(sum(dfReport['support']))
                    
                    dfReport1 = pd.DataFrame({'Category':['avg/total'],'fScore':[fScoreavg],'precision':[precisionavg],'recall':[recallavg],'support':[supportavg],'Acceptability':['']})
                    dfReport = dfReport.append(dfReport1)
                    df_60 = dfReport.to_html(escape = False)
                    print "---------------------------------------------------------------------------\n"
                    cmat_100 = folderName+"/"+'Train_Validation_cmatDf_100Coverage.csv'
                    scores_100 = folderName+"/"+'Train_Validation_scores_100Coverage.csv'
                    
                    results_100 = os.path.join(folderName,'Train_Validation_dfReport_100Coverage.csv')
                    
                    validation_results_100 = folderName+"/"+"Train_ValidationResults_100coverage.csv"
                    cross_validation_results_100 = folderName+"/"+'Train_Validation_cvResults_100Coverage.csv'
                    cmat_60 = folderName+"/"+'Train_Validation_'+str(coverage)+'cmatDf.csv'
                    scores_60 = folderName+"/"+'Train_Validation_'+str(coverage)+'scores.csv'
                    results_60 = folderName+"/"+'Train_Validation_'+str(coverage)+'dfReport.csv'
                    validation_results_60 = folderName+"/"+"Train_ValidationResults_"+str(coverage)+"coverage.csv"
                    """
                    
                    return df_100,df_60,cmat_100,scores_100,cmat_60,scores_60,results_100,validation_results_100,cross_validation_results_100,results_60,validation_results_60
    
            elif 'test' in mode:
    
                sessionIdTest = []
                tag = []
    
    
                #The following snippet identifies the manual tag corresponding to each sessionId.
                #This manual tag is present on any one of the lines of the sessionId. Hence the first non-value in the tagged column is taken.
    
                for sessionId,dfgroup in df.groupby(colNames['sessionId']):
                    tag_array = dfgroup.loc[dfgroup[colNames['L3']].notnull(),colNames['L3']].tolist()
                    if len(tag_array)>0:
                        sessionIdTest.append(sessionId)
                        tag.append(tag_array[0])
    
                dfTagged = pd.DataFrame({colNames['sessionId']:sessionIdTest,'y_true':tag}) #dfTagged contains one record for each sessionId
                #The number of sessionIds in dfTagged may be less than that in df since some sessionIds may not have been tagged.
    
                print "Number of tagged chats:", dfTagged.shape[0]
    
    
    
                if 'preprocess_and_test' in mode:
    
                    print '\n-------------------------Primary Line Extraction on Test File-------------------------'
    
                    #Primary line classification on the test file (First stage)
    
                    try:
                        model_prim = self.fromPickle(folderName+"/PrimaryLineClassification/Classifier.p", pickler=None)
                    except:
                        print "cannot open model file"
                    try:
                        vec_prim = self.fromPickle(folderName+"/PrimaryLineClassification/FeatureTransformer.p", pickler=None)
                    except:
                        print "cannot open vectorizer file"
    
    
                    df = self.issueLineClassifier(df, folderName, model_prim, vec_prim, topNLinesPrimary,colNames, lineByLevels, featureSelection, speaker = filterBySpeaker)
    
    
                #print modelFileName
    
                #issueVocabList = self.fromPickle(folderName+"/"+'features.p','')
                #print type(issueVocabList)
    
                #df = self.preprocess(df, colNames, lineByLevels, folderName, clfType, vecType, topNLinesPrimary, issueVocabList = issueVocabList,  algo = None  )
                #print modelFileName
    
                #Check if preprocessFlag == True ------------------------
                if preprocessFlag == True:
                    df = self.preprocess(df, colNames, lineByLevels, folderName= folderName, clfType = clfType, vecType= vecType,topNLinesPrimary= topNLinesPrimary,algo=None)
                    if spellCorrect:
                        df = self.spellCorrectDf(df, colNames, lineByLevels, minCharLen=5, maxDissimilarity=0.4, reportErrors = True, ignoreSpaces = True)
                    fileTokens = fileName.split('.')
                    preprocessedFileName = '%s_preprocessed.%s' % (fileTokens[0], fileTokens[1])
                    df.to_csv(preprocessedFileName, index=False)
                    shutil.move(preprocessedFileName,folderName)
    
                #---------------------------------------------------------
    
                print '\n-------------------------Classification on Test File-------------------------'
    
                #Classification on the output of the previous step (Stage 2)
    
                corpus = df[colNames['lineText']]
                try:
                    model = self.fromPickle(folderName+"/Classifier.p", pickler='joblib')
                except:
                    print "cannot open model file"
                try:
                    vec = self.fromPickle(folderName+"/FeatureTransformer.p", pickler='joblib')
                except:
                    print "cannot open vectorizer file"
    
                transformedXTest = self.vectorizeTransform(vec, corpus)
    
                #featureSelection
                if featureSelection == True:
                    try:
                        featureSelector = self.fromPickle(folderName+"/FeatureSelector.p", pickler='joblib')
                    except:
                        print "cannot open feature selector file"
    
                    #featureSelector, transformedXTest  = self.topPercentileSelection(transformedXTest, vec, quantile=10)
                    transformedXTest = featureSelector.transform(transformedXTest)
    
                y_pred = self.predict(vec, model, transformedXTest)
                y_probs = self.probabilities(vec, model, transformedXTest)
                y_prob_max= [max(i) for i in y_probs]
    
                df['y_pred'] = y_pred;
                df['y_prob_max'] = y_prob_max;
    
                print "\nNo of chats on which prediction has been made (at 100% Coverage):", df.shape[0]
    
                """
                dfProb = pd.DataFrame(y_probs, columns = model.classes_)
                #not sure if the column order and the y_probs order is the same
                #print "%%%%%%%%%%%%%%%%%%%%"
                #print modelVariable
                dfProb['y_prob_max'] = dfProb.apply(lambda x: max(x), axis = 1) #ignore session id
                dfProb['sessionId'] = df['sessionId'].tolist()
                #dfProb['y_true'] = df[modelVariable].tolist()
                dfProb['y_pred'] = df['y_pred'].tolist()
                dfProb['lineText'] = df['lineText'].tolist()
    
                #print dfProb
                """
                #prob_threshold = np.percentile(dfProb['y_prob_max'],100-coverage)
                #dfProb_threshold = dfProb[dfProb['y_prob_max']>prob_threshold]
                #dfNew = pd.concat(df, dfProb, axis=1)
                #dfProb_threshold.to_csv(folderName+'/classification_results_updated_'+str(coverage)+'.csv', index = False)
                #y_true = y
                #results = self.getConfusionMat(y, y_pred)
                #cmat = results[0]
                #labels = results[1]
                #print 'confusion matrix'
                #print cmat
                #print 'labels'
                #print labels
    
                # The following code compares the manual tag versus the prediction on a sessionId level.
                # The dataframe with the manual tag (y_true) and the dataframe with the prediction (y_pred) are merged and the validation metrics are printed
    
    
                dfMerged = pd.merge(dfTagged, df,how='inner',left_on=colNames['sessionId'], right_on=colNames['sessionId'], left_index=True, suffixes=('_x','_y'))
    
                print "Number of chats after merging tagged and predicted chats:", dfMerged.shape[0]
    
    
                print dfMerged.columns
                y_true = dfMerged['y_true'].tolist();
                y_pred = dfMerged['y_pred'].tolist();
                y_prob_max = dfMerged['y_prob_max']
                lineText = dfMerged[colNames['lineText']].tolist()
    
    
                pd.DataFrame({'XTest': lineText,'y__true': y_true, 'y_pred': y_pred, 'y_prob_max':y_prob_max}).to_csv(folderName+"/"+"Test_ValidationResults_100%coverage.csv",index = False)
    
    
                print "\n-------Precision, Recall and F1-Scores at different Coverages---------------\n"
    
                self.printValidationResults(dfMerged,coverage=50)
                self.printValidationResults(dfMerged,coverage=60)
                self.printValidationResults(dfMerged,coverage=70)
                self.printValidationResults(dfMerged,coverage=80)
                self.printValidationResults(dfMerged,coverage=90)
                self.printValidationResults(dfMerged,coverage=100)
    
                print "\n--------------------Category-wise Results at 100% coverage---------------------------------\n"
    
                cmatDf = self.getConfusionMat(y_true, y_pred)
                #print cmatDf
                cmatDf.to_csv(folderName+'/Test_cmatDf_100%Coverage.csv')
                scores = self.validate(y_true, y_pred, avgType = avgType, validationIndex = validationIndex)
                #print scores
                scores.to_csv(folderName+'/Test_scores_100%Coverage.csv')
                dfReport = self.multiClassPrecisionCurve(y_true, y_pred, avgType = avgType , validationIndex = validationIndex)
                #print dfReport
                dfReport.to_csv(folderName+'/Test_dfReport_100%Coverage.csv')
                
                """
                ###Latchi#####
                
                pd.set_option('display.max_colwidth', -1)
                correct_src = settings.STATIC_URL+"images"+"/"+"correct.png"
                correct_html = "<img src='%s' height='25' width='25' ></img>" %correct_src
                 
                wrong_src = settings.STATIC_URL+"images"+"/"+"wrong.png"
                wrong_html = "<img src='%s' height='25' width='25' ></img>" %wrong_src
                src = settings.STATIC_URL+"images"+"/"+"orange-minus-md.png"
                broken_html = "<img src='%s' height='25' width='25'></img>" %src
                
                dfReport['Acceptability']=dfReport['fScore'].apply(lambda x: correct_html if x>=0.7  else broken_html if  x >= 0.4 else wrong_html)
                dfReport = dfReport.sort(columns=['fScore'],ascending=[False])
                
                
                fScoreavg = self.roundToDecimal(sum([fscore*support for fscore,support in zip(dfReport['fScore'],dfReport['support'])])/sum(dfReport['support']))
                precisionavg = self.roundToDecimal(sum([precision*support for precision,support in zip(dfReport['precision'],dfReport['support'])])/sum(dfReport['support']))
                recallavg = self.roundToDecimal(sum([recall*support for recall,support in zip(dfReport['recall'],dfReport['support'])])/sum(dfReport['support']))
                supportavg = self.roundToDecimal(sum(dfReport['support']))
                
                dfReport1 = pd.DataFrame({'Category':['avg/total'],'fScore':[fScoreavg],'precision':[precisionavg],'recall':[recallavg],'support':[supportavg],'Acceptability':['']})
                dfReport = dfReport.append(dfReport1)
                
                df_100 = dfReport.to_html(escape = False)
                
                
                """
                #cvResults = self.crossValidate(model, transformedXTest, np.array(y_true), n_times=5)
                #print cvResults
                #cvResults.to_csv(folderName+'/cvScores.csv')
                dfMerged = self.reduceToCoverage(dfMerged,'y_prob_max', coverage)
    
                XTest = dfMerged['lineText'].tolist()
                y_true = dfMerged['y_true'].tolist()
                y_pred = dfMerged['y_pred'].tolist()
                y_prob_max = dfMerged['y_prob_max'].tolist()
    
                print "\n"
                pd.DataFrame({'XTest': XTest, 'y__true': y_true, 'y_pred': y_pred, 'y_prob_max':y_prob_max}).to_csv(folderName+"/"+"Test_ValidationResults_"+str(coverage)+"%coverage.csv",index = False)
                #pd.DataFrame([XTest, y_true, y_pred]).transpose().to_csv(folderName+"/Train_ValidationResults.csv",index = False, columns= ('lineText','y_true','y_pred'))
                #cmat, labels = self.getConfusionMat(y_true, y_pred)
                #print cmat, labels
    
                print "--------------------Category-wise Results at "+str(coverage),"%coverage---------------------------------\n"
    
                cmatDf = self.getConfusionMat(y_true, y_pred)
                cmatDf.to_csv(folderName+"/"+'Test_'+str(coverage)+'%cmatDf.csv')
                scores = self.validate(y_true, y_pred, avgType = avgType, validationIndex = validationIndex)
                scores.to_csv(folderName+"/"+'Test_'+str(coverage)+'%scores.csv')
                dfReport = self.multiClassPrecisionCurve(y_true, y_pred, avgType = avgType , validationIndex = validationIndex)
                
                #print dfReport
                dfReport.to_csv(folderName+"/"+'Test_'+str(coverage)+'%dfReport.csv')
                
                """
                
                ####Latchi####
                
                dfReport['Acceptability']=dfReport['fScore'].apply(lambda x: correct_html if x>=0.7  else broken_html if  x >= 0.4 else wrong_html)
                dfReport = dfReport.sort(columns=['fScore'],ascending=[False])
                fScoreavg = self.roundToDecimal(sum([fscore*support for fscore,support in zip(dfReport['fScore'],dfReport['support'])])/sum(dfReport['support']))
                precisionavg = self.roundToDecimal(sum([precision*support for precision,support in zip(dfReport['precision'],dfReport['support'])])/sum(dfReport['support']))
                recallavg = self.roundToDecimal(sum([recall*support for recall,support in zip(dfReport['recall'],dfReport['support'])])/sum(dfReport['support']))
                supportavg = self.roundToDecimal(sum(dfReport['support']))
                
                dfReport1 = pd.DataFrame({'Category':['avg/total'],'fScore':[fScoreavg],'precision':[precisionavg],'recall':[recallavg],'support':[supportavg],'Acceptability':['']})
                dfReport = dfReport.append(dfReport1)
                
                df_60 = dfReport.to_html(escape=False)
                
                cmat_100 = folderName+'/Test_cmatDf_100%Coverage.csv'
                scores_100 = folderName+'/Test_scores_100%Coverage.csv'
                results_100 = folderName+'/Test_dfReport_100%Coverage.csv'
                
                cmat_60 = folderName+"/"+'Test_'+str(coverage)+'%cmatDf.csv'
                scores_60 = folderName+"/"+'Test_'+str(coverage)+'%scores.csv'
                results_60 = folderName+"/"+'Test_'+str(coverage)+'%dfReport.csv'
                """
                return df_100,df_60,cmat_100,scores_100,results_100,cmat_60,scores_60,results_60
    
            elif 'execute' in mode:
    
                if 'preprocess_and_execute' in mode:
    
    
                    print '\n****************************Primary Line Extraction on Execution File******************************'
    
                    #Primary line classification on the execution file (First stage)
                    
                    try:
                        model_prim = self.fromPickle(folderName+"/PrimaryLineClassification/Classifier.p", pickler=None)
                    except:
                        pass #print "cannot open model file"
                    try:
                        vec_prim = self.fromPickle(folderName+"/PrimaryLineClassification/FeatureTransformer.p", pickler=None)
                    except:
                        pass #print "cannot open vectorizer file"
    
                    if typePrep == 'classifier':
                        df = self.issueLineClassifier(df, folderName, model_prim, vec_prim, topNLinesPrimary,colNames, lineByLevels, featureSelection, speaker= filterBySpeaker)
                    else:
                        df = self.preprocess(df,colNames,lineByLevels,folderName,clfType,vecType,topNLinesPrimary,mode)
    
    
                #Check if preprocessFlag == True ------------------------
                if preprocessFlag == True:
                    df = self.preprocess(df, colNames, lineByLevels, folderName= folderName, clfType = clfType, vecType= vecType,topNLinesPrimary= topNLinesPrimary,algo=None)
                    if spellCorrect:
                        df = self.spellCorrectDf(df, colNames, lineByLevels, minCharLen=5, maxDissimilarity=0.4, reportErrors = True, ignoreSpaces = True)
                    fileTokens = fileName.split('.')
                    preprocessedFileName = '%s_preprocessed.%s' % (fileTokens[0], fileTokens[1])
                    df.to_csv(preprocessedFileName, index=False)
    
                #---------------------------------------------------------
    
                print '\n***********************************Classifying Execution File***********************************'
    
                #Classification on the output of the previous step (Stage 2)
                
                try:
                    model = self.fromPickle(folderName+"/Classifier.p", pickler=None)
                except:
                    print "cannot open model file"
                try:
                    vec = self.fromPickle(folderName+"/FeatureTransformer.p", pickler=None)
                except:
                    print "cannot open vectorizer file"
    
                corpus = df[colNames['lineText']]
    
                transformedXTest = self.vectorizeTransform(vec, corpus)
    
    
                #featureSelection
                if featureSelection == True:
                    try:
                        featureSelector = self.fromPickle(folderName+"/FeatureSelector.p", pickler='joblib')
                    except:
                        print "cannot open feature selector file"
    
                    #featureSelector, transformedXTest  = self.topPercentileSelection(transformedXTest, vec, quantile=10)
                    transformedXTest = featureSelector.transform(transformedXTest)
    
    
                y_pred = self.predict(vec, model, transformedXTest)
                y_probs = self.probabilities(vec, model, transformedXTest)
                y_prob_max= [max(i) for i in y_probs]
                df['Predicted Category'] = y_pred
                df['Prediction Score'] = y_prob_max
    
                df = self.reduceToCoverage(df,'Prediction Score',coverage) #Reducing output to desired coverage before printing out
    
                """
                prob_threshold = np.percentile(df['L3 Prediction Score'],100-coverage)
                df = df[df['L3 Prediction Score']>prob_threshold]
    
    
                dfProb = pd.DataFrame(y_probs, columns = model.classes_)
                dfProb_New = pd.DataFrame()
                #not sure if the column order and the y_probs order is the same
                dfProb_New['sessionId'] = df['sessionId']
                dfProb_New['lineText'] = df['lineText']
                dfProb_New['L3 Predicted'] = df['y_pred']
                dfProb_New['L3 Prediction Score'] = dfProb.apply(lambda x: max(x), axis = 1)
                prob_threshold = np.percentile(dfProb_New['L3 Prediction Score'],100-coverage)
                dfProb_New = dfProb_New[dfProb_New['L3 Prediction Score']>prob_threshold]
    
    
                category_map = "L2L3Mapping.csv"
    
                map = pd.read_csv(category_map)
                map_dict = dict(zip(map['L3'].tolist(),map['L2'].tolist()))
                dfProb_New['L2 Predicted'] = [map_dict[i] for i in y_pred]
                """
                #dfNew = pd.concat(df, dfProb, axis=1)
                outputfilename = folderName+"/"+'Final_Categorization.csv'
                df.to_csv(folderName+"/"+'Final_Categorization.csv', index = False)
                sessionId = df[colNames['sessionId']]
                predictedCategory = df['Predicted Category']
                print "\nOutput with ",coverage,"% coverage written into the file: '%s'", folderName+"/"+'Final_Categorization.csv'
                return sessionId,predictedCategory,outputfilename
                #y_true = y
    
            else:
                print 'Incorrect mode selected'
            return 0
