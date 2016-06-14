# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 23:01:56 2015

@author: bhupinder.s
"""
import os
#from mysettings import *
import pandas as pd

from afinn import Afinn
import sentlex
import sentlex.sentanalysis
#from vaderSentiment import sentiment as vaderSentiment 
#import vaderSentiment as vs
from vaderSentiment import vaderSentiment as vs
from pattern.en import sentiment,polarity

from senti_classifier import senti_classifier
from nltk import word_tokenize
from nltk.corpus import sentiwordnet as swn
import nltk
import statistics

"""
cran.r-project.org/src/contrib/Archive/sentiment/sentiment_0.2.tar.gz
sentistrength, tapor, sentiwordnet
AFINN
sentilex
LabMT
SentiWordNet
EmoLex
General Inquirer
ANEW, wordnetAffect
sentistrength
https://github.com/darenr/afinn
http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/6006/pdf/imm6006.pdf
https://github.com/bohana/sentlex
https://cran.r-project.org/web/packages/qdapDictionaries/qdapDictionaries.pdf
http://fjavieralba.com/basic-sentiment-analysis-with-python.html
http://finzi.psych.upenn.edu/library/qdap/html/polarity.html
http://sentiment.christopherpotts.net/
http://sentiment.christopherpotts.net/lexicons.html
http://wndomains.fbk.eu/wnaffect.html
https://www.aclweb.org/anthology/J/J11/J11-2001.pdf
https://github.com/hitesh915/sentimentstrength
http://www.cs.waikato.ac.nz/~eibe/pubs/ijcai15.pdf
http://www.lrec-conf.org/proceedings/lrec2014/pdf/483_Paper.pdf
http://www.public.iastate.edu/~ekkekaki/pdfs/ekkekakis_2012.pdf 
https://shaktitechnology.com/whissel-dictionary-of-affect/index.htm
http://neuro.imm.dtu.dk/wiki/Sentiment_analysis
http://wndomains.fbk.eu/download.html - ANEW, WordnetAffect are for research purposes only

Carlo Strapparava and Alessandro Valitutti.
"WordNet-Affect: an affective extension ofWordNet".
In Proceedings of the 4th International Conference on Language Resources and Evaluation (LREC 2004), Lisbon, May 2004, pp. 1083-1086.

Finn Årup Nielsen, 
A new ANEW: evaluation of a word list for sentiment analysis in microblogs, 
Proceedings of the ESWC2011 Workshop on ‘Making Sense of Microposts’: 
Big things come in small packages 718 in CEUR Workshop Proceedings: 93-98. 2011 May. 
Matthew Rowe, Milan Stankovic, Aba-Sah Dadzie, Mariann Hardey (editors)

"""

class Sentiment:

    def __init__(self):
        pass
        #self.pattern = pattern
        #self.params = params
        
    
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
    
    def afinnSentiScore(self, doc, emoticons = True):
        """
        The output is a float variable that if larger than zero indicates a
        positive sentiment and less than zero indicates negative sentiment.
        """
        
        afinn = Afinn(emoticons=emoticons)
        result = afinn.score(doc)
        return result
    
    def sentilexScore(self, doc):
        SWN = sentlex.SWN3Lexicon()
        classifier = sentlex.sentanalysis.BasicDocSentiScore()
        result = classifier.classify_document(doc, tagged=False, L=SWN, a=True, v=True, n=False, r=False, negation=False, verbose=False)
        result_detailed = classifier.resultdata
        print result_detailed
        #from senti_classifier import senti_classifier
        # = ['The movie was the worst movie', 'It was the worst acting by the actors']
        #from senti_classifier.senti_classifier import synsets_scores
        #print synsets_scores['peaceful.a.01']['pos']
        #pos_score, neg_score = senti_classifier.polarity_scores(doc)
        #return pos_score, neg_score
        result = {'resultneg':result_detailed['resultneg'],'resultpos':result_detailed['resultpos']}
        return result

    def vaderSentiScore(self, doc):
        result = vs.sentiment(doc)
        return result
    
    
    def get_polarity(self, sentence):
        """
        given sentence provide the polarity {1,-1}

        """
        return polarity(sentence)
    
    def get_sentiment_class(self, sentence, lowerBound = -0.75, upperBound=0.75):
        p = self.get_polarity(sentence) 
        #senti = -1 if p < lowerBound and p >= -1 else (1 if p > upperBound and p <= 1 else 0)
        
        negSplits = (-0.75,-0.5,-0.25)
        posSplits = (0.75,0.5,0.25)
        
        if p >= -1 and p < negSplits[0]:
            senti = 'highly negative'
        elif p >= negSplits[0] and p < negSplits[1]:
            senti = 'negative'
        elif p >= negSplits[1] and p < negSplits[2]:
            senti = 'mildly negative'
        elif p >= negSplits[2] and p< posSplits[2]:
            senti = 'neutral'
        elif p >= posSplits[2] and p < posSplits[1]:
            senti = 'postive'
        elif p >= posSplits[1] and p < posSplits[0]:
            senti = 'mildly positive'
        else:
            senti = 'highly positive'
         
            
        #senti = (-0.75, -0.5,-0.25)
        
        return senti
    def get_sentiment_class1(self, sentence, lowerBound = -0.75, upperBound=0.75):
        p = self.get_polarity(sentence) 
        #senti = -1 if p < lowerBound and p >= -1 else (1 if p > upperBound and p <= 1 else 0)
        
        negSplits = (-0.75,-0.5,-0.25)
        posSplits = (0.75,0.5,0.25)
        
        if p <= lowerBound:
            senti = 'negative'
        elif p >= upperBound:
            senti = 'positive'
        else:
            senti = 'neutral'
         
            
        #senti = (-0.75, -0.5,-0.25)
        
        return senti
    def patternSentiScore(self, doc):
        """
        returns polarity,subjectivity of a doc as tuple
        polarity is a value between -1.0 to  +1.0
        subjectivity between 0.0 to 1.0
        
        """
        
        result = sentiment(doc)
        sentiLabel = self.get_sentiment_class1(doc,lowerBound = -0.65, upperBound=0.65)
        result = {'polarity':result[0],'subjectivity':result[1],'category':sentiLabel}
        return result
    
    def sentiClassfierScore(self,doc):
        pos_score,neg_score = senti_classifier.polarity_scores([doc])
        result = {'pos_score':pos_score,'neg_score':neg_score}
        return result
        
        
    def labMtMoodScore(self, doc):
        import pandas as pd
 
        url = 'http://www.plosone.org/article/fetchSingleRepresentation.action?uri=info:doi/10.1371/journal.pone.0026752.s001'
        labmt = pd.read_csv(url, skiprows=2, sep='\t', index_col=0)
         
        average = labmt.happiness_average.mean()
        happiness = (labmt.happiness_average - average).to_dict()
        
        words = doc.split()
        return sum([happiness.get(word.lower(), 0.0) for word in words]) / len(words)
   
    def sentiWnScore(self,text):
        tokens = word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        posScore = []
        negScore = []
        objScore = []

        for j in range(len(tagged)):
            avgNeg = 0
            avgPos = 0
            avgObj = 0
            #print tagged[j][0]
            if 'NN' in tagged[j][1] and len(swn.senti_synsets(tagged[j][0],'n'))>0:
                avgPos = statistics.mean((list(swn.senti_synsets(tagged[j][0],'n'))[k]).pos_score() for k in range(len(swn.senti_synsets(tagged[j][0],'n'))))
                avgNeg = statistics.mean((list(swn.senti_synsets(tagged[j][0],'n'))[k]).neg_score() for k in range(len(swn.senti_synsets(tagged[j][0],'n'))))
                avgObj = statistics.mean((list(swn.senti_synsets(tagged[j][0],'n'))[k]).obj_score() for k in range(len(swn.senti_synsets(tagged[j][0],'n'))))
            if 'VB' in tagged[j][1] and len(swn.senti_synsets(tagged[j][0],'v'))>0:
                avgPos = statistics.mean((list(swn.senti_synsets(tagged[j][0],'v'))[k]).pos_score() for k in range(len(swn.senti_synsets(tagged[j][0],'v'))))
                avgNeg = statistics.mean((list(swn.senti_synsets(tagged[j][0],'v'))[k]).neg_score() for k in range(len(swn.senti_synsets(tagged[j][0],'v'))))
                avgObj = statistics.mean((list(swn.senti_synsets(tagged[j][0],'v'))[k]).obj_score() for k in range(len(swn.senti_synsets(tagged[j][0],'v'))))
            if 'JJ' in tagged[j][1] and len(swn.senti_synsets(tagged[j][0],'a'))>0:
                avgPos = statistics.mean((list(swn.senti_synsets(tagged[j][0],'a'))[k]).pos_score() for k in range(len(swn.senti_synsets(tagged[j][0],'a'))))
                avgNeg = statistics.mean((list(swn.senti_synsets(tagged[j][0],'a'))[k]).neg_score() for k in range(len(swn.senti_synsets(tagged[j][0],'a'))))
                avgObj = statistics.mean((list(swn.senti_synsets(tagged[j][0],'a'))[k]).obj_score() for k in range(len(swn.senti_synsets(tagged[j][0],'a'))))
            if 'RB' in tagged[j][1] and len(swn.senti_synsets(tagged[j][0],'r'))>0:
                avgPos = statistics.mean((list(swn.senti_synsets(tagged[j][0],'r'))[k]).pos_score() for k in range(len(swn.senti_synsets(tagged[j][0],'r'))))
                avgNeg = statistics.mean((list(swn.senti_synsets(tagged[j][0],'r'))[k]).neg_score() for k in range(len(swn.senti_synsets(tagged[j][0],'r'))))
                avgObj = statistics.mean((list(swn.senti_synsets(tagged[j][0],'r'))[k]).obj_score() for k in range(len(swn.senti_synsets(tagged[j][0],'r'))))
            posScore.append(avgPos)
            negScore.append(avgNeg)
            objScore.append(avgObj)
        posList = [float(x) for x in posScore if x > 0]
        negList = [float(x) for x in negScore if x > 0]
        objList = [float(x) for x in objScore if x > 0]
        posScore = [statistics.mean(posList) if len(posList)> 0 else 0][0]
        negScore = [statistics.mean(negList) if len(negList)> 0 else 0][0]
        objScore = [statistics.mean(objList) if len(objList)> 0 else 0][0]
        
        return {'posScore':posScore,'negScore':negScore,'objScore':objScore}
    
    def run(self, doc, modelType = 'pattern',  args = [], kwargs = {}):
        
        sentiMap = {'pattern': self.patternSentiScore, 'afinn':self.afinnSentiScore,'vader':self.vaderSentiScore,'sentlex':self.sentilexScore,
                    'senti_classifier':self.sentiClassfierScore,'sentiwordnet':self.sentiWnScore
                    }
        
        results = sentiMap[modelType](doc)
        #twentiment
        return results
        

'''

if __name__ == '__main__':    
    baseDir = os.path.dirname(os.path.abspath(__file__))
    baseDir =  baseDir.replace('\\', '/')
    os.chdir(baseDir)
    
    inFileName = 'Twc_care_q8_Tagged_v2_train - Copy.csv'
    outFileName = 'results.csv'
    pattern = 'afinn' # ['pattern','afinn','sentlex','vader','senti_classifier','sentiwordnet']
    
    emo = Sentiment(pattern,threshold = 0.2)
    df = emo.readDf(inFileName)
    if pattern == 'pattern':
        df['pattern_senti'] = df['lineText'].apply(lambda x: emo.patternSentiScore(x))
    elif pattern == 'afinn':
        df['affin_senti'] = df['lineText'].apply(lambda x: emo.afinnSentiScore(x))
    elif pattern == 'vader':
        df['vader_senti'] = df['lineText'].apply(lambda x: emo.vaderSentiScore(x))
    elif pattern == 'sentlex':
        df['sentilex_senti'] = df['lineText'].apply(lambda x: emo.sentilexScore(x))
    elif pattern == 'senti_classifier':
        df['senticlassfier_senti'] = df['lineText'].apply(lambda x: emo.sentiClassfierScore(x))
    elif pattern == 'sentiwordnet':
        df['wn_senti'] = df['lineText'].apply(lambda x: emo.sentiWnScore(x))
    else:
        print "pattern not found !"
    
    #twentiment
    #sasa (sail)
    
    df.to_csv(outFileName, index=False)
'''