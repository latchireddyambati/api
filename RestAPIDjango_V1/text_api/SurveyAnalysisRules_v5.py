# -*- coding: utf-8 -*-
"""
Created on Fri Aug 08 16:22:19 2014

@author: Bhupinder.S
"""

"""
To do:
derive rules from baseclass - Done
complex rule parsers (rule plus strings)
unigram, bigram rules
greater than 2 words
any combination of word lists
integration with dataframes
rule tfidf
append rule id to text
rule diagnostics - precision, recall, f1, support
rule clustering
rule conflict resolution
rule merging
automatic rule generation
user defined operators
multiprocessing, whoosh
symbol class
rule + string searches
different return types: flag, count, tfidf, text, doc index, previous line,
    next line, previous line by customer, next line by customer, previous line by cust index
rule scoping across documents 
"""


import pandas as pd
import os
import re

#from sentimentAnalyser_v4 import sentimentAnalyser
#import viewResults_v3
from sentiment_api.Sentiment import Sentiment

import codecs


baseDir = os.path.dirname(os.path.abspath(__file__))
#os.chdir(baseDir)

returnTypes = ["count", "flag", "l1Category", "l2Category", "l1l2Categories"]
           
class Rule:
    """
    Base class for rules
    """
    def __init__(self, operator, phrase1, phrase2, l1Category, l2Category, returnType="flag", ruleId = None, ruleDesc = None):
        self.ruleId = ruleId
        self.ruleDesc = ruleDesc
        self.operator = operator
        self.phrase1 = phrase1
        self.phrase2 = phrase2
        self.l1Category = l1Category
        self.l2Category = l2Category
        self.returnType = "flag"
        #self.rexTemplate = None
        #self.rex = None
    
    def parse(self, docText):
        print "Error: Not implemented"
        return 0
    
    def evaluate(self, docText):
        print "Error: Not implemented"
        return 0
        

class AndRule(Rule):
    def __init__(self, phrase1, phrase2, l1Category, l2Category, returnType="flag", ruleId = None, ruleDesc = None):
        self.operator = "AND"
        #super(Rule, self).__init__()
        Rule.__init__(self, self.operator, phrase1, phrase2, l1Category, l2Category, returnType=returnType, ruleId = ruleId, ruleDesc=ruleDesc)
    
    def evaluate(self, docText):
        flag = False
        result = False
        flag = self.phrase1 in docText and self.phrase2 in docText
        if flag == True:
            result = "%s - %s" % (self.l1Category, self.l2Category)
        return flag
        
class OrRule(Rule):
    def __init__(self, phrase1, phrase2, l1Category, l2Category, returnType="flag", ruleId = None, ruleDesc = None):
        self.operator = "OR"
        Rule.__init__(self, self.operator, phrase1, phrase2, l1Category, l2Category, returnType=returnType, ruleId = ruleId, ruleDesc=ruleDesc)
        
    def evaluate(self, docText):
        flag = False
        result = False
        flag = self.phrase1 in docText or self.phrase2 in docText
        if flag == True:
            result = "%s - %s" % (self.l1Category, self.l2Category)
        return flag

class AdjRule(Rule):
    """
    rule with 'ADJ' unidirectional proximity operator (default distance - within 6 words)
    """
    def __init__(self, phrase1, phrase2, l1Category, l2Category, returnType="flag", ruleId = None, ruleDesc = None):
        self.operator = "ADJ"
        Rule.__init__(self, self.operator, phrase1, phrase2, l1Category, l2Category, returnType=returnType, ruleId = ruleId, ruleDesc=ruleDesc)       
        self.rexTemplate = "\bphrase1\W+(?:\w+\W+){1,6}?phrase2\b"
        #self.rex = "\b%s\W+(?:\w+\W+){1,6}?%s\b" % (phrase1, phrase2)
        self.rex = "%s\W+(?:\w+\W+){1,6}?%s" % (phrase1, phrase2)
        
        
    def evaluate(self, docText):
        flag = False
        result = False
        result = re.findall(self.rex, docText)
        flag = True if len(result) > 0 else False
        if flag == True:
            result = "%s - %s" % (self.l1Category, self.l2Category)
        return flag

class NearRule(Rule):
    """
    rule with 'NEAR' bidirectional proximity operator (default distance - within 6 words)
    """
    def __init__(self, phrase1, phrase2, l1Category, l2Category, returnType="flag", ruleId = None, ruleDesc = None, dist = 6):
        self.operator = "NEAR%d" % dist
        Rule.__init__(self, self.operator, phrase1, phrase2, l1Category, l2Category, returnType=returnType, ruleId = ruleId, ruleDesc=ruleDesc)
        self.rexTemplate = "\b(?:phrase1\W+(?:\w+\W+){1,6}?phrase2|phrase2\W+(?:\w+\W+){1,6}?phrase1)\b"
        #self.rex = "\b(?:%s\W+(?:\w+\W+){1,6}?%s|%s\W+(?:\w+\W+){1,6}?%s)\b" % (phrase1, phrase2, phrase2, phrase1)
        self.rex = "(?:%s\W+(?:\w+\W+){1,6}?%s|%s\W+(?:\w+\W+){1,6}?%s)" % (phrase1, phrase2, phrase2, phrase1)
    
    def evaluate(self, docText):
        flag = False
        result = False
        result = re.findall(self.rex, docText)
        flag = True if len(result) > 0 else False
        if flag == True:
            result = "%s - %s" % (self.l1Category, self.l2Category)
        return flag
        
#df['r1'] = df['lineText'].apply(r1.evaluate)

class RuleBase:
    """
    Document classification based on scikit-learn
    """
    def __init__(self):
        self.rulesDf = None
        self.rules = []
        #self.dataSet = None
        #self.ruleType = "AND"
        #self.rule = None
        #self.category = None
    
    def _readDf(self, csvFile, fileFormat="csv"):
        """
        Reads an input csv/tab file and returns a pandas DataFrame object
        """
        if fileFormat == "csv":
            df= pd.read_csv(csvFile, sep=",")
        elif fileFormat == "tab":
            df= pd.read_table(csvFile, sep="\t")
        else:
            print "File IO Error: Unsupported file format!"
        print '-------Reading DataFrame----------'
        print "Total number of records: %s" % len(df.index)
        print "Columns imported: "
        print df.columns        
        print "Preview of first few records: %s" % len(df.index)        
        #print df.head().to_string() #causes errors for some encodings
        print '----------------------------------'
        return df
    
    def loadRules(self, fileName = 'ruleBase.csv'):
        self.rulesDf = self._readDf(fileName, fileFormat='csv')
        print 'loading rule base'
        print '-------------------------------'
        for count, row in self.rulesDf.iterrows():
            row['phrase1'] = row['phrase1'].lower()
            row['phrase2'] = row['phrase2'].lower()
            if row['operator'] == 'AND':
                rule = AndRule(row['phrase1'], row['phrase2'], row['l1Category'], 
                               row['l2Category'], ruleId=row['ruleId'], ruleDesc="%s AND %s => %s - %s" %(row['phrase1'], row['phrase2'], row['l1Category'], row['l2Category'] ))
            elif row['operator'] == 'OR':
                rule = OrRule(row['phrase1'], row['phrase2'], row['l1Category'], 
                               row['l2Category'], ruleId=row['ruleId'], ruleDesc="%s OR %s => %s - %s" %(row['phrase1'], row['phrase2'], row['l1Category'], row['l2Category'] ))
            elif row['operator'] == 'ADJ':
                rule = AdjRule(row['phrase1'], row['phrase2'], row['l1Category'], 
                               row['l2Category'], ruleId=row['ruleId'],  ruleDesc="%s ADJ %s => %s - %s" %(row['phrase1'], row['phrase2'], row['l1Category'], row['l2Category'] ))
            elif row['operator'] == 'NEAR':
                rule = NearRule(row['phrase1'], row['phrase2'], row['l1Category'], 
                               row['l2Category'], ruleId=row['ruleId'], ruleDesc="%s NEAR %s => %s - %s" %(row['phrase1'], row['phrase2'], row['l1Category'], row['l2Category'] ))
            elif row['operator'][0:4] == 'NEAR':
                nearDist = int(re.search(r'NEAR(\d*)', row['operator']).groups(0)[0])
                rule = NearRule(row['phrase1'], row['phrase2'], row['l1Category'], 
                               row['l2Category'], ruleId=row['ruleId'], ruleDesc="%s NEAR%d %s => %s - %s" %(row['phrase1'], nearDist, row['phrase2'], row['l1Category'], row['l2Category'] ), dist = nearDist)
            else:
                print "Rule operator not supported for ruleId %s" % rule.ruleId
            self.rules.append(rule)
            #print "%s %s" % (rule.ruleId, rule.ruleDesc)
        
        print '-------------------------------'
        print "Done loading rule base"
        return 0
        
        
class RuleParser:
    """
    Document classification based on scikit-learn
    """
    def __init__(self):
        self.ruleSet = None
        self.dataSet = None
        self.ruleType = "AND"
        self.rule = None
        self.category = None
    
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
        print '-------Reading DataFrame----------'
        print "Total number of records: %s" % len(df.index)
        print "Columns imported: "
        print df.columns        
        print "Preview of first few records: %s" % len(df.index)        
        #print df.head().to_string() #causes errors for some encodings
        print '----------------------------------'
        return df
    
    def readLargeDf(self, csvFile, chunkSize=20, selectedCols = None, naValues = None, compressionFormat=None, fileFormat="csv"):
        if fileFormat == "csv":
            chunkIterator = pd.read_csv(csvFile, iterator=True, chunksize=chunkSize, na_values = naValues, compression=compressionFormat) #usecols = selectedCols
        elif fileFormat == "tab":
            chunkIterator = pd.read_csv(csvFile, iterator=True, chunksize=chunkSize, na_values = naValues, compression=compressionFormat) #usecols = selectedCols
        else:
            print "file format not supported"
        
        #nrows=nRows ->  returns df not iterator nRows = 1000, 
        #reader = pd.read_table('tmp.sv', sep='|', iterator=True)
        #reader.get_chunk(5)
        #dfSegment = chunkIterator.get_chunk() #returns df with default chunk size 
        return chunkIterator 
        
    def buildReport(self, rBase, colNames, df, resultFile = 'results.csv', ruleFile = 'ruleReport.csv'):
        #rb = RuleBase()
        rb = rBase;
        #df[colNames['lineText']] = df[colNames['lineText']].apply(str).apply(lambda x: codecs.decode(x, 'ascii', ignore = False))
        
        try:
            df[colNames['lineText']] = df[colNames['lineText']].apply(lambda x: codecs.decode(str(x), "ascii",'ignore'))
        except Exception as e:
            try:
                df[colNames['lineText']] = df[colNames['lineText']].apply(lambda x: codecs.decode(str(x), "latin-1",'ignore'))
            except Exception as e:
                try:
                    df[colNames['lineText']] = df[colNames['lineText']].apply(lambda x: codecs.decode(str(x), "utf-8",'ignore'))
                except Exception as e:
                    print 'Decode Exception'
        
        #rp = rParser;    
        #df = rp.readDf(csvFile, fileFormat='csv')
        #chunkIterator = rp.readLargeDf(csvFile , chunkSize=chunkSize, selectedCols = None, naValues = None, compressionFormat=None, fileFormat=fileFormat)
        #dfRules = df
        dfCols = df.columns.tolist()
        ruleCols = []
        categoryCols = []
        categories = []  
        #df['adj1'] = df[ 'ExitSurveyQ4'].apply(str).apply(adj1.evaluate)
        ##df['and1'] = df[ 'ExitSurveyQ4'].apply(str).apply(and1.evaluate)
        for r in rb.rules:
            ruleCols.append("rule_%s" % r.ruleId)
            # df[colNames['lineText']] = df[colNames['lineText']].apply(str).apply(str.lower)
            # df["rule_%s" % r.ruleId] = df[colNames['lineText']].apply(r.evaluate)
            
            df["rule_%s" % r.ruleId] = df[colNames['lineText']].apply(str).apply(str.lower).apply(r.evaluate)
            
            if "%s - %s" % (r.l1Category, r.l2Category) in df.columns:
                
                df["%s - %s" % (r.l1Category, r.l2Category)] = df.apply(lambda row: any([row["%s - %s" % (r.l1Category, r.l2Category)], row["rule_%s" % r.ruleId] ]), axis=1)
                #df.apply(lambda x: x["%s - %s" % (r.l1Category, r.l2Category)] or x["rule_%s" % r.ruleId], axis=1)
                #df["%s - %s" % (r.l1Category, r.l2Category)].apply(lambda x: x or df[ colNames['lineText']].apply(str).apply(r.evaluate)
            else:
                df["%s - %s" % (r.l1Category, r.l2Category)] = df[colNames['lineText']].apply(str).apply(r.evaluate)
                categoryCols.append("%s - %s" % (r.l1Category, r.l2Category))
            nonRuleCols = [x for x in df.columns.tolist() if x not in ruleCols]
        
        dfCats = df.ix[:, nonRuleCols]
        
        #Appending a column with multiple category labels
        
        for ix, row in dfCats.iterrows():
            rowDict = dict(row)
            categories.append(' ; '.join([k for k, v in rowDict.iteritems() if isinstance(v,bool) if v == True]))
        #print categories
        dfCats['categories'] = categories
       
        #dfRules = df.ix[:,  [colNames['sessionId'], colNames['lineText']] + ruleCols]
        try:
            dfCats.to_csv(resultFile, index=False)
        except Exception as e:
            print 'Exception:',e
        #dfRules.to_csv(ruleFile, index=False)
        return resultFile
    
    #def writeReportChunks(self, inFileName, rBase, nChunks = -1, chunkSize = 10000, fileFormat='csv', prefixResultFile = 'results', prefixRuleFile = 'ruleReport'):
    #    if nChunks == -1:
    #        self.writeReport(inFileName, rBase, fileFormat=fileFormat, resultFile = 'results.csv', ruleFile = 'ruleReport.csv')
    #    else:
    #        chunkIterator = rParser.readLargeDf(inFileName , chunkSize=chunkSize, selectedCols = None, naValues = None, compressionFormat=None, fileFormat=fileFormat)
    #        for idx, df in enumerate(chunkIterator):
    #            print idx
    #            if idx < nChunks:
    #                print "Processing chunk: %s " % str(idx)
    #                self.buildReport(rBase, df, resultFile = '%s_%s.%s' % (prefixResultFile, str(idx), fileFormat), ruleFile = '%s_%s.%s' % (prefixRuleFile, str(idx), fileFormat))
    #            else:
    #                break
    #    return 0
    
    def writeReportChunks(self, rParser ,inFileName,colNames,rBase,chunkSize = 10000, nChunks = 5 , prefixResultFile = 'results', prefixRuleFile = 'ruleReport'):
        #rParser = RuleParser()
        outputfiles = []
        chunkIterator = rParser.readLargeDf(inFileName , chunkSize=chunkSize, selectedCols = None, naValues = None, compressionFormat=None, fileFormat=fileFormat)
        for idx, df in enumerate(chunkIterator):
            if nChunks == -1 or idx < nChunks:
                print "Processing chunk: %s " % str(idx)
                outputfile = self.buildReport(rBase, colNames,df, resultFile = '%s_%s.csv' % (prefixResultFile, str(idx)), ruleFile = '%s_%s.csv' % (prefixRuleFile, str(idx)))
                outputfiles.append(outputfile)
            else:
                break
        return outputfiles
    
    def writeReport(self, inFileName, rBase, fileFormat='csv', resultFile = 'results.csv', ruleFile = 'ruleReport.csv'):
        df = self.readDf(inFileName, fileFormat=fileFormat)
        print "Writing report..." 
        self.buildReport(rBase, df, resultFile = resultFile, ruleFile = ruleFile)
        return 0
    
    def buildReport_api(self, doc,rBase):
        
        sa = Sentiment()
        rb = rBase;
        categories = []
        doc = str(doc).lower()
        for r in rb.rules:
            
            category = r.evaluate(doc)
            if category:
                
                categories.append(r.l1Category+"-"+r.l2Category)   #'polarity':sa.get_polarity(doc),'sentiment':sa.get_sentiment_class(doc,lowerBound = -0.75, upperBound=0.75),})
        categories = list(set(categories))
        
        results = {}
        results['categories'] = categories
        results['polarity'] = sa.get_polarity(doc)
        results['sentiment'] = sa.get_sentiment_class(doc,lowerBound = -0.75, upperBound = 0.75)
          
        return results
    
    
class SurveyAnalysis:
    def __init__(self):
        pass
    
    def run(self,doc,ruleBaseFile = None):
        
        rBase = RuleBase()
        rBase.loadRules(fileName = ruleBaseFile)
        rParser = RuleParser()
        result = rParser.buildReport_api(doc,rBase)
        
        return result


def run(inRuleFileName = None,inFileName = None, colNames = None, chunkSize = 10000, nChunks = -1, prefixResultFile = 'results' , prefixRuleFile = 'ruleReport', metricCol = 'sentimentClass_3Levels', outFolder = None):
    rBase = RuleBase()
    rBase.loadRules(fileName = inRuleFileName)
    rParser = RuleParser()
    outputFiles = rParser.writeReportChunks(rParser, inFileName,colNames,rBase,chunkSize = chunkSize, nChunks = nChunks, prefixResultFile = prefixResultFile, prefixRuleFile = prefixRuleFile)
    outputfilenames = {}
    sa = sentimentAnalyser()
    
    for outputfile in outputFiles:
        prefix = outputfile
        idx = prefix.rsplit('.',1)
        idx = idx[0].split('_')[-1]
        try:
            df = sa.readDF(outputfile)
            
            df['sentimentClass_7Levels'] = df[colNames['lineText']].apply(lambda x: sa.get_sentiment_class(x, lowerBound = -0.65, upperBound=0.65))
            df['sentimentClass_3Levels'] = df[colNames['lineText']].apply(lambda x: sa.get_sentiment_class1(x, lowerBound = -0.65, upperBound=0.65))
            df['sentimentPolarity'] = df[colNames['lineText']].apply(sa.get_polarity)
            df['sentimentSubjectivity'] = df[colNames['lineText']].apply(sa.get_subjectivity)
            df.to_csv(outputfile, index=False)
        except Exception as e:
            print 'Exception:',e
            #
        
        metricCol_output,imageSrc = viewResults_v3.results_sentimentClass(idx,outputfile,colNames[metricCol],outFolder)
        #os.chdir(baseDir)
        metricCol_output = outFolder+"/"+metricCol_output #os.path.join(os.getcwd()+"/"+ metricCol_output)
        
        outputfilenames[prefix] = [metricCol_output,imageSrc]
        return outputfilenames,metricCol_output,imageSrc,metricCol
    
    

def test():
    """
    test rules
    """
    txt1 = 'I have a payment related problem'
    txt2 = 'I have a problem related to my payment'
    txt3 = 'I have a billing related issue'
    corpus = [txt1, txt2, txt3]
    
    and1 = AndRule("payment", "problem", "payment", "none")
    adj1 = AdjRule("payment", "problem", "payment", "none")
    or1 = OrRule("payment", "bill", "payment", "none")
    near1 = NearRule("payment", "problem", "payment", "none")
    
    print and1.evaluate(txt1)
    print adj1.evaluate(txt1)
    print  or1.evaluate(txt1)
    print near1.evaluate(txt1)
    
    print map(and1.evaluate, corpus)
    return 0    




   

    
# baseDir = os.path.dirname(os.path.abspath(__file__))
# os.chdir(baseDir)
##print 'Current working directory: %s' % baseDir
#
##test()
##---------------------------------
##test()
##User Inputs
#colNames = mysettings.colNames
#inRuleFileName = mysettings.inRuleFileName
#inFileName =  mysettings.inFileName
fileFormat = "csv"
#resultsFile = mysettings.resultsFileName
#ruleFile = mysettings.ruleReportFileName
#
#chunkSize = mysettings.chunkSize
#nChunks = mysettings.nChunks
#prefixResultFile = mysettings.prefixResultFile
#prefixRuleFile = mysettings.prefixRuleFile
#delim = mysettings.delim
#
##---------------------------------    
#rBase = RuleBase()
#rBase.loadRules(fileName = inRuleFileName)
#rParser = RuleParser()
##---------------------------------
##rParser.writeReport(inFileName, rBase, fileFormat='csv', resultFile = 'results.csv', ruleFile = 'ruleReport.csv')
#rParser.writeReportChunks(inFileName, rBase, nChunks = nChunks, chunkSize = chunkSize, fileFormat='csv', prefixResultFile = 'results', prefixRuleFile = 'ruleReport')
#
