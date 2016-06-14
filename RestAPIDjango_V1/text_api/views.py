from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
import os
import json
import pandas as pd
import re

from SklCat import SklCat
from Sentiment import Sentiment
from SurveyAnalysisRules_v5 import SurveyAnalysis
# Create your views here.

projectDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
baseDir = os.path.dirname(os.path.abspath(__file__))

colNames = {'sessionId': 'sessionId',
            'lineNum': 'lineNum',
            'lineText': 'lineText',
            'lineBy': 'lineBy',
            'L2':'L2',
            'L3': 'L3',
            'issueLine':'issueLine',
            'Comment':'Comment'}

#lineByLevels
lineByLevels = {'system': 0, 
                'agent': 1,
                'customer': 2,}    

topNLinesPrimary = 5


def concatenateLinesWindowBySpeaker(df, colNames, lineByLevels, fromLineNum=0, toLineNum=-1, speaker=2, sep=' ||| '):
        #concatenateWindowOfLinesBySpeaker
        """
        Concatenates a window of lines (from, to) for a specified speaker in chats
        """
        concatSentences = {}
        L2 = {}
        L3 = {}
        sessionId = {}
        for idx, dfSegment in df.groupby(colNames['sessionId']):
            sessionId[idx] = idx  # Pushes the 'Session_ID' into the dict 'session'            
            dfFilteredSegment = dfSegment[dfSegment['lineBy'] == lineByLevels[speaker]]    #lineByLevels['customer']
            if dfFilteredSegment.empty:
                concatSentences[idx] = ''
                L2[idx] = ''
                L3[idx] = ''
            else:
                #print dfFilteredSegment.index
                if toLineNum < 0:
                    toLineNum = len(dfFilteredSegment.index)-1 #max(dfFilteredSegment.index)
                if fromLineNum < 0:
                    fromLineNum = 0 #min(dfFilteredSegment.index)
                if fromLineNum == toLineNum:
                    #print fromLineNum
                    concatSentences[idx] = dfFilteredSegment[colNames['lineText']][fromLineNum:fromLineNum+1]
                elif fromLineNum < toLineNum:
                    concatSentences[idx] = sep.join(dfFilteredSegment[colNames['lineText']][fromLineNum:toLineNum].tolist())  # Joins chatlines from line no. fromLineNum to toLineNum
                else:
                    print 'fromLineNum cannot be greater than toLineNum'
                    concatSentences[idx] = ''
                #print dfFilteredSegment['lineText'][1:4].tolist()
                # Filters 'prim_sec'= 1 indicating primary line
                #L2[idx] = dfFilteredSegment.ix[dfFilteredSegment[colNames['L2']].notnull(), colNames['L2']][:1][]
                L2[idx] = dfFilteredSegment[dfFilteredSegment[colNames['L2']].notnull()]['L2'].tolist() #ix[dfFilteredSegment[colNames['L2']].notnull()]            
                # Filters to rows where L2 is not empty and picks first such L2 value
                L3[idx] = dfFilteredSegment.ix[dfFilteredSegment[colNames['L3']].notnull(), colNames['L3']].tolist()
                #L3[idx] = df_seg_filtered[df_seg_filtered.L3.notnull()].L3[:1]  # Same for L3
                if len(L2[idx]) > 0:    
                    L2[idx] = L2[idx][0]
                else:
                    L2[idx] = ''
                    
                if len(L3[idx]) > 0:
                    L3[idx] = L3[idx][0]
                else:
                    L3[idx] = 'None'
                #print concatSentences[idx], L2[idx]
            
        transformedDf = pd.DataFrame.from_dict([sessionId, concatSentences, L2, L3])
        transformedDf = pd.DataFrame.transpose(transformedDf)
        transformedDf.columns = ["sessionId", "lineText", "L2", "L3"]
        #transformedDf = pd.DataFrame({'sessionId' : sessionId, 'lineText': concatSentences, 'L2': L2, 'L3': L3})
        return transformedDf


def sentimentCategories(sentimentList):
        
        sentimentList_categories = [i['category'] for i in sentimentList]
        sentiment_categories_str = ':'.join(sentimentList_categories)
        return sentiment_categories_str

def getNumberOfSwitches(sentimentList):
        
        sentiment_categories_str = sentimentCategories(sentimentList)
        totalSwitches = re.findall(r'negative:positive',sentiment_categories_str)
        
        return len(totalSwitches)

def overallSentiment(sentimentList):
        
        sentimentList_categories = [i['category'] for i in sentimentList]
        noOfPositives = sentimentList_categories.count('positive')
        noOfNegatives = sentimentList_categories.count('negative')
        noOfNeutrals = sentimentList_categories.count('neutral')
        
        overallSentiment = (noOfPositives+0.5*noOfNeutrals+0.001)/(noOfPositives+noOfNeutrals+5*noOfNegatives+0.002)
        return overallSentiment

def getPositivieSwitches(sentimentList):
        negativeToPositiveSwitches = getNumberOfSwitches(sentimentList)
        sentiment_categories_str = sentimentCategories(sentimentList)
        positiveToNegativeSwitches = len(re.findall(r'positive:negative',sentiment_categories_str))
        positiveSwitches = (negativeToPositiveSwitches+0.001)/(negativeToPositiveSwitches+ 3*positiveToNegativeSwitches+0.002)
        return positiveSwitches

def dfProbflag(row):
        print 'row',row
def text_api(request):
                
        text = str(request.GET['text'])
        usecase = str(request.GET['usecase'])
        modelId = request.GET['modelId']
        modelId = eval(modelId)
        clientId = str(request.GET['clientId'])
        params = str(request.GET['params'])
        
        modelMap_df = pd.read_csv(projectDir+"/config/modelMapping.csv")
        results = []
        records = eval(text)
        
        
        if modelId != None:
            usecaseMap = ''.join(modelMap_df.ix[modelMap_df['modelId'] == modelId ,'usecase'].tolist())
            modelName = modelMap_df.ix[modelMap_df['modelId'] == modelId ,'modelName'].tolist()
            
            
            if text == '' :
                results = {'Predicted Category':None,'Prediction Score':None}
                return HttpResponse(json.dumps(results))
            if usecase.lower() != usecaseMap.lower():
                return HttpResponse("<h1 style='color: red'>Please provide the correct usecase for the modelId:%s<h1>" %modelId)
            if usecase.lower() not in ['sentiment','survey','ltv']:
                df = pd.read_json(json.dumps(records))  # converting json string into dataframe        
                folderName = modelName
                folderName = baseDir+"/"+''.join(folderName)+"/out"
                
                try:
                        
                        skl = SklCat()
                        #df = df.ix[df['lineBy']==2,:]   #filtering customer lines         
                        df['chat_intent'] = df['lineText'].apply(lambda x:skl.run_text_vectorizeTransform([str(x)], folderName = folderName, vectorizerFile = 'FeatureTransformer.p', pickler = None)).apply(lambda x: skl.run_text_execute_api_bulkDoc(x, folderName = folderName, modelFile = 'Classifier.p', pickler=None, modelType = 'classification'))
                        dfResults = df.ix[:,['sessionId','chat_intent']]
                        results = dfResults.to_json(orient='records')
                        '''
                        for doc in records:
                                
                                text = doc['chat_text']
                            
                                transformedXTest = skl.run_text_vectorizeTransform([text], folderName = folderName, vectorizerFile = 'FeatureTransformer.p', pickler = None)    
                                y_pred,y_prob_max = skl.run_text_execute(transformedXTest, folderName = folderName, modelFile = 'Classifier.p', pickler=None, modelType = 'classification')
                                output = {'session_id':doc['session_id'],'chat_text':text ,'Predicted Category':list(y_pred),'Prediction Score':list(y_prob_max)}
                                results.append(output)
                        '''
                        return HttpResponse(json.dumps(results))
                except Exception as e:
                        
                        return HttpResponse("<h1 style='color: red'>Error Occured:%s</h1>" %e)
            elif usecase.lower() == 'sentiment':
                               
                
                sentimentPattern= modelName
                sentimentPattern = ''.join(sentimentPattern)
                records = eval(text)
                
                
                try:
                        
                        
                        emo = Sentiment()
                        
                        results = []
                        jsonResults = {}
                        for doc in records:
                                jsonTemp = {}
                                jsonTemp['sessionId'] = doc['sessionId']
                                df_chat_data = pd.read_json(json.dumps(doc['chat_data']))
                                
                                df_chat_data['sentiment'] = df_chat_data['lineText'].apply(lambda row: emo.run(row,modelType = sentimentPattern,args = [],kwargs={}))
                                df = df_chat_data.ix[:,['lineNum','sentiment']]
                                
                                jsonObj = df.to_json(orient ='records')
                                each_line = json.loads(json.dumps(jsonObj))
                                jsonTemp['each_line'] = each_line
                                chat_sentiment = {}
                                chat_sentiment['overall'] = overallSentiment(df_chat_data['sentiment'])
                                lastlines = 2
                                chat_sentiment['last_line_sentiment'] = df_chat_data['lineText'].tail(lastlines).apply(lambda row: emo.run(row,modelType = sentimentPattern,args = [],kwargs={})).to_json(orient='records')
                                chat_sentiment['switches_to_postive'] = getPositivieSwitches(df_chat_data['sentiment'].tolist())
                                jsonTemp['chat_sentiment'] = chat_sentiment
                                
                                results.append(jsonTemp)
                                
                        
                                
                                
                        '''
                        
                        for doc in records:
                            text  = doc['lineText']
                            output = emo.run(text, modelType = sentimentPattern, args = [], kwargs = {})
                            output['sessionId']= doc['sessionId']
                            output['lineText'] = text
                            results.append(output)
                        '''
                        return HttpResponse(json.dumps(results ,indent=5))
                except Exception as e:
                    return HttpResponse("<h1 style='color: red'>Error Occured:%s</h1>" %e)
            elif usecase.lower() == 'survey':
                
                ruleBaseFile = projectDir+"/config/ruleBase_v3.csv"
                try:
                        
                        sa = SurveyAnalysis()
                        results = sa.run(text,ruleBaseFile = ruleBaseFile)
                except Exception as e:
                        
                        return HttpResponse("<h1 style='color: red'>Error Occured:%s</h1>" %e)
            elif usecase.lower() == 'ltv':
                
                folderName = modelName
                folderName = baseDir+"/"+''.join(folderName)+"/out"
                skl = SklCat()
                results = []
                jsonResults = {}
                
                
                try:
                        
                        for doc in records:
                                
                                jsonTemp = {}
                                jsonTemp['sessionId'] = doc['sessionId']
                                df_chat_data = pd.read_json(json.dumps(doc['chat_data']))
                                if 'lineNum' not in df_chat_data.columns.tolist():
                                        df_chat_data['lineNum'] = pd.Series(range(1, df_chat_data.shape[0]+1))
                                df_chat_data = df_chat_data.ix[df_chat_data['lineBy'] == 1,:]           #filtering only agent lines
                                #ltv = df_chat_data['lineText'].apply(lambda x:skl.run_text_vectorizeTransform([str(x)], folderName = folderName, vectorizerFile = 'FeatureTransformer.p', pickler = None)).apply(lambda x: skl.run_text_execute_api_bulkDoc(x, folderName = folderName, modelFile = 'Classifier.p', pickler=None, modelType = 'classification',usecase='ltv'))
                                transformedXTest = skl.run_text_vectorizeTransform(df_chat_data['lineText'].tolist(), folderName = folderName, vectorizerFile = 'FeatureTransformer.p', pickler = None)
                                dfProb = skl.run_text_execute_api_bulkDoc(transformedXTest, folderName = folderName, modelFile = 'Classifier.p', pickler=None, modelType = 'classification',usecase='ltv')
                                
                                #dfTest = df_chat_data.ix[df_chat_data['ltv'].apply(lambda x : len(x)!=0)]
                                
                                
                                
                                if len(dfProb) != 0:
                                                                                
                                        dfProb['lineNum'] = pd.Series(range(1, dfProb.shape[0]+1))
                                        threshold = 0.2
                                        dfMerged = pd.merge(df_chat_data, dfProb, how='left', left_on='lineNum', right_on='lineNum',suffixes = ('_x','_y'))
                                        dfMerged = dfMerged.ix[:,['lineNum','y_pred','y_prob_max']]
                                        dfMerged['flag'] = dfMerged.apply(lambda row: row['y_pred'] == 1  and row['y_prob_max'] > threshold,axis = 1) 
                                        ltv_agg = 1 if dfMerged['flag'].any() else 0
                                        jsonTemp['chat_ltv'] = {'category':ltv_agg}
                                        jsonObj = dfMerged.to_json(orient = 'records')
                                        
                                        each_line = json.loads(json.dumps(jsonObj))
                                
                                jsonTemp['each_line'] = each_line
                                
                                results.append(jsonTemp)
                        
                        
                        return HttpResponse(json.dumps(results))
                except Exception as e:
                        
                        return HttpResponse("<h1 style='color: red'>Error Occured:%s</h1>" %e)
            
            
            else:
                return HttpResponse("<h1 style='color: red'>Please provide the proper usecase..</h1>")
            
            #return HttpResponse(json.dumps(results))
        else:
            return HttpResponse("<h1>Please provide the modelId in URL</h1>")
    








def text_api_doc(request):
    
    text = str(request.GET['text'])
    usecase = str(request.GET['usecase'])
    modelId = request.GET['modelId']
    modelId = eval(modelId)
    clientId = str(request.GET['clientId'])
    params = str(request.GET['params'])
    
    modelMap_df = pd.read_csv(projectDir+"/config/modelMapping.csv")
    
    if modelId != None:
        usecaseMap = ''.join(modelMap_df.ix[modelMap_df['modelId'] == modelId ,'usecase'].tolist())
        modelName = modelMap_df.ix[modelMap_df['modelId'] == modelId ,'modelName'].tolist()
        
        if text == '' :
            results = {'Predicted Category':None,'Prediction Score':None}
            return HttpResponse(json.dumps(results))
        if usecase.lower() != usecaseMap.lower():
            return HttpResponse("<h1 style='color: red'>Please provide the correct usecase for the modelId:%s<h1>" %modelId)
        if usecase.lower() not in ['sentiment','survey']:
            folderName = modelName
            folderName = baseDir+"/"+''.join(folderName)+"/out"
            try:
                skl = SklCat()    
                transformedXTest = skl.run_text_vectorizeTransform([text], folderName = folderName, vectorizerFile = 'FeatureTransformer.p', pickler = None)    
                y_pred,y_prob_max = skl.run_text_execute_api_doc(transformedXTest, folderName = folderName, modelFile = 'Classifier.p', pickler=None, modelType = 'classification')
                results = {'Predicted Category':list(y_pred),'Prediction Score':list(y_prob_max)}
            except Exception as e:
                return HttpResponse("<h1 style='color: red'>Error Occured:%s</h1>" %e)
        elif usecase.lower() == 'sentiment':
            sentimentPattern= modelName
            sentimentPattern = ''.join(sentimentPattern)
            try:            
                emo = Sentiment()
                results = emo.run(text, modelType = sentimentPattern, args = [], kwargs = {})
            except Exception as e:
                return HttpResponse("<h1 style='color: red'>Error Occured:%s</h1>" %e)
        elif usecase.lower() == 'survey':
            ruleBaseFile = projectDir+"/config/ruleBase_v3.csv"
            try:
                sa = SurveyAnalysis()
                results = sa.run(text,ruleBaseFile = ruleBaseFile)
            except Exception as e:
                return HttpResponse("<h1 style='color: red'>Error Occured:%s</h1>" %e)
        else:
            return HttpResponse("<h1 style='color: red'>Please provide the proper usecase..</h1>")
        
        return HttpResponse(json.dumps(results))
    else:
        return HttpResponse("<h1>Please provide the modelId in URL</h1>")

