from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
import os
import json
import pandas as pd

from SklCat import SklCat
from Sentiment import Sentiment
from SurveyAnalysisRules_v5 import SurveyAnalysis
# Create your views here.

projectDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
baseDir = os.path.dirname(os.path.abspath(__file__))


def text_api(request):
    
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
                y_pred,y_prob_max = skl.run_text_execute(transformedXTest, folderName = folderName, modelFile = 'Classifier.p', pickler=None, modelType = 'classification')
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


