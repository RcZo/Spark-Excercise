# -*- coding: UTF-8 -*-
from math import sqrt
from operator import add
from time import time
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.mllib.recommendation import ALS
from pyspark import SparkConf, SparkContext


#import logging

def SetLogger( sc ):
    logger = sc._jvm.org.apache.log4j
    sc.setLogLevel("FATAL")
    logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
    logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )
    logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)    

def SetPath(sc):
    global Path
    if sc.master[0:5]=="local" :
        Path="file:/home/hduser/pyspark/PythonProject/"
    else:   
        Path="hdfs://master:9000/user/hduser/"

    
def PrepareData(sc): 
    #-----------------------------------
    rawUserData = sc.textFile(Path+"data/u.data")
    print "Total row: "+str(rawUserData.count())
    rawRatings = rawUserData.map(lambda line: line.split("\t")[:3] )
    ratingsRDD = rawRatings.map(lambda fields: (fields[0],fields[1],fields[2]))
    #----------------------random split 3 part-------------
    trainData, validationData, testRDD =ratingsRDD.randomSplit([8, 1, 1], seed=0L)
    print(" trainData:" + str(trainData.count()) +  
           " validationData:" + str(validationData.count()) + 
           " testData:" + str(testRDD.count()))
    return(trainData, validationData, testRDD)

def computeRMSE(alsmodel, RatingRDD):
    n=RatingRDD.count()
    predictRDD= alsmodel.predictAll(RatingRDD.map(lambda x: (x[0], x[1])))
    predictionsAndRatings = predictRDD.map(lambda x: ((x[0], x[1]), x[2])).join(RatingRDD.map(lambda x: ((float(x[0]), float(x[1])), float(x[2]))) ).values()
        
    return sqrt(predictionsAndRatings.map(lambda x: (x[0] - x[1]) ** 2).reduce(add)  / float(n))

         

def trainModel(trainData, validationData, rank, iterations, lambdaParm):
    startTime = time()
    model = ALS.train(trainData, rank, iterations, lambdaParm)
    Rmse = computeRMSE(model, validationData)
    duration = time() - startTime

    print("Parameters：" +"rank="+str(rank)+"lambda="+str(lambdaParm) +"iterations="+ str(iterations)  +"time="+str(duration) + "RMSE " +str(Rmse))
                       
    return (Rmse,duration, rank, iterations, lambdaParm,model)


def evalParameter(trainData, validationData, evaparm,
                  rankList, numIterationsList, lambdaList):
    metrics = [trainModel(trainData, validationData,  rank,numIter,  lambas ) 
                       for rank in rankList 
                       for numIter in numIterationsList  
                       for lambas in lambdaList ]
    if evaparm=="rank":
        IndexList=rankList[:]
    elif evaparm=="numIterations":
        IndexList=numIterationsList[:]
    elif evaparm=="lambda":
        IndexList=lambdaList[:]
    df = pd.DataFrame(metrics,index=IndexList,columns=
         ['RMSE', 'duration' , 'rank', 'iterations', 'lambdaParm','model'])
    showchart(df,evaparm,'RMSE','duration',0.8,5)

def showchart(df,evalparm ,barData,lineData,yMin,yMax):
    ax = df[barData].plot(kind='bar', title =evalparm,figsize=(10,6),legend=True, fontsize=12)
    ax.set_xlabel(evalparm,fontsize=12)
    ax.set_ylim([yMin,yMax])
    ax.set_ylabel(barData,fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(df[[lineData ]].values, linestyle='-', marker='o', linewidth=2.0,color='r')
    plt.show()
    
def evalAllParameter(trainData, validationData, rankList, numIterationsList, lambdaList):    
    metrics = [trainModel(trainData, validationData,  rank,numIter,  lambas  )  
                      for rank in rankList for numIter in numIterationsList  for lambas in lambdaList ]
    Smetrics = sorted(metrics, key=lambda k: k[0])
    print 'Best:'
    bestParameter=Smetrics[0]
    print bestParameter
       
    print("Best parameters：rank:" + str(bestParameter[2]) + "  ,numIterations:" + str(bestParameter[3]) + "  ,lambda:" + str(bestParameter[4])+ "  ,RMSE = " + str(bestParameter[0]))
    
    return bestParameter[5]
     

def  parametersTunning(trainData, validationData):

    print("----- Evaluation rank---------")
    evalParameter(trainData, validationData,"rank", 
            rankList=[5,10,15,20,50,100], 
            numIterationsList=[10],        
            lambdaList=[1.0 ])      

    print("----- Evaluation numIterations---------")
    evalParameter(trainData, validationData,"numIterations", 
            rankList=[8],                     
            numIterationsList=[5,10,15,20,25],     
            lambdaList=[1.0])       
    
    print("----- Evaluation lambda---------")
    evalParameter(trainData, validationData,"lambda", 
            rankList=[8],                 
            numIterationsList=[10],     
            lambdaList=[0.05,0.1,1.0,5.0,10.0 ])   

    print("-----Best parameter set---------")   
    bestModel=evalAllParameter(
         trainData, validationData,
         rankList=[5,10,15,20],
         numIterationsList=[5, 10, 15, 20,30],
         lambdaList=[0.05,0.1,1.0,5.0 ])


    return bestModel

def SaveModel(model,sc): 
    try:        
        model.save(sc,Path+"ALSmodel")
        print("Model has been saved at ALSmodel folder.")
    except Exception :
        print "Model exist, delete first."      

def CreateSparkContext():
    sparkConf = SparkConf().setAppName("RecommendTrainEval").set("spark.ui.showConsoleProgress", "false") 
    sc = SparkContext(conf = sparkConf)
    print ("master="+sc.master)    
    SetLogger(sc)
    SetPath(sc)
    return (sc)
          

if __name__ == "__main__":
    sc=CreateSparkContext()
    print("==========Data Prepare===============")
    (trainData, validationData, testRDD)=PrepareData(sc)
    trainData.persist() ;    validationData.persist();    testRDD.persist()
    print("========== Train Eval===============")
    bestModel = parametersTunning(trainData, validationData)
    print("========== Test===============")
    testRmse = computeRMSE(bestModel, testRDD)
    print("test best model," + " RMSE = " + str(testRmse))
    print("========== Save Model========== ==")
    SaveModel(bestModel ,sc)
    print("saved")
    trainData.unpersist(); validationData.unpersist(); testRDD.unpersist()
    print("persisted")
    
