# -*- coding: UTF-8 -*-
from pyspark.mllib.recommendation import ALS
from pyspark import SparkConf, SparkContext

def SetLogger( sc ):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
    logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )
    logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)    

def SetPath(sc):
    global Path
    if sc.master[0:5]=="local" :
        Path="file:/home/hduser/pyspark/PythonProject/"
    else:   
        Path="hdfs://master:9000/user/hduser/"

  
def CreateSparkContext():
    sparkConf = SparkConf().setAppName("RecommendTrain").set("spark.ui.showConsoleProgress", "false") 
    sc = SparkContext(conf = sparkConf)
    print ("master="+sc.master)    
    SetLogger(sc)
    SetPath(sc)
    return (sc)
    
  
def PrepareData(sc): 
    #----------------------user rate data-------------
    print("read user rate...")
    rawUserData = sc.textFile(Path+"data/u.data")
    rawRatings = rawUserData.map(lambda line: line.split("\t")[:3] )
    ratingsRDD = rawRatings.map(lambda x: (x[0],x[1],x[2]))
    
    #-------------------------------------------------
    numRatings = ratingsRDD.count()
    numUsers = ratingsRDD.map(lambda x: x[0] ).distinct().count()
    numMovies = ratingsRDD.map(lambda x: x[1]).distinct().count() 
    print("Total：ratings: " + str(numRatings) +    
             " User:" + str(numUsers) +  
             " Movie:" +    str(numMovies))
    return(ratingsRDD)

def SaveModel(sc): 
    try:        
        model.save(sc,Path+"ALSmodel")
        print("Model has been saved.")
    except Exception :
        print "Model exist, delete first."        
    
if __name__ == "__main__":
    sc=CreateSparkContext()
    print("==========Data Prepare===========")
    ratingsRDD = PrepareData(sc)
    print("==========Train===============")
    print("Start ALS train");
    model = ALS.train(ratingsRDD, 5, 20, 0.1)
    print("========== Save Model============")
    SaveModel(sc)

