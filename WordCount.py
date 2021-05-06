from pyspark import SparkContext
from pyspark import SparkConf

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
    sparkConf = SparkConf().setAppName("WordCounts").set("spark.ui.showConsoleProgress", "false")
              
    sc = SparkContext(conf = sparkConf)
    print("master="+sc.master)
    SetLogger(sc)
    SetPath(sc)
    return (sc)
    

if __name__ == "__main__":
    print("RunWordCount")
    sc=CreateSparkContext()
 
    print("Read file...")
    textFile = sc.textFile(Path+"data/README.md")
    print("Total: "+str(textFile.count())+"rows")
     
    countsRDD = textFile.flatMap(lambda line: line.split(' ')).map(lambda x: (x, 1)).reduceByKey(lambda x,y :x+y)
                  
    print("Total word: "+str(countsRDD.count()))                  
    print("Save file...")
    try:
        countsRDD.saveAsTextFile(Path+ "data/output")
        
    except Exception as e:
        print("Folder exist, please delete first.")
    sc.stop()
