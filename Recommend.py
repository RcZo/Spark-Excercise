import sys
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import  MatrixFactorizationModel
  
def CreateSparkContext():
    sparkConf = SparkConf().setAppName("Recommend").set("spark.ui.showConsoleProgress", "false")               
    sc = SparkContext(conf = sparkConf)
    print("master="+sc.master)
    SetLogger(sc)
    SetPath(sc)
    return (sc)

def SetPath(sc):
    global Path
    if sc.master[0:5]=="local" :
        Path="file:/home/hduser/pyspark/PythonProject/"
    else:   
        Path="hdfs://master:9000/user/hduser/"

def SetLogger( sc ):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
    logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )
    logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)
    
def PrepareData(sc): 
    print("Read movie ID & Name")
    itemRDD = sc.textFile(Path+"data/u.item") 
    movieTitle= itemRDD.map( lambda line : line.split("|")).map(lambda a: (float(a[0]),a[1])).collectAsMap()                          
    return(movieTitle)

def RecommendMovies(model, movieTitle, inputUserID): 
    RecommendMovie = model.recommendProducts(inputUserID, 10) 
    print("For user" + str(inputUserID) + "recommend:")
    for rmd in RecommendMovie:
        print("For user {0} recommend {1} socre {2}".format( rmd[0],movieTitle[rmd[1]],rmd[2]))

def RecommendUsers(model, movieTitle, inputMovieID) :
    RecommendUser = model.recommendUsers(inputMovieID, 10) 
    print "For movie {0} movie name {1} recommend user:". \
           format( inputMovieID,movieTitle[inputMovieID])
    for rmd in RecommendUser:
        print  "For user {0}  score {1}".format( rmd[0],rmd[2])


def loadModel(sc):
    try:        
        model = MatrixFactorizationModel.load(sc, Path+"ALSmodel")
        print("load ALSModel")
    except Exception:
        print("Can not find model, please train the model first.")
    return model 



def Recommend(model):
    if sys.argv[1]=="--U":
        RecommendMovies(model, movieTitle,int(sys.argv[2]))
    if sys.argv[1]=="--M": 
        RecommendUsers(model, movieTitle,int(sys.argv[2]))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("input 2 parameters")
        exit(-1)
    sc=CreateSparkContext()
    print("==========PrepareData===============")
    (movieTitle) = PrepareData(sc)
    print("==========LoadModel===============")
    model=loadModel(sc)
    print("==========Recommend===============")
    Recommend(model)

    

