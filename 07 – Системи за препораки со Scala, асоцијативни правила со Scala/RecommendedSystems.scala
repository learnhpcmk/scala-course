package EuroCC

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.SparkSession

object RecommendedSystems {
  def main(args: Array[String]): Unit =
  {
    var spark = SparkSession
      .builder()
      .master("local[*]")
      .appName("Recommended System Example")
      .getOrCreate()

    var entire_data = spark.read
      .option("header","true")
      .option("inferSchema","true")
      .format("com.databricks.spark.csv")
      .load("data/ml-latest-small.csv")

      entire_data.cache()
      entire_data.printSchema()
      entire_data.show(10)

    val Array(trainingData, testingData) = entire_data.randomSplit(Array(0.8, 0.2), 123L)

    val als = new ALS()
      .setMaxIter(5)
      .setRegParam(1.0)
      .setBlockSize(10)
      .setUserCol("userId")
      .setItemCol("movieId")
      .setRatingCol("rating")
    val ALSModel = als.fit(trainingData)

    ALSModel.setColdStartStrategy("drop")
    val ALSTestModel = ALSModel.transform(testingData)

    val predviduvanja = new RegressionEvaluator()
      .setMetricName("mse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")
    val mse = predviduvanja.evaluate(ALSTestModel)
    println("MSE vrednosta e = " + mse)

    //Preporaki za TOP10 filmovi za sekoj korisnik
    val TOP10UserRecommendations = ALSModel.recommendForAllUsers(10)
    TOP10UserRecommendations.show()
    //Preporaki za TOP10 korisnici za sekoj film
    val TOP10MovieRecommendations = ALSModel.recommendForAllItems(10)
    TOP10MovieRecommendations.show()

    //Preporaki za TOP10 filmovi za specificen broj na korisnici
    val users = entire_data.select(als.getUserCol).distinct().limit(5)
    val NumberUsersTOP10UserRecommendations = ALSModel.recommendForUserSubset(users, 10)
    NumberUsersTOP10UserRecommendations.show()
    //Prepooraki za TOP10 korisnici za specificen broj na filmovi
    val movies = entire_data.select(als.getItemCol).distinct().limit(5)
    val NumberMoviesTOP10MovieRecommendations = ALSModel.recommendForItemSubset(movies, 10)
    NumberMoviesTOP10MovieRecommendations.show()
    
  }
}
