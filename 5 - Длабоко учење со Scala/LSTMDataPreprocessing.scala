package EuroCC

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DataTypes

object LSTMDataPreprocessing {
  def main(args: Array [String]): Unit =
  {
    val spark = SparkSession
      .builder()
      .master("local[*]")
      .appName("Deep Learning Example")
      .config("spark.sql.crossJoin.enabled","true")
      .getOrCreate()

    var trainingData = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .format("com.databricks.spark.csv")
      .load("data/ForestTypes_training.csv")

    var testingData = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .format("com.databricks.spark.csv")
      .load("data/ForestTypes_testing.csv")

    //trainingData.show(10)
    //testingData.show(10)

    var indexer = new StringIndexer()
      .setInputCol("class")
      .setOutputCol("label")
    var indexedDF_train = indexer.fit(trainingData).transform(trainingData).select(col("label").cast(DataTypes.IntegerType))
    var indexedDF_test = indexer.fit(testingData).transform(testingData).select(col("label").cast(DataTypes.IntegerType))

    var DF_training = trainingData.join(indexedDF_train)
    var DF_testing = testingData.join(indexedDF_test)
    DF_training = DF_training.drop("class")
    DF_testing = DF_testing.drop("class")
    DF_training.show(10)
    DF_testing.show(10)

    DF_training.coalesce(1).write
      .format("com.databricks.spark.csv")
      .option("header", "false")
      .option("delimiter",",")
      .save("data/ForestTypes_train.csv")

    DF_testing.coalesce(1).write
      .format("com.databricks.spark.csv")
      .option("header", "false")
      .option("delimiter",",")
      .save("data/ForestTypes_test.csv")
  }
}
