package EuroCC

import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.PCA
import org.apache.spark.sql.functions._

object PCA {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local[*]")
      .appName("PCA Example")
      .getOrCreate()

    var entire_data = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .format("com.databricks.spark.csv")
      .load("data/Facebook_Shops.csv")

    //Pred-procesiranje
    entire_data = entire_data.drop("Column1")
    entire_data = entire_data.drop("Column2")
    entire_data = entire_data.drop("Column3")
    entire_data = entire_data.drop("Column4")
    entire_data = entire_data.drop("status_id")
    entire_data = entire_data.drop("status_published")

    //entire_data.printSchema()
    //entire_data.show(10)

    //filling missing values
    //var avgnum_shares = entire_data.select(avg("num_shares")).first()(0).asInstanceOf[Int]
    //entire_data = entire_data.na.fill(avgnum_shares, Seq("num_shares"))

    val numericIndexer = new StringIndexer()
      .setInputCol("status_type")
      .setOutputCol("status_type_numeric")
    var newDF = numericIndexer.fit(entire_data).transform(entire_data)
    newDF = newDF.drop("status_type")
    //newDF.show(10)
    val featuresCol = newDF.columns
    val assembler = new VectorAssembler()
      .setInputCols(featuresCol)
      .setOutputCol("features")

    //Prikazi Input Algorithms Data
    val assemblerDF = assembler.transform(newDF)
    assemblerDF.show(10)

    //Povik za PCA
    val PCA = new PCA()
      .setInputCol("features")
      .setOutputCol("PCANewFeatures")
      .setK(5)
      .fit(assemblerDF)

    val PCAResults = PCA.transform(assemblerDF).select("features", "PCANewFeatures")
    PCAResults.show(10)


  }
}
