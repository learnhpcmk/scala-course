package EuroCC

import org.apache.spark.ml.feature.{ChiSqSelector, StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object ChiSqSelector {
  def main(args: Array[String]): Unit =
  {
    val spark = SparkSession
      .builder()
      .master("local[*]")
      .appName("Feature Selection Example")
      .getOrCreate()

    var entire_data = spark.read
      .option("header","true")
      .option("inferSchema","true")
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
    newDF = newDF.withColumnRenamed("status_type_numeric","label")
    newDF.show(10)
    val featuresCol = newDF.columns.drop(-1)
    val assembler = new VectorAssembler()
      .setInputCols(featuresCol)
      .setOutputCol("features")

    //Prikazi Input Algorithms Data
    val assemblerDF = assembler.transform(newDF)
    assemblerDF.show(10)

    //Povik na algoritamot za selekcija na karakteristiki
    var FeSelect = new ChiSqSelector()
      .setNumTopFeatures(3)
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setOutputCol("SelectedFeatures")

    val ChiSqResults = FeSelect.fit(assemblerDF).transform(assemblerDF)
    ChiSqResults.show(10)


  }
}
