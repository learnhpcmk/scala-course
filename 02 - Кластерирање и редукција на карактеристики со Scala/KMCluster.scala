package EuroCC

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object KMCluster {
  def main(args: Array[String]): Unit =
  {
   val spark = SparkSession
     .builder()
     .master("local[*]")
     .appName("Clustering Example")
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
    //newDF.show(10)
    val featuresCol = newDF.columns
    val assembler = new VectorAssembler()
      .setInputCols(featuresCol)
      .setOutputCol("features")

    //Prikazi Input Algorithms Data
     val assemblerDF = assembler.transform(newDF)
    //assemblerDF.show(10)

    //povik na klastering algritamot
    var km = new KMeans()
      .setK(4)
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setSeed(123L)

    val KMModel = km.fit(assemblerDF)

    val KMClusters = KMModel.transform(assemblerDF)
    KMClusters.show()

    KMClusters.groupBy("prediction").count().show()

    //Evaluacija na Silhouette score
    var evaluator = new ClusteringEvaluator()
    val SilScore = evaluator.evaluate(KMClusters)
    println("Silhouette Score e = " + SilScore)

    //Prikazi Claster Centrite
    println("Claster Centrite se ")
    KMModel.clusterCenters.foreach(println)

    var maxSilScore = new Array[Double](10)
    //Evaluate
    for (k <- 2 to 10)
      {
        val SilKModel = new KMeans().setK(k).setSeed(123L)
        val KMModelEval = SilKModel.fit(assemblerDF)
        val estimatedDF = KMModelEval.transform(assemblerDF)
        val SilouetteScoreEval = evaluator.evaluate(estimatedDF)
          maxSilScore(k-2) = SilouetteScoreEval
      }
      for (i <- 0 to (maxSilScore.length - 2))
        {
          val t = i + 2
          println("Vrednosta na SilScore = " + maxSilScore(i) + "e dobiena za k = " + t )
        }


  }
}
