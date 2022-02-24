package EuroCC

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object RFRT {
  def main(args: Array[String]): Unit =
  {
    val spark = SparkSession
      .builder()
      .master("local[*]")
      .appName("Regression Example")
      .getOrCreate()

    var entire_data = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .format("com.databricks.spark.csv")
      .load("data/energydata.csv")

    //entire_data.printSchema()
    //entire_data.show(10)

    entire_data = entire_data.drop("date")
    entire_data = entire_data.drop("rv1")
    entire_data = entire_data.drop("rv2")
    entire_data = entire_data.withColumnRenamed("Appliances","label")

    //var avgT1 = entire_data.select(mean("T1")).first()(0).asInstanceOf[Double]
    //entire_data = entire_data.na.fill(avgT1,Seq("T1"))

    val seed = 123L
    val split_data = entire_data.randomSplit(Array(0.8, 0.2),seed)
    val (trainingData, testingData) = (split_data(0), split_data(1))
    trainingData.cache()
    testingData.cache()

    var featuresCol = entire_data.columns.drop(1)
    var assembler = new VectorAssembler()
      .setInputCols(featuresCol)
      .setOutputCol("features")

    val rf = new RandomForestRegressor()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setSeed(1234L)

    val pipeline = new Pipeline()
      .setStages(Array(assembler,rf))

    var paramGrid = new ParamGridBuilder()
      .addGrid(rf.maxBins, 5 :: 10 :: Nil)
      .addGrid(rf.maxDepth, 5 :: 10 :: Nil)
      .build()

    val crossval = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator())
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10)

    val cvModel = crossval.fit(trainingData)

    val bestModel = cvModel.bestModel
    println("The Best Optimal Parameters: \n _________________")
    println(bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel].stages.last.extractParamMap)
    println("Features Importance vector: \n __________________")
    val forestModel = bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel].stages.last.asInstanceOf[RandomForestRegressionModel]
    forestModel.featureImportances.toArray.zip(featuresCol).sorted.reverse.foreach(println)

    //testing
    val modelCVResults = cvModel.transform(testingData)
    val predictionAndLabels = modelCVResults
      .select("prediction", "label")
      .rdd.map(x => (x(0).asInstanceOf[Double], x(1)
      .asInstanceOf[Double]))
    val RFRMetrics = new RegressionMetrics(predictionAndLabels)

    val MSE_metrika = RFRMetrics.meanSquaredError
    val RMSE_metrika = RFRMetrics.rootMeanSquaredError
    val Rsquared_metrika = RFRMetrics.r2
    val MAE_metrika = RFRMetrics.meanAbsoluteError
    val Explained_Variance_metrika = RFRMetrics.explainedVariance

    println ("MSE = " +MSE_metrika + "\n RMSE = " + RMSE_metrika + "\n R^2 = " + Rsquared_metrika + "\n MAE = " + MAE_metrika + "\n Exp_Var = "+ Explained_Variance_metrika)

  }
}





