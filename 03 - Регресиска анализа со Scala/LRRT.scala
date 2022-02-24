package EuroCC

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{LinearRegression}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.SparkSession

object LRRT {
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

    val lr = new LinearRegression()
      .setLabelCol("label")
      .setFeaturesCol("features")

    val pipeline = new Pipeline()
      .setStages(Array(assembler,lr))

    var paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, 0.001 :: 0.01 :: 0.1 :: Nil)
      .addGrid(lr.tol, 0.1 :: 0.01 :: Nil)
      .addGrid(lr.maxIter, Array (10,20,30,50))
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

    //testing
    val modelCVResults = cvModel.transform(testingData)
    val predictionAndLabels = modelCVResults
      .select("prediction", "label")
      .rdd.map(x => (x(0).asInstanceOf[Double], x(1)
      .asInstanceOf[Double]))
    val LRRegressionMetrics = new RegressionMetrics(predictionAndLabels)

    val MSE_metrika = LRRegressionMetrics.meanSquaredError
    val RMSE_metrika = LRRegressionMetrics.rootMeanSquaredError
    val Rsquared_metrika = LRRegressionMetrics.r2
    val MAE_metrika = LRRegressionMetrics.meanAbsoluteError
    val Explained_Variance_metrika = LRRegressionMetrics.explainedVariance

    println ("MSE = " +MSE_metrika + "\n RMSE = " + RMSE_metrika + "\n R^2 = " + Rsquared_metrika + "\n MAE = " + MAE_metrika + "\n Exp_Var = "+ Explained_Variance_metrika)
  }
}






