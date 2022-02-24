package EuroCC

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object RFCT {
  def main(args: Array[String]): Unit =
  {
    val spark = SparkSession
      .builder()
      .master("local[*]")
      .appName("Classification Example")
      .getOrCreate()

    var trainingData = spark.read
      .option("header","true")
      .option("inferSchema","true")
      .format("com.databricks.spark.csv")
      .load("data/ForestTypes_training.csv")

    var testingData = spark.read
      .option("header","true")
      .option("inferSchema","true")
      .format("com.databricks.spark.csv")
      .load("data/ForestTypes_testing.csv")

    trainingData.cache()
    testingData.cache()

    /*
    println("Ova prikaz na podatocnoto mnozestvo za treniranje na modelot")
    trainingData.printSchema()
    trainingData.show(10)
    println("Ova e prikaza na podatocnoto mnozestvo za evaluacija na modelot")
    testingData.printSchema()
    testingData.show(10)

    trainingData = trainingData.drop("b1")
    var b1AVG = trainingData.select(mean("b1")).first()(0).asInstanceOf[Double]
    trainingData = trainingData.na.fill(b1AVG, Seq("b1"))
    trainingData = trainingData.na.fill("S", Seq("class"))
    */

    val labelIndexer = new StringIndexer()
      .setInputCol("class")
      .setOutputCol("label")
    val featuresCol = trainingData.columns.drop(1)
    val assembler = new VectorAssembler()
      .setInputCols(featuresCol)
      .setOutputCol("features")

    val lable_col = labelIndexer.fit(trainingData).transform(trainingData).select("label")
    var assembledDF = assembler.transform(trainingData).select("features", "class")
    assembledDF = assembledDF.drop("class")
    assembledDF = assembledDF.join(lable_col)
    assembledDF.show(10)

    val rf = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setSeed(12345L)

    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer,assembler, rf))

    var paramGrid = new ParamGridBuilder()
      .addGrid(rf.impurity, "gini" :: Nil)
      .addGrid(rf.maxBins, 5 :: Nil)
      .addGrid(rf.maxDepth, 5 :: Nil)
      .build()

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")

    val crossval = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10)

    //val seed = 123L
    //val split = trainingData.randomSplit(Array(0.8, 0.2),seed)
    //val (trainingData,testingData) = (split(0), split(1))

    val RFModel = crossval.fit(trainingData)

    val OptimalenModel = RFModel.bestModel
    println("Optimalniot Model i negovite parametri se : \n -------------------")
    println(OptimalenModel.asInstanceOf[org.apache.spark.ml.PipelineModel].stages.last.extractParamMap())

    val forestModel = OptimalenModel.asInstanceOf[org.apache.spark.ml.PipelineModel].stages.last.asInstanceOf[RandomForestClassificationModel]
    println("Feature importance vector ->")
    forestModel.featureImportances.toArray.zip(featuresCol).sorted.reverse.foreach(println)

    //Evaluacija na modelot
    val RFTestModel = RFModel.transform(testingData)
    val EvalMetricResults = RFTestModel.select("label", "prediction", "probability")
    val EvalMetricResultsDF = EvalMetricResults.withColumnRenamed("prediction", "Predicted_Label")
    EvalMetricResultsDF.show(10)

    val predictionAndLabels = RFTestModel
      .select("prediction", "label")
      .rdd.map(x => (x(0).asInstanceOf[Double], x(1)
      .asInstanceOf[Double]))

    val EvalMetrics = new MulticlassMetrics(predictionAndLabels)

    val ModelAccuracy = EvalMetrics.accuracy
    println("Evaluation statistics")
    println(s"Classification Accuracy = $ModelAccuracy")
    println("Confusion Matrix: ")
    println(EvalMetrics.confusionMatrix)
    //Precision
    val labels = EvalMetrics.labels
    labels.foreach {
      l => println("Precision(" + l + ") = " + EvalMetrics.precision(l))
    }
    //Recall
    labels.foreach {
      l => println("recall(" + l + ") = " + EvalMetrics.recall(l))
    }
    //TruePositiveRate
    labels.foreach {
      l => println("TruePositiveRate(" + l + ") = " + EvalMetrics.truePositiveRate(l))
    }
    //FalsePositiveRate
    labels.foreach {
      l => println("FalsePositiveRate(" + l + ") = " + EvalMetrics.falsePositiveRate(l))
    }
    //F-measure za sekoja labela
    labels.foreach {
      l => println("F1-score(" + l + ") = " + EvalMetrics.fMeasure(l))
    }
    //Weighted statistics
    println("Weighted Precision = " + EvalMetrics.weightedPrecision)
    println("Weighted Recall = " + EvalMetrics.weightedRecall)
    println("Weighted TruePositiveRate = " + EvalMetrics.weightedTruePositiveRate)
    println("Weighted FalsePositiveRate = " + EvalMetrics.weightedFalsePositiveRate)
    println("Weighted F1-Score = " + EvalMetrics.weightedFMeasure)
  }
}
