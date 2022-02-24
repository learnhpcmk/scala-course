package EuroCC

import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.feature.{CountVectorizer, StopWordsRemover, StringIndexer, Tokenizer}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.SparkSession

object DocumentClassification {

  def main(args: Array[String]): Unit =
  {
    var spark = SparkSession
      .builder()
      .master("local[*]")
      .appName("Document Classification Example")
      .getOrCreate()

    var tekst_data = spark.read
      .option("header","true")
      .option("inferSchema","true")
      .format("com.databricks.spark.csv")
      .load("data/Doc_classification.csv")

    tekst_data.cache()
    tekst_data.printSchema()
    tekst_data.show()

    val tokenizer = new Tokenizer()
      .setInputCol("tekst")
      .setOutputCol("lista_zborovi")
    val zborovi = tokenizer.transform(tekst_data)
    zborovi.show()

    val brisac_sluzbeni_zborovi = new StopWordsRemover()
      .setInputCol("lista_zborovi")
      .setOutputCol("lista_zborovi_filter")
    val bez_sluzbeni_zborovi = brisac_sluzbeni_zborovi.transform(zborovi)
    bez_sluzbeni_zborovi.show()

    val CVWords = new CountVectorizer()
      .setInputCol("lista_zborovi_filter")
      .setOutputCol("count_vektor")

    val labelIndexer = new StringIndexer()
      .setInputCol("klasa")
      .setOutputCol("label")

    var indexed = labelIndexer.fit(bez_sluzbeni_zborovi).transform(bez_sluzbeni_zborovi)
    val CVModel = CVWords.fit(indexed)
    val vektori_CVModel = CVModel.transform(indexed)

    var InputData = vektori_CVModel.select("label","count_vektor")
    InputData = InputData.withColumnRenamed("count_vektor", "features")
    InputData.show(10)

    var Array(trainingData, testingData) = InputData.randomSplit(Array(0.8, 0.2), 123L)

    val nb = new NaiveBayes()
      .setLabelCol("label")
      .setFeaturesCol("features")

    val NBModel = nb.fit(trainingData)
    val NBTestModel = NBModel.transform(testingData)
    NBTestModel.show(10)

    val predictionANDLabels = NBTestModel
      .select("prediction","label")
      .rdd.map(x => (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double]))

    val EvalMetrics = new MulticlassMetrics(predictionANDLabels)

    val ModelAccuracy = EvalMetrics.accuracy
    println("Evaluation Statistics")
    println("Classification Accuracy of the model is = " + ModelAccuracy)
    println("Confusion matrix:" + EvalMetrics.confusionMatrix)
    val labels = EvalMetrics.labels
    //Precision
    labels.foreach{
      l => println("Precision(" + l + ") = " + EvalMetrics.precision(l))
    }
    //recall
    labels.foreach{
      l => println("Recall(" + l + ") = " + EvalMetrics.recall(l))
    }
    //TruePositiveRate
    labels.foreach{
      l => println("TruePositiveRate(" + l + ") = " + EvalMetrics.truePositiveRate(l))
    }
    //FalsePositiveRate
    labels.foreach{
      l => println("FalsePositiveRate(" + l + ") = " + EvalMetrics.falsePositiveRate(l))
    }
    //F1-Score
    labels.foreach{
      l => println("F1-Score(" + l + ") = " + EvalMetrics.fMeasure(l))
    }
    //Weighted Statistics
    println("Weighted Precision = " + EvalMetrics.weightedPrecision)
    println("Weighted Recall = " + EvalMetrics.weightedRecall)
    println("Weighted TruePositiveRate = " + EvalMetrics.weightedTruePositiveRate)
    println("Weighted FalsePositiveRate = " + EvalMetrics.weightedFalsePositiveRate)
    println("Weighted F1-Score = " + EvalMetrics.weightedFMeasure)
 }
}
