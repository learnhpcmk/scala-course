package EuroCC

import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.feature.{CountVectorizer, HashingTF, IDF, NGram, RegexTokenizer, StopWordsRemover, Tokenizer, Word2Vec}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, udf}

object LDAAnaliza {
  def main(args: Array[String]): Unit =
  {
    var spark = SparkSession
      .builder()
      .master("local[*]")
      .appName("LDA Analiza")
      .getOrCreate()

    val primeroci =spark.createDataFrame(Seq(
      (1, "Machine Learning is very interesting"),
      (2, "NLP provides a possibility for very interesting analysis"),
      (3, "Do you like programming in Scala?"),
      (4, "Maybe it is time for little break"),
      (5, "What is your favourite language that you are using for programming?"))).toDF("id","tekst")

    primeroci.show()

    //tokenizacija
    val tokenizator = new Tokenizer()
      .setInputCol("tekst")
      .setOutputCol("lista_zborovi")
    val brojTokens = udf {(words: Seq[String]) => words.length}
    val zborovi = tokenizator.transform(primeroci)
    zborovi.select("tekst","lista_zborovi").withColumn("tokens", brojTokens(col("lista_zborovi"))).show

    //token_regex
    val token_regex = new RegexTokenizer()
      .setInputCol("tekst")
      .setOutputCol("lista_zborovi_regex")
      .setPattern("is")
    val zborovi_regex = token_regex.transform(primeroci)
    zborovi_regex.select("tekst","lista_zborovi_regex").withColumn("tokens", brojTokens(col("lista_zborovi_regex"))).show

    //ostranuvanje na sluzbeni zborovi
    val brisac_sluzbeni_zborovi = new StopWordsRemover()
      .setInputCol("lista_zborovi")
      .setOutputCol("lista_zborovi_filter")
    val bez_sluzbeni_zborovi = brisac_sluzbeni_zborovi.transform(zborovi)
    bez_sluzbeni_zborovi.show()

    //N-grams
    val ngram = new NGram()
      .setInputCol("lista_zborovi")
      .setOutputCol("N-grams")
      .setN(1)  //bigrams
    var ngramDF = ngram.transform(zborovi)
    ngramDF.show()

    //TF-IDF
    var TF = new HashingTF()
      .setInputCol("lista_zborovi")
      .setOutputCol("TF_tezina")
      .setNumFeatures(3)
    val TF_tezina = TF.transform(zborovi)

    val IDF = new IDF()
      .setInputCol("TF_tezina")
      .setOutputCol("IDF_tezina")
    val IDF_model = IDF.fit(TF_tezina)
    var IDF_tezini = IDF_model.transform(TF_tezina)
    IDF_tezini.show()

    //Word2Vec
    val word2vec_transformer = new Word2Vec()
      .setInputCol("lista_zborovi")
      .setOutputCol("word2vec_vektor")
      .setVectorSize(3)
      .setMinCount(0)
    val W2VModel = word2vec_transformer.fit(zborovi)
    val W2VResults = W2VModel.transform(zborovi)
    W2VResults.show()

    val CountVector = new CountVectorizer()
      .setInputCol("lista_zborovi")
      .setOutputCol("count_vektor")
      .setVocabSize(3)
      .setMinDF(2)
    val CVModel = CountVector.fit(zborovi)
    val VectorCVModel = CVModel.transform(zborovi)
    VectorCVModel.show()

    //Topic modelling
    val lda = new LDA()
      .setK(5)
      .setMaxIter(5)

    var vektori = IDF_tezini
    vektori = vektori.withColumnRenamed("id","label")
    vektori = vektori.withColumnRenamed("IDF_tezina", "features")
    vektori.show()

    val LDAModel = lda.fit(vektori)
    val logLikelihood = LDAModel.logLikelihood(vektori)
    val logPerplexity = LDAModel.logPerplexity(vektori)
    println("Dolnata granica na log likelihood na celiot korpus e = " + logLikelihood)
    println("Gornata granica na log likelihood na celiot korpus e = " + logPerplexity)
    val temi = LDAModel.describeTopics(10)
    temi.show()

  }
}
