package EuroCC

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.learning.config.Adam
import org.datavec.api.records.reader.RecordReader
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.conf.layers.{LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

import java.io.File

object LSTMDeepLearning {

  def readIteratorCSVData(csvFilePath:String, batchSize: Int, numFeatures: Int, numClasses: Int): DataSetIterator = {
      val recorder: RecordReader = new CSVRecordReader()
      val input: File = new File(csvFilePath)
      recorder.initialize(new FileSplit(input))
      val iterator: DataSetIterator = new RecordReaderDataSetIterator(recorder,batchSize,numFeatures,numClasses)
      return iterator  }
  def main(args: Array[String])
  {
      val trainDir = "data/ForestType_train.csv"
      val testDir = "data/ForestType_test.csv"
      val numEpochs = 10
      var seed = 1234L
      val numFeatures = 27
      val numClasses = 4
      val batchSize = 32
      var AdamConfig = new Adam(5e-1, 5e-1, 5e-1,5e-3)
      val numInputs = numFeatures
      val numOutputs = numClasses
      val numHiddenNodes = 50

      var trainingDataIter: DataSetIterator = readIteratorCSVData(trainDir,batchSize,numFeatures,numClasses)
      var testingDataIter: DataSetIterator = readIteratorCSVData(testDir,batchSize,numFeatures,numClasses)

      //Prv LSTM sloj
      val layer_0 = new LSTM.Builder()
        .nIn(numInputs)
        .nOut(numHiddenNodes)
        .activation(Activation.RELU)
        .build()

      //Vtor LSTM sloj
      val layer_1 = new LSTM.Builder()
        .nIn(numHiddenNodes)
        .nOut(numHiddenNodes)
        .activation(Activation.RELU)
        .build()

      //Tret LSTM sloj
      val layer_2 = new LSTM.Builder()
         .nIn(numHiddenNodes)
        .nOut(numHiddenNodes)
        .activation(Activation.RELU)
        .build()

      //RNN output sloj
      val layer_3 = new RnnOutputLayer.Builder()
        .activation(Activation.SOFTMAX)
        .lossFunction(LossFunction.MCXENT)
        .nIn(numHiddenNodes)
        .nOut(numClasses)
        .build()

     val LSTMconf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
       .seed(seed)
       .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
       .weightInit(WeightInit.XAVIER)
       .updater(AdamConfig)
       .l2(1e-5)
       .list()
       .layer(0,layer_0)
       .layer(1, layer_1)
       .layer(2, layer_2)
       .layer(3,layer_3)
       .pretrain(false)
       .backprop(true)
       .build()

    val model: MultiLayerNetwork = new MultiLayerNetwork(LSTMconf)
    model.init()

    model.setListeners(new ScoreIterationListener(1));

    val layersInfo = model.getLayers()
    var VkupnoParametri = 0
    var i = 0

    for (i <- 0 to layersInfo.length -1)
      {
        var nParameters = layersInfo(i).numParams()
        println("Broj na parametri za sekoj sloj " + i + " e : " +nParameters)
        VkupnoParametri = VkupnoParametri + nParameters
      }
    println("Vkupen broj na parametri za LSTM mrezata " + i + " e : " +VkupnoParametri)

    var j = 0
    println("Treniranje na mrezata.....")
    for (j <- 0 to numEpochs -1)
      {
        model.fit(trainingDataIter)
      }

    println("Evaluacija na mrezata .....")
    val eval: Evaluation = new Evaluation(4)
    while (testingDataIter.hasNext())
      {
        val next: DataSet = testingDataIter.next()
        val output: INDArray = model.output(next.getFeatureMatrix())
        eval.eval(next.getLabels(), output)
      }

    println(eval.stats())
    println("Kraj")


  }
}
