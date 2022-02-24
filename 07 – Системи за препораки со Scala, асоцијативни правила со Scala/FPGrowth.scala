package EuroCC

import org.apache.spark.ml.fpm.FPGrowth
import org.apache.spark.sql.SparkSession

object FPGrowth {
  def main(args: Array[String]): Unit =
  {
    var spark = SparkSession
      .builder()
      .master("local[*]")
      .appName("FPG Example")
      .getOrCreate()

    val primeroci = spark.createDataFrame(Seq(
      (1, Array(1,2,3,4,5,6,7)),
      (2, Array(3,4,5,7,8,11,2,23,45)),
      (3, Array(1,3,45,23,11,31,5)),
      (4, Array(1,4,6,9,3,45,65)),
      (5, Array(34,56,23,2,12,24)))).toDF("id","numbers")

    var fpgrowth = new FPGrowth()
      .setItemsCol("numbers")
      .setMinSupport(0.5)
      .setMinConfidence(0.6)
    val FPGModel = fpgrowth.fit(primeroci)

    FPGModel.freqItemsets.show()
  }
}
