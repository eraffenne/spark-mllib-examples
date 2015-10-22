package utils

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD

object Data {

    val binaryLabeledPoints = MLUtils.loadLibSVMFile(Constants.sc, Constants.sparkHome + "/data/mllib/sample_libsvm_data.txt")

    val multiLabeledPoints = MLUtils.loadLibSVMFile(Constants.sc, "data/sample_multiclass_classification_data.txt")

    val regressionLabeledPoints = parseDataset(Constants.sparkHome + "/data/mllib/ridge-data/lpsa.data")


    private def parseDataset(path: String): RDD[LabeledPoint] = {
        Constants.sc.textFile(path).map { line =>
            val parts = line.split(',')
            LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
        }
    }

}
