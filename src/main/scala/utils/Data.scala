package utils

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD

object Data {

    val binaryLabeledPoints = MLUtils.loadLibSVMFile(Constants.sc, Constants.sparkHome + "/data/mllib/sample_libsvm_data.txt")

    val multiLabeledPoints = MLUtils.loadLibSVMFile(Constants.sc, "data/sample_multiclass_classification_data.txt")

    val regressionLabeledPoints = parseDataset(Constants.sparkHome + "/data/mllib/ridge-data/lpsa.data")

    val ratings = Constants.sc.textFile(Constants.sparkHome + "/data/mllib/als/test.data").map {
        line =>
            val parts = line.split(",")
            new Rating(parts(0).toInt, parts(1).toInt, parts(2).toDouble)
    }

    val confidences = Constants.sc.textFile("data/sample_als_implicit.txt").map {
        line =>
            val parts = line.split(",")
            new Rating(parts(0).toInt, parts(1).toInt, parts(2).toDouble)
    }

    val points = Constants.sc.textFile(Constants.sparkHome + "/data/mllib/kmeans_data.txt").map(parseVector)

    val corpus = Constants.sc.textFile(Constants.sparkHome + "/data/mllib/sample_lda_data.txt")
            .map(parseVector)
            // .zipWithIndex FIXME
            // .map(_.swap)

    private def parseDataset(path: String): RDD[LabeledPoint] = {
        Constants.sc.textFile(path).map { line =>
            val parts = line.split(',')
            LabeledPoint(parts(0).toDouble, parseVector(parts(1)))
        }
    }

    private def parseVector(s: String): Vector = {
        Vectors.dense(s.split(' ').map(_.toDouble))
    }

}
