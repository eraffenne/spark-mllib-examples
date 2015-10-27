package onlineLearning.regression

import java.util.Calendar

import org.apache.spark.mllib.classification.StreamingLogisticRegressionWithSGD
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression._
import org.apache.spark.rdd.RDD
import org.apache.spark.streaming.dstream.DStream
import utils.{Datasets, Constants}

object StreamingLinearReg {
    val trainingDir = "data/streaming/training"
    val testingDir = "data/streaming/testing"
    val outputDir = "data/streaming/predictions"

    def main(args: Array[String]) {


        // Write some files to initiate the process
        Datasets.regressionLabeledPoints.saveAsTextFile(trainingDir)

        val iterations = 10
        val numFeatures = 8

        val trainingData: DStream[LabeledPoint] = Constants.ssc.textFileStream(trainingDir).map(LabeledPoint.parse)
        val testingData: DStream[Vector] = Constants.ssc.textFileStream(testingDir).map(Vectors.parse)

        val model: StreamingLinearRegressionWithSGD = new StreamingLinearRegressionWithSGD()
                .setNumIterations(iterations)
                .setInitialWeights(Vectors.zeros(numFeatures))

        model.trainOn(trainingData)

        val predictions: DStream[Double] = model.predictOn(testingData)

        val vectorLabel: DStream[LabeledPoint] = testingData.transformWith(predictions, zipRDDs _)

        vectorLabel.foreachRDD { rdd =>
            if (rdd.count() > 0) {
                val dateString: String = Calendar.getInstance().getTime.toString.replace(" ", "-").replace(":", "-")
                rdd.saveAsTextFile(s"$outputDir-$dateString")
            }
        }

        Constants.ssc.start()
        Constants.ssc.awaitTermination()
    }

    private def zipRDDs(points: RDD[Vector], labels: RDD[Double]): RDD[LabeledPoint] = {
        points.zip(labels).map { case (vector, label) =>
            new LabeledPoint(label, vector)
        }
    }
}
