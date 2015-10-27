package onlineLearning.clustering

import java.util.Calendar

import org.apache.spark.mllib.clustering.StreamingKMeans
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.streaming.dstream.DStream
import utils.{Constants, Datasets}

object StreamingKmeans {

    val trainingDir = "data/streaming/training"
    val testingDir = "data/streaming/testing"
    val outputDir = "data/streaming/predictions"

    def main(args: Array[String]) {


        // Write some files to initiate the process
        Datasets.points.saveAsTextFile(trainingDir)

        val k: Int = 3
        val numDimensions: Int = 3

        val trainingData: DStream[Vector] = Constants.ssc.textFileStream(trainingDir).map(Vectors.parse)
        val testingData: DStream[Vector] = Constants.ssc.textFileStream(testingDir).map(Vectors.parse)

        val model: StreamingKMeans = new StreamingKMeans()
                .setK(k)
                .setRandomCenters(numDimensions, 0.0)

        model.trainOn(trainingData)

        val predictions: DStream[Int] = model.predictOn(testingData)

        val vectorCluster: DStream[(Vector, Int)] = testingData.transformWith(predictions, zipRDDs _)

        vectorCluster.foreachRDD { rdd =>
            if (rdd.count() > 0) {
                val dateString: String = Calendar.getInstance().getTime.toString.replace(" ", "-").replace(":", "-")
                rdd.saveAsTextFile(s"$outputDir-$dateString")
            }
        }

        Constants.ssc.start()
        Constants.ssc.awaitTermination()
    }

    private def zipRDDs(points: RDD[Vector], clusters: RDD[Int]): RDD[(Vector, Int)] = {
        val previous: RDD[Vector] = Constants.ssc.sparkContext.textFile(trainingDir).map(Vectors.parse)
        points.union(previous).saveAsTextFile(trainingDir)
        points.zip(clusters)
    }
}
