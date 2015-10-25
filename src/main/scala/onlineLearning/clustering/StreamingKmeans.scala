package onlineLearning.clustering

import org.apache.spark.mllib.clustering.StreamingKMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import utils.Constants

object StreamingKmeans {

    def main(args: Array[String]) {

        // FIXME: that's a raw copy&paste
        // TODO: Find a case for streaming in data
        val trainingData = Constants.ssc.textFileStream("/training/data/dir").map(Vectors.parse)
        val testData = Constants.ssc.textFileStream("/testing/data/dir").map(LabeledPoint.parse)

        val numDimensions = 3
        val numClusters = 2
        val model = new StreamingKMeans()
                .setK(numClusters)
                .setDecayFactor(1.0)
                .setRandomCenters(numDimensions, 0.0)

        model.trainOn(trainingData)
        model.predictOnValues(testData.map(lp => (lp.label, lp.features))).print()

        Constants.ssc.start()
        Constants.ssc.awaitTermination()
    }
}
