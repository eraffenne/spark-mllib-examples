package online.regression

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression._
import utils.Constants

object StreamRegressor {

    val iterations = 100

    def main(args: Array[String]) {

        // TODO: find a way to stream some data in
        // Each line should be a data point formatted as (y,[x1,x2,x3])
        val trainingSet = Constants.ssc.textFileStream("/training/data/dir").map(LabeledPoint.parse).cache()
        val testSet = Constants.ssc.textFileStream("/testing/data/dir").map(LabeledPoint.parse)

        val numFeatures = 3
        val model: StreamingLinearRegressionWithSGD = new StreamingLinearRegressionWithSGD()
                .setInitialWeights(Vectors.zeros(numFeatures))

        model.trainOn(trainingSet)
        model.predictOnValues(testSet.map(lp => (lp.label, lp.features))).print()

        Constants.ssc.start()
        Constants.ssc.awaitTermination()
    }
}
