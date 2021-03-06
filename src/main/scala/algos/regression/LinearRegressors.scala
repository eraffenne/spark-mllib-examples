package algos.regression

import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.regression._
import org.apache.spark.rdd.RDD
import utils.{Functions, Datasets}

object LinearRegressors {

    val iterations = 100
    val stepSize = 1.0
    val lambda = 0.05

    def main(args: Array[String]) {

        val Array(trainingSet: RDD[LabeledPoint], testSet: RDD[LabeledPoint]) = Datasets.regressionLabeledPoints.randomSplit(Array(0.7, 0.3), seed = 13L)

        // Linear Least Square Regression
        val model: LinearRegressionModel = LinearRegressionWithSGD.train(trainingSet, iterations)
        val regPredValue: RDD[(Double, Double)] = Functions.scoresAndLabels(testSet, model.predict)

        val metrics: RegressionMetrics = new RegressionMetrics(regPredValue)
        println("\n--- Linear least square")
        println(s"Explained variance: ${metrics.explainedVariance}")
        println(s"RMSE: ${metrics.rootMeanSquaredError}")

        // Lasso (L1) regression
        val lassoModel = LassoWithSGD.train(trainingSet, iterations, stepSize, lambda)
        val lassoPredValue: RDD[(Double, Double)] = Functions.scoresAndLabels(testSet, lassoModel.predict)

        val lassoMetrics: RegressionMetrics = new RegressionMetrics(lassoPredValue)
        println("\n--- Lasso (L1)")
        println(s"Explained variance: ${lassoMetrics.explainedVariance}")
        println(s"RMSE: ${lassoMetrics.rootMeanSquaredError}")


        // Ridge (L2) regression
        val ridgeModel = RidgeRegressionWithSGD.train(trainingSet, iterations, stepSize, lambda)
        val ridgePredValue: RDD[(Double, Double)] = Functions.scoresAndLabels(testSet, ridgeModel.predict)

        val ridgeMetrics: RegressionMetrics = new RegressionMetrics(ridgePredValue)
        println("\n--- Ridge (L2)")
        println(s"Explained variance: ${ridgeMetrics.explainedVariance}")
        println(s"RMSE: ${ridgeMetrics.rootMeanSquaredError}")
    }
}
