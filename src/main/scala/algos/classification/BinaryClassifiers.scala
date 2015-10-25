package algos.classification

import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.optimization.{GradientDescent, SquaredL2Updater}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import utils.{Functions, Datasets}

object BinaryClassifiers {

    val iterations = 100
    val lambda = 0.05
    val regularization = new SquaredL2Updater
    
    def main(args: Array[String]) {

        // Training and test sets
        val Array(trainingSet: RDD[LabeledPoint], testSet: RDD[LabeledPoint]) = Datasets.binaryLabeledPoints.randomSplit(Array(0.7, 0.3), seed = 13L)

        // SVM
        val svm: SVMWithSGD = new SVMWithSGD()
        val svmOptimizer: GradientDescent = svm.optimizer
        svm.optimizer
                .setNumIterations(iterations)
                .setRegParam(lambda)
                .setUpdater(regularization)

        val svmModel: SVMModel = svm.run(trainingSet)
        val svmScoreLabel: RDD[(Double, Double)] = Functions.scoresAndLabels(testSet, svmModel.predict)

        val svmMetrics: BinaryClassificationMetrics = new BinaryClassificationMetrics(svmScoreLabel)
        println(s"SVM AUC: ${svmMetrics.areaUnderROC()}")

        // Logistic Regression
        val lr: LogisticRegressionWithSGD = new LogisticRegressionWithSGD()
        val lrOptimizer: GradientDescent = lr.optimizer
        lr.optimizer
                .setNumIterations(iterations)
                .setRegParam(lambda)
                .setUpdater(regularization)

        val lrModel: LogisticRegressionModel = lr.run(trainingSet)
        val lrScoreLabel: RDD[(Double, Double)] = Functions.scoresAndLabels(testSet, lrModel.predict)

        val lrMetrics: BinaryClassificationMetrics = new BinaryClassificationMetrics(lrScoreLabel)
        println(s"LR AUC: ${lrMetrics.areaUnderROC()}")
    }
}
