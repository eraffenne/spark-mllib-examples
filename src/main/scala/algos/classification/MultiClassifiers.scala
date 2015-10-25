package algos.classification

import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.optimization.{LBFGS, SquaredL2Updater}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.rdd.RDD
import utils.{Functions, Datasets}

object MultiClassifiers {

    val iterations = 100
    val lambda = 0.05
    val regularization = new SquaredL2Updater
    val numClasses = 3

    def main(args: Array[String]) {

        // Training and test sets
        val Array(trainingSet: RDD[LabeledPoint], testSet: RDD[LabeledPoint]) = Datasets.multiLabeledPoints.randomSplit(Array(0.7, 0.3), seed = 13L)

        // Logistic Regression
        val lr: LogisticRegressionWithLBFGS = new LogisticRegressionWithLBFGS()
        val lrOptimizer: LBFGS = lr.optimizer
        lr.optimizer
                .setNumIterations(iterations)
                .setRegParam(lambda)
                .setUpdater(regularization)
        lr.setNumClasses(numClasses)

        val lrModel: LogisticRegressionModel = lr.run(trainingSet)

        val lrScoreLabel: RDD[(Double, Double)] = Functions.scoresAndLabels(testSet, lrModel.predict)
        printMetrics("Logistic Regression", lrScoreLabel)

        // Naive Bayes (requires non negative feature values!)
        val bayes: NaiveBayes = new NaiveBayes().setLambda(lambda).setModelType("multinomial")
        val bayesModel: NaiveBayesModel = bayes.run(trainingSet)

        val bayesScoreLabel: RDD[(Double, Double)] = Functions.scoresAndLabels(testSet, bayesModel.predict)
        printMetrics("Naive Bayes", bayesScoreLabel)

        // Random Forest
        val rfModel: RandomForestModel = RandomForest.trainClassifier(trainingSet, numClasses, Map[Int,Int](), 3, "auto", "gini", 4, 32)

        val rfScoreLabel: RDD[(Double, Double)] = Functions.scoresAndLabels(testSet, rfModel.predict)
        printMetrics("Random Forest", rfScoreLabel)
    }

    private def printMetrics(title: String, scoreLabels: RDD[(Double, Double)]): Unit = {
        val metrics: MulticlassMetrics = new MulticlassMetrics(scoreLabels)
        println(s"\n--- ${title}")
        println(s"\tPrecision/Recall: ${metrics.precision}/${metrics.recall}")
        println(s"\tF-measure: ${metrics.fMeasure}")
    }

}
