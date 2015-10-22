package offline.classification

import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.evaluation.MultilabelMetrics
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.optimization.{LBFGS, SquaredL2Updater}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.rdd.RDD
import utils.Data

object MultiLabelClassifiers {

    val iterations = 100
    val lambda = 0.05
    val regularization = new SquaredL2Updater
    val numClasses = 3

    // TODO: randomforest
    // http://spark.apache.org/docs/latest/mllib-decision-tree.html
    def main(args: Array[String]) {

        // Training and test sets
        val Array(trainingSet: RDD[LabeledPoint], testSet: RDD[LabeledPoint]) = Data.multiLabeledPoints.randomSplit(Array(0.7, 0.3), seed = 13L)

        // Logistic Regression
        val lr: LogisticRegressionWithLBFGS = new LogisticRegressionWithLBFGS()
        val lrOptimizer: LBFGS = lr.optimizer
        lr.optimizer
                .setNumIterations(iterations)
                .setRegParam(lambda)
                .setUpdater(regularization)
        lr.setNumClasses(numClasses)

        val lrModel: LogisticRegressionModel = lr.run(trainingSet)
        val lrScoreLabel: RDD[(Array[Double], Array[Double])] = scoresAndLabels(testSet, lrModel.predict)
        val lrMetrics: MultilabelMetrics = new MultilabelMetrics(lrScoreLabel)

        println("\n--- Logistic Regression")
        printMetrics(lrMetrics.labels, lrMetrics)

        // Naive Bayes (requires non negative feature values!)
        val bayes: NaiveBayes = new NaiveBayes().setLambda(lambda).setModelType("multinomial")
        val bayesModel: NaiveBayesModel = bayes.run(trainingSet)

        val bayesScoreLabel: RDD[(Array[Double], Array[Double])] = scoresAndLabels(testSet, bayesModel.predict)
        val bayesMetrics: MultilabelMetrics = new MultilabelMetrics(bayesScoreLabel)

        println("\n--- Naive Bayes")
        printMetrics(bayesModel.labels, bayesMetrics)

        // Random Forest
        val rfModel: RandomForestModel = RandomForest.trainClassifier(trainingSet, numClasses, Map[Int,Int](), 3, "auto", "gini", 4, 32)

        val rfScoreLabel: RDD[(Array[Double], Array[Double])] = scoresAndLabels(testSet, rfModel.predict)
        val rfMetrics: MultilabelMetrics = new MultilabelMetrics(rfScoreLabel)

        println("\n--- Random Forest")
        printMetrics(bayesModel.labels, rfMetrics)
    }


    private def scoresAndLabels(testSet: RDD[LabeledPoint], predict: Vector => Double): RDD[(Array[Double], Array[Double])] = {
        testSet.map { point =>
            (Array(predict(point.features)), Array(point.label))
        }
    }

    private def printMetrics(labels: Array[Double], metrics: MultilabelMetrics): Unit = {
        println(s"\tAccuracy: ${metrics.accuracy}")
        labels.foreach { label =>
            val precision = math.round(metrics.precision(label) * 100d) / 100d
            val recall = math.round(metrics.recall(label) * 100d) / 100d
            println(s"\tPrecision/Recall for label $label: ${precision}/${recall}")
        }
    }

}
