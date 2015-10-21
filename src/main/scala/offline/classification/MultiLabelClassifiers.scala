package offline.classification

import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.evaluation.MultilabelMetrics
import org.apache.spark.mllib.optimization.{LBFGS, GradientDescent, SquaredL2Updater}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import utils.{Data, Scoring}

object MultiLabelClassifiers {

    val iterations = 100
    val lambda = 0.05
    val regularization = new SquaredL2Updater

    // TODO: LR, tree, randomforest
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
        lr.setNumClasses(3)

        val lrModel: LogisticRegressionModel = lr.run(trainingSet)
        val lrScoreLabel: RDD[(Array[Double], Array[Double])] = Scoring.scoresAndLabels(testSet, lrModel.predict).map {
            case (score: Double, label: Double) =>
                (Array(score), Array(label))
        }

        val lrMetrics = new MultilabelMetrics(lrScoreLabel)
        lrMetrics.labels.foreach { label =>
            println(s"LR precision for label $label: ${lrMetrics.precision(label)}")
        }

        println(s"num label: ${lrModel.numClasses}")
    }
}
