package utils

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

object Scoring {

    def scoresAndLabels(testSet: RDD[LabeledPoint], predict: Vector => Double): RDD[(Double,Double)] = {
        testSet.map { point =>
            (predict(point.features), point.label)
        }
    }

}
