package utils

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

object Functions {

    def scoresAndLabels(testSet: RDD[LabeledPoint], predict: Vector => Double): RDD[(Double, Double)] = {
        testSet.map { point =>
            (predict(point.features), point.label)
        }
    }

    def ratingToPair(r: Rating): ((Int, Int), Double) = {
        ((r.user, r.product), r.rating)
    }

    def scaledRating(r: Rating): Rating = {
        val scaledRating = math.max(math.min(r.rating, 1.0), 0.0)
        Rating(r.user, r.product, scaledRating)
    }
}
