package offline.recommendation

import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.recommendation.{Rating, MatrixFactorizationModel, ALS}
import org.apache.spark.rdd.RDD
import utils.Data

object CollaborativeFiltering {

    def main(args: Array[String]) {

        val rank: Int = 10
        val numIterations: Int = 100
        val alpha: Double = 0.5
        val lambda: Double = 0.1

        // Explicit feedback
        val als: ALS = new ALS()
                .setAlpha(alpha)
                .setImplicitPrefs(false)
                .setIterations(numIterations)
                .setLambda(lambda)
                .setRank(rank)

        val model: MatrixFactorizationModel = als.run(Data.ratings)

        val testSet: RDD[(Int, Int)] = Data.ratings.map { rating =>
            (rating.user, rating.product)
        }

        val predictions: RDD[((Int, Int), Double)] = model.predict(testSet).map(ratingToPair)

        val rateAndPred: RDD[(Double, Double)] = Data.ratings.map(ratingToPair).join(predictions).map { case (key, value) =>
            value
        }

        val metrics = new RegressionMetrics(rateAndPred)

        println("\n--- ALS with explicit feedback")
        println(s"RMSE: ${metrics.rootMeanSquaredError}")
    }

    def ratingToPair(rating: Rating): ((Int, Int), Double) = {
        ((rating.user, rating.product), rating.rating)
    }
}
