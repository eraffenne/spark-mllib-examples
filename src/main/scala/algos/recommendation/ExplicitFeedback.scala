package algos.recommendation

import org.apache.spark.mllib.recommendation.{MatrixFactorizationModel, ALS}
import org.apache.spark.rdd.RDD
import utils.{Functions, Datasets}

object ExplicitFeedback {

    def main(args: Array[String]) {

        val rank: Int = 5
        val numIterations: Int = 10
        val alpha: Double = 0.01
        val lambda: Double = 0.1

        // Explicit feedback
        val als: ALS = new ALS()
                .setAlpha(alpha)
                .setImplicitPrefs(false)
                .setIterations(numIterations)
                .setLambda(lambda)
                .setRank(rank)

        val model: MatrixFactorizationModel = als.run(Datasets.ratings)

        val testSet: RDD[(Int, Int)] = Datasets.ratings.map { rating =>
            (rating.user, rating.product)
        }

        val predictions: RDD[((Int, Int), Double)] = model.predict(testSet).map(Functions.ratingToPair)

        val mse: Double = Datasets.ratings.map(Functions.ratingToPair).join(predictions).map { case (key, value) =>
            math.pow( value._1 - value._2, 2.0)
        }.mean()
        val rmse: Double = math.sqrt(mse)

        println("\n--- ALS with explicit feedback")
        println(s"\tMSE: $mse")
        println(s"\tRMSE: $rmse")
    }
}
