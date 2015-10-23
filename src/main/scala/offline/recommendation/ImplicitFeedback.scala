package offline.recommendation

import org.apache.spark.mllib.recommendation.{Rating, MatrixFactorizationModel, ALS}
import org.apache.spark.rdd.RDD
import utils.{Functions, Data}

object ImplicitFeedback {
    def main(args: Array[String]) {

        val rank: Int = 5
        val numIterations: Int = 10
        val alpha: Double = 0.01
        val lambda: Double = 0.01

        // Explicit feedback
        val als: ALS = new ALS()
                .setAlpha(alpha)
                .setImplicitPrefs(true)
                .setIterations(numIterations)
                .setLambda(lambda)
                .setRank(rank)

        val model: MatrixFactorizationModel = als.run(Data.confidences)

        val testSet: RDD[(Int, Int)] = Data.confidences.map { r =>
            (r.user, r.product)
        }

        val predictions: RDD[((Int, Int), Double)] = model.predict(testSet).map(Functions.ratingToPair)

        val rateAndPreds = Data.confidences.map(Functions.ratingToPair).join(predictions)

        println("\n--- ALS with implicit feedback")
        rateAndPreds.collect().foreach { case (key, value) =>
            println(s"\t$key: $value")
        }

    }
}
