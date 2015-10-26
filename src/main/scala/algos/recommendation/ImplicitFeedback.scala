package algos.recommendation

import org.apache.spark.mllib.recommendation.{MatrixFactorizationModel, ALS}
import org.apache.spark.rdd.RDD
import utils.{Functions, Datasets}

object ImplicitFeedback {
    def main(args: Array[String]) {

        val rank: Int = 5
        val numIterations: Int = 10
        val alpha: Double = 0.01
        val lambda: Double = 0.01

        val als: ALS = new ALS()
                .setAlpha(alpha)
                .setImplicitPrefs(true)
                .setIterations(numIterations)
                .setLambda(lambda)
                .setRank(rank)

        val model: MatrixFactorizationModel = als.run(Datasets.confidences)

        val testSet: RDD[(Int, Int)] = Datasets.confidences.map { r =>
            (r.user, r.product)
        }

        val predictions: RDD[((Int, Int), Double)] = model.predict(testSet).map(Functions.ratingToPair)

        val rateAndPreds = Datasets.confidences.map(Functions.ratingToPair).join(predictions)

        println("\n--- ALS with implicit feedback")
        rateAndPreds.collect().foreach { case (key, value) =>
            println(s"\t$key: $value")
        }

        // To use RankingMetrics, normalize the dataset
    }
}
