package algos.recommendation

import org.apache.spark.mllib.evaluation.RankingMetrics
import org.apache.spark.mllib.recommendation.{Rating, MatrixFactorizationModel, ALS}
import org.apache.spark.rdd.RDD
import utils.{Functions, Datasets}

object ExplicitFeedback {

    def main(args: Array[String]) {

        val rank: Int = 5
        val numIterations: Int = 10
        val alpha: Double = 0.01
        val lambda: Double = 0.1

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

        val recomendations: RDD[(Int, Array[Rating])] = model.recommendProductsForUsers(10).map { case (user, recs) =>
            (user, recs.map(Functions.scaledRating))
        }

        // Metrics
        val userMovies: RDD[(Int, Iterable[Rating])] = Datasets.ratings.map { r =>
            Rating( r.user, r.product, if (r.rating > 0) 1.0 else 0.0)
        }.groupBy(_.user)
        val relevantProducts: RDD[(Array[Int], Array[Int])] = userMovies.join(recomendations).map { case (user, (actual, prediction)) =>
            (prediction.map(_.product), actual.filter(_.rating > 0.0).map(_.product).toArray)
        }

        val metrics: RankingMetrics[Int] = new RankingMetrics(relevantProducts)
        println("\n--- Matrix Factorization")
        println(s"\tPrecision at 10: ${metrics.precisionAt(10)}")
        println(s"\tNDCG at 10: ${metrics.ndcgAt(10)}")
        println(s"\tMAP: ${metrics.meanAveragePrecision}")
    }
}
