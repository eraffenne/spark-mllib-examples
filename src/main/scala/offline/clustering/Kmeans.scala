package offline.clustering

import org.apache.spark.mllib.clustering.{GaussianMixture, KMeans, KMeansModel}
import utils.Data

object Kmeans {

    def main(args: Array[String]) {

        val epsilon: Double = 1e-4
        val numClusters: Int = 2
        val numIterations: Int = 20

        // Kmeans parallel
        val kmeans: KMeans = new KMeans()
                .setEpsilon(epsilon)
                .setK(numClusters)
                .setMaxIterations(numIterations)

        val clusters: KMeansModel = kmeans.run(Data.points)
        val wssse: Double = clusters.computeCost(Data.points)

        println(s"--- Kmeans WSSSE: $wssse")
    }
}
