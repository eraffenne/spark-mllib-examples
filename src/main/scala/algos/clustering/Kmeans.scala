package algos.clustering

import org.apache.spark.mllib.clustering.{GaussianMixture, KMeans, KMeansModel}
import utils.Datasets

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

        val clusters: KMeansModel = kmeans.run(Datasets.points)
        val wssse: Double = clusters.computeCost(Datasets.points)

        println(s"--- Kmeans WSSSE: $wssse")
    }
}
