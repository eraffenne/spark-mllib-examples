package others

import org.apache.spark.mllib.linalg.{Matrix, Vectors}
import org.apache.spark.mllib.stat.test.{ChiSqTestResult, KolmogorovSmirnovTestResult}
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, KernelDensity, Statistics}
import org.apache.spark.rdd.RDD
import utils.Constants

import scala.util.Random

object BasicStatistics {

    def main(args: Array[String]) {

        // --- Summary
        val stats: MultivariateStatisticalSummary = Statistics.colStats(Constants.vectorRDD)

        println("\n--- Basic Statistics on vectorRDD")
        println(s"\tCount: ${stats.count}")
        println(s"\tMax: ${stats.max}")
        println(s"\tMin: ${stats.min}")
        println(s"\tL1 norm: ${stats.normL1}")
        println(s"\tL2 norm: ${stats.normL2}")
        println(s"\tVariance: ${stats.variance}")
        println(s"\tNon-zero entries: ${stats.numNonzeros}")

        // --- Correlation
        val corr: Double = Statistics.corr(Constants.x, Constants.y, "pearson")
        val corrMatrix: Matrix = Statistics.corr(Constants.vectorRDD, "pearson")

        println("\n--- Correlation")
        println(s"\tSerie correlation: $corr")
        println(s"\tCorrelation matrix: \n$corrMatrix")

        // --- Stratified sampling
        val fractions: Map[String, Double] = Map( "a" -> 0.2, "b" -> 0.3)
        val seed: Long = Random.nextLong()
        val sample: RDD[(String, Double)] = Constants.pairRDD.sampleByKey(withReplacement = false, fractions, seed)
        val exactSample: RDD[(String, Double)] = Constants.pairRDD.sampleByKeyExact(withReplacement = false, fractions, seed)

        println(s"\n--- Sampling")
        println(s"\tAproximative: ${sample.collect().toList}")
        println(s"\tExact: ${exactSample.collect().toList}")

        // --- Chi square test
        // see http://web.cs.ucla.edu/~mtgarip/statistics.html

        val goodnessOfFit1: ChiSqTestResult = Statistics.chiSqTest(Vectors.dense(1.0, 2.0, 1.0, 4.0))
        println(s"\n--- First Goodness of fit:\n $goodnessOfFit1")

        val goodnessOfFit2: ChiSqTestResult = Statistics.chiSqTest(
            Vectors.dense(1.0, 2.0, 1.0, 4.0),
            Vectors.dense(1.2, 2.0, 1.0, 0.5)
        )
        println(s"\n--- Second Goodness of fit:\n $goodnessOfFit2")

        val independenceTest1: ChiSqTestResult = Statistics.chiSqTest(Constants.contingencyMatrix)
        println(s"\n--- First Independence test:\n $independenceTest1")

        val independenceTest2: Array[ChiSqTestResult] = Statistics.chiSqTest(Constants.labeledPointRDD)
        independenceTest2.foreach { test =>
            println(s"\n--- Second Independence test:\n $test")
        }

        // --- Kolmogorov-Smirnov test
        val ksTest: KolmogorovSmirnovTestResult = Statistics.kolmogorovSmirnovTest(Constants.normalRDD, "norm", 0, 1)
        println(s"\n--- Kolmogoros-Smirnov test:\n $ksTest")

        // --- Kernel density estimation (bandwidth = standard deviation)
        val kernelDensity: KernelDensity = new KernelDensity().setSample(Constants.normalRDD).setBandwidth(1.0)
        val densities: Array[Double] = kernelDensity.estimate( Array(1.0, 0, -0.2, 15.8) )
        println(s"\n--- Kernel density estimation:\n ${densities.toList}")
    }

}
