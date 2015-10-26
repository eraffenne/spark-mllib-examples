package algos.frequentItems

import org.apache.spark.mllib.fpm.AssociationRules.Rule
import org.apache.spark.mllib.fpm.FPGrowth.FreqItemset
import org.apache.spark.mllib.fpm.{AssociationRules, FPGrowthModel, FPGrowth}
import org.apache.spark.rdd.RDD
import utils.Datasets

object FreqItemMiner {

    val minSupport: Double = 0.2
    val numPartitions: Int = 10
    val minConfidence: Double = 0.8

    def main(args: Array[String]) {
        // FP-Growth
        val fPGrowth: FPGrowth = new FPGrowth()
                .setMinSupport(minSupport)
                .setNumPartitions(numPartitions)
        val fpModel: FPGrowthModel[String] = fPGrowth.run(Datasets.baskets)
        val freqItemSets: RDD[FreqItemset[String]] = fpModel.freqItemsets
        val fpRules = fpModel.generateAssociationRules(minConfidence)

        println("\n--- FP Growth")
        println("Items frequency:")
        freqItemSets.sortBy( _.freq, false).collect().foreach { itemset =>
            println("\t" + itemset.items.mkString("[", ",", "]") + ", " + itemset.freq)
        }

        printRules(fpRules)

        // Association Rules
        val ar = new AssociationRules().setMinConfidence(minConfidence)
        val arRules: RDD[Rule[String]] = ar.run(freqItemSets)

        println("\n--- Association Rule")
        printRules(arRules)
    }

    private def printRules(rules: RDD[Rule[String]]): Unit = {
        println(s"\nRules for confidence >= $minConfidence")
        rules.collect().foreach { rule =>
            println(
                rule.antecedent.mkString("\t[", ",", "]")
                        + " => " + rule.consequent .mkString("[", ",", "]")
                        + ": " + rule.confidence)
        }
    }
}
