package algos.clustering

import org.apache.spark.mllib.clustering.LDA
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import utils.Datasets

object LatentDirichletAllocation {

    def main(args: Array[String]) {

        val numTopics: Int = 3
        val numIterations: Int = 5

        // Prepare corpus
        var idx: Long = -1l
        val corpus: RDD[(Long, Vector)] = Datasets.corpus.map { value =>
            idx = idx + 1
            (idx, value)
        }

        // LDA
        val lda: LDA = new LDA().setK(numTopics).setMaxIterations(numIterations)
        val ldaModel = lda.run(corpus)

        println("--- LDA")
        println(s"Number of words: ${ldaModel.vocabSize}")

        val topics = ldaModel.topicsMatrix
        for (topic <- Range(0, 3)) {
            print("Topic " + topic + ":")
            for (word <- Range(0, ldaModel.vocabSize)) {
                print(" " + topics(word, topic))
            }
            println()
        }
    }

}
