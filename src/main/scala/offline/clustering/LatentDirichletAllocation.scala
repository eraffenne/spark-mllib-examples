package offline.clustering

import org.apache.spark.mllib.clustering.{LDAModel, LDA}
import utils.Data

object LatentDirichletAllocation {

    def main(args: Array[String]) {

        val numTopics: Int = 3
        val numIterations: Int = 1

        // LDA
        val lda: LDA = new LDA().setK(numTopics).setMaxIterations(numIterations)
        val ldaModel: LDAModel = lda.run(Data.corpus)

        println("--- LDA")
        println(s"Number of words: ${ldaModel.vocabSize}")

//        val topics = ldaModel.topicsMatrix
//        for (topic <- Range(0, 3)) {
//            print("Topic " + topic + ":")
//            for (word <- Range(0, ldaModel.vocabSize)) { print(" " + topics(word, topic)); }
//            println()
//        }
    }

}
