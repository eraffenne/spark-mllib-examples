package pipelines

import org.apache.spark.ml.{PipelineModel, Pipeline}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.sql.DataFrame
import utils.Constants

object SimplePipeline {

    def main(args: Array[String]) {
        // Datasets
        val training: DataFrame = Constants.sqlc.createDataFrame(Seq(
            (0L, "a b c d e spark", 1.0),
            (1L, "b d", 0.0),
            (2L, "spark f g h", 1.0),
            (3L, "hadoop mapreduce", 0.0)
        )).toDF("id", "text", "label")

        val testing: DataFrame = Constants.sqlc.createDataFrame(Seq(
            (4L, "spark i j k"),
            (5L, "l m n"),
            (6L, "mapreduce spark"),
            (7L, "apache hadoop")
        )).toDF("id", "text")


        // Three stages pipeline
        val tokenizer: Tokenizer = new Tokenizer()
                .setInputCol("text")
                .setOutputCol("words")
        // hashing trick
        val hashingTF: HashingTF = new HashingTF()
                .setNumFeatures(10)
                .setInputCol(tokenizer.getOutputCol)
                .setOutputCol("features")
        val lr: LogisticRegression = new LogisticRegression()
                .setMaxIter(20)
                .setRegParam(0.01)
        val pipeline: Pipeline = new Pipeline()
                .setStages(Array(tokenizer, hashingTF, lr))

        // Fit the pipeline to training documents.
        val model: PipelineModel = pipeline.fit(training)

        // Make predictions on test documents.
        model.transform(testing)
                .select("id", "text", "probability", "prediction")
                .show()
    }
}
