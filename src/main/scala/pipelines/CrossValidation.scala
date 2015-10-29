package pipelines

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.DataFrame
import utils.Constants

object CrossValidation {

    def main(args: Array[String]) {

        val logger = Logger.getRootLogger
        logger.setLevel(Level.INFO)

        // Datasets
        val training: DataFrame = Constants.sqlc.createDataFrame(Seq(
            (0L, "a b c d e spark", 1.0),
            (1L, "b d", 0.0),
            (2L, "spark f g h", 1.0),
            (3L, "hadoop mapreduce", 0.0),
            (4L, "b spark who", 1.0),
            (5L, "g d a y", 0.0),
            (6L, "spark fly", 1.0),
            (7L, "was mapreduce", 0.0),
            (8L, "e spark program", 1.0),
            (9L, "a e c l", 0.0),
            (10L, "spark compile", 1.0),
            (11L, "hadoop software", 0.0)
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
                .setInputCol(tokenizer.getOutputCol)
                .setOutputCol("features")
        val lr: LogisticRegression = new LogisticRegression()
                .setMaxIter(10)
        val pipeline: Pipeline = new Pipeline()
                .setStages(Array(tokenizer, hashingTF, lr))

        // Build a param grid for learning models
        val paramGrid: Array[ParamMap] = new ParamGridBuilder()
                .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
                .addGrid(lr.regParam, Array(0.1, 0.01))
                .build()

        // Pipeline used as estimator for each combination of paramGrid
        // Cross-validate on numFolds using a BinaryClassificationEvaluator (areaUnderROC)
        val cv: CrossValidator = new CrossValidator()
                .setEstimator(pipeline)
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(5)
                .setEvaluator(new BinaryClassificationEvaluator)

        // Run cross-validation, and choose the best set of parameters.
        val cvModel: CrossValidatorModel = cv.fit(training)

        // Make predictions on test documents.
        val predictions: DataFrame = cvModel.transform(testing)
        predictions.select("id", "text", "probability", "prediction").show()

    }
}
