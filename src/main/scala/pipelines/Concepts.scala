package pipelines

import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.DataFrame
import utils.{Constants, Datasets}

object Concepts {

    def main(args: Array[String]) {

        // Datasets
        import Constants.sqlc.implicits._
        val Array(training, testing) = Datasets.binaryLabeledPoints.toDF.randomSplit(Array(0.8, 0.2))

        // Estimator: LogisticRegression
        val lr: LogisticRegression = new LogisticRegression()

        // Params
        val paramMap: ParamMap = ParamMap(
            lr.maxIter -> 30,
            lr.regParam -> 0.05,
            lr.threshold -> 0.45,
            lr.predictionCol -> "prediction"
        )

        // Learn the model (transformer)
        val model: LogisticRegressionModel = lr.fit(training, paramMap)
        println("Model was fit using parameters: " + model.parent.extractParamMap)

        // Get predictions
        val predictions: DataFrame = model.transform(testing)

        predictions.select("features", "label", "prediction").show
    }

}
