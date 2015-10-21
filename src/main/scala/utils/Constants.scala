package utils

import org.apache.spark.mllib.linalg.{Matrices, Matrix, Vectors}
import org.apache.spark.mllib.random.RandomRDDs
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.Second
import org.apache.spark.streaming.{Seconds, Duration, StreamingContext}
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.Random

object Constants {

    val sparkHome = "/opt/spark"
    private val master = "local[2]"

    // Getting contexts
    val conf: SparkConf = new SparkConf().setAppName("Spark MLlib examples").setMaster(Constants.master)
    val sc: SparkContext = new SparkContext(conf)
    val ssc: StreamingContext = new StreamingContext(sc, Seconds(20))

    // A few convenient RDDs
    private val vectors = Array(
        Vectors.dense(1.0, 2.0, 3.0),
        Vectors.dense(1.0, 0.3, 2.5),
        Vectors.dense(0.0, 1.2, 1.2),
        Vectors.dense(3.0, 3.5, 3.0)
    )
    val vectorRDD = sc.parallelize(vectors)

    val x = sc.parallelize(Array(1, 2, 3)).map(_.toDouble)
    val y = sc.parallelize(Array(2, 3, 1)).map(_.toDouble)

    val z = sc.parallelize(Array(-1, -0.5, -0.1, -0.0001, 0, 0, 0.2, 0.23, 1))

    // generate pairs
    private val pairs = List("a", "b").flatMap { k =>
        List.fill(10)(Random.nextDouble()).map((k, _))
    }
    val pairRDD = sc.parallelize(pairs)

    val contingencyMatrix: Matrix = Matrices.dense(2,2, Array(43.0, 44.0, 9.0, 4.0))

    private val labeledPoints = Array(
        LabeledPoint(1.0, Vectors.dense(1.0, 2.0, 3.0)),
        LabeledPoint(0.0, Vectors.dense(1.0, 2.0, 5.0))
    )
    val labeledPointRDD = sc.parallelize(labeledPoints)

    val normalRDD: RDD[Double] = RandomRDDs.normalRDD(Constants.sc, 100L, 2)
}
