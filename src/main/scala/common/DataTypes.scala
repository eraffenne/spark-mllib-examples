package common

import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.mllib.linalg.{Matrices, Matrix, Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import utils.Constants

object DataTypes {

    def main(args: Array[String]) {

        // --- Local vectors
        val denseVector: Vector = Vectors.dense(1.0, 0.0, 3.0)
        val sparseVector: Vector = Vectors.sparse(3, Array(0,2), Array(1.0, 3.0))

        // --- Labeled Point
        val positivePoint: LabeledPoint = LabeledPoint( 1.0, denseVector)
        val negativePoint: LabeledPoint = LabeledPoint(0.0, sparseVector)

        // Local matrices: stored by column
        val denseMatrix: Matrix = Matrices.dense(2,2, Array(1.0, 0.0, 3.0, 4.0))
        val sparseMatrix: Matrix = Matrices.sparse(2, 2, Array(0, 1, 3), Array(0, 0, 1), Array(1.0, 3.0, 4.0))

        // --- Distributed matrices
        val rows: RDD[Vector] = Constants.sc.parallelize( Array(denseVector, sparseVector) )
        val rowMatrix: RowMatrix = new RowMatrix(rows)

        val indexedRows: RDD[IndexedRow] = Constants.sc.parallelize( Array( new IndexedRow(0, denseVector), new IndexedRow(1, sparseVector)) )
        val indexedRowMatrix: IndexedRowMatrix = new IndexedRowMatrix(indexedRows)

        val entries: RDD[MatrixEntry] = Constants.sc.parallelize( Array(new MatrixEntry(0, 0, 1.0), new MatrixEntry(0, 1, 3.0)) )
        val coordinateMatrix: CoordinateMatrix = new CoordinateMatrix(entries)

        val blockMatrixFromIndexed: BlockMatrix = indexedRowMatrix.toBlockMatrix()
        val blockMatrixFromCoordinate: BlockMatrix = coordinateMatrix.toBlockMatrix(1,1)
    }
}
