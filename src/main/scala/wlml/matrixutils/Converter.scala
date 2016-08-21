package wlml.matrixutils

import breeze.linalg._

class DataMatrix(featArray: Array[Array[Double]], outArray: Array[Double]) extends MatrixNormalizer {

    /*
     *** Dense Type is not supported yet!!! ***
    def denseDataMatrix(xs: Array[Array[Double]], ys: Array[Double]): (DenseMatrix[Double], DenseVector[Double]) = {
      (DenseMatrix(xs.map( row => 1.0 +: row ):_*), DenseVector(ys))
    }
    */
  
  private def sparseDataMatrix(xs: Array[Array[Double]], ys: Array[Double]): (CSCMatrix[Double], SparseVector[Double]) = {
    (CSCMatrix(xs.map(row => 1.0 +: row): _*), SparseVector(ys))
  }

  val (features, outputs) = sparseDataMatrix(featArray, outArray)

}

object DataMatrix {

  def apply(featArray: Array[Array[Double]], outArray: Array[Double]) =
    new DataMatrix(featArray, outArray)

  def sparseToArray(sparse: SparseVector[Double]) =
    sparse.toArray

}

