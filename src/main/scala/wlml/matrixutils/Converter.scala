package wlml.matrixutils

import breeze.linalg._

trait Converter {
    def sparseDataMatrix(xs: Array[Array[Double]], ys: Array[Double]): (CSCMatrix[Double], SparseVector[Double]) = {
      (CSCMatrix(xs.map( row => 1.0 +: row ):_*), SparseVector(ys))
    }
    def denseDataMatrix(xs: Array[Array[Double]], ys: Array[Double]): (DenseMatrix[Double], DenseVector[Double]) = {
      (DenseMatrix(xs.map( row => 1.0 +: row ):_*), DenseVector(ys))
    }
}