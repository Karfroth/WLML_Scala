package wlml.matrixutils

import breeze.linalg._

object Converter {
    def features(xs: Array[Array[Double]]): CSCMatrix[Double] = {
      CSCMatrix(xs.map( row => 1.0 +: row ):_*)
    }
    def outputs(ys: Array[Double]): SparseVector[Double] = SparseVector(ys)
}