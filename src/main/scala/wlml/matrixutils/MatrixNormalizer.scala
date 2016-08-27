package wlml.matrixutils

import breeze.linalg._

trait MatrixNormalizer {

  implicit object SparseType
  implicit object DenseType

  case class NormalizeParam(var ranges: SparseVector[Double], var means: SparseVector[Double])

  def featureNormalizer(matrix: CSCMatrix[Double])(implicit s: SparseType.type): (CSCMatrix[Double], SparseVector[Double], SparseVector[Double]) = {
    val normalizedMatrix = matrix.copy
    val nrows = normalizedMatrix.rows
    val ncols = normalizedMatrix.cols
    val normalizeRange = for (i <- 0 until ncols) yield {
      if (i == 0) 1.0
      else max(normalizedMatrix(0 until nrows, i)) - min(normalizedMatrix(0 until nrows, i))
    }

    val normalizeMeans = for (i <- 0 until ncols) yield {
      if (i == 0) 0.0
      else sum(normalizedMatrix(0 until nrows, i)) / nrows
    }

    for (i <- 0 until ncols; j <- 0 until nrows) {
      normalizedMatrix(j, i) = (normalizedMatrix(j, i) - normalizeMeans(i)) / normalizeRange(i)
    }

    (normalizedMatrix, SparseVector(normalizeRange:_*), SparseVector(normalizeMeans:_*))
  }

  def predictionNormalizer(normalizer: NormalizeParam, targetFeatures: CSCMatrix[Double]): CSCMatrix[Double] = {
    val normalizedMatrix = targetFeatures.copy
    val nrows = normalizedMatrix.rows
    val ncols = normalizedMatrix.cols

    for (i <- 0 until ncols; j <- 0 until nrows) {
      normalizedMatrix(j, i) = (normalizedMatrix(j, i) - normalizer.means(i)) / normalizer.ranges(i)
    }

    normalizedMatrix
  }
  /*
  def featureNormalizer(matrix: DenseMatrix[Double])(implicit d: DenseType.type): (DenseMatrix[Double], List[Double], List[Double]) = {
    val normalizedMatrix = matrix.copy
    val nrows = normalizedMatrix.rows
    val ncols = normalizedMatrix.cols
    val normalizeRange = for (i <- 0 until ncols) yield {
      if (i == 0) 1.0
      else max(normalizedMatrix(0 until nrows, i)) - min(normalizedMatrix(0 until nrows, i))
    }

    val normalizeMeans = for (i <- 0 until ncols) yield {
      if (i == 0) 0.0
      else sum(normalizedMatrix(0 until nrows, i)) / nrows
    }

    for (i <- 0 until ncols; j <- 0 until nrows) {
      normalizedMatrix(j, i) = (normalizedMatrix(j, i) - normalizeMeans(i)) / normalizeRange(i)
    }

    (normalizedMatrix, normalizeRange.toList, normalizeMeans.toList)
  }
  */



}