package wlml.matrixutils

import breeze.linalg._

trait MatrixNormalizer {

  implicit object SparseType
  implicit object DenseType
  
  // Sparse Type Normalizer and Denormalizer
  def featureNormalizer(matrix: CSCMatrix[Double])(implicit s: SparseType.type): (CSCMatrix[Double], List[Double]) = {
    val normalizedMatrix = matrix.copy
    val nrows = normalizedMatrix.rows
    val ncols = normalizedMatrix.cols
    val normalizeNums = for (i <- 0 until ncols) yield {
      if (i == 0) 1.0
      else max(normalizedMatrix(0 until nrows, i)) - min(normalizedMatrix(0 until nrows, i))
    }

    for (i <- 0 until ncols; j <- 0 until nrows) {
      normalizedMatrix(j, i) = normalizedMatrix(j, i) / normalizeNums(i)
    }

    (normalizedMatrix, normalizeNums.toList)
  }
  
  // Dense Type Normalizer and Denormalizer

  def weightsDenormalizer(normalizedWeights: SparseVector[Double], 
      normalizerList: List[Double])(implicit s: SparseType.type): SparseVector[Double] = {
    val denormalizedWeights = normalizedWeights.copy
    val ncols = denormalizedWeights.length

    for (i <- 0 until ncols) {
      denormalizedWeights(i) = denormalizedWeights(i) / normalizerList(i)
    }

    denormalizedWeights
  }
  
    def featureNormalizer(matrix: DenseMatrix[Double])(implicit d: DenseType.type): (DenseMatrix[Double], List[Double]) = {
    val normalizedMatrix = matrix.copy
    val nrows = normalizedMatrix.rows
    val ncols = normalizedMatrix.cols
    val normalizeNums = for (i <- 0 until ncols) yield {
      if (i == 0) 1.0
      else max(normalizedMatrix(0 until nrows, i)) - min(normalizedMatrix(0 until nrows, i))
    }

    for (i <- 0 until ncols; j <- 0 until nrows) {
      normalizedMatrix(j, i) = normalizedMatrix(j, i) / normalizeNums(i)
    }

    (normalizedMatrix, normalizeNums.toList)
  }

  def weightsDenormalizer(normalizedWeights: DenseVector[Double], 
      normalizerList: List[Double])(implicit d: DenseType.type): DenseVector[Double] = {
    val denormalizedWeights = normalizedWeights.copy
    val ncols = denormalizedWeights.length

    for (i <- 0 until ncols) {
      denormalizedWeights(i) = denormalizedWeights(i) / normalizerList(i)
    }

    denormalizedWeights
  }

}