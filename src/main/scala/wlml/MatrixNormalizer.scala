package wlml
import breeze.linalg._

trait MatrixNormalizer {
  def featureNormalizer(matrix: CSCMatrix[Double]): (CSCMatrix[Double], List[Double]) = {
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

  def weightsDenormalizer(normalizedWeights: SparseVector[Double], normalizerList: List[Double]): SparseVector[Double] = {
    val denormalizedWeights = normalizedWeights.copy
    val ncols = denormalizedWeights.length

    for (i <- 0 until ncols) {
      denormalizedWeights(i) = denormalizedWeights(i) / normalizerList(i)
    }

    denormalizedWeights
  }
  
}