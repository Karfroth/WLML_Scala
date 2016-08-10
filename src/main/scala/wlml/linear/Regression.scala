package wlml.linear

import breeze.linalg._
import wlml.matrixutils._
import wlml.ml._

object Regressor {

  class Regression(tolerance: Double, maxiter: Int, stepSize: Double,
      l1_penalty: Double, l2_penalty: Double) extends MatrixNormalizer with Optimizer {

    val parameters = Parameters(tolerance, maxiter, stepSize, l1_penalty, l2_penalty, Double.PositiveInfinity)

    private def coeeDotFeat(feat: CSCMatrix[Double], coee: SparseVector[Double]): SparseVector[Double] = {
      feat * coee
    }
    private def distanceChecker(feat: CSCMatrix[Double], out: SparseVector[Double],
      wei: SparseVector[Double], l2: Double, earlier: Double): (Double, Double) = {
      val error = (norm((feat * wei) - out) / norm(out))
      val check = earlier - error
      (error, check)
    }

    def buildSparseModel(featureMatrix: CSCMatrix[Double], outputs: SparseVector[Double],
      parameters: Parameters): SparseVector[Double] = {
      val (featMatrix, normalizerRanges, normalizerMeans) = featureNormalizer(featureMatrix)
      val initialWeights = SparseVector(Array.fill(featMatrix.cols)(1.0))
      gradientdescent(initialWeights, featMatrix, outputs, parameters, 0)(coeeDotFeat)(distanceChecker)
    }

    def buildQRModel(featureMatrix: DenseMatrix[Double], outputs: DenseVector[Double],
      parameters: Parameters): DenseVector[Double] = {
      val (featMatrix, normalizerRanges, normalizerMeans) = featureNormalizer(featureMatrix)
      val initialWeights = DenseVector(Array.fill(featMatrix.cols)(1.0))
      qrdecomposition(featMatrix, outputs, parameters)
    }

  }

}