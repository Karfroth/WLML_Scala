package wlml.linear

import breeze.linalg._
import wlml.matrixutils._

object Regressor {
  
  case class Parameters(tolerance: Double, maxiter: Int, stepSize: Double,
      l1_penalty: Double, l2_penalty: Double, earlierErrors: Double)

  class Regression(tolerance: Double, maxiter: Int, stepSize: Double,
      l1_penalty: Double, l2_penalty: Double) extends MatrixNormalizer {

    def parameters = Parameters(tolerance, maxiter, stepSize, l1_penalty, l2_penalty, Double.PositiveInfinity)

    def buildModel(featureMatrix: CSCMatrix[Double], outputs: SparseVector[Double],
      parameters: Parameters)(implicit s: Optimizer.SparseType.type): SparseVector[Double] = {
      val (featMatrix, normalizerList) = featureNormalizer(featureMatrix)
      val initialWeights = SparseVector(Array.fill(featMatrix.cols)(1.0))
      val coefficient = Optimizer.gradientdescent(initialWeights, featMatrix, outputs, parameters, 0)
      weightsDenormalizer(coefficient, normalizerList)
    }
    
    def buildModel(featureMatrix: DenseMatrix[Double], outputs: DenseVector[Double],
      parameters: Parameters)(implicit d: Optimizer.DenseType.type): DenseVector[Double] = {
      val (featMatrix, normalizerList) = featureNormalizer(featureMatrix)
      val initialWeights = DenseVector(Array.fill(featMatrix.cols)(1.0))
      val coefficient = Optimizer.gradientdescent(initialWeights, featMatrix, outputs, parameters, 0)
      weightsDenormalizer(coefficient, normalizerList)
    }

  }
}