package wlml.linear

import breeze.linalg._
import wlml.matrixutils._

object Regressor {

  class Regression(tolerance: Double, maxiter: Int, stepSize: Double,
      l1_penalty: Double, l2_penalty: Double) extends MatrixNormalizer {

    protected case class Parameters(tolerance: Double, maxiter: Int, stepSize: Double,
      l1_penalty: Double, l2_penalty: Double, earlierErrors: Double)

    type CostResult = Tuple3[SparseVector[Double], Double, SparseVector[Double]]

    def params = Parameters(tolerance, maxiter, stepSize, l1_penalty, l2_penalty, Double.PositiveInfinity)

    def buildModel(featureMatrix: CSCMatrix[Double], outputs: SparseVector[Double],
      parameters: Parameters): SparseVector[Double] = {
      val (featMatrix, normalizerList) = featureNormalizer(featureMatrix)
      val initialWeights = SparseVector(Array.fill(featMatrix.cols)(1.0))
      val coefficient = gradientdescent(initialWeights, featMatrix, outputs, parameters, 0)
      weightsDenormalizer(coefficient, normalizerList)
    }

    private def gradientdescent(weights: SparseVector[Double], features: CSCMatrix[Double],
      output: SparseVector[Double], parameters: Parameters, iter: Int): SparseVector[Double] = {

      def regulizer(m: Double, weights: SparseVector[Double],
        penalty: Double, is_l2: Boolean): Double = {
        val target_weights = weights(1 to (weights.length - 1))
        (penalty / (1.0 * m)) * (target_weights.t * target_weights)
      }

      def getCost(outputs: SparseVector[Double], weights: SparseVector[Double],
        features: CSCMatrix[Double], params: Parameters): CostResult = {
        val m: Double = features.rows
        val errors: SparseVector[Double] = (features * weights) - outputs
        val regul_term: Double = regulizer(m, weights, params.l2_penalty, true)
        val cost: Double = (1.0 / m) * ((errors.t * errors) + regul_term)
        val theta: SparseVector[Double] = weights.copy
        theta(0) = 0.0

        val gradient: SparseVector[Double] = ((features.t * errors) + (theta * params.l2_penalty)) * (m / 2.0)
        weights :-= gradient * params.stepSize

        (weights, cost, errors)
      }

      def checkNecessity(params: Parameters, errorNorm: Double, iter: Int): Boolean = {
        if (((params.earlierErrors - errorNorm) < params.tolerance) || (iter >= params.maxiter)) false
        else true
      }

      val (weightNext: SparseVector[Double], cost: Double, errors: SparseVector[Double]) =
        getCost(output, weights, features, parameters)

      val relative_r = norm(errors)

      println(relative_r, iter)
      if (checkNecessity(parameters, relative_r, iter)) {
        val newParameters = Parameters(parameters.tolerance, parameters.maxiter,
          parameters.stepSize, parameters.l1_penalty, parameters.l2_penalty, norm(errors))
        gradientdescent(weightNext, features, output, newParameters, iter + 1)
      } else weightNext

    }

  }
}