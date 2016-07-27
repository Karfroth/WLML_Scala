package wlml
import breeze.linalg._

object Regressor {

  case class Parameters(stepSize: Double, l1_penalty: Double, l2_penalty:Double, earlierErrors: Double = Double.PositiveInfinity)
  case class StopConditions(tolerance: Double, maxiter: Int)
      
  class Regression(ys: Array[Double], xs: Array[Array[Double]]) extends MatrixNormalizer{
    
    type CostResult = Tuple3[SparseVector[Double], Double, SparseVector[Double]]
    
    private def features: CSCMatrix[Double] = {
      CSCMatrix(xs.map( row => 1.0 +: row ):_*)
    }
    private def outputs: SparseVector[Double] = SparseVector(ys)

    def buildModel(featureMatrix: CSCMatrix[Double], outputs: SparseVector[Double], 
        parameters: Parameters, stopConditions: StopConditions): SparseVector[Double] = {
      val (featMatrix, normalizerList) = featureNormalizer(featureMatrix)
      val initialWeights = SparseVector(Array.fill(featMatrix.cols)(1.0))
      val coefficient = gradientdescent(initialWeights, featMatrix, outputs, stopConditions, parameters, 0)
      weightsDenormalizer(coefficient, normalizerList)
    }
    
    def gradientdescent(weights: SparseVector[Double], features: CSCMatrix[Double],
      output: SparseVector[Double], stopCons:StopConditions, 
      parameters: Parameters, iter: Int): SparseVector[Double] = {

      def regulizer(m: Double, weights: SparseVector[Double], 
          penalty: Double, is_l2: Boolean) : Double = {
        val target_weights = weights(1 to ( weights.length -1 ))
        (penalty / (1.0 * m)) * (target_weights.t * target_weights)
      }
      
      def getCost(outputs: SparseVector[Double], weights: SparseVector[Double],
        features: CSCMatrix[Double], params: Parameters): CostResult = {
        val m: Double = features.rows
        val errors:SparseVector[Double] = (features * weights) - outputs
        val regul_term:Double = regulizer(m, weights, params.l2_penalty, true)
        val cost:Double = (1.0 / m) * ((errors.t * errors) + regul_term)
        val theta:SparseVector[Double] = weights.copy
        theta(0) = 0.0

        val gradient: SparseVector[Double] = ((features.t * errors) + (theta * params.l2_penalty)) * (m / 2.0)
        weights :-= gradient * params.stepSize
        
        (weights, cost, errors)
      }

      def checkNecessity(errorNorm: Double, earlierErrorNorm:Double, iter: Int, stCons: StopConditions): Boolean = {
        if (((earlierErrorNorm - errorNorm) < stCons.tolerance) || (iter >= stCons.maxiter)) false
        else true
      }

      val (weightNext: SparseVector[Double], cost: Double, errors: SparseVector[Double]) = getCost(output, weights, features, parameters)

      val relative_r = norm(errors)
      
      println(relative_r, iter)
      if (checkNecessity(relative_r, parameters.earlierErrors, iter, stopCons)) {
        val newParameters = Parameters(parameters.stepSize, parameters.l1_penalty, parameters.l2_penalty, norm(errors))
        gradientdescent(weightNext, features, output, stopCons, newParameters, iter + 1)
      } else weightNext

    }

  }
}