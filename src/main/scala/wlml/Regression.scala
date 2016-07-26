package main.scala.wlml
import breeze.linalg._

object Regressor {

  case class Parameters(stepSize: Double, tolerance: Double, maxiter: Int, 
      l1_penalty: Double, l2_penalty:Double)
  
  class Regression(ys: Array[Double], xs: Array[Array[Double]]) {
    
    type CostResult = Tuple3[SparseVector[Double], Double, SparseVector[Double]]
    
    private def features: CSCMatrix[Double] = {
      CSCMatrix(xs.map( row => 1.0 +: row ):_*)
    }
    private def outputs: SparseVector[Double] = SparseVector(ys)

    def gradientdescent(weights: SparseVector[Double], features: CSCMatrix[Double],
      output: SparseVector[Double], parameters: Parameters, iter: Int): SparseVector[Double] = {

      def regulizer(m: Double, weights: SparseVector[Double], 
          penalty: Double, is_l2: Boolean) : Double = {
        val target_weights = weights(1 to ( weights.length -1 ))
        (penalty / (1.0 * m)) * (target_weights.t * target_weights)
      }
      
      def getCost(outputs: SparseVector[Double], weights: SparseVector[Double],
        features: CSCMatrix[Double], param: Parameters): CostResult = {
        val m: Double = features.rows
        val errors = (features * weights) - outputs
        val regul_term = regulizer(m, weights, param.l2_penalty, true)
        val cost = (1.0 / m) * ((errors.t * errors) + regul_term)
        val theta = weights.copy
        theta(0) = 0.0

        val gradient: SparseVector[Double] = ((features.t * errors) + (theta * param.l2_penalty)) * (m / 2.0)
        weights :-= gradient * param.stepSize
        
        (weights, cost, errors)
      }

      def checkNecessity(distance: Double, tol: Double, iter: Int, maxiter: Int): Boolean = {
        if ((distance < tol) || (iter >= maxiter)) false
        else true
      }

      val (weightNext: SparseVector[Double], cost: Double, errors: SparseVector[Double]) = getCost(output, weights, features, parameters)

      val relative_r = norm(errors) / norm(output)

      if (checkNecessity(relative_r, parameters.tolerance, iter, parameters.maxiter)) {
        gradientdescent(weightNext, features, output, parameters, iter + 1)
      } else weightNext

    }

  }
}