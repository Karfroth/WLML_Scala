package main.scala.wlml
import breeze.linalg._

object Regressor {

  case class Parameters(stepSize: Double, tolerance: Double, maxiter: Int)
  
  class Regression(ys: Array[Double], xs: Array[Array[Double]]) {
    private def features: CSCMatrix[Double] = {
      CSCMatrix(xs.map( row => 1.0 +: row ):_*)
    }
    private def outputs: SparseVector[Double] = SparseVector(ys)

    def gradientdescent(weights: SparseVector[Double], features: CSCMatrix[Double],
      output: SparseVector[Double], parameters: Parameters, iter: Int): SparseVector[Double] = {

      def getCost(output: SparseVector[Double], weights: SparseVector[Double],
        features: CSCMatrix[Double], param: Parameters): Tuple3[SparseVector[Double], Double, SparseVector[Double]] = {
        val m: Double = features.rows
        val errors = (features * weights) - output
        val cost = (1.0 / m) * (errors.t * errors)
        val gradient = (features.t * errors) * (m / 2.0)
        weights :-= gradient * param.stepSize

        (weights, cost, errors)
      }

      def checkNecessity(distance: Double, tol: Double, iter: Int, maxiter: Int): Boolean = {
        if ((distance < tol) || (iter >= maxiter)) false
        else true
      }

      //result._1 = weights
      //result._2 = cost
      //result._3 = errors
      val (weightNext: SparseVector[Double], cost: Double, errors: SparseVector[Double]) = getCost(output, weights, features, parameters)

      val relative_r = norm(errors) / norm(output)

      if (checkNecessity(relative_r, parameters.tolerance, iter, parameters.maxiter)) {
        gradientdescent(weightNext, features, output, parameters, iter + 1)
      } else weightNext

    }

  }
}