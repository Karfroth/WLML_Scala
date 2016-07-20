package main.scala.wlml
import breeze.linalg._

class Regression(ys: Array[Double], xs: Array[Array[Double]]) {
  private def features: DenseMatrix[Double] = {
    DenseMatrix.horzcat(
      DenseMatrix(List.fill(xs.length)(1.0): _*),
      DenseMatrix(xs: _*))
  }
  private def outputs: DenseVector[Double] = DenseVector(ys)

  def gradientdescent(weights: DenseVector[Double], features: DenseMatrix[Double],
    output: DenseVector[Double], step_size: Double, tol: Double, 
    iter: Int, maxiter: Int):DenseVector[Double] = {

    def getCost(output: DenseVector[Double], weights: DenseVector[Double],
      features: DenseMatrix[Double], step_size: Double): Tuple3[DenseVector[Double], Double, DenseVector[Double]] = {
      val m: Double = features.rows
      val errors = (features * weights) - output
      val cost = (1.0 / m) * (errors.t * errors)
      val gradient = (features.t * errors) * (m / 2.0)
      weights :-= gradient * step_size
      
      (weights, cost, errors)
    }
    
    def checkNecessity(distance: Double, tol: Double, iter: Int, maxiter: Int): Boolean = {
      if (distance > tol) true
      else if (iter < maxiter) true
      else false
    }
    
    //result._1 = weights
    //result._2 = cost
    //result._3 = errors
    val (weightNext: DenseVector[Double], cost:Double, errors: DenseVector[Double]) = getCost(output, weights, features, step_size)
    
    val relative_r = norm(errors) / norm(output)
      
    if (checkNecessity(relative_r, tol, iter, maxiter)) {
      gradientdescent(weightNext, features, output, step_size, tol, iter+1, maxiter)
    }
    else weights,

  }

}