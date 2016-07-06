package main.scala.wlml
import breeze.linalg._

class Regression(ys: Array[Double], xs: Array[Array[Double]]) {
  private def features: DenseMatrix[Double] = {
    DenseMatrix.horzcat(
      DenseMatrix(List.fill(xs.length)(1.0): _*),
      DenseMatrix(xs: _*))
  }
  private def outputs: DenseMatrix[Double] = DenseMatrix(ys: _*)
  def build: DenseMatrix[Double] = {

    def gradientdescent(weights: DenseVector[Double], features: DenseMatrix[Double],
      output: DenseMatrix[Double], stepsize: Double, tol: Double,
      iter: Int, maxiter: Int) = {
      def checkcvg(tol: Double, maxiter: Int, iter: Int, relative: Double,
        newrelative: Double): Boolean = {
        if (iter < maxiter) (true)
        else if ((relative - newrelative) > tol) (true)
        else false
      }
    }
    this.features
  }
  def predict: DenseMatrix[Double] = this.features

}