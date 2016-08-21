package wlml.classification

import breeze.linalg._
import wlml.ml._
import wlml.matrixutils._

class LogisticRegressor(tolerance: Double, maxiter: Int, stepSize: Double,
    l1_penalty: Double, l2_penalty: Double) extends MatrixNormalizer with Optimizer {

  private val parameters = Parameters(tolerance, maxiter, stepSize, l1_penalty, l2_penalty)

  private def costGrad(feat: CSCMatrix[Double], out: SparseVector[Double],
    wei: SparseVector[Double], pa: Parameters): (Double, SparseVector[Double]) = {

    def sigmoid(w: SparseVector[Double]): SparseVector[Double] = {
      w.map(x => if (x > 0) 1.0 / (1.0 + Math.exp(-x)) else Math.exp(x) / (1.0 + Math.exp(x)))
    }

    def logsig(w: SparseVector[Double]): SparseVector[Double] = {
      w.map(x => Math.log(x))
    }

    val m = feat.rows.toDouble
    val theta = wei.copy
    theta(0) = 0.0

    val hx = sigmoid(feat * wei)
    val j0 = -sum((logsig(hx) :* out) + (logsig(-hx + 1.0) :* (-out + 1.0))) / m
    val cost = j0 + (((wei.t * wei) * 0.5 * pa.l2_penalty) / m)
    val gradient = ((feat.t * (hx - out)) + (theta * pa.l2_penalty)) / m

    (cost, gradient)
  }

  def buildSparseModel(trainData: DataMatrix): SparseVector[Double] = {
    val (featMatrix, normalizerRanges, normalizerMeans) = featureNormalizer(trainData.features)
    val initialWeights = SparseVector(Array.fill(trainData.features.cols)(1.0))
    gdSolver(initialWeights, featMatrix, trainData.outputs, this.parameters, 0)(costGrad)
  }

  def buildLBFGSModel(trainData: DataMatrix): SparseVector[Double] = {
    val (featMatrix, normalizerRanges, normalizerMeans) = featureNormalizer(trainData.features)
    val initialWeights = SparseVector(Array.fill(trainData.features.cols)(1.0))
    lbfgsSolver(initialWeights, featMatrix, trainData.outputs, this.parameters)(costGrad)
  }

}

object LogisticRegressor {
  def apply(tolerance: Double, maxiter: Int, stepSize: Double, l1_penalty: Double, l2_penalty: Double) =
    new LogisticRegressor(tolerance, maxiter, stepSize, l1_penalty, l2_penalty)
}
