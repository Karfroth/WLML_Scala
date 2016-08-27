package wlml.classification

import breeze.linalg._
import wlml.ml._
import wlml.matrixutils._

class LogisticRegressor(tolerance: Double, maxiter: Int, stepSize: Double,
                        l1_penalty: Double, l2_penalty: Double) extends MatrixNormalizer with Optimizer {

  private val parameters = Parameters(tolerance, maxiter, stepSize, l1_penalty, l2_penalty)

  private var coefficients: SparseVector[Double] = SparseVector()
  def getCoefficients = coefficients

  private val normalizer: NormalizeParam = NormalizeParam(ranges = SparseVector(), means = SparseVector())

  private def sigmoid(w: SparseVector[Double]): SparseVector[Double] = {
    w.map(x => if (x > 0) 1.0 / (1.0 + Math.exp(-x)) else Math.exp(x) / (1.0 + Math.exp(x)))
  }

  private def logsig(w: SparseVector[Double]): SparseVector[Double] = {
    w.map(x => Math.log(x))
  }

  private def costGrad(feat: CSCMatrix[Double], out: SparseVector[Double],
                       wei: SparseVector[Double], pa: Parameters): (Double, SparseVector[Double]) = {

    val m = feat.rows.toDouble
    val theta = wei.copy
    theta(0) = 0.0

    val hx = sigmoid(feat * wei)
    val j0 = -sum((logsig(hx) :* out) + (logsig(-hx + 1.0) :* (-out + 1.0))) / m
    val cost = j0 + (((wei.t * wei) * 0.5 * pa.l2_penalty) / m)
    val gradient = ((feat.t * (hx - out)) + (theta * pa.l2_penalty)) / m

    (cost, gradient)
  }

  def buildSparseModel(trainData: DataMatrix, initialWeights: SparseVector[Double]): Unit = {
    assert(trainData.features.cols == initialWeights.length, "Wrong Initial Weights!")
    val (featMatrix, normalizerRanges, normalizerMeans) = featureNormalizer(trainData.features)
    normalizer.ranges = normalizerRanges
    normalizer.means = normalizerMeans
    coefficients = gdSolver(initialWeights, featMatrix, trainData.outputs, this.parameters, 0)(costGrad)
  }

  def buildLBFGSModel(trainData: DataMatrix, initialWeights: SparseVector[Double]): Unit = {
    assert(trainData.features.cols == initialWeights.length, "Wrong Initial Weights!")
    val (featMatrix, normalizerRanges, normalizerMeans) = featureNormalizer(trainData.features)
    normalizer.ranges = normalizerRanges
    normalizer.means = normalizerMeans
    coefficients = lbfgsSolver(initialWeights, featMatrix, trainData.outputs, this.parameters)(costGrad)
  }

  def predict(targetFeatures: CSCMatrix[Double]): SparseVector[Double] = {
    sigmoid(predictionNormalizer(normalizer, targetFeatures) * coefficients)
  }

  def evaluate(targetFeatures: CSCMatrix[Double], validOutputs: SparseVector[Double], verbosa: Boolean): Double = {
    val hx = sigmoid(targetFeatures * coefficients)
    val j0 = sum((logsig(hx) :* validOutputs) + (logsig(-hx + 1.0) :* (-validOutputs + 1.0)))
    if (verbosa) println(s"Log Likelihood: ${j0}")
    j0
  }

}

object LogisticRegressor {
  def apply(tolerance: Double, maxiter: Int, stepSize: Double, l1_penalty: Double, l2_penalty: Double) =
    new LogisticRegressor(tolerance, maxiter, stepSize, l1_penalty, l2_penalty)
}
