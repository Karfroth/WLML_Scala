package wlml.linear

import breeze.linalg._
import wlml.matrixutils._
import wlml.ml._

class Regressor(tolerance: Double, maxiter: Int, stepSize: Double,
    l1_penalty: Double, l2_penalty: Double) extends Optimizer with MatrixNormalizer {

  private val parameters = Parameters(tolerance, maxiter, stepSize, l1_penalty, l2_penalty)

  private var coefficients:SparseVector[Double] = SparseVector()
  def getCoefficients = coefficients

  private val normalizer: NormalizeParam = NormalizeParam(ranges = SparseVector(), means = SparseVector())
  private var isNormalized: Boolean = false

  private def costGrad(feat: CSCMatrix[Double], out: SparseVector[Double],
    wei: SparseVector[Double], pa: Parameters): (Double, SparseVector[Double]) = {

    val m = feat.rows.toDouble
    val theta: SparseVector[Double] = wei.copy
    theta(0) = 0.0

    val regul_term: Double = ((wei.t * wei) * 0.5 * pa.l2_penalty) / m
    val errors: SparseVector[Double] = (feat * wei) - out
    val j0: Double = ((errors.t * errors) * 0.5) / m
    val cost = j0 + regul_term
    val gradient = ((feat.t * errors) + (theta * pa.l2_penalty)) / m

    (cost, gradient)
  }

  def buildSparseModel(trainData: DataMatrix, initialWeights: SparseVector[Double], normalize: Boolean = false): Unit = {
    assert(trainData.features.cols == initialWeights.length, "Wrong Initial Weights!")
    if (normalize) {
      val (featMatrix, normalizerRanges, normalizerMeans) = featureNormalizer(trainData.features)
      normalizer.ranges = normalizerRanges
      normalizer.means = normalizerMeans
    }
    coefficients = gdSolver(initialWeights, trainData.features, trainData.outputs, this.parameters, 0)(costGrad)
  }

  def buildLBFGSModel(trainData: DataMatrix, initialWeights: SparseVector[Double], normalize: Boolean = false): Unit = {
    assert(trainData.features.cols == initialWeights.length, "Wrong Initial Weights!")
    if (normalize) {
      val (featMatrix, normalizerRanges, normalizerMeans) = featureNormalizer(trainData.features)
      normalizer.ranges = normalizerRanges
      normalizer.means = normalizerMeans
    }
    coefficients = lbfgsSolver(initialWeights, trainData.features, trainData.outputs, this.parameters)(costGrad)
  }

  def predict(targetFeatures: CSCMatrix[Double]): SparseVector[Double] = {
    if (isNormalized) predictionNormalizer(normalizer, targetFeatures) * coefficients
    else targetFeatures * coefficients
  }


  def evaluate(targetFeatures: CSCMatrix[Double], validOutputs: SparseVector[Double], verbosa: Boolean): Double = {
    val err = predict(targetFeatures) - validOutputs
    val rmse = Math.sqrt((err.t * err) / validOutputs.length)
    if (verbosa) println(s"RMSE: ${rmse}")
    rmse
  }
  /*
  def buildQRModel(featureMatrix: DenseMatrix[Double], outputs: DenseVector[Double])(implicit parameters: Parameters): DenseVector[Double] = {
    // val (featMatrix, normalizerRanges, normalizerMeans) = featureNormalizer(featureMatrix)
    val initialWeights = DenseVector(Array.fill(featureMatrix.cols)(1.0))
    qrSolver(featureMatrix, outputs, parameters)
  }
  */

}

object Regressor {
  def apply(tolerance: Double, maxiter: Int, stepSize: Double, l1_penalty: Double, l2_penalty: Double) =
    new Regressor(tolerance, maxiter, stepSize, l1_penalty, l2_penalty)
}
