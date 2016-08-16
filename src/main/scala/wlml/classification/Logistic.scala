package wlml.classification

import breeze.linalg._
import wlml.ml._
import wlml.matrixutils._

object Logistic {

  class LogisticRegression(tolerance: Double, maxiter: Int, stepSize: Double,
      l1_penalty: Double, l2_penalty: Double) extends MatrixNormalizer with Optimizer {

    val parameters = Parameters(tolerance, maxiter, stepSize, l1_penalty, l2_penalty, Double.PositiveInfinity)

    private def sigmoid(feat: CSCMatrix[Double], coee: SparseVector[Double]): SparseVector[Double] = {
      val z = (feat * coee)
      z.map(x => if (x > 0) (1.0 / (1.0 + Math.exp(-x))) else (Math.exp(x) / (1.0 + Math.exp(x))))
    }

    def loglikelihoodTest(feat: CSCMatrix[Double], out: SparseVector[Double],
      wei: SparseVector[Double], l2: Double, earlier: Double): (Double, Double) = {
      val scores = feat * wei
      val theta = wei.copy
      theta(0) = 0.0
      val lp = sum((out - 1.0) :* scores - breeze.numerics.log(breeze.numerics.exp(-scores) + 1.0)) - (l2 * (theta.t * theta))
      (lp, (earlier - lp) / earlier)
    }

    private def costGrad(feat: CSCMatrix[Double], out: SparseVector[Double],
      wei: SparseVector[Double], pa: Parameters): (Double, SparseVector[Double]) = {

      def sigmoid(w: SparseVector[Double]): SparseVector[Double] = {
        w.map(x => if (x > 0) 1.0 / (1.0 + Math.exp(-x)) else Math.exp(x) / (1.0 + Math.exp(x)))
      }

      def logsig(w: SparseVector[Double]): SparseVector[Double] = {
        w.map(x => Math.log(x))
      }
      
      val nextWeights = wei.copy

      val theta = wei.copy
      theta(0) = 0.0

      val hx = sigmoid(feat * wei)
      val m = feat.rows.toDouble
      val j0 = -sum((logsig(hx) :* out) + (logsig(-hx + 1.0) :* (-out + 1.0))) / m
      val cost = j0 + (((wei.t * wei) * 0.5 * pa.l2_penalty) / m)
      val gradient = ((feat.t * (hx - out)) + (theta * pa.l2_penalty)) / m

      (cost, gradient)
    }

    def buildSparseModel(featureMatrix: CSCMatrix[Double], outputs: SparseVector[Double],
      parameters: Parameters): SparseVector[Double] = {
      val (featMatrix, normalizerRanges, normalizerMeans) = featureNormalizer(featureMatrix)
      val initialWeights = SparseVector(Array.fill(featMatrix.cols)(1.0))
      gdSolver(initialWeights, featMatrix, outputs, parameters, 0)(costGrad)
    }
    
    def buildLBFGSModel(featureMatrix: CSCMatrix[Double], outputs: SparseVector[Double],
      parameters: Parameters): SparseVector[Double] = {
      val (featMatrix, normalizerRanges, normalizerMeans) = featureNormalizer(featureMatrix)
      val initialWeights = SparseVector(Array.fill(featMatrix.cols)(1.0))
      lbfgsSolver(initialWeights, featMatrix, outputs, parameters)(costGrad)
    }

  }

}
