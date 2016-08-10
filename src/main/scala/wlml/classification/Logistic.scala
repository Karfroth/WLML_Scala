package wlml.classification

import breeze.linalg._
import wlml.ml._
import wlml.matrixutils._

object Logistic {
  
  class LogisticRegression(tolerance: Double, maxiter: Int, stepSize: Double,
      l1_penalty: Double, l2_penalty: Double) extends MatrixNormalizer with Optimizer {
    
    val parameters = Parameters(tolerance, maxiter, stepSize, l1_penalty, l2_penalty, Double.PositiveInfinity)
    
    private def sigmoid(feat: CSCMatrix[Double], coee: SparseVector[Double]):SparseVector[Double] = {
      val z = (feat * coee)
      (breeze.numerics.exp(-z) + 1.0).map( x => 1.0 / x )
    }
    
    def loglikelihoodTest(feat: CSCMatrix[Double], out: SparseVector[Double], 
        wei: SparseVector[Double], l2: Double, earlier: Double): (Double, Double) = {
      val scores = feat * wei
      val theta = wei.copy
      theta(0) = 0.0
      val lp = sum((out - 1.0) :* scores - breeze.numerics.log(breeze.numerics.exp(-scores) + 1.0)) - (l2 * (theta.t * theta))
      (lp, (earlier - lp) / earlier)
    }

    def buildSparseModel(featureMatrix: CSCMatrix[Double], outputs: SparseVector[Double],
      parameters: Parameters): SparseVector[Double] = {
      val (featMatrix, normalizerRanges, normalizerMeans) = featureNormalizer(featureMatrix)
      val initialWeights = SparseVector(Array.fill(featMatrix.cols)(1.0))
      gradientdescent(initialWeights, featMatrix, outputs, parameters, 0)(sigmoid)(loglikelihoodTest)
    }

  }
  
}