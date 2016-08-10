package wlml.ml

import breeze.linalg._
import breeze.linalg.qr.QR

trait Optimizer extends wlml.ml.Opts {

  // Start Gradient Descent for Sparse Types

  def gradientdescent(weights: SparseVector[Double], features: CSCMatrix[Double],
    outputs: SparseVector[Double], params: Parameters, iter: Int)
  (predFunc: (CSCMatrix[Double], SparseVector[Double]) => SparseVector[Double]): SparseVector[Double] = {

    def regulizer(m: Double, weights: SparseVector[Double],
      penalty: Double, is_l2: Boolean): Double = {
      val target_weights = weights(1 to (weights.length - 1))
      (penalty / (1.0 * m)) * (target_weights.t * target_weights)
    }
    
    def checkNecessity(ps: Parameters, errorNorm: Double, iter: Int): Boolean = {
      if ((((ps.earlierErrors - errorNorm) < ps.tolerance) && (iter != 0)) || (iter >= ps.maxiter)) false
      else true
    }

    // Calculate cost
    val m: Double = features.rows
    val errors: SparseVector[Double] = predFunc(features, weights) - outputs
    val regul_term: Double = regulizer(m, weights, params.l2_penalty, true)
    val cost: Double = (1.0 / m) * ((errors.t * errors) + regul_term)
    // theta : Regularizer term
    val theta: SparseVector[Double] = weights.copy
    theta(0) = 0.0
    val gradient: SparseVector[Double] = ((features.t * errors) + (theta * params.l2_penalty)) * (m / 2.0)
    weights :-= gradient * params.stepSize

    if (checkNecessity(params, norm(errors), iter)) {
      val paramsNext = params.copy(earlierErrors = norm(errors))
      gradientdescent(weights, features, outputs, paramsNext, iter + 1)(predFunc)
    } else weights

  }

  // End of Gradient Descent for Sparse Types

  // Start of QR Decomposition for Dense Types

  def qrdecomposition(features: DenseMatrix[Double], outputs: DenseVector[Double],
    params: Parameters): DenseVector[Double] = {

    val n: Int = features.cols

    val feats = DenseMatrix.vertcat(features, (DenseMatrix.eye[Double](n) * params.l2_penalty))
    val outs = DenseVector.vertcat(outputs, DenseVector.zeros[Double](n))

    val QR(qValue, rValue) = qr.reduced(feats)
    inv(rValue) * (qValue.t * outs)

  }

}