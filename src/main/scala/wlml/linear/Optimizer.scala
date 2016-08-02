package wlml.linear

import breeze.linalg._

object Optimizer {

  implicit object SparseType
  implicit object DenseType

  def gradientdescent(weights: SparseVector[Double], features: CSCMatrix[Double],
    outputs: SparseVector[Double], params: Regressor.Parameters,
    iter: Int)(implicit s: SparseType.type): SparseVector[Double] = {

    def regulizer(m: Double, weights: SparseVector[Double],
      penalty: Double, is_l2: Boolean): Double = {
      val target_weights = weights(1 to (weights.length - 1))
      (penalty / (1.0 * m)) * (target_weights.t * target_weights)
    }

    def checkNecessity(ps: Regressor.Parameters, errorNorm: Double, iter: Int): Boolean = {
      if (((ps.earlierErrors - errorNorm) < ps.tolerance) || (iter >= ps.maxiter)) false
      else true
    }

    // Calculate cost
    val m: Double = features.rows
    val errors: SparseVector[Double] = (features * weights) - outputs
    val regul_term: Double = regulizer(m, weights, params.l2_penalty, true)
    val cost: Double = (1.0 / m) * ((errors.t * errors) + regul_term)
    // theta : Regularizer term
    val theta: SparseVector[Double] = weights.copy
    theta(0) = 0.0
    val gradient: SparseVector[Double] = ((features.t * errors) + (theta * params.l2_penalty)) * (m / 2.0)
    weights :-= gradient * params.stepSize

    println(norm(errors), iter)
    if (checkNecessity(params, norm(errors), iter)) {
      params.copy(earlierErrors = norm(errors))
      gradientdescent(weights, features, outputs, params, iter + 1)(SparseType)
    } else weights

  }

  def gradientdescent(weights: DenseVector[Double], features: DenseMatrix[Double],
    outputs: DenseVector[Double], params: Regressor.Parameters,
    iter: Int)(implicit d: DenseType.type): DenseVector[Double] = {

    def regulizer(m: Double, weights: DenseVector[Double],
      penalty: Double, is_l2: Boolean): Double = {
      val target_weights = weights(1 to (weights.length - 1))
      (penalty / (1.0 * m)) * (target_weights.t * target_weights)
    }

    def checkNecessity(ps: Regressor.Parameters, errorNorm: Double, iter: Int): Boolean = {
      if (((ps.earlierErrors - errorNorm) < ps.tolerance) || (iter >= ps.maxiter)) false
      else true
    }

    // Calculate cost
    val m: Double = features.rows
    val errors: DenseVector[Double] = (features * weights) - outputs
    val regul_term: Double = regulizer(m, weights, params.l2_penalty, true)
    val cost: Double = (1.0 / m) * ((errors.t * errors) + regul_term)
    // theta : Regularizer term
    val theta: DenseVector[Double] = weights.copy
    theta(0) = 0.0
    val gradient: DenseVector[Double] = ((features.t * errors) + (theta * params.l2_penalty)) * (m / 2.0)
    weights :-= gradient * params.stepSize

    println(norm(errors), iter)
    if (checkNecessity(params, norm(errors), iter)) {
      params.copy(earlierErrors = norm(errors))
      gradientdescent(weights, features, outputs, params, iter + 1)(DenseType)
    } else weights

  }

}