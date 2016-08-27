package wlml.ml

import breeze.linalg._
import breeze.linalg.qr.QR
import breeze.optimize._

trait Optimizer extends wlml.ml.Opts {

  // Start Gradient Descent for Sparse Types

  def gdSolver(weights: SparseVector[Double], features: CSCMatrix[Double],
    outputs: SparseVector[Double], params: Parameters, iter: Int)
  (costFunc: (CSCMatrix[Double], SparseVector[Double],SparseVector[Double], Parameters) => (Double,SparseVector[Double])): SparseVector[Double] = {

    def checkNecessity(chVal: Double, iter: Int, ps: Parameters): Boolean = {
      if (((chVal < ps.tolerance) && (iter != 0)) || (iter >= ps.maxiter)) false
      else true
    }
    
    def descentAmountChecker(cost: Double, pa: Parameters): Double = {
      (pa.earlierCost - cost) / pa.earlierCost
    }
    
    // Calculate cost

    val (cost:Double, gradient:SparseVector[Double]) = costFunc(features, outputs, weights, params)
    
    val checkValue = descentAmountChecker(cost, params)
    val nextWeights = weights.copy
    nextWeights :-= gradient * params.stepSize
    
    println(iter, cost, checkValue)
    
    if (checkNecessity(checkValue, iter, params)) {
      val paramsNext = params.copy(earlierCost = cost)
      gdSolver(nextWeights, features, outputs, paramsNext, iter + 1)(costFunc)
    } else weights

  }

  // End of Gradient Descent for Sparse Types

  // Start of QR Decomposition for Dense Types

  def qrSolver(features: DenseMatrix[Double], outputs: DenseVector[Double],
    params: Parameters): DenseVector[Double] = {

    val n: Int = features.cols

    val feats = DenseMatrix.vertcat(features, DenseMatrix.eye[Double](n) * params.l2_penalty)
    val outs = DenseVector.vertcat(outputs, DenseVector.zeros[Double](n))

    val QR(qValue, rValue) = qr.reduced(feats)
    inv(rValue) * (qValue.t * outs)

  }

  def lbfgsSolver(initialWeights: SparseVector[Double], features: CSCMatrix[Double], outputs: SparseVector[Double], params: Parameters)
  (costFunc: (CSCMatrix[Double], SparseVector[Double],SparseVector[Double], Parameters) => (Double,SparseVector[Double])) = {
    val obj = new DiffFunction[SparseVector[Double]] {
      override def calculate(weights: SparseVector[Double]): (Double, SparseVector[Double]) = {
        
        costFunc(features, outputs, weights, params)

      }
    }
    //val initWeights = SparseVector(Array.fill(features.cols)(1.0))
    new LBFGS[SparseVector[Double]](tolerance = params.tolerance).minimize(obj, initialWeights)

  }
}
