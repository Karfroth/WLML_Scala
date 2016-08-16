package wlml.ml

trait Opts {
  
  case class Parameters(tolerance: Double, maxiter: Int, stepSize: Double,
    l1_penalty: Double, l2_penalty: Double, earlierCost: Double)
    
}