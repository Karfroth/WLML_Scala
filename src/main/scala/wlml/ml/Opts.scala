package wlml.ml

trait Opts {
  
  case class Parameters(tolerance: Double = 1E-4, maxiter: Int = 1000, stepSize: Double=0.01,
    l1_penalty: Double=0.05, l2_penalty: Double=0.05, earlierCost: Double=Double.PositiveInfinity)
    
}