package main.scala.wlml
import breeze.linalg._

class Mlobj(df: Array[Array[Double]]) {
  val matrix = DenseMatrix(df:_*)
}

object asdf {
  val a = Array(Array(1.0, 2.0, 6.0), Array(3.0, 4.0, 5.0))
  val b = DenseMatrix(a:_*)
  b
}