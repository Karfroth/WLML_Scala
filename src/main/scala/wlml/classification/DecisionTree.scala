package wlml.classification

import breeze.linalg._

/**
  * Created by Woosang Lee
  */

trait DecisionTree {

  type Idx = Int

  case class SplittingParam(index: Idx, err: Double)

  abstract class DTree

  case class Leaf(prediction: Int) extends DTree

  case class Node(left: DTree, right: DTree, depth: Int, splittingIndex: Idx) extends DTree

  def checkNodeMistakes(outputs: Vector[Int]): Double = {

    if (outputs.length == 0) 0
    else {
      val positives: Double = sum(outputs.map(x => if (x == 1) 1 else 0))
      val negatives: Double = sum(outputs.map(x => if (x == 0) 1 else 0))
      if (positives > negatives) negatives
      else positives
    }
  }

  def buildLeaf(outputs: Vector[Int]): Int = {
    val numOnes = sum((0 until outputs.length).filter(outputs(_) == 1))
    val numZeroes = sum((0 until outputs.length).filter(outputs(_) == 0))
    if (numOnes > numZeroes) 1 else 0
  }

  def findBestFeature(features: Matrix[_], outputs: Vector[Int], featuresSet: Set[Idx]): SplittingParam = {

    assert(features.rows == outputs.length, "Number of rows in Features Matrix and Length of Outputs Vector must be same!")

    def splitting(acc: SplittingParam, featIDX: Idx): SplittingParam = {
      val left = outputs((0 until features.rows).filter(features(_, featIDX) == 1))
      val right = outputs((0 until features.rows).filter(features(_, featIDX) == 0))

      val leftMistake = checkNodeMistakes(left)
      val rightMistake = checkNodeMistakes(right)
      val error = (leftMistake + rightMistake) / features.rows
      if (error < acc.err) SplittingParam(index = featIDX, err = error)
      else acc
    }

    featuresSet.foldLeft(SplittingParam(index = 0, err = 1.0))(splitting)

  }


  //def findBestFeature(features: Matrix[_], outputs: Vector[_], featuresSet: Set[Idx]): SplittingParam
  implicit def intToDoubleMatrix(mat: CSCMatrix[Int]): CSCMatrix[Double] = mat.map( x => x * 1.0)

  def buildTree(features: Matrix[Double], outputs: Vector[Int], depth: Int = 0, maxDepth: Int, featuresIdxSet: Set[Idx]): DTree = {

    if (depth >= maxDepth || featuresIdxSet.isEmpty) {
      Leaf(buildLeaf(outputs))
    }
    else {
      //Find best feature index and Assign row index to split
      val splittingFeatureIdx: SplittingParam = findBestFeature(features, outputs, featuresIdxSet)
      val leftIdx: Seq[Idx] = (0 until features.rows).filter(features(_, splittingFeatureIdx.index) == 0)
      val rightIdx: Seq[Idx] = (0 until features.rows).filter(features(_, splittingFeatureIdx.index) == 1)

      // Spliting Features
      val leftF: Matrix[Double] = features(leftIdx, 0 until features.cols)
      val rightF: Matrix[Double] = features(rightIdx, 0 until features.cols)

      // Spliting Outputs
      val leftO: Vector[Int] = outputs(leftIdx)
      val rightO: Vector[Int] = outputs(rightIdx)

      // Remove Best Feature Index
      val featuresSetNew: Set[Idx] = featuresIdxSet - splittingFeatureIdx.index

      // If one of split has same observation to input, it means this is perfect.
      if (leftO.size == outputs.length || rightO.size == outputs.length) {
        Leaf(buildLeaf(outputs))
      } else { //else build node
        Node(left = buildTree(leftF, leftO, depth + 1, maxDepth, featuresSetNew),
          right = buildTree(rightF, rightO, depth + 1, maxDepth, featuresSetNew),
          depth = depth + 1, splittingIndex = splittingFeatureIdx.index)
      }
    }
  }

  def predict(tree: DTree, testRow: Matrix[Double], annotate:Boolean = false): Int = tree match {
    //tree is Leaf
    case Leaf(p) => p
    case _ => ???
  }
}
