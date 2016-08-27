package wlml.classification

import breeze.linalg._

/**
  * Created by Woosang Lee
  */

trait DecisionTree {

  type Idx = Int
  case class SplitingParam(index: Idx, err: Double)
  case class TreeParameter(maxDepth: Int, featureIdxs: Set[Idx])

  abstract class DTree
  class Leaf(outputs: Vector[_]) extends DTree {
    private val numOnes = sum((0 until outputs.length).filter(outputs(_) == 1))
    private val numZeroes = sum((0 until outputs.length).filter(outputs(_) == 0))
    val prediction = if (numOnes > numZeroes) 1 else 0
  }
  case class Node(left: DTree, right: DTree, depth: Int, splitingIndex: Idx) extends DTree

  def checkNodeMistakes(outputs: SliceVector[_,_]): Double = {
    //Corner case: If labels_in_node is empty, return 0
    if (outputs.length == 0) 0 //Count the number of 1 's (safe loans)
    else {
      val positives: Double = sum(outputs.map(x => if(x == 1) 1 else 0)) // Count the number of -1 's (risky loans)
      val negatives: Double = sum(outputs.map(x => if(x == 0) 1 else 0)) // Return the number of mistakes that the majority classifier makes.
      if (positives > negatives) negatives
      else positives
    }
  }

  def findBestFeature(features: Matrix[_], outputs: Vector[_], featuresSet: Set[Idx]): SplitingParam = {

    assert(features.rows == outputs.length, "Number of rows in Features Matrix and Length of Outputs Vector must be same!")

    def spliting(acc: SplitingParam, featIDX: Idx): SplitingParam = {
      val left = outputs((0 until features.rows).filter(features(_, featIDX) == 1))
      val right = outputs((0 until features.rows).filter(features(_, featIDX) == 0))
      val leftMistake = checkNodeMistakes(left)
      val rightMistake = checkNodeMistakes(right)
      val error = (leftMistake + rightMistake) / features.rows
      if (error < acc.err) SplitingParam(index = featIDX, err = error)
      else acc
    }

    featuresSet.foldLeft(SplitingParam(index = 0, err = 1.0))(spliting)

  }

  def buildTree(features: Matrix[_], outputs: Vector[_], depth: Int = 0, param: TreeParameter): DTree = {
    if ((depth >= param.maxDepth) || (param.featureIdxs.isEmpty)) {
      new Leaf(outputs)
    }
    else {
      val splittingFeatureIdx = findBestFeature(features, outputs, param.featureIdxs)
      ////// Need to develop from here!!!
    }
  }


}
