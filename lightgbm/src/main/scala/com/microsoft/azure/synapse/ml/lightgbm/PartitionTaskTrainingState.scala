// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.azure.synapse.ml.lightgbm

import com.microsoft.azure.synapse.ml.lightgbm.booster.LightGBMBooster
import com.microsoft.azure.synapse.ml.lightgbm.dataset.BaseAggregatedColumns

/**
  * Object to encapsulate all training state on a single partition, or require a booster
  * to exist before calculating (some properties are mutable)
  */
case class PartitionTaskTrainingState(ctx: TrainingContext,
                                      partitionId: Int,
                                      booster: LightGBMBooster) {
  val log = ctx.log

  val evalNames = booster.getEvalNames()
  val evalCounts = evalNames.length
  val bestScore = new Array[Double](evalCounts)
  val bestScores = new Array[Array[Double]](evalCounts)
  val bestIter = new Array[Int](evalCounts)

  var iteration: Int = 0
  var isFinished: Boolean = false
  var learningRate: Double = ctx.initialLearningRate
  var bestIterResult: Option[Int] = None
}
