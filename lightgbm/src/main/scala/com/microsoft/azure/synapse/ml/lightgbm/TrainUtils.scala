// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.azure.synapse.ml.lightgbm

import com.microsoft.azure.synapse.ml.core.env.StreamUtilities._
import com.microsoft.azure.synapse.ml.core.utils.FaultToleranceUtils
import com.microsoft.azure.synapse.ml.lightgbm.booster.LightGBMBooster
import com.microsoft.azure.synapse.ml.lightgbm.dataset.LightGBMDataset
import com.microsoft.azure.synapse.ml.lightgbm.params.BaseTrainParams
import com.microsoft.ml.lightgbm._
import org.apache.spark.BarrierTaskContext
import org.slf4j.Logger

import java.io._
import java.net._
import scala.annotation.tailrec

private object TrainUtils extends Serializable {

  def createBooster(trainParams: BaseTrainParams,
                    trainDataset: LightGBMDataset,
                    validDatasetOpt: Option[LightGBMDataset]): LightGBMBooster = {
    // Create the booster
    val parameters = trainParams.toString()
    val booster = new LightGBMBooster(trainDataset, parameters)
     trainParams.generalParams.modelString.foreach { modelStr =>
      booster.mergeBooster(modelStr)
    }
    validDatasetOpt.foreach { dataset =>
      booster.addValidationDataset(dataset)
    }
    booster
  }

  def beforeTrainIteration(state: PartitionTaskTrainingState): Unit = {
    if (state.ctx.trainingParams.delegate.isDefined) {
      state.ctx.trainingParams.delegate.get.beforeTrainIteration(state.ctx.trainingCtx.batchIndex,
                                                    state.ctx.partitionId,
                                                    state.iteration,
                                                    state.ctx.log,
                                                    state.ctx.trainingParams,
                                                    state.booster,
                                                    state.ctx.trainingCtx.hasValidationData)
    }
  }

  def afterTrainIteration(state: PartitionTaskTrainingState,
                          trainEvalResults: Option[Map[String, Double]],
                          validEvalResults: Option[Map[String, Double]]): Unit = {
    val ctx = state.ctx
    val trainingCtx = ctx.trainingCtx
    if (ctx.trainingParams.delegate.isDefined) {
      ctx.trainingParams.delegate.get.afterTrainIteration(trainingCtx.batchIndex,
        ctx.partitionId,
        state.iteration,
        trainingCtx.log,
        trainingCtx.trainingParams,
        state.booster,
        trainingCtx.hasValidationData,
        state.isFinished,
        trainEvalResults,
        validEvalResults)
    }
  }

  def getLearningRate(state: PartitionTaskTrainingState): Double = {
    state.ctx.trainingParams.delegate match {
      case Some(delegate) => delegate.getLearningRate(state.ctx.trainingCtx.batchIndex,
                                                      state.ctx.partitionId,
                                                      state.iteration,
                                                      state.ctx.log,
                                                      state.ctx.trainingParams,
                                                      state.learningRate)
      case None => state.learningRate
    }
  }

  /* TODO decide fate of this
  def createReferenceDataset(numRows: Int,
                             numCols: Int,
                             datasetParams: String,
                             sampleData: SampledData,
                             featureNames: Array[String],
                             log: Logger): Array[Byte] = {
    log.info(s"LightGBM task generating schema for empty dense dataset with $numRows rows and $numCols columns")
    // Generate the dataset for features
    val dataset = lightgbmlib.voidpp_handle()
    LightGBMUtils.validate(lightgbmlib.LGBM_DatasetCreateFromSampledColumn(
      sampleData.getSampleData(),
      sampleData.getSampleIndices(),
      numCols,
      sampleData.getRowCounts(),
      sampleData.numRows,
      numRows, // TODO does this allocate?
      datasetParams,
      dataset), "Dataset create")

    val datasetHandle = lightgbmlib.voidpp_value(dataset)

    if (featureNames.nonEmpty) {
      LightGBMUtils.validate(lightgbmlib.LGBM_DatasetSetFeatureNames(datasetHandle, featureNames, numCols),
        "Dataset set feature names")
    }

    val buffer = lightgbmlib.voidpp_handle()
    val lenPtr = lightgbmlib.new_intp()
    LightGBMUtils.validate(lightgbmlib.LGBM_DatasetSerializeReferenceToBinary(
      datasetHandle,
      buffer,
      lenPtr), "Serialize ref")
    val bufferLen: Int = lightgbmlib.intp_value(lenPtr)
    lightgbmlib.delete_intp(lenPtr)

    val serializedReference = new Array[Byte](bufferLen)
    val valPtr = lightgbmlib.new_bytep()
    val bufferHandle = lightgbmlib.voidpp_value(buffer)
    (0 until bufferLen).foreach(i => {
      LightGBMUtils.validate(lightgbmlib.LGBM_ByteBufferGetAt(bufferHandle, i, valPtr),"Buffer getat")
      serializedReference(i) = lightgbmlib.bytep_value(valPtr).toByte
    })
    lightgbmlib.delete_bytep(valPtr)

    LightGBMUtils.validate(lightgbmlib.LGBM_ByteBufferFree(bufferHandle),"Buffer free")
    LightGBMUtils.validate(lightgbmlib.LGBM_DatasetFree(datasetHandle),"Dataset free")
    serializedReference
  } */

  /* TODO decide fate   def createReferenceDataset(numRows: Int,
                             numCols: Int,
                             datasetParams: String,
                             sampleData: SampledData,
                             featureNames: Array[String],
                             log: Logger): Array[Byte] = {
    log.info(s"LightGBM task generating schema for empty dense dataset with $numRows rows and $numCols columns")
    // Generate the dataset for features
    val dataset = lightgbmlib.voidpp_handle()
    LightGBMUtils.validate(lightgbmlib.LGBM_DatasetCreateFromSampledColumn(
      sampleData.getSampleData(),
      sampleData.getSampleIndices(),
      numCols,
      sampleData.getRowCounts(),
      sampleData.numRows,
      numRows, // TODO does this allocate?
      datasetParams,
      dataset), "Dataset create")

    val datasetHandle = lightgbmlib.voidpp_value(dataset)

    if (featureNames.nonEmpty) {
      LightGBMUtils.validate(lightgbmlib.LGBM_DatasetSetFeatureNames(datasetHandle, featureNames, numCols),
        "Dataset set feature names")
    }

    val buffer = lightgbmlib.voidpp_handle()
    val lenPtr = lightgbmlib.new_intp()
    LightGBMUtils.validate(lightgbmlib.LGBM_DatasetSerializeReferenceToBinary(
      datasetHandle,
      buffer,
      lenPtr), "Serialize ref")
    val bufferLen: Int = lightgbmlib.intp_value(lenPtr)
    lightgbmlib.delete_intp(lenPtr)

    val serializedReference = new Array[Byte](bufferLen)
    val valPtr = lightgbmlib.new_bytep()
    val bufferHandle = lightgbmlib.voidpp_value(buffer)
    (0 until bufferLen).foreach(i => {
      LightGBMUtils.validate(lightgbmlib.LGBM_ByteBufferGetAt(bufferHandle, i, valPtr),"Buffer getat")
      serializedReference(i) = lightgbmlib.bytep_value(valPtr).toByte
    })
    lightgbmlib.delete_bytep(valPtr)

    LightGBMUtils.validate(lightgbmlib.LGBM_ByteBufferFree(bufferHandle),"Buffer free")
    LightGBMUtils.validate(lightgbmlib.LGBM_DatasetFree(datasetHandle),"Dataset free")
    serializedReference
  } */

  def updateOneIteration(state: PartitionTaskTrainingState): Unit = {
    try {
      state.log.info("LightGBM running iteration: " + state.iteration + " with is finished: " + state.isFinished)
        val fobj = state.ctx.trainingParams.objectiveParams.fobj
        if (fobj.isDefined) {
          val isClassification = state.ctx.trainingCtx.isClassification
          val (gradient, hessian) = fobj.get.getGradient(
            state.booster.innerPredict(0, isClassification), state.booster.trainDataset.get)
          state.isFinished = state.booster.updateOneIterationCustom(gradient, hessian)
        } else {
          state.isFinished = state.booster.updateOneIteration()
        }
      state.log.info("LightGBM completed iteration: " + state.iteration + " with is finished: " + state.isFinished)
    } catch {
      case e: java.lang.Exception =>
        state.log.warn("LightGBM reached early termination on one task," +
          " stopping training on task. This message should rarely occur." +
          " Inner exception: " + e.toString)
        state.isFinished = true
    }
  }

  def executeTrainingIterations(state: PartitionTaskTrainingState): Option[Int] = {
    @tailrec
    def iterationLoop(maxIterations: Int): Option[Int] = {
      beforeTrainIteration(state)
      val newLearningRate = getLearningRate(state)
      if (newLearningRate != state.learningRate) {
        state.log.info(s"LightGBM task calling booster.resetParameter to reset learningRate" +
          s" (newLearningRate: $newLearningRate)")
        state.booster.resetParameter(s"learning_rate=$newLearningRate")
        state.learningRate = newLearningRate
      }

      updateOneIteration(state)

      val trainEvalResults: Option[Map[String, Double]] =
        if (state.ctx.trainingCtx.isProvideTrainingMetric && !state.isFinished) getTrainEvalResults(state)
        else None

      val validEvalResults: Option[Map[String, Double]] =
        if (state.ctx.trainingCtx.hasValidationData && !state.isFinished) getValidEvalResults(state)
        else None

      afterTrainIteration(state, trainEvalResults, validEvalResults)

      state.iteration = state.iteration + 1
      if (!state.isFinished && state.iteration < maxIterations) {
        iterationLoop(maxIterations)  // tail recursion
      } else {
        state.bestIterResult
      }
    }

    state.log.info(s"Beginning training on LightGBM Booster for partition ${state.ctx.partitionId}")
    state.ctx.measures.markTrainingIterationsStart()
    val result = iterationLoop(state.ctx.trainingParams.generalParams.numIterations)
    state.ctx.measures.markTrainingIterationsStart()
    result
  }

  def getTrainEvalResults(state: PartitionTaskTrainingState): Option[Map[String, Double]] = {
    val evalResults: Array[(String, Double)] = state.booster.getEvalResults(state.evalNames, 0)
    evalResults.foreach { case (evalName: String, score: Double) => state.log.info(s"Train $evalName=$score") }
    Option(Map(evalResults: _*))
  }

  def getValidEvalResults(state: PartitionTaskTrainingState): Option[Map[String, Double]] = {
    val evalResults: Array[(String, Double)] = state.booster.getEvalResults(state.evalNames, 1)
    val results: Array[(String, Double)] = evalResults.zipWithIndex.map { case ((evalName, evalScore), index) =>
      state.log.info(s"Valid $evalName=$evalScore")
      val cmp =
        if (evalName.startsWith("auc")
            || evalName.startsWith("ndcg@")
            || evalName.startsWith("map@")
            || evalName.startsWith("average_precision"))
          (x: Double, y: Double, tol: Double) => x - y > tol
        else
          (x: Double, y: Double, tol: Double) => x - y < tol
      if (state.bestScores(index) == null
          || cmp(evalScore, state.bestScore(index), state.ctx.trainingCtx.improvementTolerance)) {
        state.bestScore(index) = evalScore
        state.bestIter(index) = state.iteration
        state.bestScores(index) = evalResults.map(_._2)
      } else if (state.iteration - state.bestIter(index) >= state.ctx.trainingCtx.earlyStoppingRound) {
        state.isFinished = true
        state.log.info("Early stopping, best iteration is " + state.bestIter(index))
        state.bestIterResult = Some(state.bestIter(index))
      }

      (evalName, evalScore)
    }
    Option(Map(results: _*))
  }

  def beforeGenerateTrainDataset(ctx: PartitionTaskContext): Unit = {
    val trainingCtx = ctx.trainingCtx
    if (trainingCtx.trainingParams.delegate.isDefined) {
      trainingCtx.trainingParams.delegate.get.beforeGenerateTrainDataset(
        trainingCtx.batchIndex,
        ctx.partitionId,
        trainingCtx.columnParams,
        trainingCtx.schema,
        trainingCtx.log,
        trainingCtx.trainingParams)
    }
  }

  def afterGenerateTrainDataset(ctx: PartitionTaskContext): Unit = {
    val trainingCtx = ctx.trainingCtx
    if (trainingCtx.trainingParams.delegate.isDefined) {
      trainingCtx.trainingParams.delegate.get.afterGenerateTrainDataset(
        trainingCtx.batchIndex,
        ctx.partitionId,
        trainingCtx.columnParams,
        trainingCtx.schema,
        trainingCtx.log,
        trainingCtx.trainingParams)
    }
  }

  def beforeGenerateValidDataset(ctx: PartitionTaskContext): Unit = {
    val trainingCtx = ctx.trainingCtx
    if (ctx.trainingCtx.trainingParams.delegate.isDefined) {
      trainingCtx.trainingParams.delegate.get.beforeGenerateValidDataset(
        trainingCtx.batchIndex,
        ctx.partitionId,
        trainingCtx.columnParams,
        trainingCtx.schema,
        trainingCtx.log,
        trainingCtx.trainingParams)
    }
  }

  def afterGenerateValidDataset(ctx: PartitionTaskContext): Unit = {
    val trainingCtx = ctx.trainingCtx
    if (trainingCtx.trainingParams.delegate.isDefined) {
      trainingCtx.trainingParams.delegate.get.afterGenerateValidDataset(
        trainingCtx.batchIndex,
        ctx.partitionId,
        trainingCtx.columnParams,
        trainingCtx.schema,
        trainingCtx.log,
        trainingCtx.trainingParams)
    }
  }
}
