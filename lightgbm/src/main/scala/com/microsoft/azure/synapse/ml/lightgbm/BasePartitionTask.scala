// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.azure.synapse.ml.lightgbm

import com.microsoft.azure.synapse.ml.lightgbm.LightGBMUtils._
import com.microsoft.azure.synapse.ml.lightgbm.TrainUtils._
import com.microsoft.azure.synapse.ml.lightgbm.booster.LightGBMBooster
import com.microsoft.azure.synapse.ml.lightgbm.dataset.{BaseAggregatedColumns, LightGBMDataset}
import com.microsoft.ml.lightgbm.lightgbmlib
import org.apache.spark.TaskContext
import org.apache.spark.sql._

import scala.language.existentials

/**
  * Object to encapsulate all intermediate data calculations.
  * Note tha only bulk uses these properties, but BasePartitionTask uses this class for consistent interfaces.
  */
case class PartitionDataState(aggregatedTrainingData: Option[BaseAggregatedColumns],
                              aggregatedValidationData: Option[BaseAggregatedColumns])

/**
  * Class for handling the execution of Tasks on workers for each partition.
  * Should not contain driver-related threads.
  */
abstract class BasePartitionTask {
  /* Prepare any data objects for this particular partition.
   */
  def preparePartitionDatasets(ctx: TrainingContext, inputRows: Iterator[Row]): PartitionDataState

  /* Generate the final dataset for this task.  Override for specific execution types.
   */
  protected def generateFinalDatasetInternal(ctx: TrainingContext,
                                             dataState: PartitionDataState,
                                             forValidation: Boolean,
                                             referenceDataset: Option[LightGBMDataset]): LightGBMDataset

  /* Generate the final dataset for this task.  This should only be run be tasks that will participate in
   * the training rounds, i.e. in useSingleDataset mode it will only be 1 task/executor.
   */
  private def generateFinalDataset(ctx: TrainingContext,
                                   dataState: PartitionDataState,
                                   forValidation: Boolean,
                                   referenceDataset: Option[LightGBMDataset]): LightGBMDataset = {
    val dataset = generateFinalDatasetInternal(ctx, dataState, forValidation, referenceDataset)

    // Validate generated dataset has the correct number of rows and cols
    dataset.validateDataset()
    dataset
  }

  private def loadDatasetAndTrain(ctx: TrainingContext,
                                  dataState: PartitionDataState,
                                  shouldReturnBooster: Boolean): Iterator[LightGBMBooster] = {
    beforeGenerateTrainDataset(ctx)
    val trainDataset: LightGBMDataset = generateFinalDataset(ctx, dataState, false, None)
    try {
      afterGenerateTrainDataset(ctx)

      val validDatasetOpt: Option[LightGBMDataset] = dataState.aggregatedValidationData.map { _ =>
        beforeGenerateValidDataset(ctx)
        val out = generateFinalDataset(ctx, dataState, true, Some(trainDataset))
        afterGenerateValidDataset(ctx)
        out
      }

      try {
        val state = PartitionTaskTrainingState(
          ctx,
          TaskContext.getPartitionId,
          createBooster(ctx.trainingParams, trainDataset, validDatasetOpt))
        try {
          val bestIterResult = executeTrainingIterations(state)
          if (shouldReturnBooster) {
            val model = state.booster.saveToString(bestIterResult)
            val modelBooster = new LightGBMBooster(model)
            // Set best iteration on booster if hit early stopping criteria in trainCore
            bestIterResult.foreach(modelBooster.setBestIteration)
            Iterator.single(modelBooster)
          } else {
            Iterator.empty
          }
        } finally {
          // Free booster
          state.booster.freeNativeMemory()
        }
      } finally {
        validDatasetOpt.foreach(_.close())
      }
    } finally {
      trainDataset.close()
    }
  }

  def handlePartitionTask(ctx: TrainingContext)(inputRows: Iterator[Row]): Iterator[LightGBMBooster] = {
    val log = ctx.log
    val trainParams = ctx.trainingParams
    if (trainParams.generalParams.verbosity > 1) {
      log.info(s"LightGBM partition $getPartitionId running on executor $getExecutorId")
    }
    val emptyPartition = !inputRows.hasNext
    // Note: the first valid worker with non-empty partitions sets the main executor worker, other workers read it
    if (ctx.useSingleDatasetMode && !emptyPartition) ctx.sharedState.linkMainExecutorWorker()
    val isEnabledWorker = if (!emptyPartition) isWorkerEnabled(ctx) else false
    // Initialize the native library
    LightGBMUtils.initializeNativeLibrary()
    // Initialize the network communication
    val (nodes, localListenPort) = getNetworkInfo(ctx, isEnabledWorker)

    if (emptyPartition) {
      log.warn("LightGBM task encountered empty partition, for best performance ensure no partitions empty")
      List[LightGBMBooster]().toIterator
    } else {
      updateHelperStartSignal(ctx, isEnabledWorker, localListenPort)
      val dataState = preparePartitionDatasets(ctx, inputRows)

      // Return booster only from main worker to reduce network communication overhead
      val shouldReturnBooster = getShouldReturnBooster(ctx, isEnabledWorker, nodes, localListenPort)
      try {
        if (isEnabledWorker) {
          // If worker enabled, initialize the network ring of communication
          networkInit(nodes,
                      localListenPort,
                      log,
                      LightGBMConstants.NetworkRetries,
                      LightGBMConstants.InitialDelay)

          if (ctx.useSingleDatasetMode) ctx.sharedState.doneSignal.await()

          loadDatasetAndTrain(ctx, dataState, shouldReturnBooster)
        } else {
          log.info("Helper task finished processing rows")
          ctx.sharedState.doneSignal.countDown()
          List[LightGBMBooster]().toIterator
        }
      } finally {
        // Finalize network when done
        if (isEnabledWorker) LightGBMUtils.validate(lightgbmlib.LGBM_NetworkFree(), "Finalize network")
      }
    }
  }

  /** Prints the listening port and, in single dataset mode, forces helper tasks to wait for the main worker
    * before continuing to prepare and merge the dataset.
    *
    * @param ctx The training context.
    * @param isEnabledWorker Whether the current work is enabled to initialize the network ring of communication.
    * @param localListenPort The local port for creating the network ring of communication.
    */
  private def updateHelperStartSignal(ctx: TrainingContext,
                                      isEnabledWorker: Boolean,
                                      localListenPort: Int) = {
    if (isEnabledWorker) {
      ctx.log.info(s"LightGBM task listening on: $localListenPort")
      if (ctx.useSingleDatasetMode) ctx.sharedState.helperStartSignal.countDown()
    } else {
      ctx.sharedState.helperStartSignal.await()
    }
  }

  /** If using single dataset mode, only returns one task in JVM.
    * Otherwise, returns true for all tasks.
    * @param ctx The training context.
    * @return Whether the current task is enabled.
    */
  private def isWorkerEnabled(ctx: TrainingContext): Boolean = {
    if (ctx.useSingleDatasetMode) {
      // Find all workers in current JVM
      val isMainWorker = isCurrentTaskMainWorker(ctx)
      ctx.incrementArrayProcessedSignal()
      if (!isMainWorker) {
        ctx.incrementDoneSignal()
      }
      isMainWorker
    } else {
      true
    }
  }

  /** Determines if the current task is the main worker in the current JVM.
    *
    * @param log The logger.
    * @param sharedState The shared state.
    * @return True if the current task in the main worker, false otherwise.
    */
  private def isCurrentTaskMainWorker(ctx: TrainingContext): Boolean = {
    val mainExecutorWorker = ctx.sharedState.mainExecutorWorker.get
    val myTaskId = LightGBMUtils.getTaskId
    val isMainWorker = mainExecutorWorker == myTaskId
    ctx.log.info(s"Using singleDatasetMode.  " +
      s"Is main worker: ${isMainWorker} for task id: ${myTaskId} and main task id: ${mainExecutorWorker}")
    isMainWorker
  }
}
