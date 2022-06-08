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
  * Object to encapsulate results from mapPartitions call.
  */
case class PartitionResult(booster: Option[LightGBMBooster],
                           taskMeasures: TaskExecutionMeasures)

/**
  * Object to encapsulate all intermediate data calculations.
  * Note tha only bulk uses these properties, but BasePartitionTask uses this class for consistent interfaces.
  */
case class PartitionDataState(aggregatedTrainingData: Option[BaseAggregatedColumns],
                              aggregatedValidationData: Option[BaseAggregatedColumns],
                              streamingValidationSet: Option[LightGBMDataset] = None) // TODO might not need last one

/**
  * Object to encapsulate all training state on a single partition, plus the actual Booster
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
  var learningRate: Double = ctx.trainingParams.generalParams.learningRate
  var bestIterResult: Option[Int] = None
}

/**
  * Class for handling the execution of Tasks on workers for each partition.
  * Should not contain driver-related threads.
  */
abstract class BasePartitionTask extends Serializable {
  /**
    * Prepare any data objects for this particular partition.  Implement for specific execution modes.
    * @param ctx The training context.
    * @param inputRows The Spark rows for a partition as an iterator.
    * @param partitionId The partition id.
    * @return Any intermediate data state (used mainly by bulk execution mode) to pass to future stages.
    */
  def preparePartitionData(ctx: TrainingContext,
                           inputRows: Iterator[Row],
                           partitionId: Int,
                           taskMeasures: TaskExecutionMeasures): PartitionDataState = {
    taskMeasures.markDataPreparationStart()
    val state = preparePartitionDataInternal(ctx, inputRows, partitionId)
    taskMeasures.markDataPreparationStop()
    state
  }

  /**
    * Prepare any data objects for this particular partition.  Implement for specific execution modes.
    * @param ctx The training context.
    * @param inputRows The Spark rows for a partition as an iterator.
    * @param partitionId The partition id.
    * @return Any intermediate data state (used mainly by bulk execution mode) to pass to future stages.
    */
  def preparePartitionDataInternal(ctx: TrainingContext, inputRows: Iterator[Row], partitionId: Int): PartitionDataState

  /**
    * Generate the final dataset for this task.  Internal implementation for specific execution modes.
    * @param ctx The training context.
    * @param dataState Any intermediate data state (used mainly by bulk execution mode).
    * @param forValidation Whether to generate the final training dataset or the validation dataset.
    * @param referenceDataset A reference dataset to start with (used mainly for validation dataset).
    */
  protected def generateFinalDatasetInternal(ctx: TrainingContext,
                                             dataState: PartitionDataState,
                                             partitionIndex: Int,
                                             forValidation: Boolean,
                                             referenceDataset: Option[LightGBMDataset]): LightGBMDataset

  /**
    * Generate the final dataset for this task.  This should only be run be tasks that will participate in
    * the training rounds, i.e. in useSingleDataset mode it will only be 1 task/executor.
    * @param ctx The training context.
    * @param dataState Any intermediate data state (used mainly by bulk execution mode).
    * @param forValidation Whether to generate the final training dataset or the validation dataset.
    * @param referenceDataset A reference dataset to start with (used mainly for validation dataset).
    * @return LightGBM dataset Java wrapper.
    */
  private def generateFinalDataset(ctx: TrainingContext,
                                   dataState: PartitionDataState,
                                   partitionIndex: Int,
                                   forValidation: Boolean,
                                   referenceDataset: Option[LightGBMDataset],
                                   taskMeasures: TaskExecutionMeasures): LightGBMDataset = {
    if (forValidation) taskMeasures.markValidationDatasetStart()
    else taskMeasures.markDatasetCreationStart()

    val dataset = generateFinalDatasetInternal(ctx, dataState, partitionIndex, forValidation, referenceDataset)

    // Validate generated dataset has the correct number of rows and cols
    dataset.validateDataset()

    if (forValidation) taskMeasures.markValidationDatasetStop()
    else taskMeasures.markDatasetCreationStop()
    dataset
  }

  /**
    * Load a data partition into Datasets and execute LightGBM training iterations.
    * Note that this method should only be called for "active" threads that created a final Dataset, and not
    * for ones that were empty or were only used to load temporary Datasets that were merged into a centralized one.
    * @param ctx The training context.
    * @param dataState Any intermediate data state (used mainly by bulk execution mode).
    * @param shouldReturnBooster Whether to return the booster or an empty iterator.
    * @return LightGBM booster iterator (to comply with Spark mapPartition API), that is either empty or
    *         has the resulting booster as the only element.
    */
  private def loadDatasetAndTrain(ctx: TrainingContext,
                                  dataState: PartitionDataState,
                                  taskMeasures: TaskExecutionMeasures,
                                  shouldReturnBooster: Boolean): Iterator[PartitionResult] = {
    val partitionId = TaskContext.getPartitionId
    taskMeasures.isActiveTrainingTask = true
    beforeGenerateTrainDataset(ctx)
    val trainDataset: LightGBMDataset = generateFinalDataset(ctx, dataState, partitionId, false, None, taskMeasures)
    try {
      afterGenerateTrainDataset(ctx)

      val validDatasetOpt: Option[LightGBMDataset] = if (!ctx.hasValid) None
       else {
          beforeGenerateValidDataset(ctx)
          val out = generateFinalDataset(ctx, dataState, partitionId, true, Some(trainDataset), taskMeasures)
          afterGenerateValidDataset(ctx)
          Option(out)
        }

      try {
        val booster = createBooster(ctx.trainingParams, trainDataset, validDatasetOpt, ctx.log)
        val state = PartitionTaskTrainingState(ctx, partitionId, booster)
        try {
          taskMeasures.markTrainingIterationsStart()
          val bestIterResult = executeTrainingIterations(state)
          taskMeasures.markTrainingIterationsStop()
          if (shouldReturnBooster) {
            val model = state.booster.saveToString(bestIterResult)
            val modelBooster = new LightGBMBooster(model)
            // Set best iteration on booster if hit early stopping criteria in trainCore
            bestIterResult.foreach(modelBooster.setBestIteration)
            Iterator.single(new PartitionResult(Option(modelBooster), taskMeasures))
          } else {
            Iterator.single(new PartitionResult(None, taskMeasures))
          }
        } finally {
          // Free booster
          state.booster.freeNativeMemory()
        }
      } finally {
        // TODO validDatasetOpt.foreach(_.close())
      }
    } finally {
      trainDataset.close()
    }
  }

  /**
    * This method will be passed to Spark's mapPartition method and handle execution of training on the workers.
    * @param ctx The training context.
    * @param inputRows The Spark rows as an iterator.
    * @return result iterator (to comply with Spark mapPartition API).
    */
  def mapPartitionTask(ctx: TrainingContext)(inputRows: Iterator[Row]): Iterator[PartitionResult] = {
    val trainParams = ctx.trainingParams
    val partitionId = getPartitionId
    val taskMeasures = new TaskExecutionMeasures(partitionId)
    if (trainParams.generalParams.verbosity > 1) {
      ctx.log.info(s"LightGBM partition $partitionId running on executor $getExecutorId")
    }
    val isEmptyPartition = !inputRows.hasNext
    // Note: the first valid worker with non-empty partitions sets the main executor worker, other workers read it
    if (ctx.useSingleDatasetMode && !isEmptyPartition) ctx.sharedState().linkMainExecutorWorker()
    val isEnabledWorker = if (!isEmptyPartition) isWorkerEnabled(ctx) else false
    // Initialize the native library
    LightGBMUtils.initializeNativeLibrary()
    val (nodes, localListenPort) = getNetworkInfo(ctx, isEnabledWorker)

    if (isEmptyPartition) {
      ctx.log.warn("LightGBM task encountered empty partition, for best performance ensure no partitions empty")
      Array { PartitionResult(None, taskMeasures) }.toIterator
    } else {
      updateHelperStartSignal(ctx, isEnabledWorker, localListenPort)
      val dataIntermediateState = preparePartitionData(ctx, inputRows, partitionId, taskMeasures)

      // Return booster only from main worker to reduce network communication overhead
      val shouldReturnBooster = getShouldReturnBooster(ctx, isEnabledWorker, nodes, localListenPort)
      try {
        if (isEnabledWorker) {
          // If worker enabled, initialize the network ring of communication
          networkInit(nodes, localListenPort, ctx.log, LightGBMConstants.NetworkRetries, LightGBMConstants.InitialDelay)
          if (ctx.useSingleDatasetMode) ctx.sharedState().dataPreparationDoneSignal.await()

          loadDatasetAndTrain(ctx, dataIntermediateState, taskMeasures, shouldReturnBooster)
        } else {
          ctx.log.info("Helper task finished processing rows")
          ctx.sharedState().dataPreparationDoneSignal.countDown()
          Array { new PartitionResult(None, taskMeasures) }.toIterator
        }
      } finally {
        cleanup(ctx, isEnabledWorker, taskMeasures)
      }
    }
  }

  /** Cleanup the task
    *
    * @param ctx The training context.
    * @param isEnabledWorker Whether the current work is enabled to initialize the network ring of communication.
    * @param taskMeasures The task instrumentation.
    */
  private def cleanup(ctx: TrainingContext, isEnabledWorker: Boolean, taskMeasures: TaskExecutionMeasures): Unit = {
    // Finalize network when done
    if (isEnabledWorker) LightGBMUtils.validate(lightgbmlib.LGBM_NetworkFree(), "Finalize network")

    if (ctx.isStreaming && isEnabledWorker) {
      ctx.sharedState().trainingDoneSignal.countDown()
      if (ctx.sharedState().trainingDoneSignal.getCount == 0)
      {
        ctx.sharedState().validationDatasetState.freeSharedStreamingDatasets()
      }
    }

    taskMeasures.markTaskEnd()
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
        ctx.incrementDataPrepDoneSignal()
      } else {
        ctx.incrementTrainingDoneSignal() // TODO this should not be in a getter function
      }
      isMainWorker
    } else {
      true
    }
  }

  /** Determines if the current task is the main worker in the current JVM.
    *
    * @param ctx The training context.
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
