// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.azure.synapse.ml.lightgbm

import com.microsoft.azure.synapse.ml.lightgbm.LightGBMUtils._
import com.microsoft.azure.synapse.ml.lightgbm.TrainUtils._
import com.microsoft.azure.synapse.ml.lightgbm.booster.LightGBMBooster
import com.microsoft.azure.synapse.ml.lightgbm.dataset.{BaseAggregatedColumns, LightGBMDataset}
import com.microsoft.azure.synapse.ml.lightgbm.params.BaseTrainParams
import com.microsoft.ml.lightgbm.lightgbmlib
import org.apache.spark.sql._
import org.slf4j.Logger

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
                              aggregatedValidationData: Option[BaseAggregatedColumns])

/**
  * Object to encapsulate all training state on a single partition, plus the actual Booster
  */
case class PartitionTaskTrainingState(ctx: PartitionTaskContext,
                                      booster: LightGBMBooster) {
  val log: Logger = ctx.log

  lazy val evalNames: Array[String] = booster.getEvalNames()
  lazy val evalCounts: Int = evalNames.length
  lazy val bestScore = new Array[Double](evalCounts)
  lazy val bestScores = new Array[Array[Double]](evalCounts)
  lazy val bestIter = new Array[Int](evalCounts)

  var iteration: Int = 0
  var isFinished: Boolean = false
  var learningRate: Double = ctx.trainingCtx.trainingParams.generalParams.learningRate
  var bestIterResult: Option[Int] = None
}

/**
  * Object to encapsulate most setup information about a particular partition Task
  */
case class PartitionTaskContext(trainingCtx: TrainingContext,
                                partitionId: Int,
                                taskId: Long,
                                measures: TaskExecutionMeasures,
                                networkTopologyInfo: NetworkTopologyInfo,
                                isMainWorker: Boolean,
                                shouldExecuteTraining: Boolean,
                                isEmptyPartition: Boolean,
                                shouldReturnBooster: Boolean) {
  // Extra info that is settable by custom execution modes
  private val customJobCount = 2
  private val shouldCalcValidationDataInfoId = 0
  private val shouldCalcExecutorDatasetInfoId = 1
  private val customModeInfo: Array[Boolean] = Array.fill(customJobCount)(false)

  val lightGBMNetworkString: String = networkTopologyInfo.lightgbmNetworkString
  val localListenPort: Int = networkTopologyInfo.localListenPort
  val lightGBMNetworkMachineCount: Int = lightGBMNetworkString.split(",").length

  val log: Logger = trainingCtx.log
  def sharedState: SharedState = { trainingCtx.sharedState() }
  val trainingParams: BaseTrainParams = trainingCtx.trainingParams

  def setShouldCalcValidationData(value: Boolean): PartitionTaskContext = {
    customModeInfo(shouldCalcValidationDataInfoId) = value
    this
  }

  def setShouldCalcExecutorDataset(value: Boolean): PartitionTaskContext = {
    customModeInfo(shouldCalcExecutorDatasetInfoId) = value
    this
  }

  def shouldCalcValidationDataset: Boolean = customModeInfo(shouldCalcValidationDataInfoId)
  def shouldCalcExecutorDataset: Boolean = customModeInfo(shouldCalcExecutorDatasetInfoId)

  lazy val streamingExecutorRowCount: Int = {
    val allPartitionRowCounts = trainingCtx.partitionCounts.get
    networkTopologyInfo.executorPartitionIdList
      .map(partitionId => allPartitionRowCounts(partitionId)).sum.toInt
  }

  lazy val streamingPartitionOffset: Int = {
    val allPartitionRowCounts = trainingCtx.partitionCounts.get
    networkTopologyInfo.executorPartitionIdList.filter(id => id < partitionId)
      .map(partitionId => allPartitionRowCounts(partitionId)).sum.toInt
  }
}

/**
  * Class for handling the execution of Tasks on workers for each partition. Only runs on worker Tasks.
  */
abstract class BasePartitionTask extends Serializable {
  /**
    * This method will be passed to Spark's mapPartition method and handle execution of training on the workers.
    * Main stages: (and each execution mode has an "Internal" version to perform mode-specific operations)
    *   initialize()
    *   preparePartitionData()
    *   finalizeDatasetAndTrain()
    *   cleanup()
    * @param ctx The training context.
    * @param inputRows The Spark rows as an iterator.
    * @return result iterator (to comply with Spark mapPartition API).
    */
  def mapPartitionTask(ctx: TrainingContext)(inputRows: Iterator[Row]): Iterator[PartitionResult] = {
    // Start with initialization
    val taskCtx = initialize(ctx, inputRows)

    if (taskCtx.isEmptyPartition) {
      ctx.log.warn("LightGBM task encountered empty partition, for best performance ensure no partitions are empty")
      Array { PartitionResult(None, taskCtx.measures) }.toIterator
    } else {
      // Perform any data preparation work
      val dataIntermediateState = preparePartitionData(taskCtx, inputRows)

      try {
        if (taskCtx.shouldExecuteTraining) {
          // If participating in training, initialize the network ring of communication
          NetworkManager.initLightGBMNetwork(taskCtx, LightGBMConstants.NetworkRetries, LightGBMConstants.InitialDelay)

          if (ctx.useSingleDatasetMode) {
            ctx.log.info("Waiting for all data preparation to be done")
            ctx.sharedState().dataPreparationDoneSignal.await()
          }

          // Create the final Dataset for training and execute training iterations
          finalizeDatasetAndTrain(taskCtx, dataIntermediateState)
        } else {
          ctx.log.info("Helper task finished processing rows")
          ctx.sharedState().dataPreparationDoneSignal.countDown()
          Array { PartitionResult(None, taskCtx.measures) }.toIterator
        }
      } finally {
        cleanup(taskCtx)
      }
    }
  }

  /**
    * Perform task initialization.
    * @param ctx The training context.
    * @param inputRows The Spark rows for a partition as an iterator.
    * @return Information about the context of the task execution.
    */
  private def initialize(ctx: TrainingContext, inputRows: Iterator[Row]): PartitionTaskContext = {
    val partitionId = getPartitionId
    val taskMeasures = new TaskExecutionMeasures(partitionId)
    taskMeasures.markInitializationStart()
    val taskId = LightGBMUtils.getTaskId
    // TODO put back  if (ctx.trainingParams.generalParams.verbosity > 1)
      ctx.log.info(s"Initializing partition $partitionId and taskid $taskId running on executor $getExecutorId")
    val isEmptyPartition = !inputRows.hasNext
    // Note: the first valid worker with non-empty partitions sets the main executor worker, other workers read it
    if (!isEmptyPartition) ctx.sharedState().linkMainExecutorWorker()

    val mainExecutorWorkerId = if (isEmptyPartition) -1 else ctx.sharedState().mainExecutorWorker.get
    val isMainWorker = mainExecutorWorkerId == taskId
    val shouldExecuteTraining: Boolean =  if (isEmptyPartition) false
      else if (!ctx.useSingleDatasetMode) true
      else isMainWorker

    if (ctx.useSingleDatasetMode)
      ctx.log.info(s"Using singleDatasetMode. Task: $taskId, PartId: $partitionId. Main task: $mainExecutorWorkerId")


    taskMeasures.markLibraryInitializationStart()
    LightGBMUtils.initializeNativeLibrary()
    taskMeasures.markLibraryInitializationStop()

    val networkInfo = NetworkManager.getGlobalNetworkInfo(ctx, partitionId, shouldExecuteTraining, taskMeasures)

    // Return booster only from main worker to reduce network communication overhead
    val shouldReturnBooster = if (isEmptyPartition) false
      else if (!shouldExecuteTraining) false
      else networkInfo.localListenPort == NetworkManager.getMainWorkerPort(networkInfo.lightgbmNetworkString, ctx.log)

    ctx.log.info(s"Initializing custom partition $partitionId and taskid $taskId running on executor $getExecutorId")
    val taskCtx: PartitionTaskContext = initializeInternal(PartitionTaskContext(ctx,
                                            partitionId,
                                            taskId,
                                            taskMeasures,
                                            networkInfo,
                                            isMainWorker,
                                            shouldExecuteTraining,
                                            isEmptyPartition,
                                            shouldReturnBooster))
    // TODO put back  if (ctx.trainingParams.generalParams.verbosity > 1)
      ctx.log.info(s"Done initializing partition $partitionId and taskid $taskId running on executor $getExecutorId")
    taskMeasures.markInitializationStop()
    taskCtx
  }

  /**
    * Prepare any data objects for this particular partition.  Implement for specific execution modes.
    * @param ctx The training context.
    * @param inputRows The Spark rows for a partition as an iterator.
    * @return Any intermediate data state (used mainly by bulk execution mode) to pass to future stages.
    */
  private def preparePartitionData(ctx: PartitionTaskContext, inputRows: Iterator[Row]): PartitionDataState = {
    ctx.log.info(s"starting data preparation on partition ${ctx.partitionId}")
    ctx.measures.markDataPreparationStart()
    val state = preparePartitionDataInternal(ctx, inputRows)
    ctx.measures.markDataPreparationStop()
    ctx.log.info(s"done with data preparation on partition ${ctx.partitionId}")
    state
  }

  /**
    * Load a data partition into Datasets and execute LightGBM training iterations.
    * Note that this method should only be called for "active" threads that created a final Dataset, and not
    * for ones that were empty or were only used to load temporary Datasets that were merged into a centralized one.
    * @param ctx The training context.
    * @param dataState Any intermediate data state (used mainly by bulk execution mode).
    * @return LightGBM booster iterator (to comply with Spark mapPartition API), that is either empty or
    *         has the resulting booster as the only element.
    */
  private def finalizeDatasetAndTrain(ctx: PartitionTaskContext,
                                      dataState: PartitionDataState): Iterator[PartitionResult] = {
    ctx.measures.isActiveTrainingTask = true
    val trainDataset: LightGBMDataset = generateFinalDataset(ctx, dataState, forValidation = false, None)
    try {
      val validDatasetOpt: Option[LightGBMDataset] =
        if (!ctx.trainingCtx.hasValidationData) None
        else Option(generateFinalDataset(ctx, dataState, forValidation = true, Some(trainDataset)))

      try {
        ctx.log.info(s"Creating LightGBM Booster for partition ${ctx.partitionId}")
        val booster = createBooster(ctx.trainingCtx.trainingParams, trainDataset, validDatasetOpt)
        val state = PartitionTaskTrainingState(ctx, booster)
        try {
          val bestIterResult = executeTrainingIterations(state)
          getReturnBoosterAsIterator(state, bestIterResult)
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

  /** Cleanup the task
    *
    * @param ctx The training context.
    */
  private def cleanup(ctx: PartitionTaskContext): Unit = {
    ctx.log.info(s"Beginning cleanup for partition ${ctx.partitionId}")

    // Finalize network when done
    if (ctx.shouldExecuteTraining) LightGBMUtils.validate(lightgbmlib.LGBM_NetworkFree(), "Finalize network")

    cleanupInternal(ctx)

    ctx.measures.markTaskEnd()
    ctx.log.info(s"Done with cleanup for partition ${ctx.partitionId}")
  }

  /**
    * Initialize and customize the context for the task.
    * @param ctx The training context.
    * @return The updated context information for the task.
    */
  protected def initializeInternal(ctx: PartitionTaskContext): PartitionTaskContext = {
    ctx  // By default, just no-op
  }

  /**
    * Prepare any data objects for this particular partition.  Implement for specific execution modes.
    * @param ctx The training context.
    * @param inputRows The Spark rows for a partition as an iterator.
    * @return Any intermediate data state (used mainly by bulk execution mode) to pass to future stages.
    */
  protected def preparePartitionDataInternal(ctx: PartitionTaskContext, inputRows: Iterator[Row]): PartitionDataState

  /**
    * Generate the final dataset for this task.  Internal implementation for specific execution modes.
    * @param ctx The training context.
    * @param dataState Any intermediate data state (used mainly by bulk execution mode).
    * @param forValidation Whether to generate the final training dataset or the validation dataset.
    * @param referenceDataset A reference dataset to start with (used mainly for validation dataset).
    */
  protected def generateFinalDatasetInternal(ctx: PartitionTaskContext,
                                             dataState: PartitionDataState,
                                             forValidation: Boolean,
                                             referenceDataset: Option[LightGBMDataset]): LightGBMDataset


  /** Cleanup the task
    *
    * @param ctx The training context.
    */
  protected def cleanupInternal(ctx: PartitionTaskContext): Unit = {
  }

  /**
    * Generate the final dataset for this task.  This should only be run be tasks that will participate in
    * the training rounds, i.e. in useSingleDataset mode it will only be 1 task/executor.
    * @param ctx The training context.
    * @param dataState Any intermediate data state (used mainly by bulk execution mode).
    * @param forValidation Whether to generate the final training dataset or the validation dataset.
    * @param referenceDataset A reference dataset to start with (used mainly for validation dataset).
    * @return LightGBM dataset Java wrapper.
    */
  private def generateFinalDataset(ctx: PartitionTaskContext,
                                   dataState: PartitionDataState,
                                   forValidation: Boolean,
                                   referenceDataset: Option[LightGBMDataset]): LightGBMDataset = {
    ctx.log.info(s"Getting final Dataset for partition ${ctx.partitionId}. isValidation: $forValidation")
    if (forValidation) {
      beforeGenerateValidDataset(ctx)
      ctx.measures.markValidationDatasetStart()
    } else {
      beforeGenerateTrainDataset(ctx)
      ctx.measures.markDatasetCreationStart()
    }

    val dataset = generateFinalDatasetInternal(ctx, dataState, forValidation, referenceDataset)

    // Validate generated dataset has the correct number of rows and cols
    dataset.validateDataset()

    if (forValidation) {
      afterGenerateValidDataset(ctx)
      ctx.measures.markValidationDatasetStop()
    } else {
      ctx.measures.markDatasetCreationStop()
      afterGenerateTrainDataset(ctx)
    }

    dataset
  }

  private def getReturnBoosterAsIterator(state: PartitionTaskTrainingState,
                                         bestIterationResult: Option[Int]) = {
    if (state.ctx.shouldReturnBooster) {
      val model = state.booster.saveToString(bestIterationResult)
      val modelBooster = new LightGBMBooster(model)
      // Set best iteration on booster if hit early stopping criteria in trainCore
      bestIterationResult.foreach(modelBooster.setBestIteration)
      Iterator.single(PartitionResult(Option(modelBooster), state.ctx.measures))
    } else {
      Iterator.single(PartitionResult(None, state.ctx.measures))
    }
  }
}
