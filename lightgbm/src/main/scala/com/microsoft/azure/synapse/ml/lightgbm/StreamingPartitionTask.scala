// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.azure.synapse.ml.lightgbm

import com.microsoft.azure.synapse.ml.lightgbm.dataset.DatasetUtils.getArrayType
import com.microsoft.azure.synapse.ml.lightgbm.dataset.{LightGBMDataset, SampledData}
import com.microsoft.azure.synapse.ml.lightgbm.swig._
import com.microsoft.ml.lightgbm._
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.sql._
import org.apache.spark.sql.types.StructType

import scala.annotation.tailrec
import scala.language.existentials

case class StreamingState(ctx: PartitionTaskContext,
                          dataset: LightGBMDataset) {
  val trainingCtx: TrainingContext = ctx.trainingCtx
  val columnParams: ColumnParams = trainingCtx.columnParams
  val schema: StructType = trainingCtx.schema
  val numCols: Int = trainingCtx.numCols
  val numInitScoreClasses: Int = trainingCtx.numInitScoreClasses
  val microBatchSize: Int = trainingCtx.microBatchSize

  val isSparse: Boolean = ctx.sharedState.isSparse.get
  val isDense: Boolean = !isSparse
  val hasWeights: Boolean = trainingCtx.hasWeights
  val hasInitialScores: Boolean = trainingCtx.hasInitialScores
  val hasGroups: Boolean = trainingCtx.hasGroups

  /* Buffers for holding micro-batch data */

  // dense data
  lazy val featureDataBuffer: DoubleSwigArray = new DoubleSwigArray(microBatchSize * numCols)

  // sparse data
  lazy val indptrBuffer: IntSwigArray = new IntSwigArray(microBatchSize + 1)
  lazy val indicesBuffer: IntSwigArray = new IntSwigArray(microBatchSize * numCols) // allocate max space
  lazy val valBuffer: DoubleSwigArray = new DoubleSwigArray(microBatchSize * numCols) // allocate max space

  // metadata
  val labelBuffer: FloatSwigArray = new FloatSwigArray(microBatchSize)
  lazy val weightBuffer: FloatSwigArray = new FloatSwigArray(microBatchSize)
  lazy val initScoreBuffer: DoubleSwigArray = new DoubleSwigArray(microBatchSize * numInitScoreClasses)
  lazy val groupBuffer: IntSwigArray = new IntSwigArray(microBatchSize)

  val datasetPointer: SWIGTYPE_p_void = dataset.datasetPtr
  val featureDataPtr: SWIGTYPE_p_void  =
    if (isSparse) null  // scalastyle:ignore null
    else lightgbmlib.double_to_voidp_ptr(featureDataBuffer.array)
  val indptrPtr: SWIGTYPE_p_void  =
    if (isDense) null  // scalastyle:ignore null
    else lightgbmlib.int_to_voidp_ptr(indptrBuffer.array)
  val indicesPtr: SWIGTYPE_p_int  =
    if (isDense) null  // scalastyle:ignore null
    else indicesBuffer.array
  val valPtr: SWIGTYPE_p_void  =
    if (isDense) null  // scalastyle:ignore null
    else lightgbmlib.double_to_voidp_ptr(valBuffer.array)

  val labelPtr: SWIGTYPE_p_float = labelBuffer.array
  val weightPtr: SWIGTYPE_p_float =
    if (hasWeights) weightBuffer.array
    else null  // scalastyle:ignore null
  val initScorePtr: SWIGTYPE_p_double =
    if (hasInitialScores) initScoreBuffer.array
    else null  // scalastyle:ignore null
  val groupPtr: SWIGTYPE_p_int =
    if (hasGroups) groupBuffer.array
    else null  // scalastyle:ignore null

  val featureIndex: Int = schema.fieldIndex(columnParams.featuresColumn)
  val labelIndex: Int = schema.fieldIndex(columnParams.labelColumn)
  val weightIndex: Int = if (hasWeights) schema.fieldIndex(columnParams.weightColumn.get) else 0
  val initScoreIndex: Int = if (hasInitialScores) schema.fieldIndex(columnParams.initScoreColumn.get) else 0
  val groupIndex: Int = if (hasGroups) schema.fieldIndex(columnParams.groupColumn.get) else 0

  if (isSparse) indptrBuffer.setItem(0, 0)  // every micro-batch starts with index 0

  def delete(): Unit = {
    // Delete all the temporary micro-batch marshaling buffers
    if (isDense) lightgbmlib.delete_doubleArray(featureDataBuffer.array)
    else {
      lightgbmlib.delete_intArray(indptrBuffer.array)
      lightgbmlib.delete_intArray(indicesBuffer.array)
      lightgbmlib.delete_doubleArray(valBuffer.array)
    }

    lightgbmlib.delete_floatArray(labelBuffer.array)
    if (hasWeights) lightgbmlib.delete_floatArray(weightBuffer.array)
    if (hasInitialScores) lightgbmlib.delete_doubleArray(initScoreBuffer.array)
    if (hasGroups) lightgbmlib.delete_intArray(groupBuffer.array)
  }
}

/**
  * Class for handling the execution of streaming-based Tasks on workers for each partition.
  */
class StreamingPartitionTask extends BasePartitionTask {
  override protected def initializeInternal(ctx: PartitionTaskContext): PartitionTaskContext = {
    val shouldCalcValidationData =
      if (ctx.trainingCtx.hasValidationData) {
        ctx.sharedState.linkValidationDatasetWorker()
        ctx.sharedState.validationDatasetWorker.get == LightGBMUtils.getTaskId
      } else false

    // Update the row count for the tasks in this executor
    val rowCount = ctx.trainingCtx.partitionCounts.get(ctx.partitionId)
    ctx.sharedState.datasetState.streamingRowCounts.update(ctx.partitionId, rowCount)

    // Streaming always uses 1 Dataset per executor, so we need to add to synchronization stop
    if (!ctx.isMainWorker) ctx.sharedState.incrementDataPrepDoneSignal(log)

    ctx.setShouldCalcValidationData(shouldCalcValidationData)
  }

  protected def preparePartitionDataInternal(ctx: PartitionTaskContext,
                                             inputRows: Iterator[Row]): PartitionDataState = {
    // If this is the task that will execute training, first create the empty Dataset from sampled data
    if (ctx.shouldExecuteTraining) {
      ctx.sharedState.datasetState.streamingDataset = Option(createSharedExecutorDataset(ctx))
      ctx.sharedState.helperStartSignal.countDown()
    } else {
      // This must be a task that just loads data and exits, so wait for the shared Dataset to be created
      ctx.sharedState.helperStartSignal.await()
    }

    // Make sure isSparse is set with a value
    val rowIterator = if (ctx.sharedState.isSparse.isEmpty) {
      val (newRowsIter: Iterator[Row], isSparseHere: Boolean) =
        getArrayType(inputRows,
          ctx.trainingCtx.trainingParams.executionParams.matrixType,
          ctx.trainingCtx.columnParams.featuresColumn)
      ctx.sharedState.linkIsSparse(isSparseHere)
      newRowsIter
    } else inputRows

    insertRowsIntoTrainingDataset(ctx, rowIterator)

    // Now handle validation data, which comes from a broadcast-ed hardcoded array
    if (ctx.shouldCalcValidationDataset) {
      ctx.sharedState.validationDatasetState.streamingDataset = Option(generateOptValidationDataset(ctx))
    }

    // streaming does not use data state (it stores intermediate results in the context shared state),
    // so just return a stub
    PartitionDataState(None, None)
  }

  protected def generateFinalDatasetInternal(ctx: PartitionTaskContext,
                                             dataState: PartitionDataState,
                                             forValidation: Boolean,
                                             referenceDataset: Option[LightGBMDataset]): LightGBMDataset = {
    // We have already calculated and finished the validation Dataset in the preparation stage
    if (forValidation) ctx.sharedState.validationDatasetState.streamingDataset.get
    else {
      val dataset = ctx.sharedState.datasetState.streamingDataset.get
      LightGBMUtils.validate(lightgbmlib.LGBM_DatasetMarkFinished(dataset.datasetPtr),
        "Dataset mark finished")
      dataset
    }
  }

  private def generateOptValidationDataset(ctx: PartitionTaskContext): LightGBMDataset = {
    val validationData = ctx.trainingCtx.validationData.get.value

    val validationDataset = createSharedValidationDataset(ctx, validationData.length)

    insertRowsIntoDataset(
      ctx,
      validationDataset,
      validationData.toIterator,
      0,
      validationData.length)

    // Complete the dataset here since we only add data to it once
    LightGBMUtils.validate(lightgbmlib.LGBM_DatasetMarkFinished(validationDataset.datasetPtr),
      "Dataset mark finished")
    validationDataset
  }

  private def insertRowsIntoTrainingDataset(ctx: PartitionTaskContext, inputRows: Iterator[Row]): Unit = {
    val partitionRowCount = ctx.trainingCtx.partitionCounts.get(ctx.partitionId).toInt
    val partitionRowOffset = ctx.streamingPartitionOffset
    val isSparse = ctx.sharedState.isSparse.get
    log.info(s"Inserting rows into training Dataset from partition ${ctx.partitionId}, " +
      s"size $partitionRowCount, offset: $partitionRowOffset, sparse: $isSparse")
    val dataset = ctx.sharedState.datasetState.streamingDataset.get

    insertRowsIntoDataset(ctx, dataset, inputRows, partitionRowOffset, partitionRowOffset + partitionRowCount)

    log.info(s"inserted $partitionRowCount partition ${ctx.partitionId} " +
      s"rows into shared training dataset at offset $partitionRowOffset")
  }

  private def insertRowsIntoDataset(ctx: PartitionTaskContext,
                                    dataset: LightGBMDataset,
                                    inputRows: Iterator[Row],
                                    startIndex: Int,
                                    stopIndex: Int): Unit = {
    val state = StreamingState(ctx, dataset)
    try {
      if (ctx.sharedState.isSparse.get)
        pushSparseMicroBatches(state, inputRows, startIndex, stopIndex)
      else
        pushDenseMicroBatches(state, inputRows, startIndex, stopIndex)
    } finally {
      state.delete()
    }
  }

  /* TODO decide fate of this method
  @tailrec
  private def createDatasetChunks(ctx: PartitionTaskContext,
                                  inputRows: Iterator[Row],
                                  sharedDatasetState: SharedDatasetState,
                                  chunkSize: Int): Unit = {
    // Generate the "empty" dataset
    val isSparse = ctx.sharedState.isSparse.get
    ctx.log.info(s"LightGBM task creating Dataset in partition ${ctx.partitionId}, size $chunkSize, sparse: $isSparse")
    val dataset = getReferenceDataset(ctx, chunkSize)

    // Initialize state buffers, and load 1 dataset chunk.
    // If we run out of rows, the dataset is "partial", but that is tracked in the LightBGM layer as num_pushed_rows()
    val state = StreamingState(ctx, dataset)
    try {
      if (ctx.sharedState.isSparse.get)
        pushSparseMicroBatches(state, chunkSize, inputRows, 0)
      else
        pushDenseMicroBatches(state, chunkSize, inputRows, 0)
    } finally {
      state.delete()
    }

    // Now store it in the shared state for use later by whatever thread will make the final dataset used for training
    sharedDatasetState.addStreamingDataset(ctx.partitionId, dataset)

    // If there are still more rows in the partition, create another dataset.
    // Ideally we'd always make 1 dataset, but this gives us the flexibility to be off a little for any reason.
    // Even 1 extra row will get its own dataset and then be coalesced later.
    if (inputRows.hasNext) {
      ctx.log.info(s"LightGBM task creating more Datasets in partition ${ctx.partitionId}")
      createDatasetChunks(ctx, inputRows, sharedDatasetState, chunkSize)
    }
  } */

  @tailrec
  private def pushDenseMicroBatches(state: StreamingState,
                                    inputRows: Iterator[Row],
                                    startIndex: Int,
                                    stopIndex: Int): Unit = {
    // Account for stopping early due to partial micro-batch
    val maxBatchSize = Math.min(state.microBatchSize, stopIndex - startIndex)
    val count =
      if (maxBatchSize == 0) 0
      else loadOneDenseMicroBatchBuffer(state, inputRows, 0, maxBatchSize)
    if (count > 0) {
      log.info(s"Pushing $count rows at $startIndex")
      LightGBMUtils.validate(lightgbmlib.LGBM_DatasetPushRowsWithMetadata(
        state.datasetPointer,
        state.featureDataPtr,
        lightgbmlibConstants.C_API_DTYPE_FLOAT64,
        count,
        state.numCols,
        startIndex,
        state.labelPtr,
        state.weightPtr,
        state.initScorePtr,
        state.groupPtr), "Dataset push dense micro-batch")

      // might be more rows, so continue with tail recursion at next index
      pushDenseMicroBatches(state, inputRows, startIndex + count, stopIndex)
    }
  }

  @tailrec
  private def pushSparseMicroBatches(state: StreamingState,
                                     inputRows: Iterator[Row],
                                     startIndex: Int,
                                     stopIndex: Int): Unit = {
    // Account for stopping early due to partial micro-batch
    val maxBatchSize = Math.min(state.microBatchSize, stopIndex - startIndex)
    val (microBatchRowCount: Int, microBatchElementCount: Int) =
      if (maxBatchSize == 0) (0, 0)
      else loadOneSparseMicroBatchBuffer(state, inputRows, 0, 0, maxBatchSize)
    if (microBatchRowCount > 0) {
      // If we have only a partial micro-batch, and we have multi-class initial scores (i.e. numClass > 1),
      // we need to re-coalesce the data since it was stored column-wise based on original microBatchSize
      if (state.hasInitialScores &&  state.microBatchSize != microBatchRowCount) {
        (1 until state.numInitScoreClasses).foreach { i =>
          (0 until microBatchRowCount).foreach { j => {
            val score = state.initScoreBuffer.getItem(i * state.microBatchSize + j)
            state.initScoreBuffer.setItem(i * microBatchRowCount + j, score)}}
        }
      }
      LightGBMUtils.validate(lightgbmlib.LGBM_DatasetPushRowsByCSRWithMetadata(
        state.datasetPointer,
        state.indptrPtr,
        lightgbmlibConstants.C_API_DTYPE_INT32,
        state.indicesPtr,
        state.valPtr,
        lightgbmlibConstants.C_API_DTYPE_FLOAT64,
        microBatchRowCount + 1,
        microBatchElementCount,
        startIndex,
        state.labelPtr,
        state.weightPtr,
        state.initScorePtr,
        state.groupPtr), "Dataset push CSR micro-batch")

      // might be more rows, so continue with tail recursion at next index
      pushSparseMicroBatches(state, inputRows, startIndex + microBatchRowCount, stopIndex)
    } else {
      log.info(s"LightGBM pushed $startIndex in partition ${state.ctx.partitionId}")
    }
  }

  @tailrec
  private def loadOneDenseMicroBatchBuffer(state: StreamingState,
                                           inputRows: Iterator[Row],
                                           currentCount: Int,
                                           maxBatchCount: Int): Int = {
    if (inputRows.hasNext && currentCount < maxBatchCount) {
      val row = inputRows.next()
      // Each row might be either sparse or dense, so convert to overall dense format
      row.getAs[Any](state.featureIndex) match {
        case dense: DenseVector => dense.values.zipWithIndex.foreach { case (x, i) =>
          state.featureDataBuffer.setItem(currentCount * state.numCols + i, x) }
        case sparse: SparseVector => sparse.toArray.zipWithIndex.foreach { case (x, i) =>
          state.featureDataBuffer.setItem(currentCount * state.numCols + i, x) }
      }

      loadOneMetadataRow(state, row, currentCount)

      // We have not reached the end of the micro-batch or Rows, so continue with tail recursion
      loadOneDenseMicroBatchBuffer(state, inputRows, currentCount + 1, maxBatchCount)
    } else currentCount
  }

  @tailrec
  private def loadOneSparseMicroBatchBuffer(state: StreamingState,
                                            inputRows: Iterator[Row],
                                            batchRowCount: Int,
                                            elementCount: Int,
                                            maxBatchCount: Int): (Int, Int) = {
    if (inputRows.hasNext && batchRowCount < maxBatchCount) {
      val row = inputRows.next()
      // Each row might be either sparse or dense, so convert to overall sparse format
      val sparseVector = row.getAs[Any](state.featureIndex) match {
        case dense: DenseVector => dense.toSparse
        case sparse: SparseVector => sparse
        case _ => throw new Exception(row.getAs[Any](state.featureIndex).toString)
      }

      val rowElementCount = sparseVector.values.length
      sparseVector.values.zipWithIndex.foreach { case (value, i) =>
        state.valBuffer.setItem(elementCount + i, value) }
      sparseVector.indices.zipWithIndex.foreach { case (index, i) =>
        state.indicesBuffer.setItem(elementCount + i, index) }
      state.indptrBuffer.setItem(batchRowCount + 1, rowElementCount)

      loadOneMetadataRow(state, row, batchRowCount)

      // We have not reached the end of the micro-batch or Rows, so continue with tail recursion
      loadOneSparseMicroBatchBuffer(state, inputRows, batchRowCount + 1, elementCount + rowElementCount, maxBatchCount)
    } else (batchRowCount, elementCount)
  }

  private def loadOneMetadataRow(state: StreamingState, row: Row, index: Int): Unit = {
    state.labelBuffer.setItem(index, row.getDouble(state.labelIndex).toFloat)
    if (state.hasWeights) state.weightBuffer.setItem(index, row.getDouble(state.weightIndex).toFloat)
    if (state.hasGroups) state.groupBuffer.setItem(index, row.getAs[Int](state.groupIndex))

    // Initial scores are passed in column-based format, where the score for each class is contiguous
    if (state.hasInitialScores) {
      if (row.schema(state.initScoreIndex).dataType == VectorType) // TODO cache bool?
        row.getAs[DenseVector](state.initScoreIndex).values.zipWithIndex.foreach {
          case (value, i) => state.initScoreBuffer.setItem(index + state.microBatchSize * i, value) }
      else
        state.initScoreBuffer.setItem(index, row.getDouble(state.initScoreIndex))
    }
  }

  private def createSharedExecutorDataset(ctx: PartitionTaskContext): LightGBMDataset = {
    // The sample data is broadcast from Spark, so retrieve it
    ctx.measures.markSamplingStart()
    val numRows = ctx.streamingExecutorRowCount
    val sampledRowData = ctx.trainingCtx.broadcastedSampleData.get.value

    // create properly formatted sampled data
    log.info(s"Creating sample data with ${sampledRowData.length} samples")
    val datasetVoidPtr = lightgbmlib.voidpp_handle()
    val sampledData: SampledData = new SampledData(sampledRowData.length, ctx.trainingCtx.numCols)
    try {
      sampledRowData.zipWithIndex.foreach(rowAndIndex =>
        sampledData.pushRow(rowAndIndex._1, rowAndIndex._2, ctx.trainingCtx.columnParams.featuresColumn))
      ctx.measures.markSamplingStop()

      // Convert byte array to native memory
      log.info(s"Creating empty dense training dataset with $numRows rows, config:${ctx.trainingCtx.datasetParams}")
      // Generate the dataset for features
      val datasetVoidPtr = lightgbmlib.voidpp_handle()
      LightGBMUtils.validate(lightgbmlib.LGBM_DatasetCreateFromSampledColumn(
        sampledData.getSampleData,
        sampledData.getSampleIndices,
        ctx.trainingCtx.numCols,
        sampledData.getRowCounts,
        sampledData.numRows,
        numRows.toInt, // TODO this limits Dataset size to int?
        ctx.trainingCtx.datasetParams,
        datasetVoidPtr), "Dataset create")

      val datasetPtr: SWIGTYPE_p_void = lightgbmlib.voidpp_value(datasetVoidPtr)
      LightGBMUtils.validate(
        lightgbmlib.LGBM_DatasetSetWaitForManualFinish(datasetPtr, 1),
        "Dataset LGBM_DatasetSetWaitForManualFinish")

      new LightGBMDataset(datasetPtr)
    } finally {
      sampledData.delete()
      lightgbmlib.delete_voidpp(datasetVoidPtr)
    }
  }

  private def createSharedValidationDataset(ctx: PartitionTaskContext, rowCount: Int): LightGBMDataset = {
    val pointer = lightgbmlib.voidpp_handle()
    val reference = ctx.sharedState.datasetState.streamingDataset.get.datasetPtr
    LightGBMUtils.validate(
      lightgbmlib.LGBM_DatasetCreateByReference(reference, rowCount, pointer),
      "Dataset create from reference")

    val datasetPtr = lightgbmlib.voidpp_value(pointer)
    LightGBMUtils.validate(
      lightgbmlib.LGBM_DatasetSetWaitForManualFinish(datasetPtr, 1),
      "Dataset LGBM_DatasetSetWaitForManualFinish")

    lightgbmlib.delete_voidpp(pointer)
    new LightGBMDataset(datasetPtr)
  }
}
