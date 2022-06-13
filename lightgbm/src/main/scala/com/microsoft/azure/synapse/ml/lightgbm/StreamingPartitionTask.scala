// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.azure.synapse.ml.lightgbm

import com.microsoft.azure.synapse.ml.lightgbm.dataset.DatasetUtils.getArrayType
import com.microsoft.azure.synapse.ml.lightgbm.dataset.{LightGBMDataset}
import com.microsoft.azure.synapse.ml.lightgbm.swig._
import com.microsoft.ml.lightgbm._
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.sql._

import scala.annotation.tailrec
import scala.language.existentials

case class StreamingState(ctx: TrainingContext,
                          dataset: LightGBMDataset,
                          partitionId: Int) {
  val numCols: Int = ctx.numCols
  val numInitScoreClasses: Int = ctx.numInitScoreClasses
  val microBatchSize: Int = ctx.microBatchSize

  val isSparse: Boolean = ctx.sharedState.isSparse.get
  val isDense: Boolean = !isSparse
  val hasWeights: Boolean = ctx.hasWeights
  val hasInitialScores: Boolean = ctx.hasInitialScores
  val hasGroups: Boolean = ctx.hasGroups

  /* Buffers for holding micro-batch data */

  // dense data
  lazy val featureDataBuffer: DoubleSwigArray = new DoubleSwigArray(microBatchSize * ctx.numCols)

  // sparse data
  lazy val indptrBuffer: IntSwigArray = new IntSwigArray(microBatchSize + 1)
  lazy val indicesBuffer: IntSwigArray = new IntSwigArray(microBatchSize * ctx.numCols) // allocate max space
  lazy val valBuffer: DoubleSwigArray = new DoubleSwigArray(microBatchSize * ctx.numCols) // allocate max space

  // metadata
  val labelBuffer: FloatSwigArray = new FloatSwigArray(microBatchSize)
  lazy val weightBuffer: FloatSwigArray = new FloatSwigArray(microBatchSize)
  lazy val initScoreBuffer: DoubleSwigArray = new DoubleSwigArray(microBatchSize * numInitScoreClasses)
  lazy val groupBuffer: IntSwigArray = new IntSwigArray(microBatchSize)

  val datasetPointer: SWIGTYPE_p_void = dataset.datasetPtr
  val featureDataPtr: SWIGTYPE_p_void  =
    if (isSparse) null
    else lightgbmlib.double_to_voidp_ptr(featureDataBuffer.array)
  val indptrPtr: SWIGTYPE_p_void  =
    if (isDense) null
    else lightgbmlib.int_to_voidp_ptr(indptrBuffer.array)
  val indicesPtr: SWIGTYPE_p_int  =
    if (isDense) null
    else indicesBuffer.array
  val valPtr: SWIGTYPE_p_void  =
    if (isDense) null
    else lightgbmlib.double_to_voidp_ptr(valBuffer.array)

  val labelPtr: SWIGTYPE_p_float = labelBuffer.array
  val weightPtr: SWIGTYPE_p_float =
    if (hasWeights) weightBuffer.array
    else null
  val initScorePtr: SWIGTYPE_p_double =
    if (hasInitialScores) initScoreBuffer.array
    else null
  val groupPtr: SWIGTYPE_p_int =
    if (hasGroups) groupBuffer.array
    else null

  val featureIndex: Int = ctx.schema.fieldIndex(ctx.columnParams.featuresColumn)
  val labelIndex: Int = ctx.schema.fieldIndex(ctx.columnParams.labelColumn)
  val weightIndex: Int = if (hasWeights) ctx.schema.fieldIndex(ctx.columnParams.weightColumn.get) else 0
  val initScoreIndex: Int = if (hasInitialScores) ctx.schema.fieldIndex(ctx.columnParams.initScoreColumn.get) else 0
  val groupIndex: Int = if (hasGroups) ctx.schema.fieldIndex(ctx.columnParams.groupColumn.get) else 0

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
  def preparePartitionDataInternal(ctx: TrainingContext,
                                   inputRows: Iterator[Row],
                                   partitionId: Int): PartitionDataState = {

    // Make sure isSparse is set with a value
    val rowIterator = if (!ctx.sharedState.isSparse.isDefined) {
      val (newRowsIter: Iterator[Row], isSparseHere: Boolean) =
        getArrayType(inputRows, ctx.trainingParams.executionParams.matrixType, ctx.columnParams.featuresColumn)
      ctx.sharedState.linkIsSparse(isSparseHere)
      newRowsIter
    } else inputRows

    val numPartitionRows: Long = ctx.partitionCounts.get(partitionId)

    // TODO For now, use precalculated partition count to just force 1 chunk per partition
    val chunkSize: Int = numPartitionRows.toInt

    // We turn the inputRows into a set of "chunks", where each chunk is stored as 1 dataset.
    // These dataset chunks will later be coalesced into 1 dataset per worker for useSingleDataset mode,
    // or 1 dataset per partition otherwise.
    // Ideally we'd create 1 dataset per partition, but allowing multiple gives us flexibility.
    createDatasetChunks(ctx, rowIterator, ctx.sharedState.datasetState, partitionId, chunkSize)

    ctx.log.info("created dataset chunks")

    // Now handle validation data, which we can make as 1 Dataset since we have a hardcoded array
    // TODO Consider moving this to shared state and only calculating on main executor task
    val validationDataset = generateOptValidationDataset(ctx, partitionId)

    // streaming does not use data state (it stores intermediate results in the context shared state),
    // so just return a stub
    PartitionDataState(None, None, validationDataset)
  }

  private def generateOptValidationDataset(ctx: TrainingContext, partitionId: Int): Option[LightGBMDataset] = {
    if (ctx.shouldCreateValidationDataset()) {
      val validationData = ctx.validationData.get.value
      createDatasetChunks(ctx,
        validationData.toIterator,
        ctx.sharedState.validationDatasetState,
        partitionId,
        validationData.length)
      val dataset: LightGBMDataset = ctx.sharedState.getSharedValidationDataset()
      ctx.log.info(s"DEBUG get validation dataset id ${dataset.datasetPtr.toString}, size: ${dataset.numData()}")

      // Complete the dataset
      // TODO finalize design of this
      LightGBMUtils.validate(lightgbmlib.LGBM_DatasetMarkFinished(dataset.datasetPtr),
        "Dataset mark finished")
      Option(dataset)
    }
    else None
  }

  @tailrec
  private def createDatasetChunks(ctx: TrainingContext,
                          inputRows: Iterator[Row],
                          sharedDatasetState: SharedDatasetState,
                          partitionId: Int,
                          chunkSize: Int): Unit = {
    // Generate the "empty" dataset
    val isSparse = ctx.sharedState.isSparse.get
    ctx.log.info(s"LightGBM task creating Dataset in partition $partitionId, size $chunkSize, sparse: $isSparse")
    val dataset = getReferenceDataset(ctx, chunkSize)

    // Initialize state buffers, and load 1 dataset chunk.
    // If we run out of rows, the dataset is "partial", but that is tracked in the LightBGM layer as num_pushed_rows()
    val state = new StreamingState(ctx, dataset, partitionId)
    try {
      if (ctx.sharedState.isSparse.get)
        pushSparseMicroBatches(state, chunkSize, inputRows, 0)
      else
        pushDenseMicroBatches(state, chunkSize, inputRows, 0)
    } finally {
      state.delete()
    }

    // Now store it in the shared state for use later by whatever thread will make the final dataset used for training
    sharedDatasetState.addStreamingDataset(partitionId, dataset)

    // If there are still more rows in the partition, create another dataset.
    // Ideally we'd always make 1 dataset, but this gives us the flexibility to be off a little for any reason.
    // Even 1 extra row will get its own dataset and then be coalesced later.
    if (inputRows.hasNext) {
      ctx.log.info(s"LightGBM task creating more Datasets in partition $partitionId")
      createDatasetChunks(ctx, inputRows, sharedDatasetState, partitionId, chunkSize)
    }
  }

  @tailrec
  private def pushDenseMicroBatches(state: StreamingState,
                            maxNumRows: Int,
                            inputRows: Iterator[Row],
                            startIndex: Int): Unit = {
    // Account for stopping early due to partial micro-batch
    val maxBatchSize = Math.min(state.microBatchSize, maxNumRows - startIndex)
    val count =
      if (maxBatchSize == 0) 0
      else loadOneDenseMicroBatchBuffer(state, inputRows, 0, maxBatchSize)
    if (count > 0) {
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
      pushDenseMicroBatches(state, maxNumRows, inputRows, startIndex + count)
    }
  }

  @tailrec
  private def pushSparseMicroBatches(state: StreamingState,
                             maxNumRows: Int,
                             inputRows: Iterator[Row],
                             startIndex: Int): Unit = {
    // Account for stopping early due to partial micro-batch
    val maxBatchSize = Math.min(state.microBatchSize, maxNumRows - startIndex)
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
            state.initScoreBuffer.setItem( i * microBatchRowCount + j, score)}}
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
      pushSparseMicroBatches(state, maxNumRows, inputRows, startIndex + microBatchRowCount)
    } else {
      state.ctx.log.info(s"LightGBM pushed $startIndex in partition ${state.partitionId}")
    }
  }

  @tailrec
  private def loadOneDenseMicroBatchBuffer(state: StreamingState,
                                   inputRows: Iterator[Row],
                                   count: Int,
                                   maxBatchCount: Int): Int = {
    if (inputRows.hasNext && count < maxBatchCount) {
      val row = inputRows.next()
      // Each row might be either sparse or dense, so convert to overall dense format
      row.getAs[Any](state.featureIndex) match {
        case dense: DenseVector => dense.values.zipWithIndex.foreach { case (x, i) =>
          state.featureDataBuffer.setItem(count * state.numCols + i, x) }
        case sparse: SparseVector => sparse.toArray.zipWithIndex.foreach { case (x, i) =>
          state.featureDataBuffer.setItem(count * state.numCols + i, x) }
      }

      loadOneMetadataRow(state, row, count)

      // We have not reached the end of the micro-batch or Rows, so continue with tail recursion
      loadOneDenseMicroBatchBuffer(state, inputRows, count + 1, maxBatchCount)
    } else count
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

  protected def generateFinalDatasetInternal(ctx: TrainingContext,
                                             dataState: PartitionDataState,
                                             partitionIndex: Int,
                                             forValidation: Boolean,
                                             referenceDataset: Option[LightGBMDataset]): LightGBMDataset = {
    // We have already calculated the validation Dataset in the preparation stage
    if (forValidation) ctx.sharedState().getSharedValidationDataset()
    else getFinalTrainingDataset(ctx, partitionIndex)
  }

  protected def getFinalTrainingDataset(ctx: TrainingContext, partitionIndex: Int): LightGBMDataset = {
    if (ctx.useSingleDatasetMode)
      getExecutorTrainingDataset(ctx)
    else
      getPartitionTrainingDataset(ctx, partitionIndex)
  }

  protected def getPartitionTrainingDataset(ctx: TrainingContext, partitionIndex: Int): LightGBMDataset = {
    ctx.log.info(s"getting partition $partitionIndex dataset")

    val partitionDatasets = ctx.sharedState().datasetState.getSharedStreamingDatasets(partitionIndex)
    val partitionDataset = getCoalescedDataset(ctx, partitionDatasets)

    // Datasets are freed as part of coalescing, so remove them from the lists
    ctx.sharedState.datasetState.clearSharedStreamingDatasets(partitionIndex)

    ctx.log.info("done getting final training dataset")
    partitionDataset
  }

  protected def getExecutorTrainingDataset(ctx: TrainingContext): LightGBMDataset = {
    ctx.log.info("getting final single-node merged dataset")

    val allDatasets = ctx.sharedState.datasetState.getSharedStreamingDatasets()
    val executorDataset = getCoalescedDataset(ctx, allDatasets)

    // Datasets are freed as part of coalescing, so remove them from the lists
    ctx.sharedState.datasetState.clearSharedStreamingDatasets()

    ctx.log.info("done getting final training dataset")
    executorDataset
  }

  /**
    * Return a dataset that is a coalesced version of the input array
    * If there is only 1 dataset and it is fully loaded, the method will optimized and return that one
    * Note that input datasets not reused are freed at the native layer
    * @param ctx The training context
    * @param allDatasets The array of datasets to coalesce
    * @return the coalesced dataset, which might be the input one if optimized
    */
  protected def getCoalescedDataset(ctx: TrainingContext,
                                    allDatasets: Array[LightGBMDataset]): LightGBMDataset = {
    val firstDataset = allDatasets(0)
    val isFull = firstDataset.numData() == firstDataset.numPushedData()

    // If there is only 1 dataset and the size is already correct, we can just optimize and finish it
    // This removes the need to copy the data (as done by coalesce)
    if (allDatasets.length == 1 && isFull) {
      LightGBMUtils.validate(lightgbmlib.LGBM_DatasetMarkFinished(firstDataset.datasetPtr),
        "Dataset mark finished")
      firstDataset
    } else {
      val totalNumRows = allDatasets.map(d => d.numPushedData()).sum
      ctx.log.info(s"LightGBM task generating final dataset with $totalNumRows")
      val coalescedDataset = getReferenceDataset(ctx, totalNumRows, false)

      val datasetNativePointers = new VoidPointerSwigArray(allDatasets.length)
      allDatasets.zipWithIndex.foreach { case (ptr, i) => datasetNativePointers.setItem(i, ptr.datasetPtr) }
      LightGBMUtils.validate(lightgbmlib.LGBM_DatasetCoalesce(coalescedDataset.datasetPtr,
        datasetNativePointers.array,
        allDatasets.length),
        "Dataset coalesce")

      allDatasets.foreach(ds => LightGBMUtils.validate(lightgbmlib.LGBM_DatasetFree(ds.datasetPtr),
        "Dataset free"))
      coalescedDataset
    }
  }

  private def getReferenceDataset(ctx: TrainingContext,
                                  numRows: Long,
                                  forStreaming: Boolean = true): LightGBMDataset = {
    // The definition is broadcast from Spark, so retrieve it
    val serializedDataset: Array[Byte] = ctx.serializedReferenceDataset.get.value

    // Convert byte array to native memory
    val pointer = lightgbmlib.voidpp_handle()
    val nativeByteArray = SwigUtils.byteArrayToNative(serializedDataset)
    LightGBMUtils.validate(
      lightgbmlib.LGBM_DatasetCreateFromSerializedReference(lightgbmlib.byte_to_voidp_ptr(nativeByteArray),
                                                            serializedDataset.length,
                                                            numRows,
                                                            ctx.numInitScoreClasses,
                                                            ctx.datasetParams,
                                                            pointer),
      "Dataset create from reference")

    val datasetPtr = lightgbmlib.voidpp_value(pointer)
    if (forStreaming) {
      LightGBMUtils.validate(
        lightgbmlib.LGBM_DatasetSetWaitForManualFinish(datasetPtr, 1),
        "Dataset LGBM_DatasetSetWaitForManualFinish")
    }

    lightgbmlib.delete_voidpp(pointer)
    new LightGBMDataset(datasetPtr)
  }
}
