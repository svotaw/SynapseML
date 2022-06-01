// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.azure.synapse.ml.lightgbm

import com.microsoft.azure.synapse.ml.lightgbm.dataset.DatasetUtils.getArrayType
import com.microsoft.azure.synapse.ml.lightgbm.dataset.{LightGBMDataset, PeekingIterator}
import com.microsoft.azure.synapse.ml.lightgbm.swig.{DoubleSwigArray, FloatSwigArray, IntSwigArray, SwigUtils, VoidPointerSwigArray}
import com.microsoft.ml.lightgbm.{SWIGTYPE_p_double, SWIGTYPE_p_float, SWIGTYPE_p_int, SWIGTYPE_p_void, lightgbmlib, lightgbmlibConstants}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.sql._

import scala.language.existentials

class StreamingState(ctx: TrainingContext,
                     dataset: LightGBMDataset) { // TODO make microBatchSize part of ctx
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

  var valueCount = 0 // TODO switch to tail recursion
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
  def preparePartitionData(ctx: TrainingContext, inputRows: Iterator[Row], partitionId: Int): PartitionDataState = {

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
    val validationDataset = if (ctx.hasValid) {
      val validationData = ctx.validationData.get.value
      createDatasetChunks(ctx,
                          validationData.toIterator,
                          ctx.sharedState.validationDatasetState,
                          partitionId,
                          validationData.length)
      Option(ctx.sharedState.validationDatasetState.getSharedStreamingDatasets().head)
    }
    else None

    // streaming does not use data state (it stores intermediate results in the context shared state),
    // so just return a stub
    PartitionDataState(None, None, validationDataset)
  }

  def createDatasetChunks(ctx: TrainingContext,
                          inputRows: Iterator[Row],
                          sharedDatasetState: SharedDatasetState,
                          partitionId: Int,
                          chunkSize: Int): Unit = {
    // Generate the "empty" dataset
    ctx.log.info(s"LightGBM task generating schema for empty dense dataset with $chunkSize rows")
    val dataset = getReferenceDataset(ctx, chunkSize)

    // Initialize state buffers, and load 1 dataset chunk.
    // If we run out of rows, the dataset is "partial", but that is tracked in the LightBGM layer as num_pushed_rows()
    val state = new StreamingState(ctx, dataset)
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
      createDatasetChunks(ctx, inputRows, sharedDatasetState, partitionId, chunkSize)
    }
  }

  def pushDenseMicroBatches(state: StreamingState,
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

  def pushSparseMicroBatches(state: StreamingState,
                             maxNumRows: Int,
                             inputRows: Iterator[Row],
                             startIndex: Int): Unit = {
    // Account for stopping early due to partial micro-batch
    state.valueCount = 0  // Reset our count of sparse values for a micro-batch
    val maxBatchSize = Math.min(state.microBatchSize, maxNumRows - startIndex)
    val microBatchRowCount =
      if (maxBatchSize == 0) 0
      else loadOneSparseMicroBatchBuffer(state, inputRows, 0, maxBatchSize)
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
        state.valueCount,
        startIndex,
        state.labelPtr,
        state.weightPtr,
        state.initScorePtr,
        state.groupPtr), "Dataset push CSR micro-batch")

      // might be more rows, so continue with tail recursion at next index
      pushSparseMicroBatches(state, maxNumRows, inputRows, startIndex + microBatchRowCount)
    }
  }

  def loadOneDenseMicroBatchBuffer(state: StreamingState,
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

  def loadOneSparseMicroBatchBuffer(state: StreamingState,
                                    inputRows: Iterator[Row],
                                    count: Int,
                                    maxBatchCount: Int): Int = {
    if (inputRows.hasNext && count < maxBatchCount) {
      val row = inputRows.next()
      // Each row might be either sparse or dense, so convert to overall sparse format
      val sparseVector = row.getAs[Any](state.featureIndex) match {
        case dense: DenseVector => dense.toSparse
        case sparse: SparseVector => sparse
      }

      sparseVector.values.zipWithIndex.foreach { case (value, i) =>
        state.valBuffer.setItem(state.valueCount + i, value) }
      sparseVector.indices.zipWithIndex.foreach { case (index, i) =>
        state.indicesBuffer.setItem(state.valueCount + i, index) }
      state.valueCount += count
      state.indptrBuffer.setItem(count + 1, state.valueCount)

      loadOneMetadataRow(state, row, count)

      // We have not reached the end of the micro-batch or Rows, so continue with tail recursion
      loadOneSparseMicroBatchBuffer(state, inputRows, count + 1, maxBatchCount)
    } else count
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
                                            forValidation: Boolean,
                                            referenceDataset: Option[LightGBMDataset]): LightGBMDataset = {
    // We have already calculated the validation Dataset in the preparation stage
    if (forValidation) dataState.streamingValidationSet.get
    else getFinalTrainingDataset(ctx)
  }


  protected def getFinalTrainingDataset(ctx: TrainingContext): LightGBMDataset = {
    ctx.log.info("getting final dataset")

    val allDatasets = ctx.sharedState.datasetState.getSharedStreamingDatasets()

    // TODO optimize for 1 dataset? or use first as base?

    val totalNumRows = allDatasets.map(d => d.numPushedData()).sum
    ctx.log.info(s"LightGBM task generating final dataset with $totalNumRows")
    val dataset = getReferenceDataset(ctx, totalNumRows)

    val datasetNativePointers = new VoidPointerSwigArray(allDatasets.length)
    allDatasets.zipWithIndex.foreach { case (ptr, i) => datasetNativePointers.setItem(i, ptr.datasetPtr) }
    LightGBMUtils.validate(lightgbmlib.LGBM_DatasetCoalesce(dataset.datasetPtr,
                                                            datasetNativePointers.array,
                                                            allDatasets.length),
                "Dataset push micro-batch")

    ctx.log.info("done getting final dataset")

    dataset
  }

  private def getReferenceDataset(ctx: TrainingContext,
                                  numRows: Long): LightGBMDataset = {
    // The definition is broadcast from Spark, so retrieve it
    val serializedDataset: Array[Byte] = ctx.serializedReferenceDataset.get.value

    // Convert byte array to native memory
    val pointer = lightgbmlib.voidpp_handle()
    val nativeByteArray = SwigUtils.byteArrayToNative(serializedDataset)
    LightGBMUtils.validate(
      lightgbmlib.LGBM_DatasetCreateFromSerializedReference(lightgbmlib.byte_to_voidp_ptr(nativeByteArray),
                                                            serializedDataset.length,
                                                            numRows,
                                                            ctx.datasetParams,
                                                            pointer),
      "Dataset create from reference")

    val datasetPtr = lightgbmlib.voidpp_value(pointer);
    lightgbmlib.delete_voidpp(pointer)
    new LightGBMDataset(datasetPtr)
  }
}
