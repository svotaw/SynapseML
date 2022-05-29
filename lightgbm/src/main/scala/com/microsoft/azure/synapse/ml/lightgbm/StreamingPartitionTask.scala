// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.azure.synapse.ml.lightgbm

import com.microsoft.azure.synapse.ml.lightgbm.dataset.LightGBMDataset
import com.microsoft.azure.synapse.ml.lightgbm.swig.{DoubleSwigArray, FloatSwigArray, IntSwigArray, SwigUtils}
import com.microsoft.ml.lightgbm.{lightgbmlib, lightgbmlibConstants}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.sql._

import scala.language.existentials

/**
  * Class for handling the execution of streaming-based Tasks on workers for each partition.
  */
class StreamingPartitionTask extends BasePartitionTask {
  def preparePartitionDatasets(ctx: TrainingContext, inputRows: Iterator[Row]): PartitionDataState = {
    val partitionId: Int = ??;
    val numPartitionRows: Long = ctx.partitionCounts.get(partitionId)
    val numCols: Int = ctx.;
    val microBatchSize: Int = 1 // TODO

    // TODO For now, use precalculated partition count to just make 1 chunk per partition
    val chunkSize: Int = numPartitionRows.toInt

    def createDatasetChunks(inputRows: Iterator[Row], chunkSize: Int): Unit = {
      // Generate the dataset for features
      ctx.log.info(s"LightGBM task generating schema for empty sparse dataset with $numPartitionRows")
      val dataset = getReferenceDataset(ctx, numPartitionRows)

      if (ctx.sharedState.isSparse.get)
        pushOneSparseDataset(ctx,
          dataset,
          chunkSize,
          numCols,
          microBatchSize,
          inputRows) // TODO make sparse version
      else pushOneDenseDataset(ctx,
         dataset,
         chunkSize,
         numCols,
         microBatchSize,
         inputRows)

      ctx.sharedState.datasetState.addStreamingDataset(partitionId, dataset)

      if (inputRows.hasNext) {
        createDatasetChunks(inputRows, chunkSize)
      }
    }

    createDatasetChunks(inputRows, chunkSize)

    PartitionDataState(None, None) // streaming does not use data state
  }

  private def pushOneDenseDataset(ctx: TrainingContext,
                                  dataset: LightGBMDataset,
                                  numRows: Int,
                                  numCols: Int,
                                  microBatchSize: Int,
                                  inputRows: Iterator[Row]): Unit = {
    // Initialize all micro-batch buffers
    val featureDataBuffer = new DoubleSwigArray(microBatchSize * numCols)
    val labelBuffer = new FloatSwigArray(microBatchSize)
    val weightBuffer = if (ctx.hasWeights) Option(new FloatSwigArray(microBatchSize)) else None
    val initScoreBuffer = if (ctx.hasInitialScores) Option(new DoubleSwigArray(microBatchSize)) else None
    val groupBuffer = if (ctx.hasGroups) Option(new IntSwigArray(microBatchSize)) else None

    val datasetPointer = dataset.datasetPtr
    val featureDataPtr = lightgbmlib.double_to_voidp_ptr(featureDataBuffer.array)
    val labelPtr = labelBuffer.array
    val weightPtr = if (ctx.hasWeights) weightBuffer.get.array else null
    val initScorePtr = if (ctx.hasInitialScores) initScoreBuffer.get.array else null
    val groupPtr = if (ctx.hasGroups) groupBuffer.get.array else null

    val featureIndex = ctx.schema.fieldIndex(ctx.columnParams.featuresColumn)
    val labelIndex = ctx.schema.fieldIndex(ctx.columnParams.labelColumn)
    val weightIndex = if (ctx.hasWeights) ctx.schema.fieldIndex(ctx.columnParams.weightColumn.get) else 0
    val initScoreIndex = if (ctx.hasInitialScores) ctx.schema.fieldIndex(ctx.columnParams.initScoreColumn.get) else 0
    val groupIndex = if (ctx.hasGroups) ctx.schema.fieldIndex(ctx.columnParams.groupColumn.get) else 0

    try {
      // Generate the dataset for features

      // Push rows 1 by 1 by copying each row into same memory array
      def loadDenseMicroBatchBuffer(inputRows: Iterator[Row], count: Int, maxBatchCount: Int): Int = {
        if (inputRows.hasNext && count < maxBatchCount) {
          val row = inputRows.next()
          row.getAs[Any](featureIndex) match {
            case dense: DenseVector => dense.values.zipWithIndex.foreach(pair =>
                featureDataBuffer.setItem(count * numCols + pair._2, pair._1))
            case sparse: SparseVector => sparse.toArray.zipWithIndex.foreach(pair =>
              featureDataBuffer.setItem(count * numCols + pair._2, pair._1))
          }
          labelBuffer.setItem(count, row.getDouble(labelIndex).toFloat)
          weightBuffer.foreach(buffer => buffer.setItem(count, row.getDouble(weightIndex).toFloat))
          // TODO fix for multiple init_scores
          initScoreBuffer.foreach(buffer => buffer.setItem(count, row.getDouble(initScoreIndex).toFloat))
          groupBuffer.foreach(buffer => buffer.setItem(count, row.getAs[Int](groupIndex)))

          // tail recurse
          loadDenseMicroBatchBuffer(inputRows, count + 1, maxBatchCount)
        } else count
      }

      def pushMicroBatches(inputRows: Iterator[Row], index: Int): Unit = {
        val maxBatchSize = Math.min(microBatchSize, numRows - index)
        val count = if (maxBatchSize == 0) 0 else loadDenseMicroBatchBuffer(inputRows, 0, maxBatchSize)
        if (count > 0) {
          LightGBMUtils.validate(lightgbmlib.LGBM_DatasetPushRowsWithMetadata(
            datasetPointer,
            featureDataPtr,
            lightgbmlibConstants.C_API_DTYPE_FLOAT64,
            count,
            numCols,
            index,
            labelPtr,
            weightPtr,
            initScorePtr,
            groupPtr), "Dataset push micro-batch")

          // tail recursion
          pushMicroBatches(inputRows, index + count)
        }
      }
    } finally {
      // Delete all the temporary micro-batch marshaling buffers
      lightgbmlib.delete_doubleArray(featureDataBuffer.array)
      lightgbmlib.delete_floatArray(labelBuffer.array)
      weightBuffer.map(buffer => lightgbmlib.delete_floatArray(buffer.array))
      initScoreBuffer.map(buffer => lightgbmlib.delete_doubleArray(buffer.array))
      groupBuffer.map(buffer => lightgbmlib.delete_intArray(buffer.array))
    }

    PartitionDataState(None, None) // streaming does not use data state.  Datasets are stored in shared data.
  }

  private def pushOneSparseDataset(ctx: TrainingContext,
                                  dataset: LightGBMDataset,
                                  numRows: Int,
                                  numCols: Int,
                                  microBatchSize: Int,
                                  inputRows: Iterator[Row]): Unit = {
   // TODO
    x
  }

  protected def generateDatasetInternal(ctx: TrainingContext,
                                            forValidation: Boolean,
                                            referenceDataset: Option[LightGBMDataset]): LightGBMDataset = {
    val allDatasets = ctx.sharedState.datasetState.getSharedStreamingDatasets()

    // TODO optimize for 1 dataset? or use first as base?

    val totalNumRows = allDatasets.map(d => d.numData()).sum // TODO get only pushed rows
    ctx.log.info(s"LightGBM task generating final dataset with $totalNumRows")
    val dataset = getReferenceDataset(ctx, totalNumRows)

    val pointer = lightgbmlib.voidpp_handle()
    val allDatasetPts = allDatasets.map(d => d.datasetPtr)
    LightGBMUtils.validate(lightgbmlib.LGBM_DatasetCoalesce(
      dataset.datasetPtr,
      allDatasetPts,
      allDatasets.length), "Dataset push micro-batch")

    val datasetPtr = lightgbmlib.voidpp_value(pointer);
    lightgbmlib.delete_voidpp(pointer)
    new LightGBMDataset(datasetPtr)
  }

  private def getReferenceDataset(ctx: TrainingContext,
                                  numRows: Long): LightGBMDataset = {
    // The definition is broadcast from Spark, so retrieve it
    val serializedDataset: Array[Byte] = ctx.serializedReferenceDataset.get.value

    // Convert byte array to native memory
    val pointer = lightgbmlib.voidpp_handle()
    val nativeByteArray = SwigUtils.byteArrayToNative(serializedDataset)
    LightGBMUtils.validate(lightgbmlib.LGBM_DatasetCreateFromSerializedReference(
      lightgbmlib.byte_to_voidp_ptr(nativeByteArray),
      serializedDataset.length,
      numRows,
      ctx.datasetParams,
      pointer), "Dataset create from reference")

    val datasetPtr = lightgbmlib.voidpp_value(pointer);
    lightgbmlib.delete_voidpp(pointer)
    new LightGBMDataset(datasetPtr)
  }
}
