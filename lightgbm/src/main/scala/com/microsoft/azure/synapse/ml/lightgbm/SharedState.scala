// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.azure.synapse.ml.lightgbm

import com.microsoft.azure.synapse.ml.lightgbm.dataset.DatasetUtils._
import com.microsoft.azure.synapse.ml.lightgbm.dataset._
import com.microsoft.azure.synapse.ml.lightgbm.params.BaseTrainParams
import com.microsoft.ml.lightgbm.lightgbmlib
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.StructType
import org.slf4j.Logger

import java.util.concurrent.CountDownLatch

class SharedDatasetState(columnParams: ColumnParams,
                         schema: StructType,
                         trainParams: BaseTrainParams,
                         isForValidation: Boolean,
                         sharedState: SharedState) {
  val chunkSize: Int = trainParams.executionParams.chunkSize
  val useSingleDataset: Boolean = trainParams.executionParams.useSingleDatasetMode
  val matrixType: String = trainParams.executionParams.matrixType

  private lazy val streamingPartitionDatasets = scala.collection.mutable.Map[Int, List[LightGBMDataset]]()

  lazy val denseAggregatedColumns: BaseDenseAggregatedColumns = new DenseSyncAggregatedColumns(chunkSize)

  lazy val sparseAggregatedColumns: BaseSparseAggregatedColumns = new SparseSyncAggregatedColumns(chunkSize)

  def prepBulk(iter: Iterator[Row]): BaseChunkedColumns = {
    val (concatRowsIter: Iterator[Row], isSparseHere: Boolean) =
      getArrayType(iter, matrixType, columnParams.featuresColumn)
    val peekableIter = new PeekingIterator(concatRowsIter)
    // Note: the first worker gets to officially set "is sparse", other workers read it
    sharedState.linkIsSparse(isSparseHere)

    if (!sharedState.isSparse.get) {
      new DenseChunkedColumns(peekableIter, columnParams, schema, chunkSize)
    } else {
      new SparseChunkedColumns(peekableIter, columnParams, schema, chunkSize, useSingleDataset)
    }
  }

  def mergeBulk(ts: BaseChunkedColumns): BaseAggregatedColumns = {
    val isSparseVal = sharedState.isSparse.get
    val aggregatedColumns = if (!isSparseVal) {
      if (useSingleDataset) denseAggregatedColumns
      else new DenseAggregatedColumns(chunkSize)
    } else {
      if (useSingleDataset) sparseAggregatedColumns
      else new SparseAggregatedColumns(chunkSize)
    }
    // For the validation Dataset in useSingleDataset mode, we only want 1 copy of the data (otherwise
    // every partition appends the same broadcast-ed data). That one copy will be made by the main execution worker.
    val mergeRowsIntoDataset: Boolean =
      if (!isForValidation) true
      else !useSingleDataset || sharedState.mainExecutorWorker.get == LightGBMUtils.getTaskId
    if (mergeRowsIntoDataset) {
      aggregatedColumns.incrementCount(ts)
    }
    if (useSingleDataset) {
      arrayProcessedSignal.countDown()
      arrayProcessedSignal.await()
    }
    if (mergeRowsIntoDataset) {
      aggregatedColumns.addRows(ts)
    }
    ts.release()
    aggregatedColumns
  }

  @volatile var arrayProcessedSignal: CountDownLatch = new CountDownLatch(0)

  def incrementArrayProcessedSignal(log: Logger): Int = {
    this.synchronized {
      val count = arrayProcessedSignal.getCount.toInt + 1
      arrayProcessedSignal = new CountDownLatch(count)
      log.info(s"Task incrementing ArrayProcessedSignal to $count")
      count
    }
  }

  def addStreamingDataset(partition: Int, dataset: LightGBMDataset): Unit = {
    this.synchronized {
      if (streamingPartitionDatasets.contains(partition)) {
        val currentList = streamingPartitionDatasets(partition)
        streamingPartitionDatasets.update(partition, dataset +: currentList)
      } else {
        streamingPartitionDatasets += partition -> List(dataset)
      }
    }
  }

  def getSharedStreamingDatasets(): Array[LightGBMDataset] =
  {
    streamingPartitionDatasets.flatten(pair => pair._2).toArray
  }

  // TODO need whole array?
  def getSharedStreamingDatasetForPartition(partitionIndex: Int): LightGBMDataset =
  {
    streamingPartitionDatasets(partitionIndex).head
  }

  def freeSharedStreamingDatasets(): Unit = {
    val allDatasets = getSharedStreamingDatasets()
    allDatasets.foreach(ds => LightGBMUtils.validate(lightgbmlib.LGBM_DatasetFree(ds.datasetPtr),
      "Dataset free"))
  }
}

class SharedState(columnParams: ColumnParams,
                  schema: StructType,
                  trainParams: BaseTrainParams) {
  //val useSingleDataset: Boolean = trainParams.executionParams.useSingleDatasetMode
  //val chunkSize: Int = trainParams.executionParams.chunkSize
  //val matrixType: String = trainParams.executionParams.matrixType

  val datasetState: SharedDatasetState = new SharedDatasetState(
    columnParams,
    schema,
    trainParams,
    isForValidation = false,
    this)
  val validationDatasetState: SharedDatasetState = new SharedDatasetState(
    columnParams,
    schema,
    trainParams,
    isForValidation = true,
    this)

  @volatile var isSparse: Option[Boolean] = None
  @volatile var mainExecutorWorker: Option[Long] = None

  def linkIsSparse(isSparse: Boolean): Unit = {
    if (this.isSparse.isEmpty) {
      this.synchronized {
        if (this.isSparse.isEmpty) {
          this.isSparse = Some(isSparse)
        }
      }
    }
  }

  def linkMainExecutorWorker(): Unit = {
    if (this.mainExecutorWorker.isEmpty) {
      this.synchronized {
        if (this.mainExecutorWorker.isEmpty) {
          this.mainExecutorWorker = Some(LightGBMUtils.getTaskId)
        }
      }
    }
  }

  def incrementArrayProcessedSignal(log: Logger): Int = {
    datasetState.incrementArrayProcessedSignal(log)
    validationDatasetState.incrementArrayProcessedSignal(log)
  }

  @volatile var doneSignal: CountDownLatch = new CountDownLatch(0)

  def incrementDoneSignal(log: Logger): Unit = {
    this.synchronized {
      val count = doneSignal.getCount.toInt + 1
      doneSignal = new CountDownLatch(count)
      log.info(s"Task incrementing DoneSignal to $count")
    }
  }

  @volatile var helperStartSignal: CountDownLatch = new CountDownLatch(1)
}
