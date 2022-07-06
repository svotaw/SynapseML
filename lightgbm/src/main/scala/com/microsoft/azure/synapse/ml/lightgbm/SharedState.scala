// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.azure.synapse.ml.lightgbm

import com.microsoft.azure.synapse.ml.lightgbm.dataset._
import com.microsoft.azure.synapse.ml.lightgbm.params.BaseTrainParams
import com.microsoft.ml.lightgbm.lightgbmlib
import org.apache.spark.sql.types.StructType
import org.slf4j.Logger

import java.util.concurrent.CountDownLatch
import scala.collection.concurrent.TrieMap

class SharedDatasetState(columnParams: ColumnParams,
                         schema: StructType,
                         trainParams: BaseTrainParams,
                         isForValidation: Boolean,
                         sharedState: SharedState) {
  val chunkSize: Int = trainParams.executionParams.chunkSize
  val useSingleDataset: Boolean = trainParams.executionParams.useSingleDatasetMode
  val matrixType: String = trainParams.executionParams.matrixType

  val streamingRowCounts = new TrieMap[Long, Long]()
  val streamingRowOffsets = new TrieMap[Long, Long]()

  @volatile var streamingDataset: Option[LightGBMDataset] = None

  private lazy val streamingPartitionDatasets = scala.collection.mutable.Map[Int, List[LightGBMDataset]]()

  lazy val denseAggregatedColumns: BaseDenseAggregatedColumns = new DenseSyncAggregatedColumns(chunkSize)

  lazy val sparseAggregatedColumns: BaseSparseAggregatedColumns = new SparseSyncAggregatedColumns(chunkSize)

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

  def getSharedStreamingDatasets(partitionIndex: Int): Array[LightGBMDataset] =
  {
    streamingPartitionDatasets(partitionIndex).toArray
  }

  def clearSharedStreamingDatasets(): Unit = {
    streamingPartitionDatasets.clear()
  }

  def clearSharedStreamingDatasets(partitionIndex: Int): Unit = {
    streamingPartitionDatasets.update(partitionIndex, List.empty[LightGBMDataset])
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
  @volatile var validationDatasetWorker: Option[Long] = None

  def getSharedTrainingDataset(): LightGBMDataset = {
    // There should only be 1 Dataset in the array
    datasetState.getSharedStreamingDatasets().head
  }

  def getSharedValidationDataset(): LightGBMDataset = {
    // There should only be 1 Dataset in the array
    validationDatasetState.getSharedStreamingDatasets().head
  }

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

  def linkValidationDatasetWorker(): Unit = {
    if (this.validationDatasetWorker.isEmpty) {
      this.synchronized {
        if (this.validationDatasetWorker.isEmpty) {
          this.validationDatasetWorker = Some(LightGBMUtils.getTaskId)
        }
      }
    }
  }

  def incrementArrayProcessedSignal(log: Logger): Int = {
    datasetState.incrementArrayProcessedSignal(log)
    validationDatasetState.incrementArrayProcessedSignal(log)
  }

  @volatile var dataPreparationDoneSignal: CountDownLatch = new CountDownLatch(0)

  def incrementDataPrepDoneSignal(log: Logger): Unit = {
    this.synchronized {
      val count = dataPreparationDoneSignal.getCount.toInt + 1
      dataPreparationDoneSignal = new CountDownLatch(count)
      log.info(s"Task incrementing DataPrepDoneSignal to $count")
    }
  }

  @volatile var helperStartSignal: CountDownLatch = new CountDownLatch(1)
}
