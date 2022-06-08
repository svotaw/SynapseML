// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.azure.synapse.ml.lightgbm

import com.microsoft.azure.synapse.ml.lightgbm.dataset.{BaseChunkedColumns, LightGBMDataset}
import org.apache.spark.sql._

import scala.language.existentials

/**
  * Class for handling the execution of bulk-based Tasks on workers for each partition.
  */
class BulkPartitionTask extends BasePartitionTask {

  def preparePartitionDataInternal(ctx: TrainingContext, inputRows: Iterator[Row], partitionId: Int): PartitionDataState = {
    val aggregatedColumns = {
      val prepAggregatedColumns: BaseChunkedColumns = ctx.sharedState.datasetState.prepBulk(inputRows)
      ctx.sharedState.datasetState.mergeBulk(prepAggregatedColumns)
    }
    val aggregatedValidationColumns = ctx.validationData.map { data =>
      val prepAggregatedColumns: BaseChunkedColumns =
        ctx.sharedState.validationDatasetState.prepBulk(data.value.toIterator)
      ctx.sharedState.validationDatasetState.mergeBulk(prepAggregatedColumns)
    }
    PartitionDataState(Option(aggregatedColumns), aggregatedValidationColumns)
  }

  protected def generateFinalDatasetInternal(ctx: TrainingContext,
                                             dataState: PartitionDataState,
                                             partitionIndex: Int,
                                             forValidation: Boolean,
                                             referenceDataset: Option[LightGBMDataset]): LightGBMDataset = {
    val ac = if (forValidation) dataState.aggregatedValidationData.get else dataState.aggregatedTrainingData.get
    try {
      val datasetInner: LightGBMDataset = ac.generateDataset(referenceDataset, ctx.datasetParams)
      ctx.columnParams.groupColumn.foreach(_ => datasetInner.addGroupColumn(ac.getGroups))
      datasetInner.setFeatureNames(ctx.featureNames, ac.getNumCols)
      datasetInner
    } finally {
      ac.cleanup()
    }
  }
}
