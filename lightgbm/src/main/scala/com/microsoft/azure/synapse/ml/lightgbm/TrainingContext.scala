// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.azure.synapse.ml.lightgbm

import com.microsoft.azure.synapse.ml.io.http.SharedSingleton
import com.microsoft.azure.synapse.ml.lightgbm.params.{BaseTrainParams, ClassifierTrainParams}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.StructType
import org.slf4j.Logger

case class NetworkParams(defaultListenPort: Int,
                         addr: String,
                         port: Int,
                         barrierExecutionMode: Boolean)
case class ColumnParams(labelColumn: String,
                        featuresColumn: String,
                        weightColumn: Option[String],
                        initScoreColumn: Option[String],
                        groupColumn: Option[String])

/**
  * Object to encapsulate all information about a training that does not change during execution
  */
case class TrainingContext(batchIndex: Int,
                           sharedStateSingleton: SharedSingleton[SharedState],
                           schema: StructType,
                           numCols: Int,
                           numInitScoreClasses: Int,
                           microBatchSize: Int,
                           trainingParams: BaseTrainParams,
                           networkParams: NetworkParams,
                           columnParams: ColumnParams,
                           datasetParams: String,
                           featureNames: Option[Array[String]],
                           numTasksPerExecutor: Int,
                           validationData: Option[Broadcast[Array[Row]]],
                           serializedReferenceDataset: Option[Broadcast[Array[Byte]]],
                           partitionCounts: Option[Array[Long]],
                           log: Logger) {
  val isProvideTrainingMetric: Boolean = { trainingParams.isProvideTrainingMetric.getOrElse(false) }
  val improvementTolerance: Double = { trainingParams.generalParams.improvementTolerance }
  val earlyStoppingRound: Int = { trainingParams.generalParams.earlyStoppingRound }

  val useSingleDatasetMode = trainingParams.executionParams.useSingleDatasetMode

  val isClassification: Boolean = { trainingParams.isInstanceOf[ClassifierTrainParams] }

  val hasValid = validationData.isDefined

  val isStreaming: Boolean = trainingParams.executionParams.executionMode == LightGBMConstants.StreamingExecutionMode

  val hasWeights: Boolean = { columnParams.weightColumn.isDefined && columnParams.weightColumn.get.nonEmpty }
  val hasInitialScores: Boolean = { columnParams.initScoreColumn.isDefined &&
                                    columnParams.initScoreColumn.get.nonEmpty }
  val hasGroups: Boolean = { columnParams.groupColumn.isDefined && columnParams.groupColumn.get.nonEmpty }

  val sharedState = sharedStateSingleton.get

  def incrementArrayProcessedSignal(): Int = { sharedState.incrementArrayProcessedSignal(log) }
  def incrementDoneSignal(): Unit = { sharedState.incrementDoneSignal(log) }
}
