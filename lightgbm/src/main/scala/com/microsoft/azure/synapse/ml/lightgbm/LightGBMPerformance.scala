// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.azure.synapse.ml.lightgbm

class TaskExecutionMeasures(val partitionId: Int) extends Serializable {
  private val startTime = System.currentTimeMillis()
  private var dataPreparationStart: Long = 0
  private var dataPreparationStop: Long = 0
  private var datasetCreationStart: Long = 0
  private var datasetCreationStop: Long = 0
  private var validationDatasetCreationStart: Long = 0
  private var validationDatasetCreationStop: Long = 0
  private var trainingIterationsStart: Long = 0
  private var trainingIterationsStop: Long = 0
  private var endTime: Long = 0

  var isActiveTrainingTask: Boolean = false // TODO make private?

  def markDataPreparationStart(): Unit = { dataPreparationStart = System.currentTimeMillis() }
  def markDataPreparationStop(): Unit = { dataPreparationStop = System.currentTimeMillis() }
  def markDatasetCreationStart(): Unit = { datasetCreationStart = System.currentTimeMillis() }
  def markDatasetCreationStop(): Unit = { datasetCreationStop = System.currentTimeMillis() }
  def markValidationDatasetCreationStart(): Unit = { validationDatasetCreationStart = System.currentTimeMillis() }
  def markValidationDatasetCreationStop(): Unit = { validationDatasetCreationStop = System.currentTimeMillis() }
  def markTrainingIterationsStart(): Unit = { trainingIterationsStart = System.currentTimeMillis() }
  def markTrainingIterationsStop(): Unit = { trainingIterationsStop = System.currentTimeMillis() }
  def markTaskEnd(): Unit = { endTime = System.currentTimeMillis() }

  def dataPreparationTime: Long = { if (dataPreparationStop == 0) 0 else dataPreparationStop - dataPreparationStart }
  def datasetCreationTime: Long = { if (datasetCreationStop == 0) 0 else datasetCreationStop - datasetCreationStart }
  def validationDatasetCreationTime: Long = {
    if (validationDatasetCreationStop == 0) 0 else validationDatasetCreationStop - validationDatasetCreationStart
  }
  def trainingIterationsTime: Long = {
    if (trainingIterationsStop == 0) 0 else trainingIterationsStop - trainingIterationsStart
  }
  def totalTime: Long = { if (endTime == 0) 0 else endTime - startTime }
}

class ExecutionMeasures() extends Serializable {
  private val startTime = System.currentTimeMillis()
  private var validationDataCollectionStart: Long = 0
  private var validationDataCollectionStop: Long = 0
  private var columnStatisticsStart: Long = 0
  private var columnStatisticsStop: Long = 0
  private var rowStatisticsStart: Long = 0
  private var rowStatisticsStop: Long = 0
  private var trainingStart: Long = 0
  private var trainingStop: Long = 0
  private var endTime: Long = 0

  private var taskMeasures: Option[Seq[TaskExecutionMeasures]] = None

  def setTaskMeasures(taskStats: Seq[TaskExecutionMeasures]): Unit = { taskMeasures = Option(taskStats) }

  def markValidationDataCollectionStart(): Unit = { validationDataCollectionStart = System.currentTimeMillis() }
  def markValidationDataCollectionStop(): Unit = { validationDataCollectionStop = System.currentTimeMillis() }
  def markColumnStatisticsStart(): Unit = { columnStatisticsStart = System.currentTimeMillis() }
  def markColumnStatisticsStop(): Unit = { columnStatisticsStop = System.currentTimeMillis() }
  def markRowStatisticsStart(): Unit = { rowStatisticsStart = System.currentTimeMillis() }
  def markRowStatisticsStop(): Unit = { rowStatisticsStop = System.currentTimeMillis() }
  def markTrainingStart(): Unit = { trainingStart = System.currentTimeMillis() }
  def markTrainingStop(): Unit = { trainingStop = System.currentTimeMillis() }
  def markExecutionEnd(): Unit = { endTime = System.currentTimeMillis() }

  def getTaskMeasures(): Seq[TaskExecutionMeasures] = { taskMeasures.get }

  def validationDataCollectionTime(): Long = {
    if (validationDataCollectionStop == 0) 0
    else Math.max(1, validationDataCollectionStop - validationDataCollectionStart) // show at least 1 msec
  }
  def columnStatisticsTime(): Long = {
    if (columnStatisticsStop == 0) 0
    else Math.max(1, columnStatisticsStop - columnStatisticsStart)
  }
  def rowStatisticsTime(): Long = {
    if (rowStatisticsStop == 0) 0
    else Math.max(1, rowStatisticsStop - rowStatisticsStart) }
  def trainingTime(): Long = {
    if (trainingStop == 0) 0
    else Math.max(1, trainingStop - trainingStart) }
  def totalTime: Long = { if (endTime == 0) 0 else endTime - startTime }

  def taskDataPreparationTimes(): Seq[Long] = {
    if (taskMeasures.isDefined) taskMeasures.get.map(measures => measures.dataPreparationTime) else Seq()
  }

  def taskDatasetCreationTimes(): Seq[Long] = {
    if (taskMeasures.isDefined) taskMeasures.get.map(measures => measures.datasetCreationTime) else Seq()
  }

  def taskValidationDatasetCreationTimes(): Seq[Long] = {
    if (taskMeasures.isDefined) taskMeasures.get.map(measures => measures.validationDatasetCreationTime) else Seq()
  }

  def taskTrainingIterationTimes(): Seq[Long] = {
    if (taskMeasures.isDefined) taskMeasures.get.map(measures => measures.trainingIterationsTime) else Seq()
  }

  def taskTotalTimes(): Seq[Long] = {
    if (taskMeasures.isDefined) taskMeasures.get.map(measures => measures.totalTime) else Seq()
  }
}

trait LightGBMPerformance extends Serializable {
  private var performanceMeasures: Option[Array[Option[ExecutionMeasures]]] = None

  protected def initPerformanceMeasures(batchCount: Int): Unit = {
    performanceMeasures = Option(Array.fill(batchCount)(None))
  }

  protected def setBatchPerformanceMeasure(index: Int, measures: ExecutionMeasures): Unit = {
    performanceMeasures.get(index) = Option(measures)
  }

  protected def setBatchPerformanceMeasures(measures: Array[Option[ExecutionMeasures]]): this.type = {
    // TODO throw if already set?
    performanceMeasures = Option(measures)
    this
  }

  def getAllPerformanceMeasures(): Option[Array[ExecutionMeasures]] = {
    performanceMeasures.map(array => array.flatMap(element => element))
  }

  /** In the common case of 1 batch, there is only 1 measure, so this is a convenience method.
    *
    */
  def getPerformanceMeasures(): Option[ExecutionMeasures] = {
    performanceMeasures.flatMap(array => array(0))
  }
}