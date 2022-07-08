// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.azure.synapse.ml.lightgbm.split1

import com.microsoft.azure.synapse.ml.core.test.base.TestBase
import com.microsoft.azure.synapse.ml.lightgbm._
import com.microsoft.azure.synapse.ml.lightgbm.dataset.{ChunkedArrayUtils, SampledData}
import com.microsoft.azure.synapse.ml.lightgbm.swig.{DoubleChunkedArray, DoubleSwigArray, IntSwigArray, SwigUtils}
import com.microsoft.ml.lightgbm.{SWIGTYPE_p_p_void, SWIGTYPE_p_void, lightgbmlib}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.sql.DataFrame

// scalastyle:off magic.number
/** Tests to validate general functionality of LightGBM module. */
class VerifyLightGBMCommon extends TestBase with LightGBMTestUtils  {
  lazy val taskDF: DataFrame = loadBinary("task.train.csv", "TaskFailed10").cache()
  lazy val pimaDF: DataFrame = loadBinary("PimaIndian.csv", "Diabetes mellitus").cache()

  lazy val baseModel: LightGBMClassifier = new LightGBMClassifier()
    .setFeaturesCol(featuresCol)
    .setRawPredictionCol(rawPredCol)
    .setDefaultListenPort(getAndIncrementPort())
    .setNumLeaves(5)
    .setNumIterations(10)
    .setObjective("binary")
    .setLabelCol(labelCol)
    .setLeafPredictionCol(leafPredCol)
    .setFeaturesShapCol(featuresShapCol)
    .setExecutionMode("streaming")


  test("Verify chunked array transpose simple") {
    Array(10, 100).foreach(chunkSize => {
      LightGBMUtils.initializeNativeLibrary()
      val rows = 10
      val cols = 2
      val chunkedArray = new DoubleChunkedArray(chunkSize) // either whole chunks or 1 incomplete chunk
      val transposedArray = new DoubleSwigArray(rows * cols)

      // Create transposed array (for easier validation since transpose will convert to sequential)
      for (row <- 0L until rows) {
        for (col <- 0L until cols) {
          chunkedArray.add(row + rows * col)
        }
      }

      try {
        ChunkedArrayUtils.insertTransposedChunkedArray(chunkedArray, cols, transposedArray, rows, 0)

        // Assert row order in source (at least for first row)
        assert(chunkedArray.getItem(0, 0, 0) == 0)
        assert(chunkedArray.getItem(0, 1, 0) == rows)

        // Assert column order in source (should be sequential numbers)
        val array = SwigUtils.nativeDoubleArrayToArray(transposedArray.array, rows * cols)
        assert(array.zipWithIndex.forall(pair => pair._1 == pair._2))
      } finally {
        transposedArray.delete()
      }
    })
  }

  test("Verify chunked array transpose complex") {
    LightGBMUtils.initializeNativeLibrary()
    val rows = 10
    val cols = 2
    val chunkedArray = new DoubleChunkedArray(7) // ensure partial chunks
    val transposedArray = new DoubleSwigArray(rows * cols * 2)
    for (row <- 0L until rows) {
      for (col <- 0L until cols) {
        chunkedArray.add(row + rows * col)
      }
    }

    try {
      // copy into start and middle
      ChunkedArrayUtils.insertTransposedChunkedArray(chunkedArray, cols, transposedArray, rows * 2, 0)
      ChunkedArrayUtils.insertTransposedChunkedArray(chunkedArray, cols, transposedArray, rows * 2, rows)

      // Assert row order in source (at least for first row)
      assert(chunkedArray.getItem(0, 0, 0) == 0)
      assert(chunkedArray.getItem(0, 1, 0) == rows)

      val array = SwigUtils.nativeDoubleArrayToArray(transposedArray.array, rows * cols * 2)
      val expectedArray = ((0 until rows)
        ++ (0 until rows)
        ++ (rows until 2*rows)
        ++ (rows until 2*rows))
      assert(array.zipWithIndex.forall(pair => pair._1 == expectedArray(pair._2)))
    } finally {
      transposedArray.delete()
    }
  }

  test("Verify sample data creation") {
    LightGBMUtils.initializeNativeLibrary()
    val Array(train, _) = pimaDF.randomSplit(Array(0.8, 0.2), seed)

    val numRows = 100
    val sampledRowData = train.take(numRows)
    val featureData = sampledRowData(0).getAs[Any](featuresCol)
    val numCols = featureData match {
      case sparse: SparseVector => sparse.size
      case dense: DenseVector => dense.size
      case _ => throw new IllegalArgumentException("Unknown row data type to push")
    }

    val sampledData: SampledData = new SampledData(sampledRowData.length, numCols)
    sampledRowData.zipWithIndex.foreach(rowWithIndex =>
      sampledData.pushRow(rowWithIndex._1, rowWithIndex._2, featuresCol))

    val rowCounts: IntSwigArray = sampledData.rowCounts
    (0 until numCols).foreach(col => {
      val rowCount = rowCounts.getItem(col)
      println(s"Row counts for col $col: $rowCount")
      val values = sampledData.sampleData.getItem(col)
      val indexes = sampledData.sampleIndexes.getItem(col)
      (0 until rowCount).foreach(i => println(s"  Index: ${indexes.getItem(i)}, val: ${values.getItem(i)}"))
    })

    var datasetVoidPtr:SWIGTYPE_p_p_void = null
    try {
      println("Creating dataset")
      datasetVoidPtr = lightgbmlib.voidpp_handle()
      val resultCode = lightgbmlib.LGBM_DatasetCreateFromSampledColumn(
        sampledData.getSampleData,
        sampledData.getSampleIndices,
        numCols,
        sampledData.getRowCounts,
        numRows,
        numRows,
        s"max_bin=255 bin_construct_sample_cnt=$numRows min_data_in_leaf=1 num_threads=3",
        datasetVoidPtr)
      println(s"Result code for LGBM_DatasetCreateFromSampledColumn: $resultCode")
    } finally {
      sampledData.delete()

      val datasetPtr: SWIGTYPE_p_void = lightgbmlib.voidpp_value(datasetVoidPtr)
      LightGBMUtils.validate(lightgbmlib.LGBM_DatasetFree(datasetPtr), "Dataset LGBM_DatasetFree")

      lightgbmlib.delete_voidpp(datasetVoidPtr)
    }
  }

  test("Verify performance measures") {
    val Array(train, _) = taskDF.randomSplit(Array(0.8, 0.2), seed)
    // TODO How does this make fresh copy?
    val measuredModel = baseModel.setUseSingleDatasetMode(false).setExecutionMode("streaming").setMatrixType("dense")
    val _ = measuredModel.fit(train)
    val measuresOpt =  measuredModel.getPerformanceMeasures

    assert(measuresOpt.isDefined)
    val measures = measuresOpt.get
    val totalTime = measures.totalTime
    assert(totalTime > 0)
    println(s"Total time: $totalTime")
    val columnStatisticsTime = measures.columnStatisticsTime()
    assert(columnStatisticsTime > 0)
    println(s"Column statistics time: $columnStatisticsTime")
    val rowStatisticsTime = measures.rowStatisticsTime()
    println(s"Row statistics time: $rowStatisticsTime")
    val trainingTime = measures.trainingTime()
    assert(trainingTime > 0)
    println(s"Training time: $trainingTime")

    val taskTimes = measures.taskTotalTimes()
    assert(taskTimes.nonEmpty)
    taskTimes.foreach(t => assert(t > 0))
    println(s"Task total times: ${taskTimes.mkString(",")}")
    val taskDataPreparationTimes = measures.taskDataPreparationTimes()
    assert(taskDataPreparationTimes.nonEmpty)
    taskDataPreparationTimes.foreach(t => assert(t > 0))
    println(s"Task data preparation times: ${taskDataPreparationTimes.mkString(",")}")
    val taskDatasetCreationTimes = measures.taskDatasetCreationTimes()
    assert(taskDatasetCreationTimes.nonEmpty)
    assert(taskDatasetCreationTimes.sum > 0)
    println(s"Task dataset creation times: ${taskDatasetCreationTimes.mkString(",")}")
    val taskTrainingIterationTimes = measures.taskTrainingIterationTimes()
    assert(taskTrainingIterationTimes.nonEmpty)
    // TODO assert(taskTrainingIterationTimes.sum > 0)
    println(s"Task training iteration times: ${taskTrainingIterationTimes.mkString(",")}")

    val tasks = measures.getTaskMeasures
    val activeTasks = tasks.filter(t => t.isActiveTrainingTask).map(t => t.partitionId)
    println(s"Active task ids: ${activeTasks.mkString(",")}")

    // TODO verify all diff measures that are 0 by default
  }

  test("Performance testing") {
    // modify this test for getting some simple performance measures
    val dataset = taskDF
    val measurementCount = 1
    val executionModes = Array("bulk")  // streaming, bulk
    val microBatchSizes = Array(4000) // 1, 2, 4, 8, 16, 32, 100, 1000)
    val matrixTypes = Array("dense")  // dense, sparse, auto
    val useSingleDatasetModes = Array(true)

    executionModes.foreach(executionMode => {
      matrixTypes.foreach(matrixType => {
        microBatchSizes.foreach(microBatchSize => {
          useSingleDatasetModes.foreach(useSingleDataset => {
            println(s"*********************************************************************************************")
            println(s"**** Start ExecutionMode: $executionMode, MatrixType: $matrixType, " +
              s"useSingleDataset: $useSingleDataset, MicroBatchSize: $microBatchSize")
            measurePerformance(dataset, measurementCount, executionMode, microBatchSize, matrixType, useSingleDataset)
            println(s"**** Done ExecutionMode: $executionMode, MatrixType: $matrixType, " +
              s"useSingleDataset: $useSingleDataset, MicroBatchSize: $microBatchSize")
            println(s"*********************************************************************************************")
          })
        })
      })
    })
  }

  def measurePerformance(df: DataFrame,
                         measurementCount: Int,
                         executionMode: String,
                         microBatchSize: Int,
                         matrixType: String,
                         useSingleDataset: Boolean): Unit = {
    val Array(train, _) = df.randomSplit(Array(0.8, 0.2), seed)
    val measurements = Array.ofDim[InstrumentationMeasures](measurementCount)

    (0 until measurementCount).foreach(i => {
      println(s"** Start Measurement $i")

      val measuredModel = baseModel
        .setUseSingleDatasetMode(useSingleDataset)
        .setExecutionMode(executionMode)
        .setMatrixType(matrixType)
        .setMicroBatchSize(microBatchSize)

      val _ = measuredModel.fit(train)
      measurements(i) = measuredModel.getPerformanceMeasures.get
      println(s"Total time, ${measurements(i).totalTime}")
      println(s"Column statistics, ${measurements(i).columnStatisticsTime}")
      println(s"Row statistics time, ${measurements(i).rowStatisticsTime}")
      println(s"Row count time, ${measurements(i).rowCountTime()}")
      println(s"Sampling time, ${measurements(i).samplingTime()}")
      println(s"Training time, ${measurements(i).trainingTime}")
      println(s"Overhead time, ${measurements(i).overheadTime}")
      println(s"Task total times, ${measurements(i).taskTotalTimes.mkString(",")}")
      println(s"Task overhead times, ${measurements(i).taskOverheadTimes().mkString(",")}")
      println(s"Task initialization times, ${measurements(i).taskInitializationTimes().mkString(",")}")
      println(s"Task library initialization times, ${measurements(i).taskLibraryInitializationTimes().mkString(",")}")
      println(s"Task network initialization times, ${measurements(i).taskNetworkInitializationTimes().mkString(",")}")
      println(s"Task data preparation times, ${measurements(i).taskDataPreparationTimes.mkString(",")}")
      println(s"Task dataset wait times, ${measurements(i).taskWaitTimes().mkString(",")}")
      println(s"Task dataset creation times, ${measurements(i).taskDatasetCreationTimes.mkString(",")}")
      println(s"Task training iteration times, ${measurements(i).taskTrainingIterationTimes.mkString(",")}")
      println(s"Task cleanup times, ${measurements(i).taskCleanupTimes().mkString(",")}")
      println(s"** Completed Measurement $i")
    })
    println(s"***** Averaged results for $measurementCount runs")
    var median = getMedian(measurements.map(m => m.totalTime))
    println(s"Median Total time, $median")
    median = getMedian(measurements.map(m => m.columnStatisticsTime))
    println(s"Median Column statistics, $median")
    median = getMedian(measurements.map(m => m.rowCountTime()))
    println(s"Median Row count time, $median")
    median = getMedian(measurements.map(m => m.samplingTime()))
    println(s"Median Sampling time, $median")
    median = getMedian(measurements.map(m => m.rowStatisticsTime()))
    println(s"Median Row statistics time, $median")
    median = getMedian(measurements.map(m => m.trainingTime))
    println(s"Median Training time, $median")
    median = getMedian(measurements.map(m => m.overheadTime))
    println(s"Median Overhead time, $median")
    var medianMax = getMedian(measurements.map(m => m.taskTotalTimes.max))
    println(s"Median-max Task total times, $medianMax")
    medianMax = getMedian(measurements.map(m => m.taskOverheadTimes().max))
    println(s"Median-max Task overhead times, $medianMax")
    medianMax = getMedian(measurements.map(m => m.taskInitializationTimes().max))
    println(s"Median-max Task initialization times, $medianMax")
    medianMax = getMedian(measurements.map(m => m.taskDataPreparationTimes.max))
    println(s"Median-max Task data preparation times, $medianMax")
    medianMax = getMedian(measurements.map(m => m.taskDatasetCreationTimes.max))
    println(s"Median-max Task dataset creation times, $medianMax")
    medianMax = getMedian(measurements.map(m => m.taskTrainingIterationTimes.max))
    println(s"Median-max Task training iteration times, $medianMax")
  }

  def getMedian[T: Ordering](seq: Seq[T])(implicit conv: T => Float, f: Fractional[Float]): Float = {
    val sortedSeq = seq.sorted
    if (seq.size % 2 == 1) sortedSeq(sortedSeq.size / 2)  else {
      val (up, down) = sortedSeq.splitAt(seq.size / 2)
      import f._
      (conv(up.last) + conv(down.head)) / fromInt(2)
    }
  }
}
