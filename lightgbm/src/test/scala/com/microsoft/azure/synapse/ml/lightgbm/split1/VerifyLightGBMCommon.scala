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
}
