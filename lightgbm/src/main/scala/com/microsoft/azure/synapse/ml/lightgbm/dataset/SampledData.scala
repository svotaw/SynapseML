// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.azure.synapse.ml.lightgbm.dataset

import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.sql.Row
import com.microsoft.azure.synapse.ml.lightgbm.LightGBMUtils
import com.microsoft.azure.synapse.ml.lightgbm.swig._
import com.microsoft.ml.lightgbm._


/** SampledData: Encapsulates the sampled data need to initialize a LightGBM dataset.
  * .
  * LightGBM expects sampled data to be an array of vectors, where each feature column
  * has a sparse representation of non-zero values (i.e. indexes and data vector). It also needs
  * a #features sized array of element count per feature to know how long each column is.
  * .
  * Since we create sampled data as a self-contained set with ONLY sampled data and nothing else,
  * the indexes are trivial (0 to #elements). We don't need to maintain original raw indexes. LightGBM
  * only uses this data to get distributions, and does not care about raw row indexes.
  * .
  * This class manages keeping all the indexing in sync so callers can just push rows of data into it
  * and retrieve the resulting pointers at the end.
  * .
  * Note: sample data row count is not expected to exceed max(Int), so we index with Ints.
  */
class SampledData(val numRows: Int, val numCols: Int) {

  // Allocate full arrays for each feature column, but we will push only non-zero values and
  // keep track of actual counts in rowCounts array
  private val sampleData = new DoublePointerSwigArray(numCols)
  private val sampleIndexes = new IntPointerSwigArray(numCols)
  private val rowCounts = new IntSwigArray(numCols)

  // Initialize column vectors (might move some of this to inside XPointerSwigArray)
  (0 to numCols-1).foreach(col => {
    rowCounts.setItem(col, 0) // Initialize as 0-rowCount columns

    sampleData.setItem(col, new DoubleSwigArray(numRows))

    val columnIndexes = new IntSwigArray(numRows);
    sampleIndexes.setItem(col, columnIndexes)
  })

  // Store non-zero elements in arrays given a dense feature value row
  def pushRow(rowData: Row, featureColName: String): Unit = {
    val data = rowData.getAs[Any](featureColName)
    data match {
      case sparse: SparseVector => pushRow(sparse)
      case dense: DenseVector => pushRow(dense)
      case _ => throw new IllegalArgumentException("Unknown row data type to push")
    }
  }

  // Store non-zero elements in arrays given a dense feature value row
  def pushRow(rowData: DenseVector): Unit = pushRow(rowData.values)

  // Store non-zero elements in arrays given a dense feature value array
  def pushRow(rowData: Array[Double]): Unit = {
    require(rowData.size <= numCols, s"Row is too large for sample data.  size should be $numCols" +
                                     s", but is ${rowData.size}")
    (0 to numCols).foreach(col => pushRowElementIfNotZero(col, rowData(col)))
  }

  // Store non-zero elements in arrays given a sparse feature value row
  def pushRow(rowData: SparseVector): Unit = {
    require(rowData.size <= numCols, s"Row is too large for sample data.  size should be $numCols" +
                                     s", but is ${rowData.size}")
    (0 to rowData.numActives).foreach(i => pushRowElementIfNotZero(rowData.indices(i), rowData.values(i)))
  }

  def pushRowElementIfNotZero(col: Int, value: Double): Unit = {
    if (value != 0.0) {
      val nextIndex = rowCounts.getItem(col)
      sampleData.pushElement(col, nextIndex, value)
      rowCounts.setItem(col, nextIndex + 1) // increment row count
    }
  }

  def getSampleData(): SWIGTYPE_p_p_double = {
    sampleData.array
  }

  def getSampleIndices(): SWIGTYPE_p_p_int = {
    sampleIndexes.array
  }

  def getRowCounts(): SWIGTYPE_p_int = {
    rowCounts.array
  }
}
