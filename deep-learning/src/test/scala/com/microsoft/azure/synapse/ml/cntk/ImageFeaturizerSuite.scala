// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.azure.synapse.ml.cntk

import com.microsoft.azure.synapse.ml.Secrets
import com.microsoft.azure.synapse.ml.core.test.fuzzing.{TestObject, TransformerFuzzing}
import com.microsoft.azure.synapse.ml.image.ImageTestUtils
import com.microsoft.azure.synapse.ml.io.IOImplicits._
import com.microsoft.azure.synapse.ml.io.powerbi.PowerBIWriter
import com.microsoft.azure.synapse.ml.io.split1.FileReaderUtils
import org.apache.spark.injections.UDFUtils
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.StringType

// TODO move all to onnx package after review

trait TrainedONNXModelUtils extends ImageTestUtils {
  override def beforeAll(): Unit = {
    spark
    super.beforeAll()
  }

  lazy val resNetOnnxUrl = "https://mmlspark.blob.core.windows.net/datasets/ONNXModels/resnet50-v1-12-int8.onnx"
  lazy val resNetOnnxPayload: Array[Byte] = {
    val isr = scala.io.Source.fromURL(resNetOnnxUrl, "ISO-8859-1").reader()
    Stream.continually(isr.read()).takeWhile(_ != -1).map(_.toByte).toArray
  }

  def resNetModel(): ImageFeaturizer = new ImageFeaturizer()
    .setInputCol(inputCol)
    .setOutputCol(outputCol)
    .setModel("ResNet18")
}

class ImageFeaturizerSuite extends TransformerFuzzing[ImageFeaturizer]
  with TrainedONNXModelUtils with FileReaderUtils {

  test("structured streaming") {
    val imageDF = spark
      .readStream
      .image
      .load(cifarDirectory)
    val resultDF = resNetModel().transform(imageDF)

    val q1 = resultDF.writeStream
      .format("memory")
      .queryName("images")
      .start()

    try {
      tryWithRetries() { () =>
        assert(spark.sql("select * from images").count() == 6)
      }
    } finally {
      q1.stop()
    }
  }

  test("the Image feature should work with the modelSchema") {
    val result = resNetModel().setHeadless(false).transform(images)
    compareToTestModel(result)
  }

  test("the Image feature should work with the modelSchema + new images") {
    val newImages = spark.read.image
      .load(cifarDirectory)

    val result = resNetModel().setHeadless(false).transform(newImages)
    compareToTestModel(result)
  }

  test("Image featurizer should work with ResNet50") {
    val result = resNetModel().transform(images)
    val resVec = result.select(outputCol).collect()(0).getAs[DenseVector](0)
    assert(resVec.size == 1000)
  }

  test("Image featurizer should work with ResNet50 in greyscale") {
    val result = resNetModel()
      .setChannelNormalizationMeans(Array(0.485))
      .setChannelNormalizationStds(Array(0.229))
      .transform(greyscaleImage)
    val resVec = result.select(outputCol).collect()(0).getAs[DenseVector](0)
    assert(resVec.size == 1000)
  }

  test("Image featurizer should work with ResNet50 in greyscale binary") {
    val result = resNetModel()
      .setChannelNormalizationMeans(Array(0.485))
      .setChannelNormalizationStds(Array(0.229))
      .transform(greyscaleBinary)
    val resVec = result.select(outputCol).collect()(0).getAs[DenseVector](0)
    assert(resVec.size == 1000)
  }

  test("Image featurizer should work with ResNet50 Binary + nulls") {
    import spark.implicits._
    val corruptImage = Seq("fooo".toCharArray.map(_.toByte))
      .toDF(inputCol)
    val df = binaryImages.union(corruptImage)

    val resultDF = resNetModel().transform(df)
    val result = resultDF.select(outputCol).collect()
    assert(result(0).getAs[DenseVector](0).size == 1000)
  }

  test("Image featurizer should work with ResNet50 Binary 2 + nulls") {
    import spark.implicits._
    val corruptImage = Seq("fooo".toCharArray.map(_.toByte))
      .toDF(inputCol)
    val df = binaryImages.union(corruptImage)

    val resultDF = resNetModel().transform(df)
    val result = resultDF.select(outputCol).collect()
    assert(result(0).getAs[DenseVector](0).size == 1000)
  }

  test("Image featurizer should correctly classify an image") {
    val testImg: DataFrame = spark
      .read.image.load(s"$filesRoot/Images/Grocery/testImages/WIN_20160803_11_28_42_Pro.jpg")
      .withColumnRenamed("image", inputCol)
    val result = resNetModel().transform(testImg)
    val resVec = result.select(outputCol).collect()(0).getAs[DenseVector](0)
    assert(resVec.argmax == 760)
  }

  test("Image featurizer should work with ResNet50 and powerBI") {
    val images = groceryImages.withColumnRenamed(inputCol, "image").coalesce(1)
    println(images.count())

    val result = resNetModel().setInputCol("image").transform(images)
      .withColumn("foo", UDFUtils.oldUdf({ x: DenseVector => x(0).toString }, StringType)(col("out")))
      .select("foo")

    PowerBIWriter.write(result,sys.env.getOrElse("MML_POWERBI_URL", Secrets.PowerbiURL), Map("concurrency" -> "1"))
  }

  val reader: MLReadable[_] = ImageFeaturizer

  override def testObjects(): Seq[TestObject[ImageFeaturizer]] = Seq(
    new TestObject(resNetModel(), images)
  )
}
