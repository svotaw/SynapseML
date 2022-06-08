// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.azure.synapse.ml.lightgbm.split1

import com.microsoft.azure.synapse.ml.core.test.benchmarks.{Benchmarks, DatasetUtils}
import com.microsoft.azure.synapse.ml.core.test.fuzzing.{EstimatorFuzzing, TestObject}
import com.microsoft.azure.synapse.ml.lightgbm._
import com.microsoft.azure.synapse.ml.lightgbm.dataset.LightGBMDataset
import com.microsoft.azure.synapse.ml.lightgbm.params.FObjTrait
import com.microsoft.azure.synapse.ml.stages.MultiColumnAdapter
import org.apache.commons.io.FileUtils
import org.apache.spark.TaskContext
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{LabeledPoint, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Row}

import java.io.File
import java.nio.file.{Files, Path, Paths}
import scala.math.exp

// scalastyle:off magic.number
/** Tests to validate the functionality of LightGBM module. */
class VerifyLightGBMClassifier extends Benchmarks with EstimatorFuzzing[LightGBMClassifier]
  with LightGBMTestUtils {

  lazy val pimaDF: DataFrame = loadBinary("PimaIndian.csv", "Diabetes mellitus").cache()
  lazy val taskDF: DataFrame = loadBinary("task.train.csv", "TaskFailed10").cache()
  lazy val breastTissueDF: DataFrame = loadMulticlass("BreastTissue.csv", "Class").cache()
  lazy val au3DF: DataFrame = loadMulticlass("au3_25000.csv", "class").cache()
  lazy val unfeaturizedBankTrainDF: DataFrame = {
    val categoricalColumns = Array(
      "job", "marital", "education", "default", "housing", "loan", "contact", "y")
    val newCategoricalColumns: Array[String] = categoricalColumns.map("c_" + _)
    val df = readCSV(DatasetUtils.binaryTrainFile("bank.train.csv").toString)
      .repartition(numPartitions)
    val df2 = new MultiColumnAdapter().setInputCols(categoricalColumns).setOutputCols(newCategoricalColumns)
      .setBaseStage(new StringIndexer())
      .fit(df)
      .transform(df).drop(categoricalColumns: _*)
      .withColumnRenamed("c_y", labelCol)
    df2
    }.cache()
  lazy val indexedBankTrainDF: DataFrame = {
    LightGBMUtils.getFeaturizer(unfeaturizedBankTrainDF, labelCol, featuresCol,
      oneHotEncodeCategoricals = false).transform(unfeaturizedBankTrainDF)
    }.cache()
  lazy val indexedTaskDF: DataFrame = {
    val categoricalColumns = Array("TaskNm", "QueueName")
    val newCategoricalColumns: Array[String] = categoricalColumns.map("c_" + _)
    val df = readCSV(DatasetUtils.binaryTrainFile("task.train.csv").toString).repartition(numPartitions)
    val df2 = new MultiColumnAdapter().setInputCols(categoricalColumns).setOutputCols(newCategoricalColumns)
      .setBaseStage(new StringIndexer())
      .fit(df)
      .transform(df).drop(categoricalColumns: _*)
      .withColumnRenamed("TaskFailed10", labelCol)
      .drop(Array("IsControl10", "RanAsSystem10", "IsDAAMachine10", "IsUx", "IsClient"): _*)
    val tdf = LightGBMUtils.getFeaturizer(df2, labelCol, featuresCol, oneHotEncodeCategoricals = false).transform(df2)
    tdf
  }.cache()
  lazy val bankTrainDF: DataFrame = {
    LightGBMUtils.getFeaturizer(unfeaturizedBankTrainDF, labelCol, featuresCol).transform(unfeaturizedBankTrainDF)
    }.cache()

  val binaryObjective = "binary"
  val multiclassObject = "multiclass"

  def binaryEvaluator: BinaryClassificationEvaluator = {
    new BinaryClassificationEvaluator()
      .setLabelCol(labelCol)
      .setRawPredictionCol(rawPredCol)
  }

  def multiclassEvaluator: MulticlassClassificationEvaluator = {
    new MulticlassClassificationEvaluator()
      .setLabelCol(labelCol)
      .setPredictionCol(predCol)
      .setMetricName("accuracy")
  }

  // TODO: Look into error on abalone dataset
  // verifyLearnerOnMulticlassCsvFile("abalone.csv",                  "Rings", 2)
  verifyLearnerOnMulticlassCsvFile("BreastTissue.csv", "Class", .07)
  verifyLearnerOnMulticlassCsvFile("CarEvaluation.csv", "Col7", 2)
  verifyLearnerOnBinaryCsvFile("PimaIndian.csv", "Diabetes mellitus", 1)
  verifyLearnerOnBinaryCsvFile("data_banknote_authentication.csv", "class", 1)
  verifyLearnerOnBinaryCsvFile("task.train.csv", "TaskFailed10", 1)
  verifyLearnerOnBinaryCsvFile("breast-cancer.train.csv", "Label", 1)
  verifyLearnerOnBinaryCsvFile("random.forest.train.csv", "#Malignant", 1)
  verifyLearnerOnBinaryCsvFile("transfusion.csv", "Donated", 1)

  verifySaveBooster(
    fileName = "PimaIndian.csv",
    labelColumnName = "Diabetes mellitus",
    outputFileName = "model.txt",
    colsToVerify = Array("Diabetes pedigree function", "Age (years)"))

  test("Compare benchmark results file to generated file") {
    verifyBenchmarks()
  }

  override def testExperiments(): Unit = {
    super.testExperiments()
  }

  override def testSerialization(): Unit = {
    super.testSerialization()
  }

  def baseModel: LightGBMClassifier = {
    new LightGBMClassifier()
      .setFeaturesCol(featuresCol)
      .setRawPredictionCol(rawPredCol)
      .setDefaultListenPort(getAndIncrementPort())
      .setNumLeaves(5)
      .setNumIterations(10)
      .setObjective(binaryObjective)
      .setLabelCol(labelCol)
      .setLeafPredictionCol(leafPredCol)
      .setFeaturesShapCol(featuresShapCol)
      .setExecutionMode("streaming")
      //.setExecutionMode("bulk")
      //.setMatrixType("sparse")
  }

  test("Verify LightGBM Classifier can be run with TrainValidationSplit") {
    val model = baseModel.setUseBarrierExecutionMode(true)

    val paramGrid = new ParamGridBuilder()
      .addGrid(model.numLeaves, Array(5, 10))
      .addGrid(model.numIterations, Array(10, 20))
      .addGrid(model.lambdaL1, Array(0.1, 0.5))
      .addGrid(model.lambdaL2, Array(0.1, 0.5))
      .build()

    val fitModel = new TrainValidationSplit()
      .setEstimator(model)
      .setEvaluator(binaryEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.8)
      .setParallelism(2)
      .fit(pimaDF)

    fitModel.transform(pimaDF)
    assert(fitModel != null)

    // Validate lambda parameters set on model
    val modelStr = fitModel.bestModel.asInstanceOf[LightGBMClassificationModel].getModel.modelStr.get
    assert(modelStr.contains("[lambda_l1: 0.1]") || modelStr.contains("[lambda_l1: 0.5]"))
    assert(modelStr.contains("[lambda_l2: 0.1]") || modelStr.contains("[lambda_l2: 0.5]"))
  }

  test("Verify LightGBM Classifier with batch training") {
    val batches = Array(0, 2, 10)
    batches.foreach(nBatches => assertFitWithoutErrors(baseModel.setNumBatches(nBatches), pimaDF))
  }

  def assertBinaryImprovement(sdf1: DataFrame, sdf2: DataFrame): Unit = {
    assert(binaryEvaluator.evaluate(sdf1) < binaryEvaluator.evaluate(sdf2))
  }

  def assertBinaryEquality(sdf1: DataFrame, sdf2: DataFrame): Unit = {
    assert(Math.abs(binaryEvaluator.evaluate(sdf1) - binaryEvaluator.evaluate(sdf2)) < 1e-10)
  }

  def assertMulticlassImprovement(sdf1: DataFrame, sdf2: DataFrame): Unit = {
    assert(multiclassEvaluator.evaluate(sdf1) < multiclassEvaluator.evaluate(sdf2))
  }

  def assertBinaryImprovement(v1: LightGBMClassifier, train1: DataFrame, test1: DataFrame,
                              v2: LightGBMClassifier, train2: DataFrame, test2: DataFrame
                             ): Unit = {
    assertBinaryImprovement(v1.fit(train1).transform(test1), v2.fit(train2).transform(test2))
  }

  test("Verify LightGBM Classifier continued training with initial score") {
    val convertUDF = udf((vector: DenseVector) => vector(1))
    val scoredDF1 = baseModel.fit(pimaDF).transform(pimaDF)
    val df2 = scoredDF1.withColumn(initScoreCol, convertUDF(col(rawPredCol)))
      .drop(predCol, rawPredCol, probCol, leafPredCol, featuresShapCol)
    val scoredDF2 = baseModel.setInitScoreCol(initScoreCol).fit(df2).transform(df2)

    assertBinaryImprovement(scoredDF1, scoredDF2)
  }

  ignore("Verify LightGBM Multiclass Classifier with vector initial score") {
    val scoredDF1 = baseModel.fit(breastTissueDF).transform(breastTissueDF)
    val df2 = scoredDF1.withColumn(initScoreCol, col(rawPredCol))
      .drop(predCol, rawPredCol, probCol, leafPredCol, featuresShapCol)
    val scoredDF2 = baseModel.setInitScoreCol(initScoreCol).fit(df2).transform(df2)

    assertMulticlassImprovement(scoredDF1, scoredDF2)
  }

  test("Verify LightGBM Classifier with custom loss function") {
    class LogLikelihood extends FObjTrait {
      override def getGradient(predictions: Array[Array[Double]],
                               trainingData: LightGBMDataset): (Array[Float], Array[Float]) = {
        // Get the labels
        val labels = trainingData.getLabel()
        val probabilities = predictions.map(rowPrediction =>
          rowPrediction.map(prediction => 1.0 / (1.0 + exp(-prediction))))
        // Compute gradient and hessian
        val grad =  probabilities.zip(labels).map {
          case (prob: Array[Double], label: Float) => (prob(0) - label).toFloat
        }
        val hess = probabilities.map(probabilityArray => (probabilityArray(0) * (1 - probabilityArray(0))).toFloat)
        (grad, hess)
      }
    }
    val scoredDF1 = baseModel
      .setUseSingleDatasetMode(false)
      .fit(pimaDF)
      .transform(pimaDF)

    // Note: run for more iterations than non-custom objective to prevent flakiness
    // Note we intentionally overfit here on the training data and don't do a split
    val scoredDF2 = baseModel
      .setUseSingleDatasetMode(false)
      .setFObj(new LogLikelihood())
      .setNumIterations(300)
      .fit(pimaDF)
      .transform(pimaDF)
    assertBinaryImprovement(scoredDF1, scoredDF2)
  }

  test("Verify LightGBM Classifier with min gain to split parameter") {
    // If the min gain to split is too high, assert AUC lower for training data (assert parameter works)
    val scoredDF1 = baseModel.setMinGainToSplit(99999).fit(pimaDF).transform(pimaDF)
    val scoredDF2 = baseModel.fit(pimaDF).transform(pimaDF)
    assertBinaryImprovement(scoredDF1, scoredDF2)
  }

  test("Verify LightGBM Classifier will give reproducible results when setting seed") {
    val scoredDF1 = baseModel.setSeed(1).setDeterministic(true).fit(pimaDF).transform(pimaDF)
    (1 to 10).foreach { i =>
      val scoredDF2 = baseModel.setSeed(1).setDeterministic(true).fit(pimaDF).transform(pimaDF)
      assertBinaryEquality(scoredDF1, scoredDF2);
    }
  }

  test("Verify LightGBM Classifier with dart mode parameters") {
    // Assert the dart parameters work without failing and setting them to tuned values improves performance
    val Array(train, test) = pimaDF.randomSplit(Array(0.8, 0.2), seed)
    val scoredDF1 = baseModel.setBoostingType("dart")
      .setNumIterations(100)
      .setSkipDrop(1.0)
      .fit(train).transform(test)
    val scoredDF2 = baseModel.setBoostingType("dart")
      .setNumIterations(100)
      .setXGBoostDartMode(true)
      .setDropRate(0.6)
      .setMaxDrop(60)
      .setSkipDrop(0.4)
      .setUniformDrop(true)
      .fit(train).transform(test)
    assertBinaryImprovement(scoredDF1, scoredDF2)
  }

  test("Verify LightGBM Classifier with num tasks parameter") {
    val numTasks = Array(0, 1, 2)
    numTasks.foreach(nTasks => assertFitWithoutErrors(baseModel.setNumTasks(nTasks), pimaDF))
  }

  test("Verify performance measures") {
    val Array(train, _) = taskDF.randomSplit(Array(0.8, 0.2), seed)
    // TODO How does this make fresh copy?
    val measuredModel = baseModel.setUseSingleDatasetMode(false).setExecutionMode("streaming").setMatrixType("dense")
    val _ = measuredModel.fit(train)
    val measuresOpt =  measuredModel.getPerformanceMeasures()

    assert(measuresOpt.isDefined)
    val measures = measuresOpt.get
    val totalTime = measures.totalTime
    assert(totalTime > 0)
    println(s"Total time: $totalTime")
    val columnStatisticsTime = measures.columnStatisticsTime
    assert(columnStatisticsTime > 0)
    println(s"Column statistics time: $columnStatisticsTime")
    val rowStatisticsTime = measures.rowStatisticsTime
    println(s"Row statistics time: $rowStatisticsTime")
    val trainingTime = measures.trainingTime
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

    val tasks = measures.getTaskMeasures()
    val activeTasks = tasks.filter(t => t.isActiveTrainingTask).map(t => t.partitionId)
    println(s"Active task ids: ${activeTasks.mkString(",")}")


    // TODO verify all diff measures that are 0 by default
  }

  test("Performance testing") {
    // modify this test for getting some simple performance measures
    val dataset = taskDF
    val executionModes = Array("streaming", "bulk")
    val matrixTypes = Array("dense", "sparse")
    val useSingleDatasetModes = Array(true, false)

    executionModes.foreach(executionMode => {
      matrixTypes.foreach(matrixType => {
        useSingleDatasetModes.foreach(useSingleDataset => {
          println(s"*********************************************************************************************")
          println(s"**** ExecutionMode: $executionMode, MatrixType: $matrixType, useSingleDataset: $useSingleDataset")
          measurePerformance(dataset, executionMode, matrixType, useSingleDataset)
          println(s"*********************************************************************************************")
        })
      })
    })
  }

  def measurePerformance(df: DataFrame,
                         executionMode: String,
                         matrixType: String,
                         useSingleDataset: Boolean): Unit = {
    val Array(train, _) = df.randomSplit(Array(0.8, 0.2), seed)
    val measuredModel = baseModel
      .setUseSingleDatasetMode(useSingleDataset)
      .setExecutionMode(executionMode)
      .setMatrixType(matrixType)
    val _ = measuredModel.fit(train)
    val measuresOpt =  measuredModel.getPerformanceMeasures()
  }

  test("Verify LightGBM Classifier with max delta step parameter") {
    // If the max delta step is specified, assert AUC differs (assert parameter works)
    // Note: the final max output of leaves is learning_rate * max_delta_step, so param should reduce the effect
    val Array(train, test) = taskDF.randomSplit(Array(0.8, 0.2), seed)
    val baseModelWithLR = baseModel.setLearningRate(0.9).setNumIterations(200)
    val scoredDF1 = baseModelWithLR.fit(train).transform(test)
    val scoredDF2 = baseModelWithLR.setMaxDeltaStep(0.5).fit(train).transform(test)
    assertBinaryImprovement(scoredDF1, scoredDF2)
  }

  test("Verify LightGBM Classifier with numIterations model parameter") {
    // We expect score to improve as numIterations is increased
    val Array(train, test) = taskDF.randomSplit(Array(0.8, 0.2), seed)
    val model = baseModel.fit(train)
    val score1 = binaryEvaluator.evaluate(model.transform(test))
    model.setNumIterations(1)
    val score2 = binaryEvaluator.evaluate(model.transform(test))
    assert(score1 > score2)
    model.setNumIterations(10)
    model.setStartIteration(8)
    val score3 = binaryEvaluator.evaluate(model.transform(test))
    assert(score1 > score3)
  }

  test("Verify LightGBM Classifier with weight column") {
    val model = baseModel.setWeightCol(weightCol)

    val df = pimaDF.withColumn(weightCol, lit(1.0))
    val dfWeight = df.withColumn(weightCol, when(col(labelCol) >= 1, 100.0).otherwise(1.0))

    def countPredictions(df: DataFrame): Long = {
      model.fit(df).transform(df).where(col("prediction") === 1.0).count()
    }

    // Verify changing weight of one label significantly skews the results
    val constLabelPredictionCount = countPredictions(df)
    assert(constLabelPredictionCount * 2 < countPredictions(dfWeight))

    // Also validate with int and long values for weight column predictions are the same within some threshold
    val threshold = 0.2 * constLabelPredictionCount
    val dfInt = pimaDF.withColumn(weightCol, lit(1))
    assert(math.abs(constLabelPredictionCount - countPredictions(dfInt)) < threshold)
    val dfLong = pimaDF.withColumn(weightCol, lit(1L))
    assert(math.abs(constLabelPredictionCount - countPredictions(dfLong)) < threshold)
  }

  test("Verify LightGBM Classifier with unbalanced dataset") {
    val Array(train, test) = taskDF.randomSplit(Array(0.8, 0.2), seed)
    assertBinaryImprovement(
      baseModel, train, test,
      baseModel.setIsUnbalance(true), train, test
    )
  }

  test("Verify LightGBM Classifier with validation dataset") {
    tryWithRetries(Array(0, 100, 500)) {() =>
      val df = au3DF.orderBy(rand()).withColumn(validationCol, lit(false))

      val Array(train, validIntermediate, test) = df.randomSplit(Array(0.5, 0.2, 0.3), seed)
      val valid = validIntermediate.withColumn(validationCol, lit(true))
      val trainAndValid = train.union(valid.orderBy(rand()))

      // model1 should overfit on the given dataset
      val model1 = baseModel
        .setNumLeaves(100)
        .setNumIterations(100)
        .setLearningRate(0.9)
        .setMinDataInLeaf(2)
        .setValidationIndicatorCol(validationCol)
        .setUseBarrierExecutionMode(true) // added for DEBUG, so remove
        .setUseSingleDatasetMode(true) // added for DEBUG, so remove
        .setEarlyStoppingRound(100)

      // model2 should terminate early before overfitting
      val model2 = baseModel
        .setNumLeaves(100)
        .setNumIterations(100)
        .setLearningRate(0.9)
        .setMinDataInLeaf(2)
        .setValidationIndicatorCol(validationCol)
        .setUseBarrierExecutionMode(true) // added for DEBUG, so remove
        .setUseSingleDatasetMode(true) // added for DEBUG, so remove
        .setEarlyStoppingRound(5)

      // Assert evaluation metric improves
      Array("auc", "binary_logloss", "binary_error").foreach { metric =>
        assertBinaryImprovement(
          model1.setMetric(metric), trainAndValid, test,
          model2.setMetric(metric), trainAndValid, test
        )
      }
    }
  }

  test("Verify LightGBM Classifier model handles iterations properly when early stopping") {
    val df = au3DF.orderBy(rand()).withColumn(validationCol, lit(false))

    val Array(train, validIntermediate, test) = df.randomSplit(Array(0.5, 0.2, 0.3), seed)
    val valid = validIntermediate.withColumn(validationCol, lit(true))
    val trainAndValid = train.union(valid.orderBy(rand()))

    // model1 should overfit on the given dataset
    val model1 = baseModel
      .setNumLeaves(100)
      .setNumIterations(100)
      .setLearningRate(0.9)
      .setMinDataInLeaf(2)
      .setMetric("auc")
      .setValidationIndicatorCol(validationCol)
      .setEarlyStoppingRound(100)
    val resultModel1 = model1.fit(trainAndValid)

    // model2 should terminate early
    val model2 = baseModel
      .setNumLeaves(100)
      .setNumIterations(100)
      .setLearningRate(0.9)
      .setMinDataInLeaf(2)
      .setMetric("auc")
      .setValidationIndicatorCol(validationCol)
      .setEarlyStoppingRound(5)
    val resultModel2 = model2.fit(trainAndValid)
    val numIterationsEarlyStopped = resultModel2.getLightGBMBooster.numTotalIterations

    // Early stopping should result in fewer iterations.
    assert(resultModel1.getLightGBMBooster.numTotalIterations > numIterationsEarlyStopped)

    // The number of iterations should be the index of the best iteration + 1.
    assert(numIterationsEarlyStopped == resultModel2.getBoosterBestIteration() + 1)

    // Make sure we serialize and deserialize appropriately.
    val modelString1 = resultModel1.getModel.modelStr.get
    val deserializedModel1 =  LightGBMClassificationModel.loadNativeModelFromString(modelString1)
    val numIterations1 = resultModel1.getLightGBMBooster.numTotalIterations
    assert(deserializedModel1.getLightGBMBooster.numTotalIterations == numIterations1)
    val modelString2 = resultModel2.getModel.modelStr.get
    val deserializedModel2 =  LightGBMClassificationModel.loadNativeModelFromString(modelString2)
    assert(deserializedModel2.getLightGBMBooster.numTotalIterations == numIterationsEarlyStopped)
  }

  test("Verify LightGBM Classifier categorical parameter for sparse dataset") {
    val Array(train, test) = indexedBankTrainDF.randomSplit(Array(0.8, 0.2), seed)
    val categoricalSlotNames = indexedBankTrainDF.schema(featuresCol)
      .metadata.getMetadata("ml_attr").getMetadata("attrs").
      getMetadataArray("numeric").map(_.getString("name"))
      .filter(_.startsWith("c_"))
    val untrainedModel = baseModel.setCategoricalSlotNames(categoricalSlotNames)
    val model = untrainedModel.fit(train)
    // Verify categorical features used in some tree in the model
    assert(model.getModel.modelStr.get.contains("num_cat=1"))
    val metric = binaryEvaluator
      .evaluate(model.transform(test))
    // Verify we get good result
    assert(metric > 0.8)
  }

  test("Verify LightGBM Classifier categorical parameter for dense dataset") {
    val Array(train, test) = indexedTaskDF.randomSplit(Array(0.8, 0.2), seed)
    val categoricalSlotNames = indexedTaskDF.schema(featuresCol)
      .metadata.getMetadata("ml_attr").getMetadata("attrs").
      getMetadataArray("numeric").map(_.getString("name"))
      .filter(_.startsWith("c_"))
    val untrainedModel = baseModel
      .setCategoricalSlotNames(categoricalSlotNames)
    val model = untrainedModel.fit(train)
    // Verify non-zero categorical features used in some tree in the model
    val numCats = Range(1, 5).map(cat => s"num_cat=${cat}")
    assert(numCats.exists(model.getModel.modelStr.get.contains(_)))
    val metric = binaryEvaluator
      .evaluate(model.transform(test))
    // Verify we get good result
    assert(metric > 0.7)
  }

  test("Verify LightGBM pass through parameters") {
    val Array(train, _) = indexedBankTrainDF.randomSplit(Array(0.8, 0.2), seed)
    val untrainedModel = baseModel
      .setCategoricalSlotNames(indexedBankTrainDF.columns.filter(_.startsWith("c_")))
      .setPassThroughArgs("is_enable_sparse=false")

    val model = untrainedModel.fit(train)

    // Verify model contains correct parameter
    assert(model.getModel.modelStr.get.contains("is_enable_sparse: 0"))
  }

  test("Verify LightGBM is_enable_sparse parameters") {
    val Array(train, _) = indexedBankTrainDF.randomSplit(Array(0.8, 0.2), seed)
    val untrainedModel = baseModel
      .setCategoricalSlotNames(indexedBankTrainDF.columns.filter(_.startsWith("c_")))
      .setIsEnableSparse(false)

    val model = untrainedModel.fit(train)

    // Verify model contains correct parameter
    assert(model.getModel.modelStr.get.contains("is_enable_sparse: 0"))
  }

  test("Verify LightGBM use_missing parameters") {
    val Array(train, _) = indexedBankTrainDF.randomSplit(Array(0.8, 0.2), seed)
    val untrainedModel = baseModel
      .setCategoricalSlotNames(indexedBankTrainDF.columns.filter(_.startsWith("c_")))
      .setUseMissing(false)

    val model = untrainedModel.fit(train)

    // Verify model contains correct parameter
    assert(model.getModel.modelStr.get.contains("use_missing: 0"))
  }

  test("Verify LightGBM zero_as_missing parameters") {
    val Array(train, _) = indexedBankTrainDF.randomSplit(Array(0.8, 0.2), seed)
    val untrainedModel = baseModel
      .setCategoricalSlotNames(indexedBankTrainDF.columns.filter(_.startsWith("c_")))
      .setZeroAsMissing(true)

    val model = untrainedModel.fit(train)

    // Verify model contains correct parameter
    assert(model.getModel.modelStr.get.contains("zero_as_missing: 1"))
  }

  test("Verify LightGBM Classifier updating learning_rate on training by using LightGBMDelegate") {
    val Array(train, _) = indexedBankTrainDF.randomSplit(Array(0.8, 0.2), seed)
    val delegate = new TrainDelegate()
    val untrainedModel = baseModel
      .setCategoricalSlotNames(indexedBankTrainDF.columns.filter(_.startsWith("c_")))
      .setDelegate(delegate)
      .setLearningRate(0.1)
      .setNumIterations(2)  // expected learning_rate: iters 0 => 0.1, iters 1 => 0.005

    val model = untrainedModel.fit(train)

    // Verify updating learning_rate
    assert(model.getModel.modelStr.get.contains("learning_rate: 0.005"))
  }

  test("Verify LightGBM Classifier leaf prediction") {
    val Array(train, test) = indexedBankTrainDF.randomSplit(Array(0.8, 0.2), seed)
    val untrainedModel = baseModel
      .setCategoricalSlotNames(indexedBankTrainDF.columns.filter(_.startsWith("c_")))
    val model = untrainedModel.fit(train)

    val evaluatedDf = model.transform(test)

    val leafIndices: Array[Double] = evaluatedDf.select(leafPredCol).rdd.map {
      case Row(v: Vector) => v
    }.first.toArray

    assert(leafIndices.length == model.getModel.numTotalModel)

    // leaf index's value >= 0 and integer
    leafIndices.foreach { index =>
      assert(index >= 0)
      assert(index == index.toInt)
    }

    // if leaf prediction is not wanted, it is possible to remove it.
    val evaluatedDf2 = model.setLeafPredictionCol("").transform(test)
    assert(!evaluatedDf2.columns.contains(leafPredCol))
  }

  test("Verify Binary LightGBM Classifier local feature importance SHAP values") {
    val Array(train, test) = indexedBankTrainDF.randomSplit(Array(0.8, 0.2), seed)
    val untrainedModel = baseModel
      .setCategoricalSlotNames(indexedBankTrainDF.columns.filter(_.startsWith("c_")))
    val model = untrainedModel.fit(train)

    val evaluatedDf = model.transform(test)

    validateHeadRowShapValues(evaluatedDf, model.getModel.numFeatures + 1)

    // if featuresShap is not wanted, it is possible to remove it.
    val evaluatedDf2 = model.setFeaturesShapCol("").transform(test)
    assert(!evaluatedDf2.columns.contains(featuresShapCol))
  }

  test("Verify Binary LightGBM Classifier chunk size parameter") {
    val Array(train, test) = pimaDF.repartition(4).randomSplit(Array(0.8, 0.2), seed)
    val untrainedModel = baseModel.setUseSingleDatasetMode(true)
    val scoredDF1 = untrainedModel.setChunkSize(1000).setSeed(1).setDeterministic(true).fit(train).transform(test)
    val chunkSizes = Array(10, 100, 1000, 10000)
    chunkSizes.foreach { chunkSize =>
      val model = untrainedModel.setChunkSize(chunkSize).setSeed(1).setDeterministic(true).fit(train)
      val scoredDF2 = model.transform(test)
      assertBinaryEquality(scoredDF1, scoredDF2);
    }
  }

  test("Verify Multiclass LightGBM Classifier local feature importance SHAP values") {
    val Array(train, test) = breastTissueDF.select(labelCol, featuresCol).randomSplit(Array(0.8, 0.2), seed)

    val untrainedModel = new LightGBMClassifier()
      .setLabelCol(labelCol)
      .setFeaturesCol(featuresCol)
      .setPredictionCol(predCol)
      .setDefaultListenPort(getAndIncrementPort())
      .setObjective(multiclassObject)
      .setFeaturesShapCol(featuresShapCol)
    val model = untrainedModel.fit(train)

    val evaluatedDf = model.transform(test)

    validateHeadRowShapValues(evaluatedDf, (model.getModel.numFeatures + 1) * model.getModel.numClasses)
  }

  test("Verify LightGBM Classifier with slot names parameter") {

    val originalDf = readCSV(DatasetUtils.binaryTrainFile("PimaIndian.csv").toString).repartition(numPartitions)
      .withColumnRenamed("Diabetes mellitus", labelCol)

    val originalSlotNames = Array("Number of times pregnant",
      "Plasma glucose concentration a 2 hours in an oral glucose tolerance test",
      "Diastolic blood pressure (mm Hg)", "Triceps skin fold thickness (mm)", "2-Hour serum insulin (mu U/ml)",
      "Body mass index (weight in kg/(height in m)^2)", "Diabetes pedigree function","Age (years)")

    val newDf = new VectorAssembler().setInputCols(originalSlotNames).setOutputCol(featuresCol).transform(originalDf)
    val newSlotNames = originalSlotNames.map(name => if(name == "Age (years)") "Age_years" else name)

    // define slot names that has a slot renamed "Age (years)" to "Age_years"
    val untrainedModel = baseModel.setSlotNames(newSlotNames)

    assert(untrainedModel.getSlotNames.length == newSlotNames.length)
    assert(untrainedModel.getSlotNames.contains("Age_years"))

    val model = untrainedModel.fit(newDf)

    // Verify the Age_years column that is renamed  used in some tree in the model
    assert(model.getModel.modelStr.get.contains("Age_years"))
  }

  test("Verify LightGBM Classifier won't get stuck on empty partitions") {
    val baseDF = pimaDF.select(labelCol, featuresCol)
    val df = baseDF.mapPartitions { rows =>
      // Create an empty partition
      if (TaskContext.get.partitionId == 0) {
        Iterator()
      } else {
        rows
      }
    }(RowEncoder(baseDF.schema))

    assertFitWithoutErrors(baseModel, df)
  }

  test("Verify LightGBM Classifier won't get stuck on unbalanced classes in multiclass classification") {
    val baseDF = breastTissueDF.select(labelCol, featuresCol)
    val df = baseDF.mapPartitions({ rows =>
      // Remove all instances of some classes
      if (TaskContext.get.partitionId == 1) {
        rows.filter(_.getInt(0) > 2)
      } else {
        rows
      }
    })(RowEncoder(baseDF.schema))

    val model = new LightGBMClassifier()
      .setLabelCol(labelCol)
      .setFeaturesCol(featuresCol)
      .setPredictionCol(predCol)
      .setDefaultListenPort(getAndIncrementPort())
      .setObjective(multiclassObject)

    // Validate fit works and doesn't get stuck
    assertFitWithoutErrors(model, df)
  }

  test("Verify LightGBM Classifier won't get stuck on unbalanced classes in binary classification") {
    val baseDF = pimaDF.select(labelCol, featuresCol)
    val df = baseDF.mapPartitions({ rows =>
      // Remove all instances of some classes
      if (TaskContext.get.partitionId == 1) {
        rows.filter(_.getInt(0) < 1)
      } else {
        rows
      }
    })(RowEncoder(baseDF.schema))

    // Validate fit works and doesn't get stuck
    assertFitWithoutErrors(baseModel, df)
  }

  test("Verify LightGBM Classifier won't get stuck on " +
    "number of features in data is not the same as it was in training data") {
    val inputData = Seq(
      LabeledPoint(1.0, Vectors.dense(1.0, 0.0, 3.0)),
      LabeledPoint(0.0, Vectors.sparse(3, Array(0, 2), Array(1.0, 3.0)))
    )
    val testData = Seq(
      (
        "uuid-value",
        Vectors.sparse(15, Array(2, 6, 8, 14), Array(1.0, 1.0, 1.0, 2.0))
      )
    )

    val modelDF = spark.createDataFrame(inputData).toDF("labels", "features")
    val testDF = spark.createDataFrame(testData).toDF("uuid", "features")

    val fitModel = baseModel.fit(modelDF)
    val oldModelString = fitModel.getModel.modelStr.get

    val testModel = LightGBMClassificationModel.loadNativeModelFromString(oldModelString)
    testModel.setPredictDisableShapeCheck(true)

    assert(testModel.transform(testDF).collect().length > 0)
  }

  def verifyLearnerOnBinaryCsvFile(fileName: String,
                                   labelColumnName: String,
                                   decimals: Int): Unit = {
    test("Verify LightGBMClassifier can be trained " +
      s"and scored on $fileName") {
      boostingTypes.foreach { boostingType =>
        val df = loadBinary(fileName, labelColumnName)
        val model = baseModel
          .setBoostingType(boostingType)

        if (boostingType == "rf") {
          model.setBaggingFraction(0.9)
          model.setBaggingFreq(1)
        }

        val fitModel = model.fit(df)
        val tdf = fitModel.transform(df)
        assertProbabilities(tdf, model)

        assertImportanceLengths(fitModel, df)
        addBenchmark(s"LightGBMClassifier_${fileName}_$boostingType",
          binaryEvaluator.evaluate(fitModel.transform(df)), decimals)
      }
    }
  }

  def verifyLearnerOnMulticlassCsvFile(fileName: String,
                                       labelColumnName: String,
                                       decimals: Int): Unit = {
    verifyLearnerOnMulticlassCsvFile(fileName, labelColumnName, scala.math.pow(10, -decimals.toDouble))
  }

  def verifyLearnerOnMulticlassCsvFile(fileName: String,
                                       labelColumnName: String,
                                       precision: Double): Unit = {
    test(s"Verify LightGBMClassifier can be trained and scored " +
      s"on multiclass $fileName") {
      lazy val df = loadMulticlass(fileName, labelColumnName).cache()
      boostingTypes.foreach { boostingType =>

        val model = baseModel
          .setObjective(multiclassObject)
          .setBoostingType(boostingType)

        if (boostingType == "rf") {
          model.setBaggingFraction(0.9)
          model.setBaggingFreq(1)
        }

        val fitModel = model.fit(df)
        val tdf = fitModel.transform(df)
        assertProbabilities(tdf, model)

        assertImportanceLengths(fitModel, df)
        addBenchmark(s"LightGBMClassifier_${fileName}_$boostingType",
          multiclassEvaluator.evaluate(tdf), precision)
      }
      df.unpersist()
    }
  }

  override def testObjects(): Seq[TestObject[LightGBMClassifier]] = {
    Seq(new TestObject(baseModel, pimaDF.coalesce(1)))
  }

  def verifySaveBooster(fileName: String,
                        outputFileName: String,
                        labelColumnName: String,
                        colsToVerify: Array[String]): Unit = {
    test("Verify LightGBMClassifier save booster to " + fileName) {
      val model = baseModel
      val df = loadBinary(fileName, labelColumnName)
      val fitModel = model.fit(df)

      val targetDir: Path = Paths.get(getClass.getResource("/").toURI)
      val modelPath = targetDir.toString + "/" + outputFileName
      FileUtils.deleteDirectory(new File(modelPath))
      fitModel.saveNativeModel(modelPath, overwrite = true)
      val retrievedModelStr = fitModel.getNativeModel()
      assert(Files.exists(Paths.get(modelPath)), true)

      val oldModelString = fitModel.getModel.modelStr.get
      // Assert model string is equal when retrieved from booster and getNativeModel API
      assert(retrievedModelStr == oldModelString)

      // Verify model string contains some feature
      colsToVerify.foreach(col => oldModelString.contains(col))

      assertFitWithoutErrors(model.setModelString(oldModelString), df)

      // Verify can load model from file
      val resultsFromString = LightGBMClassificationModel
        .loadNativeModelFromString(oldModelString)
        .setFeaturesCol(featuresCol)
        .setRawPredictionCol(rawPredCol)
        .setLeafPredictionCol(leafPredCol)
        .setFeaturesShapCol(featuresShapCol)
        .transform(df)

      val resultsFromFile = LightGBMClassificationModel
        .loadNativeModelFromFile(modelPath)
        .setFeaturesCol(featuresCol)
        .setRawPredictionCol(rawPredCol)
        .setLeafPredictionCol(leafPredCol)
        .setFeaturesShapCol(featuresShapCol)
        .transform(df)

      val resultsOriginal = fitModel.transform(df)

      assert(resultsFromString === resultsOriginal)
      assert(resultsFromFile === resultsOriginal)
    }
  }
  override def reader: MLReadable[_] = LightGBMClassifier
  override def modelReader: MLReadable[_] = LightGBMClassificationModel
}
