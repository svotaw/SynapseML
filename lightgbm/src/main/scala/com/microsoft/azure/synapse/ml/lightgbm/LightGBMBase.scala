// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.azure.synapse.ml.lightgbm

import com.microsoft.azure.synapse.ml.core.utils.{ClusterUtil, ParamsStringBuilder}
import com.microsoft.azure.synapse.ml.io.http.SharedSingleton
import com.microsoft.azure.synapse.ml.lightgbm.ConnectionState.Finished
import com.microsoft.azure.synapse.ml.lightgbm.LightGBMUtils.{closeConnections, handleConnection, sendDataToExecutors}
import com.microsoft.azure.synapse.ml.lightgbm.TrainUtils._
import com.microsoft.azure.synapse.ml.lightgbm.booster.LightGBMBooster
import com.microsoft.azure.synapse.ml.lightgbm.dataset.{DatasetUtils, SampledData}
import com.microsoft.azure.synapse.ml.lightgbm.params._
import com.microsoft.azure.synapse.ml.logging.BasicLogging
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.attribute._
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.ml.param.shared.{HasFeaturesCol => HasFeaturesColSpark, HasLabelCol => HasLabelColSpark}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql._
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types._

import java.net.{ServerSocket, Socket}
import java.util.concurrent.Executors
import scala.collection.immutable.HashSet
import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.concurrent.duration.{Duration, SECONDS}
import scala.concurrent.{Await, ExecutionContext, ExecutionContextExecutor, Future}
import scala.language.existentials
import scala.math.min
import scala.util.matching.Regex

trait LightGBMBase[TrainedModel <: Model[TrainedModel]] extends Estimator[TrainedModel]
  with LightGBMParams with HasFeaturesColSpark with HasLabelColSpark with LightGBMPerformance with BasicLogging {

  /** Trains the LightGBM model.  If batches are specified, breaks training dataset into batches for training.
    *
    * @param allTrainingData The input dataset to train.
    * @return The trained model.
    */
  protected def train(allTrainingData: Dataset[_]): TrainedModel = {
    LightGBMUtils.initializeNativeLibrary()

    val isMultiBatch = getNumBatches > 0
    val numBatches = if (isMultiBatch) getNumBatches else 1
    setBatchPerformanceMeasures(Array.fill(numBatches)(None))

    logTrain({
      getOptGroupCol.foreach(DatasetUtils.validateGroupColumn(_, allTrainingData.schema))
      if (isMultiBatch) {
        val ratio = 1.0 / numBatches
        val datasetBatches = allTrainingData.randomSplit((0 until getNumBatches).map(_ => ratio).toArray)
        datasetBatches.zipWithIndex.foldLeft(None: Option[TrainedModel]) { (model, datasetWithIndex) =>
          if (model.isDefined) {
            setModelString(stringFromTrainedModel(model.get))
          }
          val batchDataset = datasetWithIndex._1
          val batchIndex = datasetWithIndex._2

          beforeTrainBatch(batchIndex, batchDataset, model)
          val newModel = trainOneDataBatch(batchDataset, batchIndex)
          afterTrainBatch(batchIndex, batchDataset, newModel)

          Some(newModel)
        }.get
      } else {
        trainOneDataBatch(allTrainingData, batchIndex = 0)
      }
    })
  }

  def beforeTrainBatch(batchIndex: Int, dataset: Dataset[_], model: Option[TrainedModel]): Unit = {
    if (getDelegate.isDefined) {
      val previousBooster: Option[LightGBMBooster] = model match {
        case Some(_) => Option(new LightGBMBooster(stringFromTrainedModel(model.get)))
        case None => None
      }

      getDelegate.get.beforeTrainBatch(batchIndex, log, dataset, previousBooster)
    }
  }

  def afterTrainBatch(batchIndex: Int, dataset: Dataset[_], model: TrainedModel): Unit = {
    if (getDelegate.isDefined) {
      val booster = new LightGBMBooster(stringFromTrainedModel(model))
      getDelegate.get.afterTrainBatch(batchIndex, log, dataset, booster)
    }
  }

  protected def castColumns(dataset: Dataset[_], trainingCols: Array[(String, Seq[DataType])]): DataFrame = {
    val schema = dataset.schema
    // Cast columns to correct types
    dataset.select(
      trainingCols.map {
        case (name, datatypes: Seq[DataType]) =>
          val index = schema.fieldIndex(name)
          // Note: We only want to cast if original column was of numeric type

          schema.fields(index).dataType match {
            case numericDataType: NumericType =>
              // If more than one datatype is allowed, see if any match
              if (datatypes.contains(numericDataType)) {
                dataset(name)
              } else {
                // If none match, cast to the first option
                dataset(name).cast(datatypes.head)
              }
            case _ => dataset(name)
          }
      }: _*
    ).toDF()
  }

    protected def prepareDataframe(dataset: Dataset[_], numTasks: Int): DataFrame = {
      val df = castColumns(dataset, getTrainingCols)
    // Reduce number of partitions to number of executor tasks
    /* Note: with barrier execution mode we must use repartition instead of coalesce when
     * running on spark standalone.
     * Using coalesce, we get the error:
     *
     * org.apache.spark.scheduler.BarrierJobUnsupportedRDDChainException:
     * [SPARK-24820][SPARK-24821]: Barrier execution mode does not allow the following
     * pattern of RDD chain within a barrier stage:
     * 1. Ancestor RDDs that have different number of partitions from the resulting
     * RDD (eg. union()/coalesce()/first()/take()/PartitionPruningRDD). A workaround
     * for first()/take() can be barrierRdd.collect().head (scala) or barrierRdd.collect()[0] (python).
     * 2. An RDD that depends on multiple barrier RDDs (eg. barrierRdd1.zip(barrierRdd2)).
     *
     * Without repartition, we may hit the error:
     * org.apache.spark.scheduler.BarrierJobSlotsNumberCheckFailed: [SPARK-24819]:
     * Barrier execution mode does not allow run a barrier stage that requires more
     * slots than the total number of slots in the cluster currently. Please init a
     * new cluster with more CPU cores or repartition the input RDD(s) to reduce the
     * number of slots required to run this barrier stage.
     *
     * Hence we still need to estimate the number of tasks and repartition even when using
     * barrier execution, which is unfortunate as repartition is more expensive than coalesce.
     */
    if (getUseBarrierExecutionMode) {
      val numPartitions = df.rdd.getNumPartitions
      if (numPartitions > numTasks) {
        df.repartition(numTasks)
      } else {
        df
      }
    } else {
      df.coalesce(numTasks)
    }
  }

  protected def getTrainingCols: Array[(String, Seq[DataType])] = {
    val colsToCheck: Array[(Option[String], Seq[DataType])] = Array(
      (Some(getLabelCol), Seq(DoubleType)),
      (Some(getFeaturesCol), Seq(VectorType)),
      (get(weightCol), Seq(DoubleType)),
      (getOptGroupCol, Seq(IntegerType, LongType, StringType)),
      (get(validationIndicatorCol), Seq(BooleanType)),
      (get(initScoreCol), Seq(DoubleType, VectorType)))

    colsToCheck.flatMap {
      case (col: Option[String], colType: Seq[DataType]) =>
        if (col.isDefined) Some(col.get, colType) else None
    }
  }

  /**
    * Retrieves the categorical indexes in the features column.
    *
    * @param featuresSchema The schema of the features column
    * @return the categorical indexes in the features column.
    */
  protected def getCategoricalIndexes(featuresSchema: StructField): Array[Int] = {
    val categoricalColumnSlotNames = get(categoricalSlotNames).getOrElse(Array.empty[String])
    val categoricalIndexes = if (getSlotNames.nonEmpty) {
      categoricalColumnSlotNames.map(getSlotNames.indexOf(_))
    } else {
      val categoricalSlotNamesSet = HashSet(categoricalColumnSlotNames: _*)
      val attributes = AttributeGroup.fromStructField(featuresSchema).attributes
      if (attributes.isEmpty) {
        Array[Int]()
      } else {
       attributes.get.zipWithIndex.flatMap {
          case (null, _) => None
          case (attr, idx) =>
            if (attr.name.isDefined && categoricalSlotNamesSet.contains(attr.name.get)) {
              Some(idx)
            } else {
              attr match {
                case _: NumericAttribute | UnresolvedAttribute => None
                // Note: it seems that BinaryAttribute is not considered categorical,
                // since all OHE cols are marked with this, but StringIndexed are always Nominal
                case _: BinaryAttribute => None
                case _: NominalAttribute => Some(idx)
              }
            }
        }
      }
    }

    get(categoricalSlotIndexes)
      .getOrElse(Array.empty[Int])
      .union(categoricalIndexes).distinct
  }

  def getSlotNamesWithMetadata(featuresSchema: StructField): Option[Array[String]] = {
    if (getSlotNames.nonEmpty) {
      Some(getSlotNames)
    } else {
      AttributeGroup.fromStructField(featuresSchema).attributes.flatMap(attributes =>
        if (attributes.isEmpty) {
          None
        } else {
          val colNames = attributes.indices.map(_.toString).toArray
          attributes.foreach(attr =>
            attr.index.foreach(index => colNames(index) = attr.name.getOrElse(index.toString)))
          Some(colNames)
        }
      )
    }
  }

  private def validateSlotNames(featuresSchema: StructField): Unit = {
    val metadata = AttributeGroup.fromStructField(featuresSchema)
    if (metadata.attributes.isDefined) {
      val slotNamesOpt = getSlotNamesWithMetadata(featuresSchema)
      val pattern = new Regex("[\",:\\[\\]{}]")
      slotNamesOpt.foreach(slotNames => {
        val badSlotNames = slotNames.flatMap(slotName =>
          if (pattern.findFirstIn(slotName).isEmpty) None else Option(slotName))
        if (!badSlotNames.isEmpty) {
          throw new IllegalArgumentException(
            s"Invalid slot names detected in features column: ${badSlotNames.mkString(",")}" +
            " \n Special characters \" , : \\ [ ] { } will cause unexpected behavior in LGBM unless changed." +
            " This error can be fixed by renaming the problematic columns prior to vector assembly.")
        }
      })
    }
  }

  /**
    * Constructs the DartModeParams
    *
    * @return DartModeParams object containing parameters related to dart mode.
    */
  protected def getDartParams: DartModeParams = {
    DartModeParams(getDropRate, getMaxDrop, getSkipDrop, getXGBoostDartMode, getUniformDrop)
  }

  /**
    * Constructs the GeneralParams.
    *
    * @return GeneralParams object containing parameters related to general LightGBM parameters.
    */
  protected def getGeneralParams(numTasks: Int, featuresSchema: StructField): GeneralParams = {
    GeneralParams(
      getParallelism,
      get(topK),
      getNumIterations,
      getLearningRate,
      get(numLeaves),
      get(maxBin),
      get(binSampleCount),
      get(baggingFraction),
      get(posBaggingFraction),
      get(negBaggingFraction),
      get(baggingFreq),
      get(baggingSeed),
      getEarlyStoppingRound,
      getImprovementTolerance,
      get(featureFraction),
      get(featureFractionByNode),
      get(maxDepth),
      get(minSumHessianInLeaf),
      numTasks,
      if (getModelString == null || getModelString.isEmpty) None else get(modelString),
      getCategoricalIndexes(featuresSchema),
      getVerbosity,
      getBoostingType,
      get(lambdaL1),
      get(lambdaL2),
      get(metric),
      get(minGainToSplit),
      get(maxDeltaStep),
      getMaxBinByFeature,
      get(minDataPerBin),
      get(minDataInLeaf),
      get(topRate),
      get(otherRate),
      getMonotoneConstraints,
      get(monotoneConstraintsMethod),
      get(monotonePenalty),
      getSlotNames)
  }

  /**
    * Constructs the DatasetParams.
    *
    * @return DatasetParams object containing parameters related to LightGBM Dataset parameters.
    */
  protected def getDatasetParams(): DatasetParams = {
    DatasetParams(
      get(isEnableSparse),
      get(useMissing),
      get(zeroAsMissing)
    )
  }

  /**
    * Constructs the ExecutionParams.
    *
    * @return ExecutionParams object containing parameters related to LightGBM execution.
    */
  protected def getExecutionParams(numTasksPerExec: Int): ExecutionParams = {
    val execNumThreads =
      if (getUseSingleDatasetMode) get(numThreads).getOrElse(numTasksPerExec - 1)
      else getNumThreads
    ExecutionParams(getChunkSize, getMatrixType, execNumThreads, getExecutionMode, getUseSingleDatasetMode)
  }

  /**
    * Constructs the ColumnParams.
    *
    * @return ColumnParams object containing the parameters related to LightGBM columns.
    */
  protected def getColumnParams: ColumnParams = {
    ColumnParams(getLabelCol, getFeaturesCol, get(weightCol), get(initScoreCol), getOptGroupCol)
  }

  /**
    * Constructs the ObjectiveParams.
    *
    * @return ObjectiveParams object containing parameters related to the objective function.
    */
  protected def getObjectiveParams: ObjectiveParams = {
    ObjectiveParams(getObjective, if (isDefined(fobj)) Some(getFObj) else None)
  }

  /**
    * Constructs the SeedParams.
    *
    * @return SeedParams object containing the parameters related to LightGBM seeds and determinism.
    */
  protected def getSeedParams: SeedParams = {
    SeedParams(
      get(seed),
      get(deterministic),
      get(baggingSeed),
      get(featureFractionSeed),
      get(extraSeed),
      get(dropSeed),
      get(dataRandomSeed),
      get(objectiveSeed),
      getBoostingType,
      getObjective)
  }

  /**
    * Constructs the CategoricalParams.
    *
    * @return CategoricalParams object containing the parameters related to LightGBM categorical features.
    */
  protected def getCategoricalParams: CategoricalParams = {
    CategoricalParams(
      get(minDataPerGroup),
      get(maxCatThreshold),
      get(catl2),
      get(catSmooth),
      get(maxCatToOnehot))
  }

  def getDatasetCreationParams(categoricalIndexes: Array[Int], numThreads: Int): String = {
    new ParamsStringBuilder(prefix = "", delimiter = "=")
      .appendParamValueIfNotThere("is_pre_partition", Option("True"))
      .appendParamValueIfNotThere("max_bin", Option(getMaxBin))
      .appendParamValueIfNotThere("bin_construct_sample_cnt", Option(getBinSampleCount))
      .appendParamValueIfNotThere("min_data_in_leaf", Option(getMinDataInLeaf))
      .appendParamValueIfNotThere("num_threads", Option(numThreads))
      .appendParamListIfNotThere("categorical_feature", categoricalIndexes)
      .appendParamValueIfNotThere("data_random_seed", get(dataRandomSeed).orElse(get(seed)))
      .result

    // TODO do we need to add any of the new Dataset parameters here? (e.g. is_enable_sparse) check
  }

  /**
    * Opens a socket communications channel on the driver, starts a thread that
    * waits for the host:port from the executors, and then sends back the
    * information to the executors.
    *
    * @param numTasks The total number of training tasks to wait for.
    * @return The address and port of the driver socket.
    */
  private def createDriverNodesThread(numTasks: Int, spark: SparkSession): (String, Int, Future[Unit]) = {
    // Start a thread and open port to listen on
    implicit val context: ExecutionContextExecutor =
      ExecutionContext.fromExecutor(Executors.newSingleThreadExecutor())
    val driverServerSocket = new ServerSocket(getDriverListenPort)
    // Set timeout on socket
    val duration = Duration(getTimeout, SECONDS)
    if (duration.isFinite()) {
      driverServerSocket.setSoTimeout(duration.toMillis.toInt)
    }
    val f = Future {
      var emptyTaskCounter = 0
      val hostAndPorts = ListBuffer[(Socket, String)]()
      val hostToMinPartition = mutable.Map[String, String]()
      if (getUseBarrierExecutionMode) {
        log.info(s"driver using barrier execution mode")

        def connectToWorkers: Boolean = handleConnection(driverServerSocket, log,
          hostAndPorts, hostToMinPartition) == Finished || connectToWorkers

        connectToWorkers
      } else {
        log.info(s"driver expecting $numTasks connections...")
        while (hostAndPorts.size + emptyTaskCounter < numTasks) {
          val connectionResult = handleConnection(driverServerSocket, log, hostAndPorts, hostToMinPartition)
          if (connectionResult == ConnectionState.EmptyTask) emptyTaskCounter += 1
        }
      }
      // Concatenate with commas, eg: host1:port1,host2:port2, ... etc
      val hostPortsList = hostAndPorts.map(_._2).sortBy(hostPort => {
        val host = hostPort.split(":")(0)
        Integer.parseInt(hostToMinPartition(host))
      })
      val allConnections = hostPortsList.mkString(",")
      log.info(s"driver writing back to all connections: $allConnections")
      // Send data back to all tasks and helper tasks on executors
      sendDataToExecutors(hostAndPorts, allConnections)
      closeConnections(log, hostAndPorts, driverServerSocket)
    }
    val host = ClusterUtil.getDriverHost(spark)
    val port = driverServerSocket.getLocalPort
    log.info(s"driver waiting for connections on host: $host and port: $port")
    (host, port, f)
  }

  /**
    * Inner train method for LightGBM learners.  Calculates the number of workers,
    * creates a driver thread, and runs mapPartitions on the dataset.
    *
    * @param dataset    The dataset to train on.
    * @param batchIndex In running in batch training mode, gets the batch number.
    * @return The LightGBM Model from the trained LightGBM Booster.
    */
  protected def trainOneDataBatch(dataset: Dataset[_], batchIndex: Int): TrainedModel = {
    val measures = new ExecutionMeasures()
    setBatchPerformanceMeasure(batchIndex, measures)
    val numTasksPerExecutor = ClusterUtil.getNumTasksPerExecutor(dataset.sparkSession, log)
    // By default, we try to intelligently calculate the number of executors, but user can override this with numTasks
    val numTasks =
      if (getNumTasks > 0) getNumTasks
      else {
        val numExecutorTasks = ClusterUtil.getNumExecutorTasks(dataset.sparkSession, numTasksPerExecutor, log)
        min(numExecutorTasks, dataset.rdd.getNumPartitions)
      }
    val df = prepareDataframe(dataset, numTasks)

    val sc = dataset.sparkSession.sparkContext

    val (trainingData, validationData) =
      if (get(validationIndicatorCol).isDefined && dataset.columns.contains(getValidationIndicatorCol))
        (df.filter(x => !x.getBoolean(x.fieldIndex(getValidationIndicatorCol))),
          Some(sc.broadcast(collectValidationData(df, measures))))
      else (df, None)

    val preprocessedDF = preprocessData(trainingData)

    val (numCols, numInitScoreClasses) = calculateColumnStatistics(preprocessedDF, measures)

    val featuresSchema = dataset.schema(getFeaturesCol)
    val trainParams: BaseTrainParams = getTrainParams(numTasks, featuresSchema, numTasksPerExecutor)
    calculateCustomTrainParams(trainParams, dataset)
    log.info(s"LightGBM parameters: ${trainParams.toString()}")

    val isStreamingMode = getExecutionMode == LightGBMConstants.StreamingExecutionMode
    val (serializedReferenceDataset: Option[Broadcast[Array[Byte]]], partitionCounts: Option[Array[Long]]) =
      if (isStreamingMode) {
        val stats = calculateRowStatistics(df, trainParams, numCols, measures)
        (Some(sc.broadcast(stats._1)), Some(stats._2))
      }
      else (None, None)

    validateSlotNames(featuresSchema)
    executeSparkTraining(
      preprocessedDF,
      validationData,
      serializedReferenceDataset,
      partitionCounts,
      trainParams,
      numCols,
      numInitScoreClasses,
      batchIndex,
      numTasks,
      numTasksPerExecutor,
      measures)
  }

  /**
    * Get the validation data from the initial dataset.
    *
    * @param dataframe The dataset to train on.
    * @return The number of feature columns and initial score classes
    */
  private def collectValidationData(df: DataFrame, measures: ExecutionMeasures): Array[Row] = {
    measures.markValidationDataCollectionStart()
    val data = preprocessData(df.filter(x =>
      x.getBoolean(x.fieldIndex(getValidationIndicatorCol)))).collect()
    measures.markValidationDataCollectionStop()
    data
  }

  /**
    * Extract column counts from the dataset.
    *
    * @param dataframe The dataset to train on.
    * @return The number of feature columns and initial score classes
    */
  protected def calculateColumnStatistics(dataframe: DataFrame, measures: ExecutionMeasures): (Int, Int) = {
    measures.markColumnStatisticsStart()
    // Use the first row to get the column count
    val firstRow: Row = dataframe.first()

    val featureData = firstRow.getAs[Any](getFeaturesCol)
    val numCols = featureData match {
      case sparse: SparseVector => sparse.size
      case dense: DenseVector => dense.size
      case _ => throw new IllegalArgumentException("Unknown row data type to push")
    }

    val numInitScoreClasses =
      if (get(initScoreCol).isEmpty) 0
      else if (dataframe.schema(getInitScoreCol).dataType == VectorType)
        firstRow.getAs[DenseVector](getInitScoreCol).size
      else 1

    measures.markColumnStatisticsStop()
    (numCols, numInitScoreClasses)
  }

  /**
    * Inner train method for LightGBM learners.  Calculates the number of workers,
    * creates a driver thread, and runs mapPartitions on the dataset.
    *
    * @param dataset    The dataset to train on.
    * @param trainingParams The training parameters.
    * @param numCols The number of feature columns.
    * @return The serialized Dataset reference and an array of partition counts.
    */
  protected def calculateRowStatistics(dataframe: DataFrame,
                                       trainingParams: BaseTrainParams,
                                       numCols: Int,
                                       measures: ExecutionMeasures): (Array[Byte], Array[Long]) = {
    measures.markRowStatisticsStart()
    // Get the row counts per partition
    val indexedRowCounts: Array[(Int, Long)] = dataframe
      .rdd
      .mapPartitionsWithIndex({case (i,rows) => Iterator((i,rows.size.toLong))}, true)
      .collect()
    val rowCounts: Array[Long] = new Array[Long](indexedRowCounts.size)
    indexedRowCounts.foreach(pair => rowCounts(pair._1) = pair._2)
    val totalNumRows = indexedRowCounts.map(partition => partition._2).sum

    // Get sample data using sample() function in Spark
    // TODO optimize with just a take() in case of user approval
    val sampleCount: Int = getBinSampleCount
    val seed: Int = getSeedParams.dataRandomSeed.getOrElse(42) // TODO data or just plain seed?
    val featureColName = getFeaturesCol
    // TODO what if error causes < sampleCount?  should we add a little buffer and let limit() do the work?
    val fraction = if (sampleCount > totalNumRows) 1.0
                   else Math.min(1.0, (sampleCount.toDouble + 10000)/totalNumRows)
    val numSamples = Math.min(sampleCount, totalNumRows).toInt
    val rawSampleData = dataframe.select(dataframe.col(featureColName)).sample(fraction, seed).limit(numSamples)

    // Make a wrapped object that formats the binary sample data as needed by LightGBM
    val sampleData = new SampledData(numSamples, numCols)
    rawSampleData.collect().foreach(row => sampleData.pushRow(row, featureColName))

    // Get reference data set using sampled data
    // Use driver node to run LightGBM in standalone mode
    val datasetParams = getDatasetCreationParams(
      trainingParams.generalParams.categoricalFeatures,
      trainingParams.executionParams.numThreads)

    val serializedReference = createReferenceDataset(totalNumRows.toInt,
      numCols,
      datasetParams,
      sampleData,
      trainingParams.generalParams.featureNames,
      log)
    measures.markRowStatisticsStop()
    (serializedReference, rowCounts)
  }

  /**
    * Run a parallel job via map partitions to initialize the native library and network,
    * translate the data to the LightGBM in-memory representation and train the models.
    *
    * @param dataframe    The dataset to train on.
    * @param batchIndex In running in batch training mode, gets the batch number.
    * @return The LightGBM Model from the trained LightGBM Booster.
    */
  protected def executeSparkTraining(dataframe: DataFrame,
                                     validationData: Option[Broadcast[Array[Row]]],
                                     serializedReferenceDataset: Option[Broadcast[Array[Byte]]],
                                     partitionCounts: Option[Array[Long]],
                                     trainParams: BaseTrainParams,
                                     numCols: Int,
                                     numInitValueClasses: Int,
                                     batchIndex: Int,
                                     numTasks: Int,
                                     numTasksPerExecutor: Int,
                                     measures: ExecutionMeasures): TrainedModel = {
    val (inetAddress, port, future) = createDriverNodesThread(numTasks, dataframe.sparkSession)
    val ctx = getTrainingContext(dataframe,
                                 validationData,
                                 serializedReferenceDataset,
                                 partitionCounts,
                                 trainParams,
                                 numCols,
                                 numInitValueClasses,
                                 batchIndex,
                                 numTasksPerExecutor,
                                 inetAddress,
                                 port)

    // Create the object that will manage the mapPartitions function
    val workerTaskHandler: BasePartitionTask = if (ctx.isStreaming) new StreamingPartitionTask()
                                               else new BulkPartitionTask()
    val mapPartitionsFunc = mapPartitions(ctx, workerTaskHandler)(_)

    val encoder = Encoders.kryo[PartitionResult]
    measures.markTrainingStart()
    val results: Array[PartitionResult] =
      if (getUseBarrierExecutionMode)
        dataframe.rdd.barrier().mapPartitions(mapPartitionsFunc).collect()
      else
        dataframe.mapPartitions(mapPartitionsFunc)(encoder).collect()

    val lightGBMBooster = results.flatMap(r => r.booster).head
    measures.setTaskMeasures(results.map(r => r.taskMeasures))

    // Wait for future to complete (should be done by now)
    Await.result(future, Duration(getTimeout, SECONDS))
    measures.markTrainingStop()
    val model = getModel(trainParams, lightGBMBooster)
    measures.markExecutionEnd()
    model
  }

  /**
    * Get the object that holds all relevant context information for the training session.
    *
    * @param dataframe    The dataset to train on.
    * @param batchIndex In running in batch training mode, gets the batch number.
    * @return The LightGBM Model from the trained LightGBM Booster.
    */
  protected def getTrainingContext(dataframe: DataFrame,
                                     validationData: Option[Broadcast[Array[Row]]],
                                     serializedReferenceDataset: Option[Broadcast[Array[Byte]]],
                                     partitionCounts: Option[Array[Long]],
                                     trainParams: BaseTrainParams,
                                     numCols: Int,
                                     numInitValueClasses: Int,
                                     batchIndex: Int,
                                     numTasksPerExecutor: Int,
                                     inetAddress: String,
                                     port: Int): TrainingContext = {
    val networkParams = NetworkParams(getDefaultListenPort, inetAddress, port, getUseBarrierExecutionMode)
    val columnParams = getColumnParams
    val schema = dataframe.schema
    val sharedState = SharedSingleton(new SharedState(columnParams, schema, trainParams))
    val datasetParams = getDatasetCreationParams(
      trainParams.generalParams.categoricalFeatures,
      trainParams.executionParams.numThreads)

    TrainingContext(batchIndex, // TODO log this context?
      sharedState,
      schema,
      numCols,
      numInitValueClasses,
      1, // TODO where to set this? (microBatchSize)
      trainParams,
      networkParams,
      columnParams,
      datasetParams,
      getSlotNamesWithMetadata(dataframe.schema(getFeaturesCol)),
      numTasksPerExecutor,
      validationData,
      serializedReferenceDataset,
      partitionCounts)
  }

  /** Method that is executed per worker as part of Spark mapPartitions call. Just sets the
    * Logger (since it's not serializable) and calls the dedicated worker task class.
    *
    * @return Iterator[T], as per Spark API
    */
  private def mapPartitions(ctx: TrainingContext,
                            task: BasePartitionTask)
                           (inputRows: Iterator[Row]): Iterator[PartitionResult] = {
    // TODO set Logger on BasePartitionTask directly?
    ctx.setLogger(log) // Logger is not serializable, so set it in the context of the worker
    task.mapPartitionTask(ctx)(inputRows)
  }

  /** Optional group column for Ranking, set to None by default.
    *
    * @return None
    */
  protected def getOptGroupCol: Option[String] = None

  /** Gets the trained model given the train parameters and booster.
    *
    * @return trained model.
    */
  protected def getModel(trainParams: BaseTrainParams, lightGBMBooster: LightGBMBooster): TrainedModel

  /** Gets the training parameters.
    *
    * @param numTasks        The total number of tasks.
    * @param dataset         The training dataset.
    * @param numTasksPerExec The number of tasks per executor.
    * @return train parameters.
    */
  protected def getTrainParams(numTasks: Int,
                               featuresSchema: StructField,
                               numTasksPerExec: Int): BaseTrainParams

  protected def calculateCustomTrainParams(params: BaseTrainParams, dataset: Dataset[_]): Unit = { }

  protected def stringFromTrainedModel(model: TrainedModel): String

  /** Allow algorithm specific preprocessing of dataset.
    *
    * @param df The data frame to preprocess prior to training.
    * @return The preprocessed data.
    */
  protected def preprocessData(df: DataFrame): DataFrame = df
}
