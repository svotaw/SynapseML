// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.azure.synapse.ml.lightgbm

import com.microsoft.azure.synapse.ml.core.env.StreamUtilities._
import com.microsoft.azure.synapse.ml.core.utils.FaultToleranceUtils
import com.microsoft.azure.synapse.ml.lightgbm.booster.LightGBMBooster
import com.microsoft.azure.synapse.ml.lightgbm.dataset.{LightGBMDataset, SampledData}
import com.microsoft.azure.synapse.ml.lightgbm.params.{BaseTrainParams, ClassifierTrainParams}
import com.microsoft.ml.lightgbm._
import org.apache.spark.{BarrierTaskContext, TaskContext}
import org.slf4j.Logger

import java.io._
import java.net._

private object TrainUtils extends Serializable {

  def createBooster(trainParams: BaseTrainParams,
                    trainDatasetPtr: LightGBMDataset,
                    validDatasetPtr: Option[LightGBMDataset]): LightGBMBooster = {
    // Create the booster
    val parameters = trainParams.toString()
    val booster = new LightGBMBooster(trainDatasetPtr, parameters)
    trainParams.generalParams.modelString.foreach { modelStr =>
      booster.mergeBooster(modelStr)
    }
    validDatasetPtr.foreach { lgbmdataset =>
      booster.addValidationDataset(lgbmdataset)
    }
    booster
  }

  def beforeTrainIteration(state: PartitionTaskTrainingState): Unit = {
    if (state.ctx.trainingParams.delegate.isDefined) {
      state.ctx.trainingParams.delegate.get.beforeTrainIteration(state.ctx.batchIndex,
                                                    state.partitionId,
                                                    state.iteration,
                                                    state.ctx.log,
                                                    state.ctx.trainingParams,
                                                    state.booster,
                                                    state.ctx.hasValid)
    }
  }

  def afterTrainIteration(state: PartitionTaskTrainingState,
                          trainEvalResults: Option[Map[String, Double]],
                          validEvalResults: Option[Map[String, Double]]): Unit = {
    val info = state.ctx
    if (info.trainingParams.delegate.isDefined) {
      info.trainingParams.delegate.get.afterTrainIteration(info.batchIndex,
        state.partitionId,
        state.iteration,
        info.log,
        info.trainingParams,
        state.booster,
        info.hasValid,
        state.isFinished,
        trainEvalResults,
        validEvalResults)
    }
  }

  def getLearningRate(state: PartitionTaskTrainingState): Double = {
    state.ctx.trainingParams.delegate match {
      case Some(delegate) => delegate.getLearningRate(state.ctx.batchIndex,
                                                      state.partitionId,
                                                      state.iteration,
                                                      state.ctx.log,
                                                      state.ctx.trainingParams,
                                                      state.learningRate)
      case None => state.learningRate
    }
  }

  def createReferenceDataset(numRows: Int,
                             numCols: Int,
                             datasetParams: String,
                             sampleData: SampledData,
                             log: Logger): Array[Byte] = {
    log.info(s"LightGBM task generating schema for empty dense dataset with $numRows rows and $numCols columns")
    // Generate the dataset for features
    val dataset = lightgbmlib.voidpp_handle()
    LightGBMUtils.validate(lightgbmlib.LGBM_DatasetCreateFromSampledColumn(
      sampleData.getSampleData(),
      sampleData.getSampleIndices(),
      numCols,
      sampleData.getRowCounts(),
      sampleData.numRows,
      numRows, // TODO does this allocate?
      datasetParams,
      dataset), "Dataset create")

    val buffer = lightgbmlib.voidpp_handle()
    val datasetHandle = lightgbmlib.voidpp_value(dataset)
    val lenPtr = lightgbmlib.new_intp()
    LightGBMUtils.validate(lightgbmlib.LGBM_DatasetSerializeReferenceToBinary(
      datasetHandle,
      buffer,
      lenPtr), "Serialize ref")
    val bufferLen: Int = lightgbmlib.intp_value(lenPtr)
    lightgbmlib.delete_intp(lenPtr)

    val serializedReference = new Array[Byte](bufferLen)
    val valPtr = lightgbmlib.new_bytep()
    val bufferHandle = lightgbmlib.voidpp_value(buffer)
    (0 until bufferLen).foreach(i => {
      LightGBMUtils.validate(lightgbmlib.LGBM_ByteBufferGetAt(bufferHandle, i, valPtr),"Buffer getat")
      serializedReference(i) = lightgbmlib.bytep_value(valPtr).toByte
    })
    lightgbmlib.delete_bytep(valPtr)

    LightGBMUtils.validate(lightgbmlib.LGBM_ByteBufferFree(bufferHandle),"Buffer free")
    LightGBMUtils.validate(lightgbmlib.LGBM_DatasetFree(datasetHandle),"Dataset free")
    serializedReference
  }

  def updateOneIteration(state: PartitionTaskTrainingState): Unit = {
    try {
        val fobj = state.ctx.trainingParams.objectiveParams.fobj
        if (fobj.isDefined) {
          val isClassification = state.ctx.isClassification
          val (gradient, hessian) = fobj.get.getGradient(
            state.booster.innerPredict(0, isClassification), state.booster.trainDataset.get)
          state.isFinished = state.booster.updateOneIterationCustom(gradient, hessian)
        } else {
          state.isFinished = state.booster.updateOneIteration()
        }
      state.log.info("LightGBM running iteration: " + state.iteration + " with is finished: " + state.isFinished)
    } catch {
      case e: java.lang.Exception =>
        state.log.warn("LightGBM reached early termination on one task," +
          " stopping training on task. This message should rarely occur." +
          " Inner exception: " + e.toString)
        state.isFinished = true
    }
  }

  def executeTrainingIterations(state: PartitionTaskTrainingState): Option[Int] = {
    def iterationLoop(maxIterations: Int): Option[Int] = {
      beforeTrainIteration(state)
      val newLearningRate = getLearningRate(state)
      if (newLearningRate != state.learningRate) {
        state.log.info(s"LightGBM task calling booster.resetParameter to reset learningRate" +
          s" (newLearningRate: $newLearningRate)")
        state.booster.resetParameter(s"learning_rate=$newLearningRate")
        state.learningRate = newLearningRate
      }

      updateOneIteration(state)

      val trainEvalResults: Option[Map[String, Double]] =
        if (state.ctx.isProvideTrainingMetric && !state.isFinished) getTrainEvalResults(state)
        else None

      val validEvalResults: Option[Map[String, Double]] =
        if (state.ctx.hasValid && !state.isFinished) getValidEvalResults(state)
        else None

      afterTrainIteration(state, trainEvalResults, validEvalResults)

      state.iteration = state.iteration + 1
      if (!state.isFinished && state.iteration < maxIterations) {
        iterationLoop(maxIterations)  // tail recursion
      } else {
        state.bestIterResult
      }
    }

    iterationLoop(state.ctx.trainingParams.generalParams.numIterations)
  }

  def getTrainEvalResults(state: PartitionTaskTrainingState): Option[Map[String, Double]] = {
    val evalResults: Array[(String, Double)] = state.booster.getEvalResults(state.evalNames, 0)
    evalResults.foreach { case (evalName: String, score: Double) => state.log.info(s"Train $evalName=$score") }
    Option(Map(evalResults: _*))
  }

  def getValidEvalResults(state: PartitionTaskTrainingState): Option[Map[String, Double]] = {
    val evalResults: Array[(String, Double)] = state.booster.getEvalResults(state.evalNames, 1)
    val results: Array[(String, Double)] = evalResults.zipWithIndex.map { case ((evalName, evalScore), index) =>
      state.log.info(s"Valid $evalName=$evalScore")
      val cmp =
        if (evalName.startsWith("auc")
            || evalName.startsWith("ndcg@")
            || evalName.startsWith("map@")
            || evalName.startsWith("average_precision"))
          (x: Double, y: Double, tol: Double) => x - y > tol
        else
          (x: Double, y: Double, tol: Double) => x - y < tol
      if (state.bestScores(index) == null
          || cmp(evalScore, state.bestScore(index), state.ctx.improvementTolerance)) {
        state.bestScore(index) = evalScore
        state.bestIter(index) = state.iteration
        state.bestScores(index) = evalResults.map(_._2)
      } else if (state.iteration - state.bestIter(index) >= state.ctx.earlyStoppingRound) {
        state.isFinished = true
        state.log.info("Early stopping, best iteration is " + state.bestIter(index))
        state.bestIterResult = Some(state.bestIter(index))
      }

      (evalName, evalScore)
    }
    Option(Map(results: _*))
  }

  def beforeGenerateTrainDataset(ctx: TrainingContext): Unit = {
    if (ctx.trainingParams.delegate.isDefined) {
      ctx.trainingParams.delegate.get.beforeGenerateTrainDataset(
        ctx.batchIndex,
        TaskContext.getPartitionId,
        ctx.columnParams,
        ctx.schema,
        ctx.log,
        ctx.trainingParams)
    }
  }

  def afterGenerateTrainDataset(ctx: TrainingContext): Unit = {
    if (ctx.trainingParams.delegate.isDefined) {
      ctx.trainingParams.delegate.get.afterGenerateTrainDataset(
        ctx.batchIndex,
        TaskContext.getPartitionId,
        ctx.columnParams,
        ctx.schema,
        ctx.log,
        ctx.trainingParams)
    }
  }

  def beforeGenerateValidDataset(ctx: TrainingContext): Unit = {
    if (ctx.trainingParams.delegate.isDefined) {
      ctx.trainingParams.delegate.get.beforeGenerateValidDataset(
        ctx.batchIndex,
        TaskContext.getPartitionId,
        ctx.columnParams,
        ctx.schema,
        ctx.log,
        ctx.trainingParams)
    }
  }

  def afterGenerateValidDataset(ctx: TrainingContext): Unit = {
    if (ctx.trainingParams.delegate.isDefined) {
      ctx.trainingParams.delegate.get.afterGenerateValidDataset(
        ctx.batchIndex,
        TaskContext.getPartitionId,
        ctx.columnParams,
        ctx.schema,
        ctx.log,
        ctx.trainingParams)
    }
  }

  private def findOpenPort(ctx: TrainingContext): Option[Socket] = {
    val defaultListenPort: Int = ctx.networkParams.defaultListenPort
    val log = ctx.log
    val basePort = defaultListenPort + (LightGBMUtils.getWorkerId * ctx.numTasksPerExecutor)
    if (basePort > LightGBMConstants.MaxPort) {
      throw new Exception(s"Error: port $basePort out of range, possibly due to too many executors or unknown error")
    }
    var localListenPort = basePort
    var taskServerSocket: Option[Socket] = None
    def findPort(): Unit = {
      try {
        taskServerSocket = Option(new Socket())
        taskServerSocket.get.bind(new InetSocketAddress(localListenPort))
      } catch {
        case _: IOException =>
          log.warn(s"Could not bind to port $localListenPort...")
          localListenPort += 1
          if (localListenPort > LightGBMConstants.MaxPort) {
            throw new Exception(s"Error: port $basePort out of range, possibly due to networking or firewall issues")
          }
          if (localListenPort - basePort > 1000) {
            throw new Exception("Error: Could not find open port after 1k tries")
          }
        findPort()
      }
    }
    findPort()
    log.info(s"Successfully bound to port $localListenPort")
    taskServerSocket
  }

  def setFinishedStatus(networkParams: NetworkParams,
                        localListenPort: Int, log: Logger): Unit = {
    using(new Socket(networkParams.addr, networkParams.port)) {
      driverSocket =>
        using(new BufferedWriter(new OutputStreamWriter(driverSocket.getOutputStream))) {
          driverOutput =>
            log.info("sending finished status to driver")
            // If barrier execution mode enabled, create a barrier across tasks
            driverOutput.write(s"${LightGBMConstants.FinishedStatus}\n")
            driverOutput.flush()
        }.get
    }.get
  }

  def getNetworkInitNodes(networkParams: NetworkParams,
                          localListenPort: Int,
                          log: Logger,
                          ignoreTask: Boolean): String = {
    using(new Socket(networkParams.addr, networkParams.port)) {
      driverSocket =>
        usingMany(Seq(new BufferedReader(new InputStreamReader(driverSocket.getInputStream)),
          new BufferedWriter(new OutputStreamWriter(driverSocket.getOutputStream)))) {
          io =>
            val driverInput = io(0).asInstanceOf[BufferedReader]
            val driverOutput = io(1).asInstanceOf[BufferedWriter]
            val partitionId = LightGBMUtils.getPartitionId
            val taskHost = driverSocket.getLocalAddress.getHostAddress
            val taskStatus =
              if (ignoreTask) {
                log.info(s"send empty status to driver with partitionId: $partitionId")
                s"${LightGBMConstants.IgnoreStatus}:$taskHost:$partitionId"
              } else {
                val taskInfo = s"$taskHost:$localListenPort:$partitionId"
                log.info(s"send current task info to driver: $taskInfo ")
                taskInfo
              }
            // Send the current host:port:partitionId to the driver
            driverOutput.write(s"$taskStatus\n")
            driverOutput.flush()
            // If barrier execution mode enabled, create a barrier across tasks
            if (networkParams.barrierExecutionMode) {
              val context = BarrierTaskContext.get()
              context.barrier()
              if (context.partitionId() == 0) {
                setFinishedStatus(networkParams, localListenPort, log)
              }
            }
            if (!taskStatus.startsWith(LightGBMConstants.IgnoreStatus)) {
              // Wait to get the list of nodes from the driver
              val nodes = driverInput.readLine()
              log.info(s"LightGBM worker got nodes for network init: $nodes")
              nodes
            } else {
              taskStatus
            }
        }.get
    }.get
  }

  def networkInit(nodes: String, localListenPort: Int, log: Logger, retry: Int, delay: Long): Unit = {
    try {
      LightGBMUtils.validate(lightgbmlib.LGBM_NetworkInit(nodes, localListenPort,
        LightGBMConstants.DefaultListenTimeout, nodes.split(",").length), "Network init")
    } catch {
      case ex@(_: Exception | _: Throwable) =>
        log.info(s"NetworkInit failed with exception on local port $localListenPort with exception: $ex")
        Thread.sleep(delay)
        if (retry > 0) {
          log.info(s"Retrying NetworkInit with local port $localListenPort")
          networkInit(nodes, localListenPort, log, retry - 1, delay * 2)
        } else {
          log.info(s"NetworkInit reached maximum exceptions on retry: $ex")
          throw ex
        }
    }
  }

  /**
    * Gets the main node's port that will return the LightGBM Booster.
    * Used to minimize network communication overhead in reduce step.
    * @return The main node's port number.
    */
  def getMainWorkerPort(nodes: String, log: Logger): Int = {
    val nodesList = nodes.split(",")
    if (nodesList.length == 0) {
      throw new Exception("Error: could not split nodes list correctly")
    }
    val mainNode = nodesList(0)
    val hostAndPort = mainNode.split(":")
    if (hostAndPort.length != 2) {
      throw new Exception("Error: could not parse main worker host and port correctly")
    }
    val mainHost = hostAndPort(0)
    val mainPort = hostAndPort(1)
    log.info(s"LightGBM setting main worker host: $mainHost and port: $mainPort")
    mainPort.toInt
  }

  /** Retrieve the network nodes and current port information.
    *
    * Establish local socket connection.
    *
    * Note: Ideally we would start the socket connections in the C layer, this opens us up for
    * race conditions in case other applications open sockets on cluster, but usually this
    * should not be a problem
    *
    * @param ctx Information about the current training session.
    * @param isEnabledWorker True if the current worker is enabled, including whether the partition
    *                        was enabled and this is the chosen worker to initialize the network connection.
    * @return A tuple containing the string with all nodes and the current worker's open socket connection.
    */
  def getNetworkInfo(ctx: TrainingContext, isEnabledWorker: Boolean): (String, Int) = {
    val networkParams = ctx.networkParams
    using(findOpenPort(ctx).get) {
      openPort =>
        val localListenPort = openPort.getLocalPort
        ctx.log.info(s"LightGBM task connecting to host: ${networkParams.addr} and port: ${networkParams.port}")
        FaultToleranceUtils.retryWithTimeout() {
          (getNetworkInitNodes(networkParams, localListenPort, ctx.log, !isEnabledWorker), localListenPort)
        }
    }.get
  }

  /** Return true if the current thread will return the booster after training is complete.
    *
    * @param isEnabledWorker True if the current worker is enabled, including whether the partition
    *                        was enabled and this is the chosen worker to initialize the network connection.
    * @param nodes The string representation of all nodes communicating in the network.
    * @param log The logger.
    * @param numTasksPerExec The number of tasks per executor.
    * @param localListenPort The local port used to setup the network ring of communication.
    * @return Boolean representing whether the current task will return the booster or not.
    */
  def getShouldReturnBooster(ctx: TrainingContext,
                             isEnabledWorker: Boolean,
                             nodes: String,
                             localListenPort: Int): Boolean = {
    if (!isEnabledWorker) {
      false
    } else {
      val mainWorkerPort = getMainWorkerPort(nodes, ctx.log)
      mainWorkerPort == localListenPort
    }
  }
}
