package ai.clipper.spark

import java.net.URLClassLoader
import java.nio.file.StandardCopyOption.REPLACE_EXISTING
import java.nio.file.StandardOpenOption.CREATE
import java.nio.file.{Files, Paths}
import java.io.File
import java.nio.charset.Charset

import org.apache.spark.SparkContext
import org.apache.spark.ml.PipelineModel
import org.apache.sysml.api.jmlc.Connection
import org.json4s._
import org.json4s.jackson.Serialization.{read, write}
import org.json4s.jackson.Serialization
import scalaj.http._

import org.apache.sysml.parser.DataExpression
import org.apache.sysml.runtime.io.MatrixReaderFactory
import org.apache.sysml.runtime.matrix.data.InputInfo
import org.apache.sysml.runtime.matrix.data.MatrixBlock
import java.io.IOException

import scala.sys.process._

sealed trait ModelType extends Serializable

case object MLlibModelType extends ModelType

case object PipelineModelType extends ModelType

object ModelTypeSerializer
    extends CustomSerializer[ModelType](
      _ =>
        (
          {
            case JString("MLlibModelType") => MLlibModelType
            case JString("PipelineModelType") => PipelineModelType
          }, {
            case MLlibModelType => JString("MLlibModelType")
            case PipelineModelType => JString("PipelineModelType")
          }
      ))

class UnsupportedEnvironmentException(e: String) extends RuntimeException(e)

class ModelDeploymentError(e: String) extends RuntimeException(e)

case class ClipperContainerConf(var className: String,
                                var jarName: String,
                                var modelType: ModelType,
                                var fromRepl: Boolean = false,
                                var replClassDir: Option[String] = None)

case class SysmlModelMeta(val dml: String,
                          val inVarName: String,
                          val outVarName: String,
                          val ncol: Int,
                          val weights: Map[String, (Array[Double], Int, Int)])

object Clipper {

  val CLIPPER_CONF_FILENAME: String = "clipper_conf.json"
  val CONTAINER_JAR_FILE: String = "container_source.jar"
  val MODEL_DIRECTORY: String = "model"
  val REPL_CLASS_DIR: String = "repl_classes"
  val CLIPPER_SPARK_CONTAINER_NAME = "test-spark-container"

  val DOCKER_NW: String = "clipper_network"
  val CLIPPER_MANAGEMENT_PORT: Int = 1338
  val CLIPPER_DOCKER_LABEL: String = "ai.clipper.container.label"


  // Imports the json serialization library as an implicit and adds our custom serializer
  // for the ModelType case classes
  implicit val json4sFormats = Serialization.formats(NoTypeHints) + ModelTypeSerializer

  def deploySysmlModel(name: String,
                       version: Int,
                       clipperHost: String,
                       weightsDir: String,
                       inVarName: String,
                       outVarName: String,
                       dml: String,
                       ncol: Int,
                       loggingMountPoint: String,
                       logPath: String,
                       gpuIndex: Int,
                       labels: List[String],
                       batchSize: Int) : Unit = {

    println("Walrus")
    val basePath = Paths.get("/tmp", name, version.toString).toString
    // create the base path if it does not already exist
    val fileObj = new File(basePath)
    if (!fileObj.exists()) {
      if (!fileObj.mkdirs())
        throw new Exception("Could not makedir")
      println("Created directory")
    }

    // Use the same directory scheme of /tmp/<name>/<version> on host
    val hostDataPath = basePath
    val localHostNames = Set("local", "localhost", "127.0.0.1")
    val islocalHost = localHostNames.contains(clipperHost.toLowerCase())

    // Construct a map representing the model and serialize to disk
    val model = Map("dml" -> dml,
      "inVarName" -> inVarName,
      "outVarName" -> outVarName,
      "dml" -> dml,
      "ncol" -> ncol,
      "weights" -> Map("UNUSED" -> (Array[Double](-1), -1, -1)))
    val modelJson = write(model)
    Files.write(Paths.get(basePath + "/model_data.json"), modelJson.getBytes, CREATE)

    // publish the model to the clipper service
    publishModelToClipper(clipperHost, name, version, labels, basePath, batchSize)

    // figure out the query frontend name. This is a bit hacky and there should be a better way to do this
    val res = "docker network inspect clipper_network" !!
    val pattern = "query_frontend-[0-9]+".r
    val query_frontend_name = pattern.findFirstIn(res).get.toString

    // start the local docker container
    println("CALLING START CONTAINER LOCAL...")
    startSysmlContainer(name, version, basePath, loggingMountPoint, logPath, gpuIndex, query_frontend_name, weightsDir)
  }

  def deploySysmlModel(name: String,
                       version: Int,
                       clipperHost: String,
                       weights: Map[String, (Array[Double], Int, Int)],
                       inVarName: String,
                       outVarName: String,
                       dml: String,
                       ncol: Int,
                       loggingMountPoint: String,
                       logPath: String,
                       gpuIndex: Int,
                       labels: List[String],
                       batchSize: Int): Unit = {

    println("Walrus")
    val basePath = Paths.get("/tmp", name, version.toString).toString
    // create the base path if it does not already exist
    val fileObj = new File(basePath)
    if (!fileObj.exists()) {
      if (!fileObj.mkdirs())
        throw new Exception("Could not makedir")
      println("Created directory")
    }

    // Use the same directory scheme of /tmp/<name>/<version> on host
    val hostDataPath = basePath
    val localHostNames = Set("local", "localhost", "127.0.0.1")
    val islocalHost = localHostNames.contains(clipperHost.toLowerCase())

    // Construct a map representing the model and serialize to disk
    val model = Map("dml" -> dml,
                    "inVarName" -> inVarName,
                    "outVarName" -> outVarName,
                    "dml" -> dml,
                    "ncol" -> ncol,
                    "weights" -> weights)
    val modelJson = write(model)
    Files.write(Paths.get(basePath + "/model_data.json"), modelJson.getBytes, CREATE)

    // publish the model to the clipper service
    publishModelToClipper(clipperHost, name, version, labels, basePath, batchSize)

    // figure out the query frontend name. This is a bit hacky and there should be a better way to do this
    val res = "docker network inspect clipper_network" !!
    val pattern = "query_frontend-[0-9]+".r
    val query_frontend_name = pattern.findFirstIn(res).get.toString

    // start the local docker container
    println("CALLING START CONTAINER LOCAL...")
    startSysmlContainer(name, version, basePath, loggingMountPoint, logPath, gpuIndex, query_frontend_name)
  }

  def loadSysmlModel(conn: Connection, basePath: String, logPath: String, gpuIndex: Int) : SysmlModelContainer = {
    val useGPU = gpuIndex > -1
    // read the serialized model file from the disk
    val modelJsonString = Files.readAllLines(Paths.get(basePath + "/model_data.json")).get(0)
    val sysmlModel = read[SysmlModelMeta](modelJsonString)

    val inputs = sysmlModel.weights.keys.toArray ++ Array[String](sysmlModel.inVarName)
    val ps = conn.prepareScript(sysmlModel.dml, inputs, Array[String](sysmlModel.outVarName), useGPU, useGPU, 0)

    for ((name, value) <- sysmlModel.weights) {
      val mb = new MatrixBlock(value._2, value._3, -1).allocateDenseBlock()
      mb.init(value._1, value._2, value._3)
      ps.setMatrix(name, mb, true)
    }

    new SysmlModelContainer(ps, sysmlModel.inVarName, sysmlModel.outVarName, sysmlModel.ncol, logPath)
  }

  def loadSysmlModel(conn: Connection,
                     basePath: String,
                     logPath: String,
                     weightsDir: String,
                     gpuIndex: Int) : SysmlModelContainer = {

    val useGPU = gpuIndex > -1
    // read the serialized model file from the disk
    if (weightsDir == "UNUSED")
      return loadSysmlModel(conn, basePath, logPath, gpuIndex)

    val weightsPath = "/external/" + weightsDir
    val modelJsonString = Files.readAllLines(Paths.get(basePath + "/model_data.json")).get(0)
    val sysmlModel = read[SysmlModelMeta](modelJsonString)

    System.err.println("WEIGHTS PATH: " + weightsPath)
    new File(weightsPath).listFiles().foreach { println(_) }
    val weightFiles = new File(weightsPath).listFiles().map(_.toString).filter(x => x.split("\\.").last == "mtx")
    val weights = weightFiles.map(x => x.split("/").last.split("\\.")(0) -> readMatrix(x)).toMap
    val inputs = weights.keys.toArray ++ Array[String](sysmlModel.inVarName)
    val ps = conn.prepareScript(sysmlModel.dml, inputs, Array[String](sysmlModel.outVarName), useGPU, useGPU, 0)

    for ((name,value) <- weights) {
      ps.setMatrix(name, value, true)
    }

    new SysmlModelContainer(ps, sysmlModel.inVarName, sysmlModel.outVarName, sysmlModel.ncol, logPath)
  }

  @throws[IOException]
  def readMatrix(fname: String): MatrixBlock = try {
    val fnamemtd = DataExpression.getMTDFileName(fname)
    val jmtd = new DataExpression().readMetadataFile(fnamemtd, false)
    //parse json meta data
    val rows = jmtd.getLong(DataExpression.READROWPARAM)
    val cols = jmtd.getLong(DataExpression.READCOLPARAM)
    val brlen = if (jmtd.containsKey(DataExpression.ROWBLOCKCOUNTPARAM))
      jmtd.getInt(DataExpression.ROWBLOCKCOUNTPARAM) else -1
    val bclen = if (jmtd.containsKey(DataExpression.COLUMNBLOCKCOUNTPARAM))
      jmtd.getInt(DataExpression.COLUMNBLOCKCOUNTPARAM) else -1
    val nnz = if (jmtd.containsKey(DataExpression.READNNZPARAM))
      jmtd.getLong(DataExpression.READNNZPARAM) else -1
    val format = jmtd.getString(DataExpression.FORMAT_TYPE)
    val iinfo = InputInfo.stringExternalToInputInfo(format)
    readMatrix(fname, iinfo, rows, cols, brlen, bclen, nnz)
  } catch {
    case ex: Exception =>
      throw new IOException(ex)
  }

  @throws[IOException]
  def readMatrix(fname: String, iinfo: InputInfo, rows: Long, cols: Long, brlen: Int, bclen: Int, nnz: Long): MatrixBlock = try {
    val reader = MatrixReaderFactory.createMatrixReader(iinfo)
    reader.readMatrixFromHDFS(fname, rows, cols, brlen, bclen, nnz)
  } catch {
    case ex: Exception =>
      throw new IOException(ex)
  }

  /**
    *
    * @param sc Spark context
    * @param name The name to assign the model when deploying to Clipper
    * @param version The model version
    * @param model The trained Spark model. Note that this _must_ be an instance of either
    *              ai.clipper.spark.MLlibModel or org.apache.spark.ml.PipelineModel
    * @param containerClass The model container which specifies how to use the trained
    *                       model to make predictions. This can include any pre-processing
    *                       or post-processing code (including any featurization). This class
    *                       must either extend ai.clipper.spark.MLlibContainer or
    *                       ai.clipper.spark.PipelineModelContainer.
    * @param clipperHost The IP address or hostname of a running Clipper instance. This can be either localhost
    *                    or a remote machine that you have SSH access to. SSH access is required to copy the model and
    *                    launch a Docker container on the remote machine.
    * @param labels A list of labels to be associated with the model.
    * @param sshUserName If deploying to a remote machine, the username associated with the SSH credentials.
    * @param sshKeyPath If deploying to a remote machine, the path to an SSH key authorized to log in to the remote
    *                   machine.
    * @param dockerRequiresSudo True if the Docker daemon on the machine hosting Clipper requires sudo to access. If
    *                           set to true, the ssh user you specify must have passwordless sudo access.
    * @tparam M The type of the model. This _must_ be an instance of either
    *              ai.clipper.spark.MLlibModel or org.apache.spark.ml.PipelineModel
    */
  def deploySparkModel[M](sc: SparkContext,
                          name: String,
                          version: Int,
                          model: M,
                          containerClass: Class[_],
                          clipperHost: String,
                          labels: List[String],
                          sshUserName: Option[String] = None,
                          sshKeyPath: Option[String] = None,
                          dockerRequiresSudo: Boolean = true): Unit = {
    val basePath = Paths.get("/tmp", name, version.toString).toString
    // Use the same directory scheme of /tmp/<name>/<version> on host
    val hostDataPath = basePath
    val localHostNames = Set("local", "localhost", "127.0.0.1")
    val islocalHost = localHostNames.contains(clipperHost.toLowerCase())

    try {
      saveSparkModel[M](sc, name, version, model, containerClass, basePath)
      if (!islocalHost) {
        // Make sure that ssh credentials were supplied
        val (user, key) = try {
          val user = sshUserName.get
          val key = sshKeyPath.get
          (user, key)
        } catch {
          case _: NoSuchElementException => {
            val err =
              "SSH user name and keypath must be supplied to deploy model to remote Clipper instance"
            println(err)
            throw new ModelDeploymentError(err)
          }
        }
        copyModelDataToHost(clipperHost, basePath, hostDataPath, user, key)
        publishModelToClipper(clipperHost, name, version, labels, hostDataPath)
        startSparkContainerRemote(name,
                                  version,
                                  clipperHost,
                                  hostDataPath,
                                  user,
                                  key,
                                  dockerRequiresSudo)
      } else {
        publishModelToClipper(clipperHost, name, version, labels, hostDataPath)
        startSparkContainerLocal(name, version, basePath)
      }
    } catch {
      case e: Throwable => {
        println(s"Error saving model: ${e.printStackTrace}")
        return
      }
    }
  }

  private def startSysmlContainer(name: String,
                                  version: Int,
                                  modelDataPath: String,
                                  externalMountPoint: String,
                                  logPath: String,
                                  gpuIndex: Int,
                                  clipper_id: String = "query_frontend",
                                  weightsDir: String = "UNUSED"): Unit = {
    println(s"MODEL_DATA_PATH: $modelDataPath")
    println("CLIPPER ID: " + clipper_id)
    println("USING GPU: " + gpuIndex)
    val useGpu = gpuIndex > -1
    val dockerCmd = if (useGpu) "nvidia-docker" else "docker"
    val nvidiaMountPoint = if (useGpu) Seq("-v", "/usr/local/cuda/lib64:/usr/local/cuda/lib64") else Seq()
    val startContainerCmd = Seq(
      dockerCmd,
      "run",
      "-d",
      s"--network=$DOCKER_NW",
      "-v", s"$externalMountPoint:/external",
      "-v", s"$modelDataPath:/model:ro") ++ nvidiaMountPoint ++
    Seq(
      "-e", s"WEIGHTS_DIR=$weightsDir",
      "-e", s"LOG_PATH=$logPath",
      "-e", s"GPU_INDEX=${gpuIndex.toString}",
      "-e", s"CLIPPER_MODEL_NAME=$name",
      "-e", s"CLIPPER_MODEL_VERSION=$version",
      "-e", s"CLIPPER_IP=$clipper_id",
      "-e", "CLIPPER_INPUT_TYPE=doubles",
      "--name", s"${name}_container",
      "-l", s"$CLIPPER_DOCKER_LABEL",
      CLIPPER_SPARK_CONTAINER_NAME
    )


    println(startContainerCmd.mkString(" "))
    if (startContainerCmd.! != 0) {
      throw new ModelDeploymentError("Error starting model container")
    }
  }

  private def startSparkContainerLocal(name: String,
                                       version: Int,
                                       modelDataPath: String,
                                       clipper_id: String = "query_frontend"): Unit = {
    println(s"MODEL_DATA_PATH: $modelDataPath")
    println("CLIPPER ID: " + clipper_id)

    val startContainerCmd = Seq(
      "docker",
      "run",
      "-d",
      s"--network=$DOCKER_NW",
      "-v", s"$modelDataPath:/model:ro",
      "-e", s"CLIPPER_MODEL_NAME=$name",
      "-e", s"CLIPPER_MODEL_VERSION=$version",
      "-e", s"CLIPPER_IP=$clipper_id",
      "-e", "CLIPPER_INPUT_TYPE=doubles",
      "-l", s"$CLIPPER_DOCKER_LABEL",
      CLIPPER_SPARK_CONTAINER_NAME
    )

    println(startContainerCmd.mkString(" "))
    if (startContainerCmd.! != 0) {
      throw new ModelDeploymentError("Error starting model container")
    }
  }

  private def startSparkContainerRemote(name: String,
                                        version: Int,
                                        clipperHost: String,
                                        modelDataPath: String,
                                        sshUserName: String,
                                        sshKeyPath: String,
                                        dockerRequiresSudo: Boolean): Unit = {

    val sudoCommand = if (dockerRequiresSudo) Seq("sudo") else Seq()
    val getDockerIPCommand = sudoCommand ++ Seq(
      "docker",
      "ps", "-aqf",
      "ancestor=clipper/query_frontend",
      "|", "xargs") ++ sudoCommand ++ Seq("docker",
      "inspect",
      "--format='{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'"
    )



    val sshCommand = Seq("ssh",
      "-o", "StrictHostKeyChecking=no",
      "-i", s"$sshKeyPath",
      s"$sshUserName@$clipperHost")

    val dockerIpCommand = sshCommand ++ getDockerIPCommand
    val dockerIp = dockerIpCommand.!!.stripLineEnd
    println(s"Docker IP: $dockerIp")

    val startContainerCmd = sudoCommand ++ Seq(
      "docker",
      "run",
      "-d",
      s"--network=$DOCKER_NW",
      "-v", s"$modelDataPath:/model:ro",
      "-e", s"CLIPPER_MODEL_NAME=$name",
      "-e", s"CLIPPER_MODEL_VERSION=$version",
      "-e", s"CLIPPER_IP=$dockerIp",
      "-e", "CLIPPER_INPUT_TYPE=doubles",
      "-l", s"$CLIPPER_DOCKER_LABEL",
      CLIPPER_SPARK_CONTAINER_NAME
    )
    val sshStartContainerCmd = sshCommand ++ startContainerCmd
    println(sshStartContainerCmd)
    if (sshStartContainerCmd.! != 0) {
      throw new ModelDeploymentError("Error starting model container")
    }
  }

  private def publishModelToClipper(host: String,
                                    name: String,
                                    version: Int,
                                    labels: List[String],
                                    hostModelDataPath: String,
                                    batchSize: Int = -1): Unit = {
    println("Host: " + host)
    println("Name: " + name)
    println("Version: " + version)
    println("Labels: " + labels)
    println("HostModelDataPath: " + hostModelDataPath)
    println("BATCH SIZE: 1")
    val data = Map(
      "model_name" -> name,
      "model_version" -> version.toString,
      "labels" -> labels,
      "input_type" -> "doubles",
      "batch_size" -> batchSize,
      "container_name" -> CLIPPER_SPARK_CONTAINER_NAME,
      "model_data_path" -> hostModelDataPath
    )
    val jsonData = write(data)

    val response = Http(s"http://$host:$CLIPPER_MANAGEMENT_PORT/admin/add_model")
      .header("Content-type", "application/json")
      .postData(jsonData)
      .asString

    if (response.code == 200) {
      println("Successfully published model to Clipper")
    } else {
      throw new ModelDeploymentError(
        s"Error publishing model to Clipper. ${response.code}: ${response.body}")
    }
  }

  private def copyModelDataToHost(host: String,
                                  localPath: String,
                                  destPath: String,
                                  sshUserName: String,
                                  sshKeyPath: String): Unit = {

    val mkdirCommand = Seq("ssh",
                           "-o", "StrictHostKeyChecking=no",
                           "-i", s"$sshKeyPath",
                           s"$sshUserName@$host",
                           "mkdir", "-p", destPath)
    val copyCommand = Seq("rsync",
                          "-r",
                          "-e", s"ssh -i $sshKeyPath -o StrictHostKeyChecking=no",
                          s"$localPath/",
                          s"$sshUserName@$host:$destPath")
    if (!(mkdirCommand.! == 0 && copyCommand.! == 0)) {
        throw new ModelDeploymentError(
        "Error copying model data to Clipper host")
    }
  }

  private[clipper] def saveSparkModel[M](sc: SparkContext,
                                         name: String,
                                         version: Int,
                                         model: M,
                                         containerClass: Class[_],
                                         basePath: String): Unit = {
    // Check that Spark is not running in the REPL
    if (getReplOutputDir(sc).isDefined) {
      throw new UnsupportedEnvironmentException(
        "Clipper cannot deploy models from Spark Shell")
    }
    val modelPath = Paths.get(basePath, MODEL_DIRECTORY).toString
    val modelType = model match {
      case m: MLlibModel => {
        m.save(sc, modelPath)
        // Because I'm not sure how to do it in the type system, check that
        // the container is of the right type
        // NOTE: this test doesn't work from the REPL
        try {
          containerClass.newInstance.asInstanceOf[MLlibContainer]
        } catch {
          case e: ClassCastException => {
            throw new IllegalArgumentException(
              "Error: Container must be a subclass of MLlibContainer")
          }
        }
        MLlibModelType
      }
      case p: PipelineModel => {
        p.save(modelPath)
        // Because I'm not sure how to do it in the type system, check that
        // the container is of the right type
        try {
          containerClass.newInstance.asInstanceOf[PipelineModelContainer]
        } catch {
          case e: ClassCastException => {
            throw new IllegalArgumentException(
              "Error: Container must be a subclass of PipelineModelContainer")
          }
        }
        PipelineModelType
      }
      case _ =>
        throw new IllegalArgumentException(
          s"Illegal model type: ${model.getClass.getName}")
    }
    val jarPath = Paths.get(
      containerClass.getProtectionDomain.getCodeSource.getLocation.getPath)
    val copiedJarName = CONTAINER_JAR_FILE
    println(s"JAR path: $jarPath")
    System.out.flush()
    Files.copy(jarPath, Paths.get(basePath, copiedJarName), REPLACE_EXISTING)
    val conf =
      ClipperContainerConf(containerClass.getName, copiedJarName, modelType)
    getReplOutputDir(sc) match {
      case Some(classSourceDir) => {
        throw new UnsupportedOperationException("Clipper does not support deploying models directly from the Spark REPL")
        // NOTE: This commented out code is intentionally committed. We hope to support
        // model deployment in the future.
//        println(
//          "deployModel called from Spark REPL. Saving classes defined in REPL.")
//        conf.fromRepl = true
//        conf.replClassDir = Some(REPL_CLASS_DIR)
//        val classDestDir = Paths.get(basePath, REPL_CLASS_DIR)
//        FileUtils.copyDirectory(Paths.get(classSourceDir).toFile,
//                                classDestDir.toFile)
      }
      case None =>
        println(
          "deployModel called from script. No need to save additionally generated classes.")
    }
    Files.write(Paths.get(basePath, CLIPPER_CONF_FILENAME),
                write(conf).getBytes,
                CREATE)
  }

  private def getReplOutputDir(sc: SparkContext): Option[String] = {
    sc.getConf.getOption("spark.repl.class.outputDir")
  }

  private[clipper] def loadSparkModel(
      sc: SparkContext,
      basePath: String): SparkModelContainer = {
    val confString = Files
      .readAllLines(Paths.get(basePath, CLIPPER_CONF_FILENAME),
                    Charset.defaultCharset())
      .get(0)

    val conf = read[ClipperContainerConf](confString)
    val classLoader = getClassLoader(basePath, conf)
    val modelPath = Paths.get(basePath, MODEL_DIRECTORY).toString
    println(s"Model path: $modelPath")

    conf.modelType match {
      case MLlibModelType => {
        val model = MLlibLoader.load(sc, modelPath)
        try {
          val container = classLoader
            .loadClass(conf.className)
            .newInstance()
            .asInstanceOf[MLlibContainer]
          container.init(sc, model)
          container.asInstanceOf[SparkModelContainer]
        } catch {
          case e: Throwable => {
            e.printStackTrace
            throw e
          }

        }
      }
      case PipelineModelType => {
        val model = PipelineModel.load(modelPath)
        try {
        val container = classLoader
          .loadClass(conf.className)
          .newInstance()
          .asInstanceOf[PipelineModelContainer]
        container.init(sc, model)
        container.asInstanceOf[SparkModelContainer]
        } catch {
          case e: Throwable => {
            e.printStackTrace
            throw e
          }
        }
      }
    }
  }

  private def getClassLoader(path: String,
                             conf: ClipperContainerConf): ClassLoader = {
    new URLClassLoader(Array(Paths.get(path, conf.jarName).toUri.toURL),
                       getClass.getClassLoader)
  }
}
