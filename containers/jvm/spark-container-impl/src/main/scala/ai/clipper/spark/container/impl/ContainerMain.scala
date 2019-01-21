package ai.clipper.spark.container.impl

import java.net.UnknownHostException

import ai.clipper.container.data.DoubleVector
import ai.clipper.rpc.RPC
import ai.clipper.spark.{Clipper, SysmlModelContainer}
import org.apache.sysml.api.jmlc.Connection

object ContainerMain {

  def main(args: Array[String]): Unit = {

    val modelPath = sys.env("CLIPPER_MODEL_PATH")
    val modelName = sys.env("CLIPPER_MODEL_NAME")
    val gpuIndex = sys.env.getOrElse("GPU_INDEX", "-1").toInt

    val modelVersion = sys.env("CLIPPER_MODEL_VERSION").toInt
    val weightsDir = sys.env.getOrElse("WEIGHTS_DIR", "UNUSED")

    val clipperAddress = sys.env.getOrElse("CLIPPER_IP", "127.0.0.1")
    val clipperPort = sys.env.getOrElse("CLIPPER_PORT", "7000").toInt
    val logPath = sys.env.getOrElse("LOG_PATH", "/external/temp/clipper_log.txt")

    val conn = new Connection()
    val container: SysmlModelContainer = Clipper.loadSysmlModel(conn, modelPath, logPath, weightsDir, gpuIndex)
    val parser = new DoubleVector.Parser

    System.err.println("USING GPU: " + gpuIndex)

    while (true) {
      println("Starting Clipper SystemML Container")
      println(s"Serving model $modelName@$modelVersion")
      println(s"Connecting to Clipper at $clipperAddress:$clipperPort")

      val rpcClient = new RPC(parser)
      try {
        rpcClient.start(container, modelName, modelVersion, clipperAddress, clipperPort)
        println("Connected...")
      } catch {
        case e: UnknownHostException => e.printStackTrace()
      }
    }
  }
}
