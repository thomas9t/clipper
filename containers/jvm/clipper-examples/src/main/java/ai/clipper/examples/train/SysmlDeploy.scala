package ai.clipper.examples.train

import ai.clipper.spark.Clipper

import scala.io.Source
import scala.sys.process._

object SysmlDeploy {

  def main(args: Array[String]): Unit = {
    val argMap = args.map(x => x.split("=")(0) -> x.split("=")(1)).toMap
    if (argMap.getOrElse("weightsDir", "unused") != "unused") {
      deployStaticModel(argMap("dmlPath"),
        argMap("weightsDir"),
        argMap("inVarName"),
        argMap("outVarName"),
        argMap("externalMountPoint"),
        argMap("logStub"),
        argMap("modelNameStub"),
        argMap("K").toInt,
        argMap("gpuIndex").toInt,
        argMap("numToDeploy").toInt,
        argMap.getOrElse("batchSize", "-1").toInt)
    } else {
      deployDynamicModel(argMap("dmlPath"),
        argMap("inVarName"),
        argMap("outVarName"),
        argMap("externalMountPoint"),
        argMap("logStub"),
        argMap("modelNameStub"),
        argMap("K").toInt,
        argMap("gpuIndex").toInt,
        argMap("numToDeploy").toInt,
        argMap.getOrElse("batchSize", "-1").toInt)
    }
  }

  def deployDynamicModel(dmlPath: String,
                         inVarName: String,
                         outVarName: String,
                         externalMountPoint: String,
                         logStub: String,
                         modelNameStub: String,
                         K: Int,
                         gpuIndex: Int,
                         numToDeploy: Int,
                         batchSize: Int = -1) : Unit = {
    val clipperHost = sys.env.getOrElse("CLIPPER_HOST", "localhost")
    val clipperVersion = sys.env.getOrElse("CLIPPER_MODEL_VERSION", "1").toInt
    val dml = Source.fromFile(dmlPath).getLines().mkString("\n")

    val RNG = scala.util.Random
    val weights = Map("b" -> ((1 to K).map(_ => RNG.nextDouble).toArray, K, 1))
    println("CALLING DEPLOY MODEL...")
    for (m <- 1 to numToDeploy) {
      val logPath = s"/external/$logStub$m.txt"
      println("LOGGING TO: " + logPath)
      println("KILLING OLD CONTAINER...")
      s"docker container stop ${modelNameStub}${m}_container".!
      s"docker container rm ${modelNameStub}${m}_container".!
      Clipper.deploySysmlModel(s"$modelNameStub$m", clipperVersion,
        clipperHost, weights, inVarName, outVarName, dml, K, externalMountPoint, logPath, gpuIndex, List("a"), batchSize)
    }
  }

  def deployStaticModel(dmlPath: String,
                        weightsDir: String,
                        inVarName: String,
                        outVarName: String,
                        externalMountPoint: String,
                        logStub: String,
                        modelNameStub: String,
                        imgSize: Int,
                        gpuIndex: Int,
                        numToDeploy: Int,
                        batchSize: Int = -1) : Unit = {
    val clipperHost = sys.env.getOrElse("CLIPPER_HOST", "localhost")
    val clipperVersion = sys.env.getOrElse("CLIPPER_MODEL_VERSION", "1").toInt
    val dml = Source.fromFile(dmlPath).getLines().mkString("\n")

    println("CALLING DEPLOY MODEL...")
    for (m <- 1 to numToDeploy) {
      val logPath = s"/external/$logStub$m.txt"
      s"docker container stop ${modelNameStub}${m}_container".!
      s"docker container rm ${modelNameStub}${m}_container".!
      Clipper.deploySysmlModel(s"$modelNameStub$m", clipperVersion,
        clipperHost, weightsDir, inVarName,
        outVarName, dml, imgSize, externalMountPoint, logPath, gpuIndex, List("a"), batchSize)
    }
  }
}
