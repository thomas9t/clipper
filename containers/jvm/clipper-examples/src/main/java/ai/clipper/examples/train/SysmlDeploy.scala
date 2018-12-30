package ai.clipper.examples.train

import ai.clipper.spark.Clipper

import scala.io.Source

object SysmlDeploy {

  def main(args: Array[String]): Unit = {
    val argMap = args.map(x => x.split("=")(0) -> x.split("=")(1)).toMap
    if (argMap("model").toLowerCase == "glm") {
      deployGlmModel(argMap("dmlPath"),
        argMap("inVarName"),
        argMap("outVarName"),
        argMap("externalMountPoint"),
        argMap("logStub"),
        argMap("modelNameStub"),
        argMap("K").toInt,
        argMap("numToDeploy").toInt)
    } else if (argMap("model").toLowerCase == "vgg") {
      deployVGGModel(argMap("dmlPath"),
        argMap("weightsDir"),
        argMap("inVarName"),
        argMap("outVarName"),
        argMap("externalMountPoint"),
        argMap("logStub"),
        argMap("modelNameStub"),
        argMap("imgSize").toInt,
        argMap("numToDeploy").toInt)
    } else {
      val m = argMap("model")
      throw new RuntimeException(s"Invalid Model: $m")
    }
  }

  def deployGlmModel(dmlPath: String,
                     inVarName: String,
                     outVarName: String,
                     externalMountPoint: String,
                     logStub: String,
                     modelNameStub: String,
                     K: Int,
                     numToDeploy: Int) : Unit = {
    val clipperHost = sys.env.getOrElse("CLIPPER_HOST", "localhost")
    val clipperVersion = sys.env.getOrElse("CLIPPER_MODEL_VERSION", "1").toInt
    val dml = Source.fromFile(dmlPath).getLines().mkString("\n")

    val RNG = scala.util.Random
    val weights = Map("b" -> ((1 to K).map(_ => RNG.nextDouble).toArray, K, 1))
    println("CALLING DEPLOY MODEL...")
    for (m <- 1 to numToDeploy) {
      val logPath = s"/external/$logStub$m.txt"
      println("LOGGING TO: " + logPath)
      Clipper.deploySysmlModel(s"$modelNameStub$m", clipperVersion,
        clipperHost, weights, inVarName, outVarName, dml, K, externalMountPoint, logPath, List("a"))
    }
  }

  def deployVGGModel(dmlPath: String,
                     weightsDir: String,
                     inVarName: String,
                     outVarName: String,
                     externalMountPoint: String,
                     logStub: String,
                     modelNameStub: String,
                     imgSize: Int,
                     numToDeploy: Int) : Unit = {
    val clipperHost = sys.env.getOrElse("CLIPPER_HOST", "localhost")
    val clipperVersion = sys.env.getOrElse("CLIPPER_MODEL_VERSION", "1").toInt
    val dml = Source.fromFile(dmlPath).getLines().mkString("\n")

    println("CALLING DEPLOY MODEL...")
    for (m <- 1 to numToDeploy) {
      val logPath = s"/external/$logStub$m.txt"
      Clipper.deploySysmlModel(s"$modelNameStub$m", clipperVersion,
        clipperHost, weightsDir, inVarName,
        outVarName, dml, imgSize, externalMountPoint, logPath, List("a"))
    }
  }
}
