package ai.clipper.examples.train

import ai.clipper.spark.Clipper

import scala.io.Source

object SysmlDeploy {

  def main(args: Array[String]): Unit = {
    val argMap = args.map(x => x.split("=")(0) -> x.split("=")(1)).toMap

    if (argMap("model") == "GLM") {
      deployGlmModel(argMap("K").toInt, argMap("dmlPath"))
    } else {
      deployVGGModel(argMap("dmlPath"),
        argMap("weightsDir"),
        argMap("inVarName"),
        argMap("outVarName"),
        argMap("imgSize").toInt)
    }
  }

  def deployGlmModel(K: Int, dmlPath: String) : Unit = {
    val clipperHost = sys.env.getOrElse("CLIPPER_HOST", "localhost")
    val clipperVersion = sys.env.getOrElse("CLIPPER_MODEL_VERSION", "1").toInt
    val dml = Source.fromFile(dmlPath).getLines().mkString("\n")

    val home = sys.env.get("HOME").get
    val externalMountPoint = home + "/SystemML/ServingProfiler"
    val logPath = "/external/temp/clipper_glm.txt"
    val RNG = scala.util.Random
    val weights = Map("b" -> ((1 to K).map(_ => RNG.nextDouble).toArray, K, 1))
    println("CALLING DEPLOY MODEL...")
    Clipper.deploySysmlModel("model1", clipperVersion,
      clipperHost, weights, "X", "predicted_y", dml, K, externalMountPoint, logPath, List("a"))
  }

  def deployVGGModel(dmlPath: String,
                     weightsDir: String,
                     inVarName: String,
                     outVarName: String,
                     imgSize: Int) : Unit = {
    val clipperHost = sys.env.getOrElse("CLIPPER_HOST", "localhost")
    val clipperVersion = sys.env.getOrElse("CLIPPER_MODEL_VERSION", "1").toInt
    val dml = Source.fromFile(dmlPath).getLines().mkString("\n")

    val home = sys.env.get("HOME").get
    val externalMountPoint = home + "/SystemML/ServingProfiler"
    val logPath = "/external/temp/clipper_vgg.txt"

    println("CALLING DEPLOY MODEL...")
    Clipper.deploySysmlModel("model1", clipperVersion,
      clipperHost, weightsDir, inVarName,
      outVarName, dml, imgSize, externalMountPoint, logPath, List("a"))
  }
}
