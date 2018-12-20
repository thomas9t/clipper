package ai.clipper.spark

import java.io.{File, PrintWriter}
import java.util.ArrayList

import ai.clipper.container.ClipperModel
import ai.clipper.container.data.{DataType, DoubleVector, SerializableString}
import org.apache.sysml.api.jmlc.PreparedScript
import org.apache.sysml.runtime.matrix.data.MatrixBlock

import scala.collection.JavaConversions._

class SysmlModelContainer(ps: PreparedScript,
                          inVarName: String,
                          outVarName: String,
                          dataLen: Int,
                          logPath: String) extends ClipperModel[DoubleVector] {

  val fh = new PrintWriter(new File(logPath))
  fh.write("batch_size,set_up_time,compute_time,clean_up_time,total_time\n")
  fh.flush()

  override def getInputType : DataType = {
    DataType.Doubles
  }

  override def predict(inputVectors: ArrayList[DoubleVector]): ArrayList[SerializableString] = {
    // package the inputs into a MatrixBlock
    println("LOGGING TO: " + logPath)
    val startTime = System.nanoTime()
    val mb: MatrixBlock = new MatrixBlock(inputVectors.length, dataLen, -1).allocateDenseBlock()
    val doubles = mb.getDenseBlockValues
    var start = 0
    for (req <- inputVectors) {
      req.getData.get(doubles, start, dataLen)
      start += dataLen
    }
    mb.setNonZeros(-1)

    ps.setMatrix(inVarName, mb, false)
    val setUpDoneTime = System.nanoTime()
    val res = ps.executeScript().getMatrixBlock(outVarName)
    val computeDoneTime = System.nanoTime()

    // unpackage the predictions and convert to string...
    val out = new ArrayList[SerializableString]()
    if (res.getNumRows == 1) {
      out.add(new SerializableString(res.getDenseBlockValues.mkString(",")))
    } else {
      // for now making the simplifying assumption that no matrix prediction requests are received
      for (ix <- 0 to res.getNumRows) {
        out.add(new SerializableString(res.slice(ix, ix + 1).getDenseBlockValues.mkString(",")))
      }
    }
    val cleanUpDoneTime = System.nanoTime()
    fh.write(Seq(inputVectors.length,
                 setUpDoneTime - startTime,
                 computeDoneTime - setUpDoneTime,
                 cleanUpDoneTime - computeDoneTime,
                 cleanUpDoneTime - startTime).mkString(",") + "\n")
    fh.flush()
    out
  }
}
