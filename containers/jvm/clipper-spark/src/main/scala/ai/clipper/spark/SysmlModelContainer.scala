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
  fh.write("batch_size,set_up_time,compute_time,clean_up_time,total_time,timestamp\n")
  fh.flush()

  override def getInputType : DataType = {
    DataType.Doubles
  }

  override def predict(inputVectors: ArrayList[DoubleVector]): ArrayList[SerializableString] = {
    // package the inputs into a MatrixBlock
    println("LOGGING TO: " + logPath)
    System.err.println("dataLen: " + dataLen)
    val startTime = System.nanoTime()
    val mb: MatrixBlock = new MatrixBlock(inputVectors.size(), dataLen, -1).allocateDenseBlock()
    val doubles = mb.getDenseBlockValues
    var start = 0
    for (req <- inputVectors) {
      req.getData.get(doubles, start, dataLen)
      req.getData.clear()
      start += dataLen
    }
    mb.setNonZeros(-1)

    ps.setMatrix(inVarName, mb, false)
    val setUpDoneTime = System.nanoTime()
    val res = ps.executeScript().getMatrixBlock(outVarName)
    val computeDoneTime = System.nanoTime()

    // unpackage the predictions and convert to string...
    val out = new ArrayList[SerializableString]()
    System.err.println("Predicted => " + res.getNumRows + " VALUES")
    if (res.getNumRows == 1) {
      out.add(new SerializableString(res.getDenseBlockValues.mkString(",")))
    } else {
      // for now making the simplifying assumption that no matrix prediction requests are received
      for (ix <- 0 until res.getNumRows) {
        out.add(new SerializableString(res.slice(ix, ix).getDenseBlockValues.mkString(",")))
      }
    }
    val cleanUpDoneTime = System.nanoTime()
    System.err.println("RECEIVED BATCH: " + inputVectors.size())
    System.err.println("TOTAL COMPUTE TIME: " + (setUpDoneTime-startTime))
    fh.write(Seq(inputVectors.size(),
                 setUpDoneTime - startTime,
                 computeDoneTime - setUpDoneTime,
                 cleanUpDoneTime - computeDoneTime,
                 cleanUpDoneTime - startTime,
                 System.nanoTime()).mkString(",") + "\n")
    fh.flush()
    out
  }
}
