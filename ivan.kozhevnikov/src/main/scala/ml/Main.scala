package ml

/* @author berkut@yandex-team.ru */
object Main {
  def main(args: Array[String]) {
    val vectors = scala.io.Source.fromInputStream(getClass.getResourceAsStream("/wdbc.data")).getLines().map(line => {
      val parts = line.split(",")
      val id = parts(0)
      val y = parts(1) match {
        case "M" => -1
        case "B" => 1
      }
      val x = new Array[Double](parts.size - 1)
      for (i <- 2 until parts.size) {
        x(i - 2) = parts(i).toDouble
      }
      x(parts.size - 2) = 1
      (x, y)
    }).toSeq
    val n = vectors.head._1.size
    val (input, test) = vectors.splitAt(vectors.size / 2)
    val classifier = new LinearPerceptron(input, n)
    var tp = 0
    var fp = 0
    var tn = 0
    var fn = 0
    var error = 0 
    for ((x, y) <- test) {
      val result = classifier.classify(x)
      if (y == 1 && result == 1) {
        tp += 1
      }
      if (y == -1 && result == 1) {
        fp += 1
      }
      if (y == -1 && result == -1) {
        tn += 1
      }
      if (y == 1 && result  == -1) {
        fn += 1
      }
      if (y != result) {
        error += 1
      }
    }
    val precision = 1.0 * tp / (fp + tp)
    val recall = 1.0 * tp / (tp + fn)
    println(s"error: ${1.0 * error / test.size}\nprecesion: $precision\nrecall: $recall")
  }
}
