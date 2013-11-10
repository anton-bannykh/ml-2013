package ml

import scala.util.Random

/* @author berkut@yandex-team.ru */

class LinearPerceptron(elementAndClasses: Iterable[(Array[Double], Int)], n: Int) {
  private final val ITERATION_LIMIT = 10000000
  private final var w = new Array[Double](n)

  for (i <- 0 until ITERATION_LIMIT) {
    for ((x, y) <- elementAndClasses) {
      if (y != classify(x)) {
        w = sum(w, x, y)
      }
    }
  }

  def classify(x: Array[Double]): Int = if (multiply(x, w) >= 0) 1 else -1

  private def multiply(a: Array[Double], b: Array[Double]) = {
    var ans = 0d
    for (i <- 0 until n) {
      ans += a(i) * b(i)
    }
    ans
  }

  private def sum(a: Array[Double], b: Array[Double], sign: Int) = {
    val ans = new Array[Double](n)
    for (i <- 0 until n) {
      ans(i) = a(i) + b(i) * sign
    }
    ans
  }
}
