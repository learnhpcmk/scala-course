package EuroCC

import scala.Array._

object Scala_Intro {
  def main(args: Array[String]): Unit = {
    val a = 1
    val b = "Mojot prv kod vo Skala"
    val c = (1,2)
    var d = 'A'
    //println(a)
    //println(b)
    //println(c)
    var r = 4
    //println(4)

    // Kontrolni strukturi

    if (a < 2) { println("Vrednosta na promelivata a e pomala od 2")}
    else if ( a > 0) (println("Vrednosta na promenlivata a e pogolema od 0"))
    else  (println("Vrednosta na promenlivata a e 1"))
    val x = if (a > r) a else r
    println("Vrednosta na promenlivata x e> " + x)

    for (t <- 1 to 4) {println(t); println(a)}
    for (s <- 1 to 20 by 2) println(s)

    while (r <= 10)
      {
        println("Vrednosta na promenlivata r e:" + r)
        r += 1
      }

    def result(i:Any) = i match {
      case 1 => "eden"
      case 2 => "dva"
      case _ => "ne e 1 nitu 2"
      case "Andreja" => 5
    }
    println(result(1))

    var niza1 = range(2,20,4)
    val niza2: Array[Int] = Array[Int] (4,2,3,4,1,6)
    val niza3 = new Array[String](3)
    niza3(0) = "My"
    niza3(1) = "first"
    niza3(2) = "code"
    println("Vrednost na elementire vo listata se: ")
    //for (i <- 0 to 2)
    //  {print(" " + niza3(i))}
    //for (i <- 0 to (niza3.length - 1))
    //{print(" " + niza3(i))}

    var min = niza2(0)
    //val min4 = niza2(0)
    for (i <- 1 to (niza2.length - 1)) {if (niza2(i) < min) min = niza2(i)}
    println("Min e :" +min)

    var MatrixDef = ofDim[Int](4,4)
    for (i <- 0 to 3) {
      for (j <- 0 to 3) {
        MatrixDef(i)(j) = j}}

    println(MatrixDef(1)(2))


    //Lists
    val lista1 = List(1, 2, 3, 4, 5, 1, 2)
    val lista2 = 1 :: 2 :: 3 :: 4:: 5:: 1:: 2 :: Nil
    println(lista2(3))
    val temp = Nil //Definicija na prazna lista
    val gradoviEU = "Brussel" :: "Paris" :: "Berlin" :: Nil
    println("Prviot element оd listata e : " + gradoviEU.head)
    println("Posledniot element od listata e : " + gradoviEU.isEmpty)
    val gradoviMK = "Skopje" :: "Bitola" :: "Ohrid" :: Nil
    val gradoviConcat = List.concat(gradoviEU, gradoviMK)   //merge dve listi
    println(gradoviConcat)

    //Sets
    var setInt : Set[Int] = Set()
    var setParni : Set[Int] = Set(2,4,6,8,10)
    val gradoviEU1 = Set("Paris", "Berlin", "Rome")
    val gradoviMK1 = Set("Skopje", "Bitola", "Ohrid")
    var citiesConcat = gradoviEU1.++(gradoviMK1)
    println("Minimalna vrednost na elementite vo setParni(2,4,6,8,10) : " + setParni.min)
    println("Unija na dve mnozestva : " + citiesConcat)

    //Tuples
    //Elementite moz e dase od razlicen tip na promeliva, GLAVNA RAZLIKA
    var torki  = (34, "Zdravo ", r)
    println(torki._2 + torki._1)
    val newinfo = "Skopje" -> 2
    println(newinfo)

    //Maps
    //Moze da se razgleduvaat kako kolekcija na torki
    val newMap = Map(1 -> 2)
    val newMap2 = Map(1 -> "Eden", 2 -> "Dva")
    println(newMap2)
    def getMinNiza2():Int =
    {
      var min = niza1(0)
      for (i <- 1 to (niza2.length - 1))
      {
        if (niza1(i) < min)
          min = niza1(1)}
      min}
    val NizaMin = Map("Mininalna vrednost e" -> getMinNiza2())
    println("Elementot so najmala vrednost e: " + NizaMin)


    //OOP concepts -method
    def min2(x1: Int, x2: Int): Int = if (x1 < x2) x1 else x2

    def Zdravo(person: String) = "Zdravo " + person + " !"

    println(Zdravo("Andreja"))

    def mnozenje(x: Int, y: Int): Int = {
      var multiply: Int = 0
      multiply = x * y
      multiply
    }

    println(mnozenje(3, 4))
    //OOP concepts - klasi
    class Point(var x: Int, var y: Int) {

      def move(dx: Int, dy: Int): Unit = {
        x = x + dx
        y = y + dy
      }
    }
    val point1 = new Point(2, 3)
    println(point1.x) // 2
    println(point1) // prints (2, 3)

    //object primer + method (singleton)

    object Box
    object Logger {
      def info(message: String): Unit = println(s"INFO: $message")
    }

    //object companion
    case class Circle(radius: Double) {

      import Circle._

      def area: Double = calculateArea(radius)
    }

    import scala.math._
    object Circle {
      private def calculateArea(radius: Double): Double = Pi * pow(radius, 2.0)
    }

    val circle1 = Circle(5.0)
    circle1.area

    //packages
    // package edenPackage {
    //class test { override def toString = "Jas sum edenPaket.Test" }
    // }
    object MainProgram {
      def main(args: Array[String]): Unit = {
        // println(new parentPackage.Test())
        // println(new parentPackage.parentChildPackage.Test())
      }
    }

    //functions
    object add {
      println("Returned Value : " + addInt(5, 7))

      def addInt(a: Int, b: Int): Int = {
        var sum: Int = 0
        sum = a + b
        return sum
      }
    }
    println(add)

  }
}
