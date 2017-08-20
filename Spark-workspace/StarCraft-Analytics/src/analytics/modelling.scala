package analytics

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassifier, GBTClassifier, NaiveBayes, LogisticRegression,MultilayerPerceptronClassifier}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator,BinaryClassificationEvaluator}
import org.apache.spark.sql.types.StructType
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.types._
import org.apache.spark.ml.param.ParamMap
//import org.apache.spark.SparkContext
//import org.apache.spark.SparkContext._
//import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.KNNClassifier

// OFF logging
import org.apache.log4j.Logger
import org.apache.log4j.Level

object modelling {
  def main(args: Array[String]) {

    val spark = SparkSession
      .builder()
      .appName("StarCraft winner prediction - Modelling")
      .getOrCreate()

      Logger.getLogger("org").setLevel(Level.WARN)
      Logger.getLogger("akka").setLevel(Level.WARN)

    val dataSchema = new StructType()
      .add("ReplayID","string")
      .add("Duration","int")
      .add("Frame", "int")
      .add("Minerals", "int")
      .add("Gas", "int")
      .add("Supply", "int")
      .add("TotalMinerals", "int")
      .add("TotalGas", "int")
      .add("TotalSupply", "int")
      .add("GroundUnitValue", "int")
      .add("BuildingValue", "int")
      .add("AirUnitValue", "int")
      .add("ObservedEnemyGroundUnitValue", "int")
      .add("ObservedEnemyBuildingValue", "int")
      .add("ObservedEnemyAirUnitValue", "int")
      .add("EnemyMinerals", "int")
      .add("EnemyGas", "int")
      .add("EnemySupply", "int")
      .add("EnemyTotalMinerals", "int")
      .add("EnemyTotalGas", "int")
      .add("EnemyTotalSupply", "int")
      .add("EnemyGroundUnitValue", "int")
      .add("EnemyBuildingValue", "int")
      .add("EnemyAirUnitValue", "int")
      .add("EnemyObservedEnemyGroundUnitValue", "int")
      .add("EnemyObservedEnemyBuildingValue", "int")
      .add("EnemyObservedEnemyAirUnitValue", "int")
      .add("ObservedResourceValue", "double")
      .add("EnemyObservedResourceValue", "double")
      .add("Winner", "string")
      .add("Races", "string")
      


    val assembler = new VectorAssembler()
      .setInputCols(Array("Frame","Minerals","Gas","Supply","TotalMinerals","TotalGas","TotalSupply",
                          "GroundUnitValue","BuildingValue","AirUnitValue",
                          "ObservedEnemyGroundUnitValue","ObservedEnemyBuildingValue","ObservedEnemyAirUnitValue",
                          "EnemyMinerals","EnemyGas","EnemySupply","EnemyTotalMinerals","EnemyTotalGas","EnemyTotalSupply",
                          "EnemyGroundUnitValue","EnemyBuildingValue","EnemyAirUnitValue",
                          "EnemyObservedEnemyGroundUnitValue","EnemyObservedEnemyBuildingValue","EnemyObservedEnemyAirUnitValue",
                          "ObservedResourceValue","EnemyObservedResourceValue"
                          ,"indexedRaces"
                        ))
      .setOutputCol("features")

    println("Loading data")
    val dataPath = "../../data/data_full.csv"
    val file = spark.read.option("header","true")
      .option("inferSchema","false")
      .schema(dataSchema)
      .csv(dataPath)

    // Split the data into training and test sets (30% held out for testing).
    val trainPath = "trainData.csv"
    val testPath = "testData.csv"

    val Array(trainData, testData) = if (! new java.io.File(trainPath).exists || ! new java.io.File(testPath).exists) {
      println("Splitting data")
      val Array(trainData, testData) = file randomSplit Array(0.7, 0.3)

      println("Writing to files")
      trainData.write.option("header", "true").csv(trainPath)
      testData.write.option("header", "true").csv(testPath)
      Array(trainData, testData)
    }
    else {
      println("Reading from existing files")
      val trainData = spark.read.option("header","true")
        .option("inferSchema","false")
        .schema(dataSchema)
        .csv(trainPath)

      val testData = spark.read.option("header","true")
        .option("inferSchema","false")
        .schema(dataSchema)
        .csv(testPath)

      Array(trainData, testData)
    }



    val winnerIndexer = new StringIndexer()
      .setInputCol("Winner")
      .setOutputCol("indexedWinner")

    val racesIndexer = new StringIndexer()
      .setInputCol("Races")
      .setOutputCol("indexedRaces")

    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 6 distinct values are treated as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(6)

    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")

    // Train a RandomForest model.
    val rf = new RandomForestClassifier()
      .setLabelCol(winnerIndexer.getOutputCol)
      .setFeaturesCol(featureIndexer.getOutputCol)
      .setNumTrees(150)
      .setMaxMemoryInMB(1024)

    val nb = new NaiveBayes()
      .setLabelCol(winnerIndexer.getOutputCol)
      .setFeaturesCol(featureIndexer.getOutputCol)

    val lr = new LogisticRegression()
      .setLabelCol(winnerIndexer.getOutputCol)
      .setFeaturesCol(featureIndexer.getOutputCol)
      .setMaxIter(150)

    val gbt = new GBTClassifier()
      .setLabelCol(winnerIndexer.getOutputCol)
      .setFeaturesCol(featureIndexer.getOutputCol)
      .setMaxIter(150)

    val mlp = new MultilayerPerceptronClassifier()
      .setLabelCol(winnerIndexer.getOutputCol)
      .setFeaturesCol(featureIndexer.getOutputCol)
      .setMaxIter(150)

    val knn = new KNNClassifier()
      .setLabelCol(winnerIndexer.getOutputCol)
      .setFeaturesCol(featureIndexer.getOutputCol)
      .setTopTreeSize(trainData.count().toInt / 500)

    val rf_paramGrid = new ParamGridBuilder()
      .addGrid(rf.maxDepth, Array(5, 10))
      .build()

    val nb_paramGrid = new ParamGridBuilder()
      .addGrid(nb.smoothing, Array(1.0, 2.0))
      .build()

    val lr_paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.3,0.5))
      .build()

    val gbt_paramGrid = new ParamGridBuilder()
      .addGrid(gbt.maxDepth, Array(5, 10))
      .build()

    val mlp_paramGrid = new ParamGridBuilder()
      .addGrid(mlp.layers, Array(Array(28,4,5,2),Array(28,10,10,2)))
      .build()

    val knn_paramGrid = new ParamGridBuilder()
      .addGrid(knn.k, Array(3,5))
      .build()

    val model_map = Map ("rf" -> Tuple2(rf, rf_paramGrid),
                         "nb" -> Tuple2(nb, nb_paramGrid),
                         "lr" -> Tuple2(lr, lr_paramGrid),
                         "gbt" -> Tuple2(gbt, gbt_paramGrid),
                         "mlp" -> Tuple2(mlp, mlp_paramGrid) //,
                         //"knn" -> Tuple2(knn, knn_paramGrid) This is not writable ATM!
                        )


    val results = model_map.map(model => {
      val model_path = "model_" + model._1 + "_grid"
      if(! new java.io.File(model_path).exists) {
        println("Fitting " + model_path)

        val param:Array[ParamMap] = model._2._2

        // Chain indexers and forest in a Pipeline.
        val pipeline:Pipeline = new Pipeline()
          .setStages(Array(racesIndexer, winnerIndexer, assembler, featureIndexer, model._2._1))

        // evaluator
        val evaluator = new MulticlassClassificationEvaluator()
          .setMetricName("accuracy")
          .setLabelCol("indexedWinner")
          .setPredictionCol("prediction")

        val crossValidator = new CrossValidator()
          .setEstimator(pipeline)
          .setEvaluator(evaluator)
          .setEstimatorParamMaps(param)
          .setNumFolds(10)


        val timeStart = System.nanoTime
        // Train model. This also runs the indexers.
        val fitted_model = crossValidator fit trainData

        val timeEnd = System.nanoTime

        val elapsed = timeEnd - timeStart

        println("Elapsed: " + elapsed + " seconds")

        fitted_model.save(model_path)
        (model._1, elapsed)
      }
      else println(model_path + " already exists!")
    })

    println("Success!")

    val csv:String = "Model,Time\n" + results.toSeq.mkString("\n")

    new java.io.PrintWriter("timeElapsed.csv") { write(csv); close() }



    // Make predictions.
//    val predictions = model transform testData
//
//    val accuracy = evaluator evaluate predictions
//
//    val evaluatorAUC = new BinaryClassificationEvaluator()
//      .setLabelCol("indexedWinner")
//      .setRawPredictionCol("rawPrediction")
//
//    val AUC = if (predictions.columns.contains("rawPrediction")) evaluatorAUC evaluate predictions else -1
//
//
//    println("Accuracy = " + accuracy)
//    println("AUC = " + AUC)
//
  }
}
