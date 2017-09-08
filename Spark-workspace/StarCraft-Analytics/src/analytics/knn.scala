/*
 * author: Antonio Álvarez Caballero (a.k.a. analca3)
 * license: GPLv3
 * 
 * This is a temporary module. When spark-knn package implements MLWritable, it will be merged to the main modules.
 */

package analytics

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator,BinaryClassificationEvaluator}
import org.apache.spark.sql.types.StructType
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.types._
import org.apache.spark.ml.param.ParamMap

import org.apache.spark.sql.types.StructType
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator,MulticlassClassificationEvaluator}

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions
import org.apache.spark.sql.Row

import org.apache.spark.ml.classification.KNNClassifier

// OFF logging
import org.apache.log4j.Logger
import org.apache.log4j.Level

object knn {
  def main(args: Array[String]) {
    val spark = SparkSession
      .builder()
      .appName("StarCraft winner prediction - KNN")
      .getOrCreate()
      
    import spark.implicits._

      Logger.getLogger("org").setLevel(Level.WARN)
      Logger.getLogger("akka").setLevel(Level.WARN)
      
    val FRAMES = (4500 to 85500 by 4500).union(Array(238556))
    val MAX_ITERS = 5
    val N_RACES = 6
    val K_KNN = 3
    val TREESIZE_DENOM = 500 // Don't touch this value unless you know what are you doing!

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
      .setMaxCategories(N_RACES)

    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")


      
    

    println("Creating evaluators")

    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
      .setLabelCol("indexedWinner")
      .setPredictionCol("prediction")
      
    println("Getting predictions")
    
    val sequence = 1 to MAX_ITERS
    
    val rddAcc = spark.sparkContext.parallelize(sequence.map(n => {
      
      val trainPath = "datasets/train_" + n.toString()
      val testPath = "datasets/test_" + n.toString()
      
      println("Loading data")
      val trainData = spark.read.option("header","true")
        .option("inferSchema","false")
        .schema(dataSchema)
        .csv(trainPath)
  
      val rawTestData = spark.read.option("header","true")
        .option("inferSchema","false")
        .schema(dataSchema)
        .csv(testPath)
      
      val knn = new KNNClassifier()
        .setLabelCol(winnerIndexer.getOutputCol)
        .setFeaturesCol(featureIndexer.getOutputCol)
        .setTopTreeSize(trainData.count().toInt / TREESIZE_DENOM)
        .setK(K_KNN)

      
      
      // Chain indexers and forest in a Pipeline.
      val pipeline:Pipeline = new Pipeline()
        .setStages(Array(racesIndexer, winnerIndexer, assembler, featureIndexer, knn))
      
      val knnModel = pipeline fit trainData
    
       
      //var strArgs = ""
      
      val row = Row.fromSeq(FRAMES.map(a => {
        
        println("Testing with frames = " + a.toString)
        
        //strArgs = "_" + a.toString
        
        val testData = rawTestData.where($"Frame" < a)
        
        val predictions = knnModel transform testData
        
        val accuracy = evaluator evaluate predictions
        accuracy
        
      }))
      row
    }))
    
    var accSchema = new StructType()
    FRAMES.map(n => accSchema = accSchema.add(n.toString, "double"))
      
    
    val dfAcc = spark.createDataFrame(rddAcc, accSchema)
    
    dfAcc.repartition(1).write.mode("overwrite").option("header","true").csv("measures/measuresKNN")
    
    
    
  }
}