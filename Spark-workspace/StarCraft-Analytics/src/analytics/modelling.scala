/*

		Copyright (C) 2017 Antonio Álvarez Caballero a.k.a. analca3

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    
 */

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
      
    
      
    val FRAMES = (4500 to 85500 by 4500).union(Array(238556))
    val MAX_ITERS = 5
    val N_RACES = 6
    val TEST_SIZE = 0.3
    val CV_FOLDS = 10
    
    /* Fixed algorithm parameters */
    val NUM_ITERS = 150
    val MAX_MEMORY_MB = 1024
    
    /* -------------------------- */

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

    // Train a RandomForest model.
    val rf = new RandomForestClassifier()
      .setLabelCol(winnerIndexer.getOutputCol)
      .setFeaturesCol(featureIndexer.getOutputCol)
      .setNumTrees(NUM_ITERS)
      .setMaxMemoryInMB(MAX_MEMORY_MB)

    val nb = new NaiveBayes()
      .setLabelCol(winnerIndexer.getOutputCol)
      .setFeaturesCol(featureIndexer.getOutputCol)

    val lr = new LogisticRegression()
      .setLabelCol(winnerIndexer.getOutputCol)
      .setFeaturesCol(featureIndexer.getOutputCol)
      .setMaxIter(NUM_ITERS)

    val gbt = new GBTClassifier()
      .setLabelCol(winnerIndexer.getOutputCol)
      .setFeaturesCol(featureIndexer.getOutputCol)
      .setMaxIter(NUM_ITERS)

    val mlp = new MultilayerPerceptronClassifier()
      .setLabelCol(winnerIndexer.getOutputCol)
      .setFeaturesCol(featureIndexer.getOutputCol)
      .setMaxIter(NUM_ITERS)

//    val knn = new KNNClassifier()
//      .setLabelCol(winnerIndexer.getOutputCol)
//      .setFeaturesCol(featureIndexer.getOutputCol)
//      .setTopTreeSize(trainData.count().toInt / 500)
      
    /* Grid of parameters of each algorithm */

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

//    val knn_paramGrid = new ParamGridBuilder()
//      .addGrid(knn.k, Array(3,5))
//      .build()
   /* ------------------------------------- */
      
    val sequence = 1 to MAX_ITERS
    
    sequence.map(n => {

      // Split the data into training and test sets (30% held out for testing).
      val trainPath = "datasets/train_" + n.toString()
      val testPath = "datasets/test_" + n.toString()
  
      val Array(trainData, testData) = if (! new java.io.File(trainPath).exists || ! new java.io.File(testPath).exists) {
        println("Splitting data")
        val Array(trainData, testData) = file randomSplit Array(1 - TEST_SIZE, TEST_SIZE)
  
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
    

      val model_map = Map ("rf" -> Tuple2(rf, rf_paramGrid),
                           "nb" -> Tuple2(nb, nb_paramGrid),
                           "lr" -> Tuple2(lr, lr_paramGrid),
                           "gbt" -> Tuple2(gbt, gbt_paramGrid),
                           "mlp" -> Tuple2(mlp, mlp_paramGrid) //,
                           //"knn" -> Tuple2(knn, knn_paramGrid) This is not writable ATM!
                          )
  
  
      model_map.map(model => {
        val model_path = "models/model_" + model._1 + "_" + n.toString()
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
            .setNumFolds(CV_FOLDS)
  
  
          // Train model. This also runs the indexers.
          val fitted_model = crossValidator fit trainData
  
          fitted_model.save(model_path)
        }
        else println(model_path + " already exists!")
      })
    
    })

    println("Success!")

  }
}
