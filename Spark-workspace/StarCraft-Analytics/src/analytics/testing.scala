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

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.StructType
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.{MultilayerPerceptronClassificationModel,GBTClassificationModel,RandomForestClassificationModel,LogisticRegressionModel, NaiveBayesModel}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator,MulticlassClassificationEvaluator}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions
import org.apache.spark.sql.Row



// OFF logging
import org.apache.log4j.Logger
import org.apache.log4j.Level

// Only for KNN
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.KNNClassifier
import org.apache.spark.ml.Pipeline

object testing {
  def DFtoSingleCSV(df:DataFrame, path:String) {

  }
  def main(args: Array[String]) {

    val spark = SparkSession
      .builder()
      .appName("StarCraft winner prediction - Modelling")
      .getOrCreate()

    //val sqlContext = new org.apache.spark.sql.SQLContext(spark.sparkContext)

    import spark.sqlContext.implicits._

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)
    
    val MAX_ITERS = 5


    val attribs = Array("Frame","Minerals","Gas","Supply","TotalMinerals","TotalGas","TotalSupply",
                          "GroundUnitValue","BuildingValue","AirUnitValue",
                          "ObservedEnemyGroundUnitValue","ObservedEnemyBuildingValue","ObservedEnemyAirUnitValue",
                          "EnemyMinerals","EnemyGas","EnemySupply","EnemyTotalMinerals","EnemyTotalGas","EnemyTotalSupply",
                          "EnemyGroundUnitValue","EnemyBuildingValue","EnemyAirUnitValue",
                          "EnemyObservedEnemyGroundUnitValue","EnemyObservedEnemyBuildingValue","EnemyObservedEnemyAirUnitValue",
                          "ObservedResourceValue","EnemyObservedResourceValue"
                          ,"indexedRaces"
                        )

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
      
    
    println("Creating evaluator")

    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
      .setLabelCol("indexedWinner")
      .setPredictionCol("prediction")
      
    var strArgs = ""
      
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
      
       
      
      
      val testData = if (args.length > 0) {
        println("Filter => " + args(0))
        strArgs = "_" + args(0)
        val limit:Double = args(0).toDouble
        rawTestData.where($"Frame" < limit)
      }
      else {
        rawTestData
      }
      
      println("Loading models")
      

      val rf = CrossValidatorModel.load("models/model_rf_" + n.toString())
        //.bestModel.asInstanceOf[PipelineModel]
        //.stages(4).asInstanceOf[RandomForestClassificationModel]
  
      val lr = CrossValidatorModel.load("models/model_lr_" + n.toString())
        //.bestModel.asInstanceOf[PipelineModel]
        //.stages(4).asInstanceOf[LogisticRegressionModel]
  
      val nb = CrossValidatorModel.load("models/model_nb_" + n.toString())
        //.bestModel.asInstanceOf[PipelineModel]
        //.stages(4).asInstanceOf[NaiveBayesModel]
  
      val gbt = CrossValidatorModel.load("models/model_gbt_" + n.toString())
        //.bestModel.asInstanceOf[PipelineModel]
        //.stages(4).asInstanceOf[GBTClassificationModel]
  
      val mlp = CrossValidatorModel.load("models/model_mlp_" + n.toString())
        //.bestModel.asInstanceOf[PipelineModel]
        //stages(4).asInstanceOf[MultilayerPerceptronClassificationModel
     
  
      val pipelines = Array(rf, lr, nb, gbt, mlp)
      
      println("Getting predictions")

      val predictions = pipelines.map( _ transform testData)
    
      println("Evaluating accuracy")
  
      val accuracy = Row.fromSeq(predictions.map( evaluator evaluate _ ))
      accuracy
            
    }))
    
    val accSchema = new StructType()
      .add("RF", "double")
      .add("LR", "double")
      .add("NB", "double")
      .add("GBT", "double")
      .add("MLP", "double")
      //.add("KNN", "double")
      
    //val dfAcc = spark.createDataFrame(rddAcc, accSchema)
    val dfAcc = spark.createDataFrame(rddAcc, accSchema)
    
    dfAcc.repartition(1).write.mode("overwrite").option("header","true").csv("measures/measures" + strArgs)
    



  }
}
