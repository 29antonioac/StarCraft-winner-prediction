package analytics

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.StructType
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.{MultilayerPerceptronClassificationModel,GBTClassificationModel,RandomForestClassificationModel,LogisticRegressionModel, NaiveBayesModel}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator,MulticlassClassificationEvaluator}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions



// OFF logging
import org.apache.log4j.Logger
import org.apache.log4j.Level

object testing {
  def DFtoSingleCSV(df:DataFrame, path:String) {
    
  }
  def main(args: Array[String]) {
    
    val spark = SparkSession
      .builder()
      .appName("StarCraft winner prediction - Modelling")
      .getOrCreate()
      
    val sqlContext = new org.apache.spark.sql.SQLContext(spark.sparkContext)

    import sqlContext.implicits._
      
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)
      
    
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
      .add("Races", "string")
      .add("Winner", "string")
      
    println("Loading data...")
    val trainData = spark.read.option("header","true")
      .option("inferSchema","false")
      .schema(dataSchema)
      .csv("trainData.csv")
      
    val testData = spark.read.option("header","true")
      .option("inferSchema","false")
      .schema(dataSchema)
      .csv("testData.csv")
      
    val rf = CrossValidatorModel.load("model_rf_grid")
      .bestModel.asInstanceOf[PipelineModel]
      //.stages(4).asInstanceOf[RandomForestClassificationModel]
    
    val lr = CrossValidatorModel.load("model_lr_grid")
      .bestModel.asInstanceOf[PipelineModel]
      //.stages(4).asInstanceOf[LogisticRegressionModel]
    
    val nb = CrossValidatorModel.load("model_nb_grid")
      .bestModel.asInstanceOf[PipelineModel]
      //.stages(4).asInstanceOf[NaiveBayesModel]
    
    val gbt = CrossValidatorModel.load("model_gbt_grid")
      .bestModel.asInstanceOf[PipelineModel]
      //.stages(4).asInstanceOf[GBTClassificationModel]
    
    val mlp = CrossValidatorModel.load("model_mlp_grid")
      .bestModel.asInstanceOf[PipelineModel]
      //stages(4).asInstanceOf[MultilayerPerceptronClassificationModel
    
    val pipelines = Array(rf, lr, nb, gbt, mlp)
    
    
    /*
     * 
     * rf.featureImportances
     * lr.weights
     * nb.pi
     * gbt.featureImportances
     */
    
    
    
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
      .setLabelCol("indexedWinner")
      .setPredictionCol("prediction")
      
    val evaluatorF1 = new MulticlassClassificationEvaluator()
      .setMetricName("f1")
      .setLabelCol("indexedWinner")
      .setPredictionCol("prediction")
      
    val evaluatorAUC = new BinaryClassificationEvaluator()
      .setLabelCol("indexedWinner")
      .setRawPredictionCol("rawPrediction")
      
    val predictions = pipelines.map( _ transform testData)
    
    val accuracy = predictions.map( evaluator evaluate _ )
    
    val AUC = predictions.map( p => {
      if (p.columns.contains("rawPrediction")) 
        evaluatorAUC evaluate p 
      else
        -1
    })
    
    val F1 = predictions.map( evaluatorF1 evaluate _ )
    
    val csv = "RandomForest,LogisticRegression,NaiveBayes,GradientBoosting,MultilayerPerceptron\n" +
      accuracy.mkString(",") + "\n" +
      AUC.mkString(",") + "\n" +
      F1.mkString(",") + "\n" 
       
      
    new java.io.PrintWriter("measures.csv") { write(csv); close() }
    
    // Extract GBT and RF models
    val gbtModel = gbt.stages(4).asInstanceOf[GBTClassificationModel]
    val rfModel = rf.stages(4).asInstanceOf[RandomForestClassificationModel]
    val nbModel = nb.stages(4).asInstanceOf[NaiveBayesModel]

    
    val featureGBT = spark.sparkContext.parallelize(attribs zip gbtModel.featureImportances.toArray).toDF("Feature","Importance")
    val featureRF = spark.sparkContext.parallelize(attribs zip rfModel.featureImportances.toArray).toDF("Feature","Importance")
    
    val nb_aux = nbModel.theta.transpose.toArray.splitAt(28)
    val featureNB = spark.sparkContext.parallelize(nb_aux._1 zip nb_aux._2).toDF("PA","PB") 
    
    
    featureGBT.repartition(1).write.option("header","true").csv("features_gbt")
    featureRF.repartition(1).write.option("header","true").csv("features_rf")
    featureNB.repartition(1).write.option("header","true").csv("features_nb")
    
       
      
  }
}
