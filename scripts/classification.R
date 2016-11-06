## ---- setup ----
library(ggplot2)
library(caret)
library(reshape2)

## ---- multiplot ----
# Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

## ---- readData ----
data.pvp <- read.csv("data/data_pvp.csv", colClasses=c("integer",rep("numeric",28),"factor"))
data.pvt <- read.csv("data/data_pvt.csv", colClasses=c("integer",rep("numeric",28),"factor"))
data.pvz <- read.csv("data/data_pvz.csv", colClasses=c("integer",rep("numeric",28),"factor"))
data.tvt <- read.csv("data/data_tvt.csv", colClasses=c("integer",rep("numeric",28),"factor"))
data.tvz <- read.csv("data/data_tvz.csv", colClasses=c("integer",rep("numeric",28),"factor"))
data.zvz <- read.csv("data/data_zvz.csv", colClasses=c("integer",rep("numeric",28),"factor"))
data.full <- rbind(cbind(data.pvp, Races = "PvP"), cbind(data.pvt, Races = "PvT"),
                   cbind(data.pvz, Races = "PvZ"), cbind(data.tvt, Races = "TvT"),
                   cbind(data.tvz, Races = "TvZ"), cbind(data.zvz, Races = "ZvZ"))

data.full$ReplayID  <- as.factor(paste(data.full$Races,data.full$ReplayID,sep = "_"))

metadata <- data.full[, colnames(data.full) %in% c("ReplayID", "Duration", "Races")]
metadata <- unique(metadata)

## ---- meltPlots ----
set.seed(1984787456)
replays.plot <- data.full[data.full$ReplayID %in% sample(unique(data.full$ReplayID[grepl("PvP",data.full$ReplayID)]),3),]

features.melt <- list(c("Minerals","Gas","Supply"),
                      c("TotalMinerals","TotalGas","TotalSupply"),
                      c("GroundUnitValue","BuildingValue","AirUnitValue"),
                      c("ObservedEnemyGroundUnitValue","ObservedEnemyBuildingValue","ObservedEnemyAirUnitValue"),
                      c("ObservedResourceValue"))

replays.melt.plots <- lapply(features.melt, function(features){
  r1 <- melt(replays.plot, id.vars = c("Frame","ReplayID"), measure.vars = features, variable_name = "Value")
  r2 <- melt(replays.plot, id.vars = c("Frame","ReplayID"), measure.vars = paste0("Enemy",features), variable_name = "Value")
  replays.melt <- merge(r1,r2, by = c("Frame","ReplayID"))
  replays.melt <- replays.melt[replays.melt$variable.y == paste0("Enemy",replays.melt$variable.x),
                               !colnames(replays.melt) %in% "variable.y"]
  colnames(replays.melt) <- c("Frame","ReplayID","Variable","Value","Value2")
  
  gg <- ggplot(replays.melt) + facet_grid(ReplayID ~ Variable) + geom_line(aes(x = Frame, y = Value),  color = "red") + geom_line(aes(x = Frame, y = Value2), color = "blue")
  gg
})

## ---- classify ---
# Save a validation dataset (20%)
indices <- createDataPartition(data.full$Winner, p = 0.8, list = FALSE)

# Pre-Compute CV folds so we can use the same ones for all models
CV.Folds <- createFolds(data.full$Winner[indices], k = 10)
train.control <- trainControl(method = "cv", index = CV.Folds)

# Fit a xgbTree
xgbGrid <-  expand.grid(eta = c(0.1, 0.2, 0.3,0.4), 
                        max_depth = c(1,2,4,8,16,32), 
                        gamma = 0,
                        colsample_bytree = c(0.6,0.8,1),
                        min_child_weight = 1,
                        nrounds = c(100,150,200)
                        )
xgbModel <- train(Winner~., data=data.full[indices,-c(1,2,31)], 
               trControl=train.control, 
               method="xgbTree", 
               nthread = 4,
               tuneGrid = xgbGrid)

# Fit a Linear SVM
LSVMGrid <- expand.grid(C = seq(0.25,1,by=0.25))
LSVMmodel <- train(Winner~., data=data.full[indices,-c(1,2,31)], 
                   method="svmLinear",
                   trControl=train.control, 
                   tuneGrid = LSVMGrid)

# Fit a Poly SVM
PSVMGrid <- expand.grid(degree = 1:3,
                        C = seq(0.25,1,by=0.25),
                        scale = c(0.001,0.01,0.1)
                        )
PSVMmodel <- train(Winner~., data=data.full[indices,-c(1,2,31)], 
                   method="svmPoly",
                   trControl=train.control, 
                   tuneGrid = PSVMGrid)

# Fit a Radial SVM
RSVMGrid <- expand.grid(C = seq(0.25,1,by=0.25))
RSVMmodel <- train(Winner~., data=data.full[indices,-c(1,2,31)], 
                   method="svmRadial",
                   trControl=train.control, 
                   tuneGrid = RSVMGrid)

# Fit a Paralell Random Forest
RFGrid <- expand.grid(mtry = 2:10)
RFModel <- train(Winner~., data=data.full[indices,-c(1,2,31)], 
                 method="parRF",
                 trControl=train.control, 
                 tuneGrid = RFGrid)

# Fit a KNN
KNNGrid <- expand.grid(k =c(3,5,7,9,13),
                       kernel = "optimal",
                       distance = 2)
KNNModel <- train(Winner~., data=data.full[indices,-c(1,2,31)], 
                  method="kknn",
                  trControl=train.control, 
                  tuneGrid = KNNGrid)

## ---- predict ----
# Predict with validation set

