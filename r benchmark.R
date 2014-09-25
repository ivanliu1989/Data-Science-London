setwd("/Users/ivan/Work_directory/Data-Science-London")
library(caret)
library(doMC)

#load data
train <- read.csv("Data/train.csv",header=F)
y <- read.csv("Data/trainLabels.csv", header= F)
names(train) <- paste0("V",1:40)
test <- read.csv("Data/test.csv",header=F)

#########
## PCA ##
#########
pca <- prcomp(rbind(train,test))
train <- as.data.frame(cbind(y[,1],pca$x[1:1000,]))
train[,1] <- as.factor(ifelse(train[,1] == 0, "zero", "one"))
names(train)[1] <- "label"
test <- as.data.frame(pca$x[1001:10000,])

#######################
## Feature Selection ##
#######################
registerDoMC(10)
rfeFuncs <- rfFuncs
rfeFuncs$summary <- twoClassSummary
rfe.control <- rfeControl(rfeFuncs, method = "repeatedcv", number=10 ,repeats = 4, verbose = FALSE, returnResamp = "final")
rfe.rf <- rfe(train[,-1], train[,1], sizes = 10:15, rfeControl = rfe.control,metric="ROC")

###############
## Save Data ##
###############
train <- train[,c("label",predictors(rfe.rf))]
test <- test[,predictors(rfe.rf)]
save(train,test,file="Data/trainData.RData")

##################
## Build Models ##
##################
load("Data/trainData.RData")
#model parameters
registerDoMC(7)
pp <- c("center","scale")
sumFunc <- function (data, lev = NULL, model = NULL) {
    require(pROC)
    if (!all(levels(data[, "pred"]) == levels(data[, "obs"]))) 
        stop("levels of observed and predicted data do not match")
    rocObject <- try(pROC:::roc(data$obs, data[, lev[1]]), silent = TRUE)
    rocAUC <- if (class(rocObject)[1] == "try-error") 
        NA
    else rocObject$auc
    out <- c(mean(data[,"pred"]==data[,"obs"]),rocAUC, sensitivity(data[, "pred"], data[, "obs"], lev[1]), specificity(data[, "pred"], data[, "obs"], lev[2]))
    names(out) <- c("ACC","ROC", "Sens", "Spec")
    out
}
tc <- trainControl(method="repeatedcv",number=10,repeats=10,classProbs=T,savePred=T,index=createMultiFolds(train$label, k=10, times=5),summaryFunction=sumFunc)
(model <- train(label~.,data=train,method="avNNet",trControl=tc,preProcess=pp,metric="ROC",tuneGrid=expand.grid(.bag=c(F),.size=c(27:28),.decay=c(0.17)),allowParallel=F))
test.pred <- predict(model,test,type="prob")[,"one"]

##################################
## Semisupervised Self Training ##
##################################
#add cases
newtraini <- which(test.pred >= 0.98 | test.pred <= 0.02)
newtrain <- test[newtraini,]
newtrain$label <- as.factor(ifelse(test.pred[newtraini] < 0.5, "zero","one"))
train <- rbind(train,newtrain[,names(train)])

#make new model
tc$index <- lapply(tc$index,function(fold) c(fold,1001:nrow(train)))
(model.semi <- train(label~.,data=train,method="avNNet",trControl=tc,preProcess=pp,metric="ROC",tuneGrid=expand.grid(.bag=c(F),.size=c(27:28),.decay=c(0.17)),allowParallel=F))
test.pred.semi <- predict(model.semi,test,type="prob")[,"one"]

######################
## Save Predictions ##
######################
test.pred <- (test.pred+test.pred.semi)/2
write.csv(data.frame(id=1:length(test.pred),solution=ifelse(test.pred < 0.5, 0 ,1)),file="Data/submission_25Sep2014.csv",row.names=F)
