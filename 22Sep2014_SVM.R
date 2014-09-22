##### Setup #####
    setwd("/Users/ivan/Work_directory/Data-Science-London")
    library(caret)

##### Load Data #####
    x <- read.csv("Data/train.csv", header = F)
    test <- read.csv("Data/test.csv", header = F)
    y <- read.csv("Data/trainLabels.csv", header= F)
    names(y)<- 'result'
    x <- cbind(x,y)
    str(x)

##### Preprocess Data #####
    x$result <- as.factor(x$result)
    levels(x$result) <- c('NO', 'YES')
    mean(is.na(x))
    png("featurePlot.png",1024,1024)
    featurePlot(x = x[, 1:40],
            y = x$result,
            plot = "box",
            ## Pass in options to bwplot() 
            scales = list(y = list(relation="free"),
                          x = list(rot = 90)),
            layout = c(4,10),
            auto.key = list(columns = 2))
    dev.off()
    # none zero variables
    nzv <- nearZeroVar(x, saveMetrics = TRUE)
    nzv
    # centering and scaling
#     preProcValues <- preProcess(x, method = c("center", "scale"))
#     x <- predict(preProcValues,x)
    # transforming predictors


##### Split Data #####
    index <- createDataPartition(x$result, p = 0.8,list = F)
    x_train <- x[index,]
    x_test <- x[-index,]

##### Tune Parameters #####
    fitControl <- trainControl(method = "repeatedcv",
        number = 10,repeats = 10, summaryFunction = twoClassSummary,
        classProbs = TRUE)
    gbmGrid <-  expand.grid(interaction.depth = c(9, 15, 20),
                            n.trees = (1:30)*50,
                            shrinkage = 0.1)
##### Modeling #####
    set.seed(825)
    gbmFit2 <- train(result ~ ., data = x_train,
                     method = "gbm",
                     trControl = fitControl,
                     verbose = FALSE,
                     tuneGrid = gbmGrid,
                     preProc = c('pca'),
                     metric = "ROC")

##### Evaluation #####
    trellis.par.set(caretTheme())
    png('tune_plot2.png')    
    plot(gbmFit2, scales = list(x = list(log = 2)))
    dev.off()
    gbmImp <- varImp(gbmFit2, scale = FALSE)
    png('varImp.png')    
    plot(gbmImp, top = 40)
    dev.off()
    getTrainPerf(gbmFit2)

##### Prediction #####
    pred_gbm <- predict(gbmFit2, x_test)

##### Evaluation #####
    confusionMatrix(pred_gbm, x_test$result)

######################## reduce dimemtionality ####################
pcaVar <- preProcess(x[,1:40], method = 'pca')

##### svm ##################################################
set.seed(1)
sigDist <- sigest(result~., data = x_train, frac = 1)
ctrl <- trainControl(method = "repeatedcv", repeats = 10, savePred = T)
svmGrid <- expand.grid(.sigma = sigDist, .C = 2^(-2:7))
set.seed(2)
svmPCAFit <- train(result~., data=x,
                   method = "svmRadial",
                   tuneGrid = svmGrid,                  
                   preProcess = c("center","scale","pca"), # if center and scale needed
                   trControl = ctrl)
pred_svm <- predict(svmPCAFit, x_test)
confusionMatrix(pred_svm, x_test$result)
trellis.par.set(caretTheme())
png('tune_plot_svmRadial.png')    
plot(svmPCAFit, scales = list(x = list(log = 2)))
dev.off()
svmImp <- varImp(svmPCAFit, scale = FALSE)
png('varImp_svmRadial.png')    
plot(svmImp, top = 40)
dev.off()
getTrainPerf(svmPCAFit)
###################################################################