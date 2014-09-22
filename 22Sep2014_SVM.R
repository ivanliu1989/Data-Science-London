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
    gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9),
                            n.trees = (1:30)*50,
                            shrinkage = 0.1)
##### Modeling #####
    set.seed(825)
    gbmFit <- train(result ~ ., data = x_train,
                     method = "gbm",
                     trControl = fitControl,
                     verbose = FALSE,
                     tuneGrid = gbmGrid,
                     preProc = c('pca'),
                     metric = "ROC")

##### Evaluation #####
    trellis.par.set(caretTheme())
    png('tune_plot.png')    
    plot(gbmFit, scales = list(x = list(log = 2)))
    dev.off()
    gbmImp <- varImp(gbmFit, scale = FALSE)
    png('varImp.png')    
    plot(gbmImp, top = 40)
    dev.off()
    getTrainPerf(gbmFit)
##### Prediction #####

##### Evaluation #####