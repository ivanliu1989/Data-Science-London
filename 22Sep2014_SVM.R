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

##### Split Data #####
    index <- createDataPartition(x$result, p = 0.8,list = F)
    x_train <- x[index,]
    x_test <- x[-index,]

##### Tune Parameters #####

##### Modeling #####

##### Prediction #####

##### Evaluation #####