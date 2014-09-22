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

##### Split Data #####
index <- createDataPartition(x$result, p = 0.8,list = F)
x_train <- x[index,]
x_test <- x[-index,]

##### Tune Parameters #####

##### Modeling #####

##### Prediction #####

##### Evaluation #####