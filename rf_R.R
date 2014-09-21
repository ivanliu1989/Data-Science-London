setwd("C:\\Users\\Ivan.Liuyanfeng\\Desktop\\Data_Mining_Work_Space\\Data-Science-London")
library(caret)
###### load data #########################################
x <- read.csv("Data/train.csv", header = F)
test <- read.csv("Data/test.csv", header = F)
y <- read.csv("Data/trainLabels.csv", header= F)
names(y)<- 'result'
x <- cbind(x,y)
str(x)
x$result <- as.factor(x$result)
index <- createDataPartition(x$result, p = 0.8,list = F)
x_train <- x[index,]
x_test <- x[-index,]

###### pre process #######################################