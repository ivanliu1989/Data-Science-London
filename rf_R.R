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
Grid1 <- expand.grid(mtry=c(5,20,50))
fitControl1 <- trainControl(method="cv",10)
set.seed(888)
fit1 <- train(result~., method='rf',data=x_train,trControl = fitControl,tuneGrid = Grid)
Grid1 <- expand.grid(n.trees=c(100,500,1000),shrinkage=.1,mtry=c(5,20,50))
