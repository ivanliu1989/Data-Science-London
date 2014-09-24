setwd("/Users/ivan/Work_directory/Data-Science-London")
library(caret)
library(gmm)

##### Pre process #####
x <- read.csv("Data/train.csv", header = F)
test <- read.csv("Data/test.csv", header = F)
y <- read.csv("Data/trainLabels.csv", header= F)
names(y)<- 'result'
x <- cbind(x,y)
str(x)
x$result <- as.factor(x$result)
mean(is.na(x))
index <- createDataPartition(x$result, p = 0.8,list = F)
x_train <- x[index,]
x_test <- x[-index,]

##### data explorasion #####
featurePlot(x=x$result, y=pr)



##### GMM #####
g1 <-as.matrix(x_train[,1:40])
g2 <- as.matrix(x_train$result)
x1 <- as.matrix(x_train)
res <- gmm(g2~g1, x1)


##### Model #####
    # cost=c(0.01,1,10,100,1000)
    # gamma=c(0.01,0.001,0.0001)1/number of features
    # range=10^c(-3:3)
    # kernel=rbf
Grid <- expand.grid(C=c(0.01,1,10,100,1000),gamma=c(0.01,0.001,0.0001),range=10^c(-3:3)) 
fitControl <- trainControl(method="repeatedcv",10,5,classProbs = T)
fit_svmLinear <- train(result~., method='svmRadialCost', data=x_train,
              trControl = fitControl, verbose=T, preProcess=c('pca'),
              metric='ROC',tuneGrid=Grid)