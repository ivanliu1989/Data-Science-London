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
levels(x$result) <- c('No','Yes')
mean(is.na(x))
index <- createDataPartition(x$result, p = 0.8,list = F)
x_train <- x[index,]
x_test <- x[-index,]
    # x_train_1 <- x_train[,-41]
    # prep <- preProcess(x = x_train_1, method=c('center','scale'))
    # x_train_1 <- predict(prep, x_train_1)
    # x_train_1 <- cbind(x_train_1, x_train[,41])

##### exploration #####
plot(density(x_train[,1]))
qqnorm(x_train[,1])


##### GMM #####
    # g1 <-as.matrix(x_train[,1:40])
    # g2 <- as.matrix(x_train$result)
    # x1 <- as.matrix(x_train)
    # res <- gmm(g2~g1, x1)


##### Model #####
    # cost=c(0.01,1,10,100,1000)
    # gamma=c(0.01,0.001,0.0001)1/number of features
    # range=10^c(-3:3)
    # kernel=rbf
fit_gaussprLinear <- train(result~., method='gaussprLinear', data=x_train,
                           verbose=T, preProcess=c('pca','center','scale'))
Pred_gaussprLinear <- predict(fit_gaussprLinear, x_train)
confusionMatrix(Pred_gaussprLinear, x_train[,41])
featureSelection <- varImp(fit_gaussprLinear)
plot(featureSelection)

x_train_gmm <- cbind(x_train, Pred_gaussprLinear)
Grid <- expand.grid(C=seq(1,10,0.5)) 
fitControl <- trainControl(method="repeatedcv",10,10,classProbs = T)
fit_svmRadialCost <- train(result~V15+V13+V19+V35+V29+V40+V37+V33+V7+V24+V12+V5+V2+V21+Pred_gaussprLinear, 
                           method='svmRadialCost', data=x_train_gmm,
                           trControl = fitControl, verbose=T, preProcess=c('pca','center','scale'),
                           tuneGrid=Grid)
Pred <- predict(fit_svmRadialCost, x_train_gmm)
confusionMatrix(Pred, x_train_gmm[,41])
featureSelection <- varImp(fit_svmRadialCost)
plot(featureSelection)

##### Prediction Cross validation #####
Pred_gaussprLinear <- predict(fit_gaussprLinear, x_test)
x_test_gmm <- cbind(x_test, Pred_gaussprLinear_test)
Pred_test <- predict(fit_svmRadialCost, x_test_gmm)
confusionMatrix(Pred_test, x_test_gmm$result) ## acc 0.88 after 1st feature selection

##### Prediction on Test datasets #####
Pred_test <- predict(fit_svmRadialCost, test)
Id <- c(1:length(Pred_test))
submission <- data.frame(Id, Pred_test)
levels(submission[,2]) <- c(0,1)
names(submission)<-c("Id","Solution")
write.table(submission, 'submission_svm_24Sep2014.csv',sep=',',row.names = F)
