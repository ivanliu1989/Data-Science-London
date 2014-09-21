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
Grid1 <- expand.grid(mtry=c(5,20,40),n.trees=c(1000))
fitControl1 <- trainControl(method="repeatedcv",10,10,
                            classProbs = TRUE, allowParallel = TRUE,
                            summaryFunction = twoClassSummary)
set.seed(888)
fit1 <- train(result~., method='rf',data=x,trControl = fitControl,
              tuneGrid = Grid, importance = TRUE, metric = "ROC")
pred1 <- predict(fit1, x_train)
# index2 <- pred1 >0.5
# index3 <- pred1 <0.5
# pred1[index2] <- 1
# pred1[index3] <- 0
confusionMatrix(pred1, x_train$result)
pred2 <- predict(fit1, x_test)
# index2 <- pred2 >.5
# index3 <- pred2 <.5
# pred2[index2] <- 1
# pred2[index3] <- 0
confusionMatrix(pred2, x_test$result)

pred_test <- predict(fit1, test)
index2 <- pred_test >.5
index3 <- pred_test <.5
pred_test[index2] <- 1
pred_test[index3] <- 0
Id <- c(1:length(pred_test))
submission <- data.frame(Id, pred_test)
names(submission)<-c("Id","Solution")
write.table(submission, 'submission_rf_1.csv',sep=',',row.names = F)

################ gbm ###################################################################
Grid2 <- expand.grid(n.trees=c(100,500,1000),shrinkage=.1,interaction.depth=c(10,22,50))
fitControl1 <- trainControl(method="repeatedcv",10,10)
fit2 <- train(result~., method='gbm', data=x_train,
              trControl = fitControl1,tuneGrid = Grid2,verbose=T)
pred3 <- predict(fit2, test)
# index2 <- pred1 >0.5
# index3 <- pred1 <0.5
# pred1[index2] <- 1
# pred1[index3] <- 0
confusionMatrix(pred3, x_test$result)
Id <- c(1:length(pred3))
submission <- data.frame(Id, pred3)
names(submission)<-c("Id","Solution")
write.table(submission, 'submission_gbm_1.csv',sep=',',row.names = F)

############### SVM ####################################################################
Grid3 <- expand.grid()
fitControl3 <- trainControl(method="repeatedcv",10,10)
fit3 <- train(result~., method='svm', data=x_train,
              trControl = fitControl3,tuneGrid = Grid3,verbose=T)





############### glm ###################################################################
Grid4 <- expand.grid()
fitControl4 <- trainControl(method="repeatedcv",10,10)
fit4 <- train(result~., method='glm', family='binomial', data=x_train,
              trControl = fitControl4,tuneGrid = Grid4,verbose=T)