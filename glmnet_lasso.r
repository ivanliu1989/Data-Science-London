#
# Use glmnet for prediction.
# We use logistic regression.
# Lasso is used to force most coefficients to zero (unused),
# so we obtain a sparse model.
#
setwd("C:\\Users\\Ivan.Liuyanfeng\\Desktop\\Data_Mining_Work_Space\\Data-Science-London")
library(glmnet)

x.test <- read.csv("Data/test.csv", header=F)
x <- read.csv("Data/train.csv", header=F)
y <- read.csv("Data/trainLabels.csv", header=F)

x = x.train
y = x.trainLabels

alpha=1.0     # Lasso
f=glmnet(x=as.matrix(x), y=as.matrix(y), alpha=alpha, family="binomial")
summary(f)
plot(f)
png('deviance.png')
plot(f, xvar="dev")   # Show graph of deviance
dev.off()

cv = cv.glmnet(x=as.matrix(x), y=as.matrix(y), alpha=alpha, family="binomial")

# Get the array of coefficients.
# lambda.min = lambda at minimum cross validation error
# lambda.1se = lambda at 1 standard error within minimum cross validation error
#              therefore fewer coefficients - a sparse model
coef(f, cv$lambda.min)
coef(f, cv$lambda.1se)

# Print out the indexes of the columns chosen.
#1  5
#2 13
#3 15
#4 19
#5 33
#6 35
#7 37
#8 40
predict(f, s=cv$lambda.1se, type="nonzero")

png('sprintf.png')
plot(cv)
title(sprintf("alpha=%.2f", alpha))
dev.off()
# Predict the training data.
y.pred <- predict(f, s=cv$lambda.1se, newx=as.matrix(x), type="response")
y.pred <- round( y.pred[,1])
mean((y.pred-y)^2)
sum(abs(y.pred-y))

# Predict the test data.
# Write the file to submit.
y.pred <- predict(f, s=cv$lambda.1se, newx=as.matrix(x.test), type="response")
y.pred <- round( y.pred[,1])
write.table(cbind(1:length(y.pred),y.pred), col.names=c("Id","Solution"), file="ypred_glmnet_10_1se.csv", row.names=F, sep=",")
