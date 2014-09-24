setwd("/Users/ivan/Work_directory/Data-Science-London")
library(caret)

x <- as.matrix(read.csv("Data/train.csv", header = F))
x_test <- as.matrix(read.csv("Data/test.csv", header = F))
y <- as.matrix(read.csv("Data/trainLabels.csv", header= F))

train <- cbind(x,y)
pca2 <- princomp()