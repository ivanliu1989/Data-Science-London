setwd("/Users/ivan/Work_directory/Data-Science-London")
library(caret)
library(GMMBoost)

x <- read.csv("Data/train.csv", header = F)
x_test <- read.csv("Data/test.csv", header = F)
y <- read.csv("Data/trainLabels.csv", header= F)

##### pca #####
train <- cbind(x,y)
pca_fit <- preProcess(x, method=c("BoxCox", "center", "scale", "pca"))
PC = predict(pca_fit, x)
head(PC,3)
head(pca_fit$rotation,3)

x_pca <- prcomp(x,center = TRUE,scale. = TRUE) 
print(x_pca)
plot(x_pca, type = "l")
summary(x_pca)

plot(x[,1], ylab = "Density", type = "l",zero.line = TRUE)
plot(density(x[,2]))
plot(density(x[,37]))
plot(density(x[,40]))
qqnorm(x[,1])
qqnorm(x[,40])

##### model #####
pca_comp <- 12
gmm_comp <- 4
covariance_type='full'
min_covar <- 0.1
gamma <- 0
c <- 1.0
training <- x[,1:pca_comp]
training_gmm