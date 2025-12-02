# installing nesscary packages
install.packages("languageserver")
install.packages("caret", dependencies = c("Depends", "Suggests"))
install.packages("timechange")
install.packages("ModelMetrics")
install.packages("pkgconfig")
install.packages("ggplot2")
install.packages("labeling")
install.packages("arm")
install.packages("lme4")
install.packages("Rdpack")
install.packages("dslabs")
install.packages("tinytex")
# calling nessecary libraries
library(tinytex)
library(languageserver)
library(caret)
library(datasets)
library(ggplot2)
library(lme4)
library(arm)
library(dslabs)
library(rmarkdown)
library(pandoc)
# 3
# gathering data from dataset package
data(iris)
# checking the first few rows of the data
head(iris)
# pulling summary statistics for the data set
summary(iris)

species <- c("setosa", "versicolor", "virginica")
mean_data <- data.frame()

for (i in seq_along(species)) {
  sepal_l_mean <- mean(iris$Sepal.Length[iris$Species == species[i]])
  sepal_w_mean <- mean(iris$Sepal.Width[iris$Species == species[i]])
  petal_l_mean <- mean(iris$Petal.Length[iris$Species == species[i]])
  petal_w_mean <- mean(iris$Petal.Width[iris$Species == species[i]])

  row <- data.frame(
    species = species[i],
    sepal_length_mean = sepal_l_mean,
    sepal_width_mean = sepal_w_mean,
    petal_length_mean = petal_l_mean,
    petal_width_mean = petal_w_mean
  )

  mean_data <- rbind(mean_data, row)
}
# checking data frame
print(mean_data)
# printing variance between classes of each feature
# this will show which features are the strongest
print(var(mean_data$sepal_length_mean))
print(var(mean_data$sepal_width_mean))
print(var(mean_data$petal_length_mean))
print(var(mean_data$petal_width_mean))
# from this we see that the petal length and petal width are the two largest

# plot of data
ggplot(iris, aes(
  x = Petal.Length,
  y = Petal.Width,
  label = Species,
  color = Species
)) + geom_point(size = 5)

# creating data partition
train <- createDataPartition(
  y = iris$Species,
  p = .7,
  list = FALSE
)

train_data <- iris[train, ]
test_data <- iris[-train, ]


# naive bayes
gnb_train <- train(
  Species ~ .,
  data = train_data,
  method = "naive_bayes"
)
# This shows using GNB we have a 95% accuracy for true False and true Positive
print(gnb_train)

# predicting test data (model, test data)
gnb_test <- predict(gnb_train, newdata = test_data)
print(gnb_test)
print(test_data$Species)

# creating confusion matrix
gnb_conf_matrix <- table(gnb_test, test_data$Species)
print(gnb_conf_matrix)
# from this we can see that all setosa where correctly classified.
# we can see that 1 versicolor and 1 virginica have been misclassified
# in this model

# lda
lda_train <- train(
  Species ~ .,
  data = train_data,
  method = "lda"
)
# This shows that we have a 96% model accuracy for classifying our species
print(lda_train)
# using predict function to test model
lda_test <- predict(lda_train, newdata = test_data)
print(lda_test)
# confusion matrix
lda_conf_matrix <- table(test_data$Species, lda_test)
print(lda_conf_matrix)
# Looking at this we can see that all setosa were correctly classified
# we can also see that 1 versicolor was inocorrectly classified as virginica

# qda
qda_train <- train(
  Species ~ .,
  data = train_data,
  method = "qda"
)
# This shows that we have a 97% model accuracy for classifying our species
print(qda_train)
# using predict to test model
qda_test <- predict(qda_train, test_data)
# confusion matrix
qda_conf_matrix <- table(test_data$Species, qda_test)
print(qda_conf_matrix)
# We can see this model is the same as LDA only incorrectly
# classifying one versicolor as virginica

# logistic regression
log_train <- train(
  Species ~ .,
  data = train_data,
  method = "glmnet"
)
# This shows that we have three classes, and that glmnet
# has a built built in parameter tuning. it shows that
# aplha = .55 and lambda = 0.000867 has the highest overall accruacy.
print(log_train)
# using predict to test model
log_test <- predict(log_train, test_data)
# confusion matrix
log_conf_matrix <- table(test_data$Species, log_test)
print(log_conf_matrix)
# From this we can see that this is the same as lda, qda in that
# there is one versicolor incorrectly labeled as virginica.

# In this we can see that the train model determined that QDA had
# the highest accuracy on the training data. On the test data we did
# not see any difference in the confusions matrix for LDA, QDA,
# and Logistic models


# 4
# gathering data from dataset package
data(mnist_127)
# printing info about data set
head(mnist_127$train)
head(mnist_127$test)
summary(mnist_127$train)
summary(mnist_127$test)


num <- c(1, 2, 7)
mean_mnist_data <- data.frame()

for (i in num) {
  feat_1_mean <- mean(mnist_127$train$x_1[mnist_127$train$y == i])
  feat_2_mean <- mean(mnist_127$train$x_2[mnist_127$train$y == i])

  row_num <- data.frame(
    class = i,
    x_1_mean = feat_1_mean,
    x_2_mean = feat_2_mean
  )

  mean_mnist_data <- rbind(mean_mnist_data, row_num)
}
print(mean_mnist_data)

# GNB on mnist data
gnb_train_mnist <- train(
  y ~ .,
  data = mnist_127$train,
  method = "naive_bayes"
)
# this shows we have a 73% accuracy for True False and
# 74% accuracy for True Positive.
print(gnb_train_mnist)
# using predict to test the model (model, test data)
gnb_mnist_test <- predict(gnb_train_mnist, mnist_127$test)
print(gnb_mnist_test)

# making confusion matrix
gnb_mnist_conf_matrix <- table(mnist_127$test$y, gnb_mnist_test)
print(gnb_mnist_conf_matrix)
# Looking at this matrix we can see that 101/142 1's were correctly
# classified, 87/123 2's were correctly classified, and 106/134 7's
# were correctly classified. This shows us that GNB has the hardest time
# classifying the label 2.

# LDA on mnist data
lda_train_mnist <- train(
  y ~ .,
  data = mnist_127$train,
  method = "lda"
)
# This shows that we have a 66% accuracy for labeling the training data
print(lda_train_mnist)

lda_test_mnist <- predict(lda_train_mnist, mnist_127$test)
# calc conf matrix use table func pred vs actual
lda_mnist_conf_matrix <- table(mnist_127$test$y, lda_test_mnist)
print(lda_mnist_conf_matrix)
# this shows us that 101/142 1's, 53/123 2's, and 97/134 7's were correctly
# classified. This means that the LDA model also struggled to classify
# 2, and performed worse the the GNB method.

# QDA on mnist data
qda_train_mnist <- train(
  y ~ .,
  data = mnist_127$train,
  method = "qda"
)
# here we can see the model has an accuracy of 75% on the trainind data
print(qda_train_mnist)

qda_test_mnist <- predict(qda_train_mnist, mnist_127$test)

qda_mnist_conf_matrix <- table(mnist_127$test$y, qda_test_mnist)
print(qda_mnist_conf_matrix)
# this shows us that 111/142 1's, 86/123 2's, and 102/ 134 7's was
# correctly classified. This is much better than the LDA method,
# and marginally better than the GNB method.

# Logistic Reg mnist data
log_mnist_train <- train(
  y ~ .,
  data = mnist_127$train,
  method = "glmnet"
)
# here we can see the model uses all three classes to train, and it has a
# built in paramerter tune feature that picked an alpha value of 0.1,
# lambda = 0.00542 with the highest accuracy on the training data.
print(log_mnist_train)

log_test_mnist <- predict(log_mnist_train, mnist_127$test)
# conf matrix
log_mnist_conf_matrix <- table(mnist_127$test$y, log_test_mnist)
print(log_mnist_conf_matrix)
# this shows us that 100/142 1's, 55/123 2's, and 96/134 7's were
# correctly classified. This is worse than the QDA method, and struggled
# most classfying the number 2.

# In general all of these metods stuggled to classify the number 2. The best
# method tested was QDA.

tinytex::install_tinytex()
# setting path for pandoc
Sys.setenv(RSTUDIO_PANDOC = "/opt/homebrew/bin")
# knit pdf
rmarkdown::render(
  "/Users/alex/stat_426_proj/stat_426_hw_6.r",
  output_format = "pdf_document"
)