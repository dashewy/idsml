install.packages("Rdimtools")
install.packages("IMIFA")
install.packages("knitr")
install.packages("rmarkdown")
install.packages("fastmap")
install.packages("e1071")
install.packages("pandoc")

library(caret)
library(arm)
library(languageserver)
library(Rdimtools)
library(IMIFA)
library(ggplot2)
library(knitr)
library(rmarkdown)
library(pandoc)
library(kernlab)
library(e1071)
library(tinytex)

# uploading data
data(iris)
# looking at the first few rows in the data set
head(iris)
# looking at the summary statistics we can see that there are three classes
summary(iris)

# 1

# uploading data
data(USPSdigits)
# looking at summary statistics
summary(USPSdigits$train)

# The usps data is filled with values correspnding to grayscale values
# between -1 & 1 along with the digit that each matrix corresponds too.
# the numbers are 8-bit on 16x16 grids. Digits range from 0 - 9.

# seperating data into labels and number dataframes
digit_labels <- USPSdigits$train[, 1]
digits <- USPSdigits$train[, -1]
print(digit_labels)
print(class(digits))

# finding the avg of all pixels in the dataframe
avg_digit <- mean(as.matrix(USPSdigits$train[, -1]))
print(avg_digit)
# finding the mean value of all the pixels for each digit
mean_digits <- data.frame()
for (i in 0:9) {
  num_mean <- mean(as.matrix(USPSdigits$train[USPSdigits$train[, 1] == i, -1]))

  row <- data.frame(
    number = i,
    num_mean = num_mean
  )
  mean_digits <- rbind(mean_digits, row)
}
# looking at all each column we can see that the number 6 is the closests to
# the overall avg, making it the avg number in the dataset.
print(mean_digits)

#2
train <- createDataPartition(
  y = iris$Species,
  p = .7,
  list = FALSE
)

train_set <- iris[train, ]
test_set <- iris[-train, ]

# naive bayes model
gnb_train <- train(
  Species ~ .,
  data = train_set,
  method = "naive_bayes"
)

gnb_test <- predict(gnb_train, newdata = test_set)

gnb_conf_matrix <- confusionMatrix(
  data = gnb_test,
  reference = test_set$Species
)
print(gnb_conf_matrix)
# Looking at this confusion matrix we can see that our model has an overall
# 95% accuracy. Looking just at Setosa classifacation we have also have
# perfect percision (TP predictions), specificty (True Positive Rate)
# meaning no errors were made in classifying setosa, as well
# as not misclassifying the others specieas as setosa.

# Linear Discriminant Analysis
lda_train <- train(
  Species ~ .,
  data = train_set,
  method = "lda"
)

lda_test <- predict(lda_train, newdata = test_set)

lda_conf_matrix <- confusionMatrix(
  data = lda_test,
  reference = test_set$Species,
)
print(lda_conf_matrix)
# Looking at this confusion matrix we can see that our model has an overall
# 97% accuracy. Looking at just Setosa classifaction we also have perfect
# percision (True positive predictions), and specificty (True Positive Rate)
# meaning no errors were made in classifying setosa, as well as not
#  misclassifying the others specieas as setosa.

# Quadratic Discriminant Analysis
qda_train <- train(
  Species ~ .,
  data = train_set,
  method = "qda"
)

qda_test <- predict(qda_train, newdata = test_set)

qda_conf_matrix <- confusionMatrix(
  data = qda_test,
  reference = test_set$Species
)

print(qda_conf_matrix)
# Looking at this confusion matrix we can see that our model has an overall
# 95% accuracy. Looking at just Setosa classifaction we also have perfect
# percsion (True positive predictions), and specificity (True positve rate)
# meaning no errors were made in classifying setosa, as well
# as not misclassifying the others specieas as setosa.

# Support Vector Machine
kfold <- trainControl(method = "cv", number = 10)
tune_grid <- c(0.1, 1, 10)

svm_train <- train(
  Species ~ .,
  data = train_set,
  method = "svmLinear",
  preProcess = c("center", "scale"),
  trControl = kfold,
  tune_grid = tune_grid
)
print(svm_train)

svm_test <- predict(svm_train, newdata = test_set)

svm_conf_matrix <- confusionMatrix(
  data = svm_test,
  reference = test_set$Species
)
print(svm_conf_matrix)
# Looking at this model we can see that the over all accuracy is
# 95%. We can also see that the the specifity and sensitivity of
# the model is perfect meaning that the model perfectly. Meaning
# that no setosas were incorectly classified.

# svm linear kernel
svm_linear_train <- train(
  Species ~ .,
  data = train_set,
  method = "svmLinear",
  trControl = kfold
)
print(svm_linear_train)

svm_linear_test <- predict(svm_linear_train, newdata = test_set)

svm_linear_conf_matrix <- confusionMatrix(
  data = svm_linear_test,
  reference = test_set$Species
)

print(svm_conf_matrix)
# Looking at this model we can see that the over all accuracy is
# 95%. We can also see that the the specifity and sensitivity of
# the model is perfect meaning that the model perfectly. Meaning
# that no setosas were incorectly classified.

# svm rbf
svm_rbf_train <- train(
  Species ~ .,
  data = train_set,
  method = "svmRadial",
  preProcess = c("center", "scale"),
  tune_grid = tune_grid,
  trControl = kfold
)
print(svm_rbf_train)

svm_rbf_test <- predict(svm_rbf_train, newdata = test_set)
print(svm_rbf_test)

svm_rbf_conf_matrix <- confusionMatrix(
  data = svm_rbf_test,
  reference = test_set$Species
)

print(svm_rbf_conf_matrix)
# Looking at this confusion matrix we have an overall accuracy of 93%.
# Looking just at the setosa label, we have a 93% percision, meaning that
# the model labeled one none setosa as setosa. This model also had perfect
# specifity meaning the model produced no false positves (saying cat when not).

# svm poly
svm_poly_train <- train(
  Species ~ .,
  data = train_set,
  method = "svmPoly",
  preProcess = c("center", "scale"),
  tune_grid = tune_grid,
  trControl = kfold
)
print(svm_poly_train)
# looking at the output we can see that the highest degree chosen was the 1
# this means that the model will be the same as the linear version. This was
# not consistent and everytime this was ran it picked different hyperparameters.

svm_poly_test <- predict(svm_poly_train, newdata = test_set)

svm_poly_conf_matrix <- confusionMatrix(
  data = svm_poly_test,
  reference = test_set$Species
)
print(svm_poly_conf_matrix)
# looking at this confusion matrix we cn see that we have an overall
# accuracy of 93%.looking just at setosa we have a perfect percision and
# specificity meaning no errors were made in classifying setosa, as well
# as not misclassifying the others specieas as setosa.

# 3
train_dig <- USPSdigits$train
train_dig[, 1] <- as.factor(train_dig[, 1])

test_dig <- USPSdigits$test
test_dig[, 1] <- as.factor(test_dig[, 1])

num_28_train <- subset(train_dig, train_dig[, 1] == 2 | train_dig[, 1] == 8)
# factoring and giving it two levels so that the models function correctly
num_28_train[, 1] <- factor(num_28_train[, 1], levels = c(2, 8))

num_28_test <- subset(test_dig, test_dig[, 1] == 2 | test_dig[, 1] == 8)
# factoring and giving it two levels so that the models function correctly
num_28_test[, 1] <- factor(num_28_test[, 1], levels = c(2, 8))

# GNB
gnb_digit <- train(
  x = num_28_train[, -1],
  y = num_28_train[, 1],
  method = "naive_bayes"
)
print(gnb_digit)

gnb_digit_test <- predict(gnb_digit, newdata = num_28_test)

gnb_digit_conf_matrix <- confusionMatrix(
  gnb_digit_test,
  reference = num_28_test[, 1]
)
print(gnb_digit_conf_matrix)
# looking at this confusion matrix we can see that this model has overall
# accuracy of 88%. This model also has a sensitivy of 85%, and specificty
# of 92%. Meaning this model is more likely to predict 2 over 8.

# lda
lda_digit <- train(
  x = num_28_train[, -1],
  y = num_28_train[, 1],
  method = "lda"
)
print(lda_digit)

lda_digit_test <- predict(lda_digit, newdata = num_28_test)

lda_digit_conf_matrix <- confusionMatrix(
  lda_digit_test,
  reference = num_28_test[, 1]
)
print(lda_digit_conf_matrix)
# looking at this confusion matrix we can see that this model has an overall
# accuracy of 93%. This model also has a sensitivity of 93% and a specificity
# of 93%. This means the model did not favor a specific digit.

# qda
qda_digit <- train(
  x = num_28_train[, -1],
  y = num_28_train[, 1],
  method = "qda",
  # model does not run without pca
  preProcess = c("pca", "center", "scale")
)

qda_digit_test <- predict(qda_digit, newdata = num_28_test[, -1])

qda_digit_conf_matrix <- confusionMatrix(
  qda_digit_test,
  reference = num_28_test[, 1]
)
print(qda_digit_conf_matrix)
# looking at this confusion matrix we can see that this model has an
# overall accuracy of 95%. As well as a 96% sensitivity, and a 93% specificty.
# meaning that this model has a higher tendecy to predict 8 when 2,
# and 2 when 8.

# svm
svm_digit <- svm(
  x = num_28_train[, -1],
  y = num_28_train[, 1],
  type = "C-classification",
  kernel = "linear",
  # increasing cost for soft margin
  cost = 10
)

svm_digit_test <- predict(svm_digit, newdata = num_28_test[, -1])

svm_digit_conf_mat <- confusionMatrix(
  svm_digit_test,
  reference = num_28_test[, 1]
)
print(svm_digit_conf_mat)

# as we can see from this confusion matrix this model has an overall accuracy
# of 95%. We can also see that this model has a sensitivity and specificity of
# 95%. meaning that this model will equally misclassify the digit causing both
# type I & II errors.

# svm linear kernel
svm_lin_digit_train <- train(
  x = num_28_train[, -1],
  y = num_28_train[, 1],
  method = "svmLinear2",
  trControl = kfold
)
print(svm_lin_digit_train)

svm_lin_k_digit_test <- predict(svm_lin_digit_train, newdata = num_28_test)
print(svm_lin_k_digit_test)

svm_lin_dgt_conf <- confusionMatrix(
  svm_lin_k_digit_test,
  reference = num_28_test[, 1]
)
print(svm_lin_dgt_conf)
# looking at this model we can see that this mdoel has an overall accuracy
# of 95%. It also has sensitivity and specificty of 95% meaning that it will
# equally and rarely misclassify a digit as one another.

# svm RBF
svm_rbf_digit_train <- train(
  x = num_28_train[, -1],
  y = num_28_train[, 1],
  method = "svmRadial",
  trControl = kfold
)
print(svm_rbf_digit_train)

svm_rbf_digit_test <- predict(svm_rbf_digit_train, newdata = num_28_test[, -1])

svm_rbf_digit_conf_matrix <- confusionMatrix(
  svm_rbf_digit_test,
  reference = num_28_test[, 1]
)

print(svm_rbf_digit_conf_matrix)

# Looking at this confusion matrix we can see that the model has an overall
# accuracy of 96%. This model also has a Sensitivity of 95% and specificty of
# 97%. This means the model has a higher tendency to choose 8 when 2,
# and 2 when 8.

# svm poly
svm_ply_digit_train <- train(
  x = num_28_train[, -1],
  y = num_28_train[, 1],
  method = "svmPoly",
  trControl = kfold
)

svm_ply_digit_test <- predict(svm_ply_digit_train, newdata = num_28_test[, -1])

svm_ply_digit_conf_mat <- confusionMatrix(
  svm_ply_digit_test,
  reference = num_28_test[, 1]
)
print(svm_ply_digit_conf_mat)

# Looking at this model we can see that the overall accuracy was 96%.
# This model also had a sensitivity of 94% and specificty of 98%.
# Meaning this model would predict 8 when 2 and 2 when 8 more often.

# 4
# only taking numeric values in iris
iris_num <- iris[, 1:4]
# procomp gives x (principal components), sdev (std dev of pc),
# rotation (rotation matrix)
pca_iris <- prcomp(iris_num, center = TRUE, scale. = TRUE)
# top two principal compenents
top_pc <- pca_iris$x[, 1:2]

ggplot(as.data.frame(top_pc), aes(x = PC1, y = PC2)) + geom_point() + labs(title = "PCA Visualization") # nolint

# 5
# getting principle componenets on number dataframes
pca_dig <- prcomp(train_dig[, -1], center = TRUE, scale. = TRUE)
# finding variance using std dev
var_pca <- pca_dig$sdev^2
# finding proportion of variance
prop_var <- var_pca / sum(var_pca)
# finding cumulative sum
cumulative_sum_pca <- cumsum(prop_var)
# finding which sum satisfies 95%
num_comp <- which(cumulative_sum_pca >= 0.95)[1]
print(num_comp)
# here we can see that we need 107 components to retain 95% of the varaince.

top_two <- pca_dig$x[, 1:2]
top_two
# subsetting data into each digit
num_1 <- top_two[train_dig[, 1] == 1, ]
num_3 <- top_two[train_dig[, 1] == 3, ]
num_4 <- top_two[train_dig[, 1] == 4, ]
num_7 <- top_two[train_dig[, 1] == 7, ]
num_8 <- top_two[train_dig[, 1] == 8, ]
# random sample of 50
num_1_fifty <- num_1[sample(nrow(num_1), size = 50), ]
num_3_fifty <- num_3[sample(nrow(num_3), size = 50), ]
num_4_fifty <- num_4[sample(nrow(num_4), size = 50), ]
num_7_fifty <- num_7[sample(nrow(num_7), size = 50), ]
num_8_fifty <- num_8[sample(nrow(num_8), size = 50), ]

# plots
plot(
  num_1_fifty[, 1],
  num_1_fifty[, 2],
  pch = 19,
  main = "number 1",
  xlab = "first pca",
  ylab = "second pca"
)

plot(
  num_3_fifty[, 1],
  num_3_fifty[, 2],
  pch = 19,
  main = "number 3",
  xlab = "first pca",
  ylab = "second pca"
)

plot(
  num_4_fifty[, 1],
  num_4_fifty[, 2],
  pch = 19,
  main = "number 4",
  xlab = "first pca",
  ylab = "second pca"
)

plot(
  num_7_fifty[, 1],
  num_7_fifty[, 2],
  pch = 19,
  main = "number 7",
  xlab = "first pca",
  ylab = "second pca"
)

plot(
  num_8_fifty[, 1],
  num_8_fifty[, 2],
  pch = 19,
  main = "number 8",
  xlab = "first pca",
  ylab = "second pca"
)

# looking at these plots you can not distinguish the numbers from two priniple
# components. looking at the varacince retention of two principle componenets
# is only 22%, so this is not too suprising.

# setting path for pandoc
Sys.setenv(RSTUDIO_PANDOC = "/opt/homebrew/bin")
# knit pdf
rmarkdown::render(
  "/Users/alex/stat_426_proj/stat_426_hw_11.r",
  output_format = "pdf_document"
)