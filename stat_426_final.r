library(caret)
library(languageserver)
library(arm)
library(IMIFA)
library(ggplot2)
library(knitr)
library(rmarkdown)
library(kernlab)
library(e1071)


# ovr train
train_one_vs_rest <- function(data, features, target,
                              classes, kernel = "linear",
                              func) {
  models <- list()
  for (class in classes) {
    # Create binary response for current class
    binary_response <- ifelse(data[[target]] == class, class, "other")
    # Train the SVM model
    if (func == "svm") {
      model <- svm(
        data[, features],
        as.factor(binary_response),
        type = "C-classification",
        kernel = kernel,
        probability = TRUE
      )
    } else if (func == "qda") {
      # creating df to simplify input in train
      df <- data.frame(
        data[, features],
        binary = as.factor(binary_response)
      )

      model <- train(
        binary ~ .,
        df,
        # had to change to rda, because full dim qda struggles with ovr model
        method = "rda",
        # had to expand grid to minimize singularity issues
        tuneGrid = expand.grid(gamma = 1, lambda = 1),
        # had to turn resample to "none" to get runtime down
        trControl = trainControl(method = "none"),
        preProcess = c("center", "scale")
      )
    }
    models[[class]] <- model
  }
  models
}

# ovr predict
predict_one_vs_rest <- function(models, data, features) {

  # Collect probability vectors for each class
  prob_list <- lapply(names(models), function(class_name) {
    model <- models[[class_name]]
    if ("svm" %in% class(model)) {
      pred <- predict(model, data[, features], probability = TRUE)
      probs <- attr(pred, "probabilities")
      # Ensure column for this class exists
      if (!(class_name %in% colnames(probs))) {
        stop(paste(
          "SVM model does not provide probability column for class:",
          class_name
        ))
      }
      return(probs[, class_name])
    }
    pred <- predict(model, data[, features], type = "prob")
    # Some caret models return tibble; convert to data.frame
    pred <- as.data.frame(pred)
    # Ensure probability column exists
    if (!(class_name %in% colnames(pred))) {
      stop(paste(
        "Caret model does not provide probability column for class:",
        class_name
      ))
    }

    pred[, class_name]
  })

  # Convert output list to clean numeric matrix
  prob_matrix <- do.call(cbind, prob_list)
  colnames(prob_matrix) <- names(models)
  # Select class with highest probability
  max_idx <- max.col(prob_matrix, ties.method = "first")
  predicted_classes <- names(models)[max_idx]

  predicted_classes
}

# ovo train
train_one_vs_one <- function(data, features, target, func, kernel = "linear") {
  models <- list()
  classes <- unique(data[[target]])
  class_pairs <- combn(classes, 2, simplify = FALSE)
  # Iterate over each pair of classes
  for (pair in class_pairs) {
    class1 <- pair[1]
    class2 <- pair[2]
    # Subset the data for the current pair of classes
    subset_data <- data[data[[target]] %in% pair, ]
    if (func == "svm") {
      # Train the SVM model
      model <- svm(
        subset_data[, features],
        as.factor(subset_data[[target]]),
        type = "C-classification",
        kernel = kernel
      )
    } else if (func == "qda") {
      model <- train(
        x = subset_data[, features],
        y = as.factor(subset_data[[target]]),
        # had to change to rda, becasue qda struggled ov
        method = "rda",
        # had to expand grid to minimize singularity issues
        tuneGrid = expand.grid(gamma = 1, lambda = 1),
        # had to turn resample to "none" to get runtime down
        trControl = trainControl(method = "none"),
        preProcess = c("center", "scale")
      )
    }
    # Store the model with a label indicating the classes it discriminates
    models[[paste(class1, class2, sep = "_vs_")]] <- model
  }
  models
}

# ovo predict
predict_one_vs_one <- function(models, data, features, target) {

  classes <- sort(unique(data[[target]]))

  # Initialize vote matrix
  votes <- matrix(
    0,
    nrow = nrow(data),
    ncol = length(classes)
  )
  colnames(votes) <- as.character(classes)

  # Iterate over models
  for (model_name in names(models)) {

    model <- models[[model_name]]

    pair_prediction <- predict(model, data[, features])
    pair_prediction <- as.character(pair_prediction)

    for (i in seq_along(pair_prediction)) {
      votes[i, pair_prediction[i]] <-
        votes[i, pair_prediction[i]] + 1
    }
  }

  # Majority vote
  predicted_classes <- apply(votes, 1, function(row) {
    colnames(votes)[which.max(row)]
  })
  predicted_classes
}


data(USPSdigits)
# 1
# training data
train_dig <- USPSdigits$train
# labels
train_dig_labels <- train_dig[, 1]
train_dig_labels <- as.factor(train_dig_labels)
# number data
train_dig_data <- train_dig[, -1]
# test data
test_dig <- USPSdigits$test
test_dig_labels <- test_dig[, 1]
test_dig_labels <- as.factor(test_dig_labels)

test_dig_data <- test_dig[, -1]

sel_dig <- c(1, 3, 7, 8)

selected_digits_train <- subset(
  train_dig,
  train_dig_labels == 1 | train_dig_labels == 3 |
    train_dig_labels == 7 | train_dig_labels == 8
)

selected_digits_test <- subset(
  test_dig,
  test_dig_labels == 1 | test_dig_labels == 3 |
    test_dig_labels == 7 | test_dig_labels == 8
)


# gettig columns for labels and features
features_sel_dig <- names(selected_digits_train[, -1])
features_sel_dig

features_sel_dig_test <- names(selected_digits_test[, -1])
dig_classes <- levels(as.factor(sel_dig))

# using qda (rda)
qda_ovo_dig <- train_one_vs_one(
  selected_digits_train,
  features_sel_dig,
  "V1",
  func = "qda"
)

qda_ovo_pred <- predict_one_vs_one(
  qda_ovo_dig,
  selected_digits_test,
  features_sel_dig,
  "V1"
)
# confusion matrix
qda_ovo_conf_mat <- confusionMatrix(
  as.factor(qda_ovo_pred),
  as.factor(selected_digits_test[, 1])
)

qda_ovo_conf_mat
# looking at this table we can see that the diagonals are large.
# the overall accuracy of this model is 92%, with the lowest
# sensitivity in class 3. Where is misclassified digit 8 the most.

# svm with linear
svm_pol_ovo_dig <- train_one_vs_one(
  selected_digits_train,
  features_sel_dig,
  "V1",
  func = "svm",
  kernel = "polynomial"
)
svm_pol_ovo_dig
# predicted svm linear
svm_pol_ovo_pred <- predict_one_vs_one(
  svm_pol_ovo_dig,
  selected_digits_test,
  features_sel_dig,
  "V1"
)
svm_pol_ovo_pred
# confusion matrix
svm_pol_ovo_conf_mat <- confusionMatrix(
  as.factor(svm_pol_ovo_pred),
  as.factor(selected_digits_test[, 1])
)
svm_pol_ovo_conf_mat
# looking at this table we can see that the diagonals are large.
# the overall accuracy of this model was 96%. and the lowest
# sensitivity was 93% in class 8, with the most misclassifactions
# in class 3, which is not too suprising due to there similar shape.

# svm with RBF
svm_rbf_ovo_dig <- train_one_vs_one(
  selected_digits_train,
  features_sel_dig,
  "V1",
  func = "svm",
  kernel = "radial"
)
# predicted svm rbf
svm_rbf_ovo_pred <- predict_one_vs_one(
  svm_rbf_ovo_dig,
  selected_digits_test,
  features_sel_dig,
  "V1"
)
# confusion matrix
svm_rbf_ovo_conf_mat <- confusionMatrix(
  as.factor(svm_rbf_ovo_pred),
  as.factor(selected_digits_test[, 1])
)

svm_rbf_ovo_conf_mat
# looking at this table we can see that the diagonlas are large.
# we can also see the overall accuracy is 96% which is the same
# as the polynomial kernel, but the sensitivitys are much higher
# with the lowest at 93% in class 8. Not suprisingly this incorrectly
# classified 3 as number 8.

# one vs rest

qda_ovr_dig <- train_one_vs_rest(
  selected_digits_train,
  features_sel_dig,
  "V1",
  dig_classes,
  func = "qda"
)
# predict
qda_ovr_dig_pred <- predict_one_vs_rest(
  qda_ovr_dig,
  selected_digits_test,
  features_sel_dig
)
# confusion matrix
qda_ovr_table <- confusionMatrix(
  as.factor(qda_ovr_dig_pred),
  as.factor(selected_digits_test[, 1])
)
qda_ovr_table
# looking at this table we can se that the diagonals are large.
# the overall accuracy of this model is 91%. and the class 3 & 8
# has the lowest sensitivity and specificity.

# svm poly kernel
svm_pol_ovr_train <- train_one_vs_rest(
  selected_digits_train,
  features_sel_dig,
  "V1",
  dig_classes,
  kernel = "polynomial",
  func = "svm"
)
svm_pol_ovr_train

svm_ovr_pol_pred <- predict_one_vs_rest(
  svm_pol_ovr_train,
  selected_digits_test,
  features_sel_dig
)

levels(as.factor(svm_ovr_pol_pred))

svm_poly_mat <- confusionMatrix(
  as.factor(svm_ovr_pol_pred),
  as.factor(selected_digits_test[, 1])
)
svm_poly_mat
# looking at this table we can see that this model has very large
# diagonls. This model misclassified 3 and 7 as label 8 the most.
# this model has a 97% accuracy. class 7 has the lowest sensitivy
# at 95% meaning it predicting other classes at 7 when not more
# than the other digits. This was slightly higher than the ovr method


# svm rbf kernel
svm_rbf_ovr_train <- train_one_vs_rest(
  selected_digits_train,
  features_sel_dig,
  "V1",
  dig_classes,
  kernel = "radial",
  func = "svm"
)

svm_rbf_ovr_pred <- predict_one_vs_rest(
  svm_rbf_ovr_train,
  selected_digits_test,
  features_sel_dig
)

svm_rbf_ovr_conf_mat <- confusionMatrix(
  as.factor(svm_rbf_ovr_pred),
  as.factor(selected_digits_test[, 1])
)
svm_rbf_ovr_conf_mat
# Looking at this table we can see that the diagonals are large.
# We can also see that the overall accuracy is 96%. Class 8 had
# the lowest sensitivity at 93%, we can see that all the
# misclassifaction in the the class were guessed as 3. Which is understable.

# 2
# pca
pca_dig <- prcomp(selected_digits_train[, -1], center = TRUE, scale. = TRUE)
# variance
var_pca <- pca_dig$sdev^2
# proportion of variance
prop_var <- var_pca / sum(var_pca)
# cummulative sum
cum_sum_var <- cumsum(prop_var)
# finding 95%
num_comps <- which(cum_sum_var >= 0.95)[1]
print(num_comps)

# new training data
pca_dig_train <- selected_digits_train[, 1:num_comps]
pca_dig_train

pca_dig_test <- selected_digits_test[, 1:num_comps]

pca_features <- names(pca_dig_train[, -1])

# using qda
pca_qda_ovo_dig <- train_one_vs_one(
  pca_dig_train,
  pca_features,
  "V1",
  func = "qda"
)

# predict
pca_qda_ovo_pred <- predict_one_vs_one(
  pca_qda_ovo_dig,
  pca_dig_test,
  pca_features,
  "V1"
)
# confusion matrix
pca_qda_ovo_conf_mat <- confusionMatrix(
  as.factor(pca_qda_ovo_pred),
  as.factor(selected_digits_test[, 1])
)
pca_qda_ovo_conf_mat
# looking at this table we can see that the diagonals are
# relativily large. It is worse than the full feature set
# with an overall accuracy of 79%. with the lowest sensitivity
# in class 7 at 63% where most of the miclassifactions went to
# digit 3.

# svm with poly
pca_svm_pol_ovo_dig <- train_one_vs_one(
  pca_dig_train,
  pca_features,
  "V1",
  func = "svm",
  kernel = "polynomial"
)
# predicted svm poly
pca_svm_pol_ovo_pred <- predict_one_vs_one(
  pca_svm_pol_ovo_dig,
  pca_dig_test,
  pca_features,
  "V1"
)

# confusion matrix
pca_svm_lin_ovo_conf_mat <- confusionMatrix(
  as.factor(pca_svm_pol_ovo_pred),
  as.factor(selected_digits_test[, 1])
)

pca_svm_lin_ovo_conf_mat
# looking at this table we can see that the diagonals are much
# larger than the previous model. This model also had a better
# overall accuracy at 86%. The lowest sensitivity was in the
# same class 7 as the qda model but higher at 72%, with the
# most misclassifactions going to class 3 again.

# svm with RBF
pca_svm_rbf_ovo_dig <- train_one_vs_one(
  pca_dig_train,
  pca_features,
  "V1",
  func = "svm",
  kernel = "radial"
)
# predicted svm rbf
pca_svm_rbf_ovo_pred <- predict_one_vs_one(
  pca_svm_rbf_ovo_dig,
  pca_dig_test,
  pca_features,
  "V1"
)

pca_svm_rbf_ovo_conf <- confusionMatrix(
  as.factor(pca_svm_rbf_ovo_pred),
  as.factor(selected_digits_test[, 1])
)
pca_svm_rbf_ovo_conf
# looking at this table we can see that the diagonals are large.
# We can also see that the overall accuracy is at 90% which is
# lower that the polynomial kernel but higher than qda method.
# the lowest sensitivity is now in class 3 at 83% with most
# misclassifaction going to class 7 which is consistent with
# all the ovo reduced dim feature sets.

pca_ovr_qda <- train_one_vs_rest(
  pca_dig_train,
  pca_features,
  "V1",
  dig_classes,
  func = "qda"
)

pca_pred_qda_ovr <- predict_one_vs_rest(
  pca_ovr_qda,
  pca_dig_test,
  pca_features
)

pca_ovr_qda_confmat <- confusionMatrix(
  as.factor(pca_pred_qda_ovr),
  as.factor(pca_dig_test[, 1])
)
pca_ovr_qda_confmat
# looking at this model we can see that the diagonals are smaller
# than the full data set. The overall accuracy of this model is less
# at 77%. class 3 had the worst sensitivity at 61%, with most of the
# misclassifacions being in calss 7, which is not what I expected. This
# also differs from the full dataset.

# ploy kernel
pca_svm_pol_ovr_train <- train_one_vs_rest(
  pca_dig_train,
  pca_features,
  "V1",
  dig_classes,
  kernel = "polynomial",
  func = "svm"
)
pca_svm_ovr_pol_pred <- predict_one_vs_rest(
  pca_svm_pol_ovr_train,
  pca_dig_test,
  pca_features
)
# confusion matrix
pca_svm_ovr_pol_conf_mat <- confusionMatrix(
  as.factor(pca_svm_ovr_pol_pred),
  as.factor(pca_dig_test[, 1])
)
pca_svm_ovr_pol_conf_mat
# looking at the table we can see that the diagonals are fairly
# large. The overall accuracy of this model is 90%, which is much
# higher than the qda/rda model, but less than the full feature set.
# The lowest sensitivity was in class 7, most of the misclassifactions
# where in class 3.

# svm rbf kernel
pca_svm_rbf_ovr_train <- train_one_vs_rest(
  pca_dig_train,
  pca_features,
  "V1",
  dig_classes,
  kernel = "radial",
  func = "svm"
)

pca_svm_rbf_ovr_pred <- predict_one_vs_rest(
  pca_svm_rbf_ovr_train,
  pca_dig_test,
  pca_features
)

svm_rbf_ovr_conf_mat <- confusionMatrix(
  as.factor(svm_rbf_ovr_pred),
  as.factor(pca_dig_test[, 1])
)
svm_rbf_ovr_conf_mat
# looking at this table we can see that the the diagonls are large.
# The overall accuracy of this model was 96%. which is the highest
# of all the lower dim feature sets. the lowest sensitivity was in
# class 8 with most of the misclassifactions going to class 3.

# 3
# data needs to be arragened labels [, 1] & data [, -1]
center_plotter <- function(data, k) {
  # stops random init
  set.seed(123)
  # kmeans using euclidian distance at the metric
  centers_data <- kmeans(
    data[, -1],
    center = k
  )
  par(mfcol = c(1, k))
  for (i in 1:k) {
    center_matrix <- centers_data$centers[i, ]
    img <- matrix(center_matrix, nrow = 16, byrow = TRUE)
    # transpose to see digit images
    img <- t(apply(img, 2, rev))
    plot_title <- paste0("Cluster Center", i)
    image(img, col = gray.colors(256), axes = FALSE, main = plot_title)
  }
  # creates table of to see which cluster got which label
  cluster_table <- table(data[, 1], centers_data$cluster)
  print(cluster_table)

  col_sum <- colSums(cluster_table)
  col_max <- apply(cluster_table, 2, max)

  for (i in 1:k) {
    maj_prop <- col_max[i] / col_sum[i]
    max_prop <- max(cluster_table[, i])
    maj_row <- rownames(cluster_table)[which(cluster_table[, i] == max_prop)]

    cat("Cluster:", colnames(cluster_table)[i], "\n")
    cat("  Majority label:", maj_row, "\n")
    cat("  Majority proportion:", round(maj_prop, 3), "\n\n")
  }
}

center_plot_3 <- center_plotter(selected_digits_train, 3)
# As we can see in the plot are clusters are digts cluster 1: 1,
# cluster 2: 8, cluster 3: 7. the largest proportion of majority labels
# was in cluster 1. Where it labeled all but 1 digit 1 correctly. Not too
# suprisingly almost all 3's where classified as number 8.

center_plot_4 <- center_plotter(selected_digits_train, 4)
# As we can see in the plot the clusters are cluster 1: 1, cluster 2: 8
# cluster 3: 7, and cluster 4: 3. Here was added the digit 3 into in the
# prevoius clusters. The largest proportion of majority labels was in
# cluster 1. Where it labeled all but three 1's as 1. there was a decent
# number of threes labeled as 8s and vice versa. Over all this model did
# a decent job of labeling everything.
center_plot_5 <- center_plotter(selected_digits_train, 5)
# As we can see in the plot we have all the same clusters with a duplicate
# 3. The proportion of majority labels is still for cluster corresponding
# to digit 1. it has mislabeled three ones in cluster 2 which corresponds to
# digit 8. The label 3 is pretty evenly spit between its two clusters 354 / 272.

center_plot_6 <- center_plotter(selected_digits_train, 6)
# As we can see in the plot we now have a duplicate 7. this leaves are clusters
# as 1, 8, 3, 7, 7, 3. Cluster one is still the largest majority proportion.
# The labels are split fairly evenly between the two duplicate 3 and 7s. Overall
# this model seems to do a better job at classifying 3 and 8 which cleans up
# the over all error in the model predictions.

# 4
# table needs to be in label vs clusters
maj_lab <- function(table) {
  col_sum <- colSums(table)
  col_max <- apply(table, 2, max)

  clusters <- dim(table)[2]
  for (i in 1:clusters) {
    maj_prop <- col_max[i] / col_sum[i]
    maj_row <- rownames(table)[which(table[, i] == max(table[, i]))]

    cat("Cluster:", colnames(table)[i], "\n")
    cat("  Majority label:", maj_row, "\n")
    cat("  Majority proportion:", round(maj_prop, 4), "\n\n")
  }
}

# h clust method
# clear plot from above
dev.off()
# making a duplicate dataset to not overwrite data for the models above
sel_dig_hclust_comp <- selected_digits_train
# distances
dig_dist_comp <- dist(sel_dig_hclust_comp[, -1])
set.seed(123)
h_comp <- hclust(dig_dist_comp, method = "complete")
h_comp
plot(h_comp)
# looking at the plot it seems that h = 20 has the best level

# clear plot
dev.off()
# cutting tree at ideal level
h_comp_cut <- cutree(h_comp, h = 20)
# assign cluster labels to training data
sel_dig_hclust_comp$cluster <- h_comp_cut
# labels vs cluster
h_comp_table <- table(sel_dig_hclust_comp[, 1], sel_dig_hclust$cluster)
h_comp_table
# majority variance for complete linkage
maj_lab_comp <- maj_lab(h_comp_table)

# looking at this table we can see that all 1s went into cluster 3. Assuming
# cluster 3 is label 1 without printing 1002 grayscale images, this model did
# a good job with that classifaction. cluster 7 had the highest prop with maj
# label of 3. Looking at this model after using kmeans above we can assume that
# clusster 6, 7, 8 belong to label 3 as its second highest label was 8.

# using ward
# creating new dataset to not overwrite previous
sel_dig_hclust_ward <- selected_digits_train
# getting distance
dist_ward <- dist(sel_dig_hclust_ward[, -1])
set.seed(123)

h_ward <- hclust(dist_ward, method = "ward.D2")
h_ward
plot(h_ward)
# looing at this plot it seems that h ~ 88 has the ideal level of clusters.
# compared to the other two dendograms this method appears to produce the
# best/easiest one to visually interperate.
h_ward_cut <- cutree(h_ward, h = 88)
# assign cluster lables to training data
sel_dig_hclust_ward$cluster <- h_ward_cut
# labels vs cluster
h_ward_table <- table(sel_dig_hclust_ward[, 1], sel_dig_hclust_ward$cluster)
h_ward_table
# looking at this table h = 88 gives us 7 clusters which is what we were aiming
# for.
# clear plot
dev.off()
# majority labels ward
maj_lab_ward <- maj_lab(h_ward_table)
# Here we can see that cluster 4 had perfect classifaction of digit 1 (assuming
# that this is actually digit one without producing 558 grayscale images).
# cluster 5 also had perfect clustering of digit 7. Compared to complete linkage
# ward.d2 has much better majority labels. With only minor misclassifaction of
# digits 8 and 3.

# using avg
# creating data set to not overwrite above models
sel_dig_hclust_avg <- selected_digits_train
# distances
hclust_avg_dist <- dist(sel_dig_hclust_avg[, -1])
set.seed(123)
h_avg <- hclust(hclust_avg_dist, method = "average")
h_avg
plot(h_avg)
# looking at this dendogram it is not very clear where to cut it. Instead
# I gave it 8 clusters which is close to what the other two methods produced.
h_avg_cut <- cutree(h_avg, k = 8)
sel_dig_hclust_avg$cluster <- h_avg_cut
# labels vs clusters
h_avg_table <- table(sel_dig_hclust_avg[, 1], sel_dig_hclust_avg$cluster)
h_avg_table
# looking at this table we can see that nearly all data points were classified
# in cluster 1. While the other ~50 labels where spread between the other 7
# clusters (most likley for label 8).

# majority labels
maj_lab_avg <- maj_lab(h_avg_table)
# looking at the table nearly every label was put into cluster 1. The proportion
# of majority labels was only 35% of the cluster. This is pretty bad and shows
# that we would need many cluster to clean up this tree. There are a few cols
# with perfect classifaction which does not accurtly show the model without
# looking at the table where they only have a few labels.

# clear plot
dev.off()

# 5
house_data <- read.csv("/Users/alex/stat_426_proj/housing.csv")
# inspecting csv
head(house_data, 5)
# getting data structure
str(house_data)
# looking at the data structure the quantative cols are currently beds,
# baths, footage, lattitude, longitude. The qualitative cols are price, zip
# city, state, price.per.sq.ft, type. However we can massage this data set
# to get price and price.per.sq.ft to be quantative.


# stripping leading "$" from prices column
house_data$Price <- gsub("^\\$", "", house_data$Price)
# dropping commas
house_data$Price <- gsub(",", "", house_data$Price)
# converting to qualatative value
house_data$Price <- as.numeric(house_data$Price)

# stripping "$"
house_data$Price.Per.Sq..Ft. <- gsub("^\\$", "", house_data$Price.Per.Sq..Ft.)
# dropping commas if there are any
house_data$Price.Per.Sq..Ft. <- gsub(",", "", house_data$Price.Per.Sq..Ft.)
# convert to qualatative
house_data$Price.Per.Sq..Ft. <- as.numeric(house_data$Price.Per.Sq..Ft.)

# for this model we will not be using price per sq ft becasue we we can get this
# using this we could easily get to price since this model has sqft

# checking data structure after massage
str(house_data)
# as we can see this data is cleaned up adding price & price per as qualatative
# features in our data set.

house_data_clean <- subset(
  house_data,
  select = c(Price, Beds, Baths, Footage, Latitude, Longitude)
)
# checking data structure
str(house_data_clean)
# making train test split with price as our predictor variable
train_test_split <- createDataPartition(
  house_data_clean$Price,
  p = 0.7,
  list = FALSE
)
# making train and test sets
train_house <- house_data_clean[train_test_split, ]
test_house <- house_data_clean[- train_test_split, ]

# manually preprocessing data to work with feature selection version
# of linear reggression only on features.
pp_train <- preProcess(train_house[, -1], method = c("center", "scale"))
train_house[, -1] <- predict(pp_train, train_house[, -1])
# run on train to ensure that the scale predictions are consistent
test_house[, -1] <- predict(pp_train, test_house[, -1])

# 10 k-cross folds
kfold <- trainControl(method = "cv", number = 10)
# using feat selection
set.seed(123)
freg_model <- train(
  Price ~ (.)^2,
  data = train_house,
  method = "leapSeq",
  trControl = kfold
)
freg_model
# getting model coeffs
freg_coef <- coef(freg_model$finalModel, unlist(freg_model$bestTune))
freg_coef
# looking at this we can see that the model has kept footage,
# Lattitude, secnod order beds:footage, baths:footage.

# the RMSE on the traning set is 784889.5 with optimal predictors = 4
freg_predict <- predict(freg_model, newdata = test_house)

# getting rmse
root_error_freg <- RMSE(test_house$Price, freg_predict)
root_error_freg
# the root mean square error is 721836, which is slightly less than
#  the training set.

freg_resid <- data.frame(
  Actual = test_house$Price,
  Predicted = freg_predict
)

freg_resid$Residual <- freg_resid$Actual - freg_resid$Predicted

# residula plot
ggplot(freg_resid, aes(x = Predicted, y = Residual)) +
  geom_point(alpha = 0.4, color = "#5e0b66") +
  geom_hline(yintercept = 0, color = "#094b09", linewidth = 1) +
  geom_smooth(method = "loess", se = FALSE, color = "black") +
  theme_minimal() +
  labs(
    title = "Residuals vs Predicted Values",
    subtitle = "Best subset regression (leapSeq)",
    x = "Predicted Price",
    y = "Residual (Actual - Predicted)"
  )

# looking at this plot we can see most predictions centered around
# zero and then there are a few large errors most likely from outlier
# houses in the data set.

# clear plot
# dev.off()
# using ridge regression instead
set.seed(123)
rid_reg_model <- train(
  Price ~ (.)^2,
  data = train_house,
  # using glmnet to get coef
  method = "glmnet",
  # alpha = 0 usese rigde regression
  tuneGrid = expand.grid(
    alpha = 0,
    lambda = seq(0.01, 0.5, length = 20)
  ),
  trControl = kfold
)
rid_reg_model
rid_coef <- coef(rid_reg_model$finalModel, s = rid_reg_model$bestTune$lambda)
rid_coef
# here we can see that this model keeps all of the features. The
# two lowest features where beds:footage and beds:lattitude.

# the RMSE on the traning set is 692937.2 with optimal lambda = 0.5
# this error is better than the prevoius feature selection model.
rid_reg_predict <- predict(rid_reg_model, newdata = test_house)

# getting rmse
root_error_rid_reg <- RMSE(test_house$Price, rid_reg_predict)
root_error_rid_reg
# the root mean square error is 597953, which is less than the
# training set, and less than the feature selection model.

rid_resid <- data.frame(
  Actual = test_house$Price,
  Predicted = rid_reg_predict
)

rid_resid$Residual <- rid_resid$Actual - rid_resid$Predicted

# residula plot
ggplot(rid_resid, aes(x = Predicted, y = Residual)) +
  geom_point(alpha = 0.4, color = "#5e0b66") +
  geom_hline(yintercept = 0, color = "#094b09", linewidth = 1) +
  geom_smooth(method = "loess", se = FALSE, color = "black") +
  theme_minimal() +
  labs(
    title = "Residuals vs Predicted Values",
    subtitle = "Best subset regression (ridge)",
    x = "Predicted Price",
    y = "Residual (Actual - Predicted)"
  )

# the loss line on this plot seems to be smoother than feature selection
# this plot also has most of the errors around zero and then a few
# outlier error points.

# clear plot
# dev.off()
# using lasso ridge
lasso_reg_model <- train(
  Price ~ (.)^2,
  data = train_house,
  method = "glmnet",
  trControl = kfold,
  # adjusting to find best range for function
  # alpha = 1 uses lasso reg
  tuneGrid = expand.grid(
    alpha = 1,
    lambda = seq(0.2, 0.7, length = 20)
  )
)
lasso_reg_model

lasso_coef <- coef(
  lasso_reg_model$finalModel,
  s = lasso_reg_model$bestTune$lambda
)
lasso_coef
# in this we can see that the two lowest coeffs are beds, and
# footage:lattitude.

# the RMSE on the traning set is 665231 with optimal lambda = 0.7
# this error is better than the prevoius feature selection model.
lasso_reg_predict <- predict(lasso_reg_model, newdata = test_house)

# getting rmse
root_error_lasso_reg <- RMSE(test_house$Price, lasso_reg_predict)
root_error_lasso_reg
# the root mean square error is 818616, which is more than the training
# set. This is worse than the feature selection, and worse than ridge

las_resid <- data.frame(
  Actual = test_house$Price,
  Predicted = lasso_reg_predict
)

las_resid$Residual <- las_resid$Actual - las_resid$Predicted

# residula plot
ggplot(las_resid, aes(x = Predicted, y = Residual)) +
  geom_point(alpha = 0.4, color = "#5e0b66") +
  geom_hline(yintercept = 0, color = "#094b09", linewidth = 1) +
  geom_smooth(method = "loess", se = FALSE, color = "black") +
  theme_minimal() +
  labs(
    title = "Residuals vs Predicted Values",
    subtitle = "Best subset regression (lasso)",
    x = "Predicted Price",
    y = "Residual (Actual - Predicted)"
  )
# this plot seems to be very similar to the other plots, but seems to
# have a few more outlier data points as price increases.

# sorting the abs value of the coef weights to get the three lowest weights
freg_feat <- data.frame(sort(abs(freg_coef))[1: 3])

rid_feat <- data.frame(sort(abs(rid_coef[, 1]))[1: 3])

las_feat <- data.frame(sort(abs(lasso_coef[, 1]))[1: 3])

# table comparing all three
error_frame <- data.frame(
  feature_selction = c(root_error_freg, rownames(freg_feat)),
  ridge = c(root_error_rid_reg, rownames(rid_feat)),
  lasso = c(root_error_lasso_reg, rownames(las_feat))
)
error_frame
# looking at this table it seems that ridge regeression has
# the lowest error and will be the best model to use for this dataset.
# We can also see that the all the models did not get much from the
# lattitide and longitude features in the dataset.