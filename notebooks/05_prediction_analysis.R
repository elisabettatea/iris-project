# Load necessary libraries
library(pROC)
library(PRROC)
library(tidyverse)


# After verifying the model, proceed to adjust the classification threshold
# Choose a lower threshold to minimize error, as the optimal threshold is unknown

threshold = 0.1

# Actual values (ground truth)
true_values = iris.datalog$class_binaria

# Predicted values based on the chosen threshold
predicted_values = as.numeric(model2$fitted.values > threshold)  # 1 if > threshold, 0 otherwise

# Confusion matrix
confusion_matrix = table(true_values, predicted_values)
confusion_matrix

# The confusion matrix contains:
# - True Positives (TP): 1s classified as 1s
# - True Negatives (TN): 0s classified as 0s
# - False Positives (FP): 0s classified as 1s
# - False Negatives (FN): 1s classified as 0s

# Calculate misclassification error to minimize it

# Performance metrics:
# Accuracy: Percentage of correctly classified cases
accuracy = round(sum(diag(confusion_matrix)) / sum(confusion_matrix), 2)
accuracy

# Misclassification rate: Percentage of incorrectly classified cases
misclassification_rate = round((confusion_matrix[1, 2] + confusion_matrix[2, 1]) / sum(confusion_matrix), 2)
misclassification_rate

# Sensitivity (True Positive Rate)
sensitivity = confusion_matrix[2, 2] / (confusion_matrix[2, 1] + confusion_matrix[2, 2])
sensitivity

# Specificity (True Negative Rate)
specificity = confusion_matrix[1, 1] / (confusion_matrix[1, 2] + confusion_matrix[1, 1])
specificity

# Use the ROC curve to select the best threshold
fit2 = model2$fitted.values

PRROC_obj <- roc.curve(scores.class0 = fit2, weights.class0 = as.numeric(iris.datalog$class_binaria), curve = TRUE)
plot(PRROC_obj)

# From the ROC plot, observe that the curve is nearly optimal
# The threshold of 0.1 can be kept as it minimizes error
