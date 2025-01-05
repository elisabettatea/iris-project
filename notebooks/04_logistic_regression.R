# Load necessary libraries
library(rms)
library(ResourceSelection)

# Filter out the "Iris-setosa" class to create a binary classification problem
iris.datalog <- iris.data[iris.data$class != "Iris-setosa", ]

# Rename variables for clarity
names(iris.datalog)[names(iris.datalog) == "sepalenght"] <- "sl"
names(iris.datalog)[names(iris.datalog) == "petalenght"] <- "pl"
names(iris.datalog)[names(iris.datalog) == "sepalwidth"] <- "sw"
names(iris.datalog)[names(iris.datalog) == "petalwidth"] <- "pw"

# Transform the 'class' variable into a binary variable
# 1 for "Iris-versicolor", 0 for other classes
iris.datalog$class_binaria <- ifelse(iris.datalog$class == "Iris-versicolor", 1, 0)

# Preview the updated dataset
head(iris.datalog)
str(iris.datalog)

# Remove the original 'class' variable since the binary variable is sufficient
iris.datalog <- iris.datalog[,-5]

# Split the dataset into training and testing sets
set.seed(42)  # Set seed for reproducibility
train_indices <- sample(1:nrow(iris.datalog), nrow(iris.datalog) * 0.7)  # 70% training set
train_data <- iris.datalog[train_indices, ]
test_data <- iris.datalog[-train_indices, ]

# Fit a logistic regression model with all predictors
model <- glm(class_binaria ~ sl + sw + pl + pw, data = iris.datalog, family = "binomial")
summary(model)
# Results: AIC = 21.899, NULL DEVIANCE = 138.629, RESIDUAL DEVIANCE = 11.899

# Fit a refined logistic regression model with selected predictors
model2 <- glm(class_binaria ~ sw + pl + pw, data = iris.datalog, family = "binomial")
summary(model2)
# Results: AIC = 21.266, NULL DEVIANCE = 138.629, RESIDUAL DEVIANCE = 13.266 (improvement)

# Compare models using ANOVA
anova(model2, model, test = "Chisq")

# Check Odds Ratios for predictors in model2
model.matrix(model2)[1:15, ]
exp(coef(model2)[2])  # Odds Ratio for sw
exp(coef(model2)[3])  # Odds Ratio for pl
exp(coef(model2)[4])  # Odds Ratio for pw

# Goodness-of-Fit check using Hosmer-Lemeshow test
hoslem.test(model2$y, fitted(model2), g = 6)  # Grouping into 6 bins
# p-value = 0.984, indicating the model is very valid.