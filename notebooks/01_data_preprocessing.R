library(car)
library(ellipse)
library(faraway)
library(leaps)
library(MASS)
library(GGally)
library(rgl)
library(corrplot)

# Initial data visualization
View(iris.data)
summary(iris.data)
str(iris.data)
sum(is.na(iris.data)) # Check for missing values (no NAs found)