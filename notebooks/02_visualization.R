# Rename variables for simplicity
names(iris.data)[names(iris.data) == "sepalenght"] <- "sl"
names(iris.data)[names(iris.data) == "petalenght"] <- "pl"
names(iris.data)[names(iris.data) == "sepalwidth"] <- "sw"
names(iris.data)[names(iris.data) == "petalwidth"] <- "pw"
View(iris.data)

# Data visualization
pairs(iris.data, pch = 16) 
ggpairs(data = iris.data, title = "Relationships between predictors and response",
        lower = list(continuous = wrap("points", alpha = 0.5, size = 0.1)))

# Convert 'class' to numeric factor
iris.data$class = as.factor(iris.data$class)
iris.data$class = as.numeric(iris.data$class)

# Correlation indices
X = iris.data[, -5]
corrplot(cor(X), method = 'color')
corrplot(cor(X), method = 'number')

# Response vs covariates plots
# Class vs Petal Length (pl)
plot(iris.data$pl, iris.data$class, pch = ifelse(iris.data$class == 1, 3, 4),
     col = ifelse(iris.data$class == 1, 'forestgreen', 'forestgreen'),
     xlab = 'pl', ylab = 'class', main = 'class vs. pl', lwd = 2, cex = 1.5)

# Class vs Petal Width (pw)
plot(iris.data$pw, iris.data$class, pch = ifelse(iris.data$class == 1, 3, 4),
     col = ifelse(iris.data$class == 1, 'red', 'red'),
     xlab = 'pw', ylab = 'class', main = 'class vs. pw', lwd = 2, cex = 1.5)

# Class vs Sepal Length (sl)
plot(iris.data$sl, iris.data$class, pch = ifelse(iris.data$class == 1, 3, 4),
     col = ifelse(iris.data$class == 1, 'blue', 'blue'),
     xlab = 'sl', ylab = 'class', main = 'class vs. sl', lwd = 2, cex = 1.5)

# Class vs Sepal Width (sw)
plot(iris.data$sw, iris.data$class, pch = ifelse(iris.data$class == 1, 3, 4),
     col = ifelse(iris.data$class == 1, 'purple', 'purple'),
     xlab = 'sw', ylab = 'class', main = 'class vs. sw', lwd = 2, cex = 1.5)
