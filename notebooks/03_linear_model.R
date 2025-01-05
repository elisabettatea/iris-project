# Fit first linear model
m1 = lm(class ~ ., data = iris.data)
summary(m1)
shapiro.test(residuals(m1))
# R2 = 0.9285, p-value = 0.4589

# Add interaction terms to improve R2
m2 = lm(class ~ . + sl*sw + pl*pw, data = iris.data)
summary(m2)
shapiro.test(residuals(m2))

qqnorm(m2$residuals)
qqline(m2$residuals, col = 'pink') # Tails are quite distant
# R2 = 0.9343, p-value = 0.105

# Analyze influential points in m2

# Plot residuals
plot(m2$res, ylab = "Residuals", main = "Plot of residuals")
sort(m2$res)
sort(m2$res)[c(1, 50)]  # Check the first and last residual

# 1. Leverage points (R2 = 0.9464, p-value = 7.6e-05)
r = m2$rank
p = 4
n = dim(iris.data)[1]
lev = hatvalues(m2)
watchout_points_lev = lev[which(lev > 2 * r / n)]
watchout_ids_lev = seq_along(lev)[which(lev > 2 * r / n)]

# Fit model without leverage points
id_to_keep = !(1:n %in% watchout_ids_lev)
m3 = lm(class ~ . + sl*pl + sl*sw + sl*pw + pl*sw + pl*pw + sw*pw, iris.data[id_to_keep, ])
summary(m3)
shapiro.test(residuals(m3))

# 2. Cook's distance
Cdist = cooks.distance(m2)
watchout_ids_Cdist = which(Cdist > 4 / (n - p - 1))
watchout_Cdist = Cdist[watchout_ids_Cdist]

# Fit model without Cook's distance leverage points
id_to_keep = !(1:n %in% watchout_ids_Cdist)
m4 = lm(class ~ . + sl*pl + sl*sw + sl*pw + pl*sw + pl*pw + sw*pw, iris.data[id_to_keep, ])
summary(m4)
shapiro.test(residuals(m4))

# 3. Standardized residuals
res_std = m2$res / summary(m2)$sigma
watchout_ids_rstd = which(abs(res_std) > 2)

# Fit model without standardized residuals
id_to_keep = !(1:n %in% watchout_ids_rstd)
m5 = lm(class ~ . + sl*pl + sl*sw + sl*pw + pl*sw + pl*pw + sw*pw, iris.data[id_to_keep, ])
summary(m5)
shapiro.test(residuals(m5))

# 4. Studentized residuals
stud = rstandard(m2)
watchout_ids_stud = which(abs(stud) > 2)

# Fit model without studentized residuals
id_to_keep = !(1:n %in% watchout_ids_stud)
m6 = lm(class ~ . + sl*pl + sl*sw + sl*pw + pl*sw + pl*pw + sw*pw, iris.data[id_to_keep, ])
summary(m6)
shapiro.test(residuals(m6))

# Model comparison
# Chosen model: m5 with standardized residuals (R2 = 0.9678, p-value = 0.08855)

qqnorm(m5$residuals)
qqline(m5$residuals, col = 'blue') # Tails are less distant

# Plot standardized residuals (needs correction)
plot(m5$fitted.values, res_std, ylab = "Standardized Residuals", main = "Standardized Residuals")
abline(h = c(-2, 2), lty = 2, col = 'orange')
points(m5$fitted.values[watchout_ids_rstd], 
       res_std[watchout_ids_rstd], col = 'red', pch = 16)
points(m5$fitted.values[watchout_ids_lev], 
       res_std[watchout_ids_lev], col = 'orange', pch = 16)
legend('topright', col = c('red', 'orange'), 
       legend = c('Standardized Residuals', 'Leverages'), pch = rep(16, 2), bty = 'n')

# Stepwise covariate selection
m5 = lm(class ~ . + sl*pl + sl*sw + sl*pw + pl*sw + pl*pw + sw*pw, iris.data[id_to_keep, ])
summary(m5)
shapiro.test(residuals(m5))

step(m5, direction = "both", trace = TRUE)
m7 = lm(formula = class ~ sl + sw + pl + pw + sl:sw + pl:pw + sw:pw, 
        data = iris.data[id_to_keep, ])
summary(m7)
shapiro.test(residuals(m7))

# After stepwise selection: R2 = 0.9685, p-value = 0.1171

# Manually remove 'pw' (as it has the largest estimate)
m8 = lm(formula = class ~ sl + sw + pl + sl:sw + pl:pw + sw:pw, 
        data = iris.data[id_to_keep, ])
summary(m8)
shapiro.test(residuals(m8))
# Final model: R2 = 0.959, p-value = 0.4222 (indicating normality of residuals).

# Q-Q plot for residuals
qqnorm(m8$residuals)
qqline(m8$residuals, col = 'violet')

# The model has reached a very good level (good fit).
