library(readr)
library(dplyr)

# Load and preprocess data
data <- read_csv("Walmart_Sales.csv")
data$Date <- as.Date(data$Date, format = "%d-%m-%Y")

# Sort data by Store and Date to compute lag per store
data <- data %>%
  arrange(Store, Date) %>%
  group_by(Store) %>%
  mutate(Last_Week_Sales = lag(Weekly_Sales)) %>%
  ungroup()

# Remove rows where lagged sales is NA
data <- na.omit(data)

# Sort by Date to split chronologically
data <- data[order(data$Date), ]
split_index <- floor(0.9 * nrow(data))
train_data <- data[1:split_index, ]
test_data  <- data[(split_index + 1):nrow(data), ]

# Convert Store to a factor (categorical)
train_data$Store <- as.factor(train_data$Store)
test_data$Store <- as.factor(test_data$Store)

# Train model with lag feature and Store
model <- lm(log(Weekly_Sales) ~ Store + Holiday_Flag + Temperature + Fuel_Price + CPI + Unemployment + Last_Week_Sales,
            data = train_data)

# Summary of model
summary(model)

# Predict on test set
log_predictions <- predict(model, newdata = test_data)
predictions <- exp(log_predictions)

# RMSE
rmse <- sqrt(mean((predictions - test_data$Weekly_Sales)^2))

# R-squared
ss_res <- sum((test_data$Weekly_Sales - predictions)^2)
ss_tot <- sum((test_data$Weekly_Sales - mean(test_data$Weekly_Sales))^2)
r_squared <- 1 - (ss_res / ss_tot)

# Print performance
cat("Test RMSE:", rmse, "\n")
cat("Test R-squared:", r_squared, "\n")

# Residual diagnostics
residuals <- residuals(model)

# Plot histogram of residuals
hist(residuals, breaks = 30, main = "Histogram of Residuals", xlab = "Residuals")

# QQ plot of residuals
qqnorm(residuals)
qqline(residuals, col = "red")

