# Setting Working Directory
getwd()
setwd('/Users/jaikrishna/Documents/College/Data Programming/Walmart Data/Data')

# Importing the data
stores <- read.csv("stores.csv")
head(stores, 3)

features <- read.csv("features.csv")
head(features, 3)

sales <- read.csv("train.csv")
head(sales, 3)

# Joining Data
library(dplyr)

# Merge the data frames
sales1 <- merge(sales, stores, by = "Store", all.x = TRUE)
sales2 <- merge(sales1, features, by = c("Store", "Date", "IsHoliday"), all.x = TRUE)

# Group by at week year level
sales_by_store <- sales2 %>% group_by(Store, Type) %>% summarize(mean_sales = mean(Weekly_Sales))
size_of_store <- sales2 %>% group_by(Store, Type) %>% summarize(mean_size = mean(Size))
count_of_dept <- sales2 %>% group_by(Store, Type) %>% summarize(n_dept = n_distinct(Dept))

# Merge the data frames
data <- inner_join(sales_by_store, size_of_store, by = c("Store", "Type"))
df <- inner_join(data, count_of_dept, by = c("Store", "Type"))

library(randomForest)

# Encode A as 1, B as 2, and C as 3
df$Type <- factor(df$Type, levels = c("A", "B", "C"), labels = c(1, 2, 3))

# Load required packages
library(randomForest)
library(caret)

# Split data set into training and testing sets
set.seed(123) # for reproducibility
trainIndex <- sample(1:nrow(df), 0.6*nrow(df))

train <- df[trainIndex,]
test <- df[-trainIndex,]

# Perform random forest classification
rf_model <- randomForest(Type ~ mean_sales + mean_size + n_dept, data = train)

# Make predictions on test set
pred_rf <- predict(rf_model, test)

# Obtain confusion matrix
cm <- table(test$Type, pred_rf)
print(cm)

# Calculate accuracy, precision, and recall
accuracy <- sum(diag(cm)) / sum(cm)
precision <- diag(cm) / rowSums(cm)
recall <- diag(cm) / colSums(cm)
accuracy
precision
recall

# Get row and column names
rows <- rownames(cm)
cols <- colnames(cm)

# Convert confusion matrix to a data frame
cm_df <- data.frame(actual = rep(rows, each = length(cols)),
                    predicted = cols,
                    value = as.vector(cm))

# Create plot
ggplot(cm_df, aes(x = actual, y = predicted, fill = value)) +
  geom_tile() +
  geom_text(aes(label = value), size = 16, color = "white") +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(x = "Actual", y = "Predicted", title = "Confusion Matrix") +
  theme(plot.title = element_text(hjust = 0.5))

# Load required packages
library(e1071)
library(caret)

# Train naive bayes model
nb_model <- naiveBayes(Type ~ mean_sales + mean_size + n_dept, data = train)

# Make predictions on test set
pred_nb <- predict(nb_model, test)

# Obtain confusion matrix
cm <- table(test$Type, pred)
print(cm)

# Get row and column names
rows <- rownames(cm)
cols <- colnames(cm)

# Convert confusion matrix to a data frame
cm_df <- data.frame(actual = rep(rows, each = length(cols)),
                    predicted = cols,
                    value = as.vector(cm))

# Create plot
ggplot(cm_df, aes(x = actual, y = predicted, fill = value)) +
  geom_tile() +
  geom_text(aes(label = value), size = 16, color = "white") +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(x = "Actual", y = "Predicted", title = "Confusion Matrix") +
  theme(plot.title = element_text(hjust = 0.5))

# Calculate accuracy, precision, and recall
accuracy <- sum(diag(cm)) / sum(cm)
precision <- diag(cm) / rowSums(cm)
recall <- diag(cm) / colSums(cm)
accuracy
precision
recall

# Perform KNN classification
k <- 5 # set the number of neighbors
pred_knn <- knn(train = train[, c("mean_sales", "mean_size", "n_dept")], 
                 test = test[, c("mean_sales", "mean_size", "n_dept")], 
                 cl = train$Type, 
                 k = k)

# Obtain confusion matrix
cm <- table(test$Type, pred_knn)
print(cm)

# Obtain classification statistics
accuracy <- sum(diag(cm)) / sum(cm)
precision <- diag(cm) / colSums(cm)
recall <- diag(cm) / rowSums(cm)
print(paste0("Accuracy: ", accuracy*100))
print(paste0("Precision: ", precision))
print(paste0("Recall: ", recall))

# Get row and column names
rows <- rownames(cm)
cols <- colnames(cm)

# Convert confusion matrix to a data frame
cm_df <- data.frame(actual = rep(rows, each = length(cols)),
                    predicted = cols,
                    value = as.vector(cm))

# Create plot
ggplot(cm_df, aes(x = actual, y = predicted, fill = value)) +
  geom_tile() +
  geom_text(aes(label = value), size = 16, color = "white") +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(x = "Actual", y = "Predicted", title = "Confusion Matrix") +
  theme(plot.title = element_text(hjust = 0.5))

# Obtain classification statistics
accuracy <- sum(diag(cm)) / sum(cm)
precision <- diag(cm) / colSums(cm)
recall <- diag(cm) / rowSums(cm)
print(paste0("Accuracy: ", accuracy))
print(paste0("Precision: ", precision))
print(paste0("Recall: ", recall))

# Nueral Network

# Merge the data frames
data <- inner_join(sales_by_store, size_of_store, by = c("Store", "Type"))
data <- inner_join(data, count_of_dept, by = c("Store", "Type"))

data <- sales_by_store %>% 
  inner_join(size_of_store, by = c("Store", "Type")) %>% 
  inner_join(count_of_dept, by = c("Store", "Type"))

# Convert Type column to a factor
data$Type <- as.factor(data$Type)

# Scale the numeric variables
data[, c("mean_sales", "mean_size", "n_dept")] <- scale(data[, c("mean_sales", "mean_size", "n_dept")])

set.seed(123) # for reproducibility
train_idx <- sample(nrow(data), nrow(data) * 0.6)
train_data <- data[train_idx, ]
test_data <- data[-train_idx, ]


nn <- neuralnet(Type ~ mean_sales + mean_size + n_dept, 
                data = train_data, 
                hidden = 2, 
                threshold = 0.01)
test_data$A <- test_data$Type=='A'
test_data$B <- test_data$Type=='B'
test_data$C <- test_data$Type=='C'
pred <- compute(nn, test_data[, c("mean_sales", "mean_size", "n_dept", "A", "B", "C")])
pred <- apply(pred$net.result, 1, which.max)



cm <- table(test_data$Type, pred)
pred_nn = pred

table(pred, test_data$Type)
mean(pred == test_data$Type)

install.packages("NeuralNetTools")
library(NeuralNetTools)

# Plot neural net
par(mfcol=c(1,1))
plotnet(nn)
plot(nn)
# get the neural weights
neuralweights(nn)
# Plot the importance
olden(nn)

# Obtain classification statistics
accuracy <- sum(diag(cm)) / sum(cm)
precision <- diag(cm) / colSums(cm)
recall <- diag(cm) / rowSums(cm)
print(paste0("Accuracy: ", accuracy))
print(paste0("Precision: ", precision))
print(paste0("Recall: ", recall))

# Ensemble Model

# Combine predictions into a data frame
pred_df <- data.frame(rf = pred_rf, nb = pred_nb, knn = pred_knn, nn = pred_nn)



# Calculate majority vote prediction
ensemble_pred <- apply(pred_df, 1, function(x) {
  ifelse(sum(x == "1") >= sum(x == "2") & sum(x == "1") >= sum(x == "3"), "1",
         ifelse(sum(x == "2") >= sum(x == "1") & sum(x == "2") >= sum(x == "3"), "2", "3"))
})

# Obtain confusion matrix
cm <- table(ensemble_pred, test$Type)
print(cm)

# Get row and column names
rows <- rownames(cm)
cols <- colnames(cm)

# Convert confusion matrix to a data frame
cm_df <- data.frame(actual = rep(rows, each = length(cols)),
                    predicted = cols,
                    value = as.vector(cm))

# Create plot
ggplot(cm_df, aes(x = actual, y = predicted, fill = value)) +
  geom_tile() +
  geom_text(aes(label = value), size = 16, color = "white") +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(x = "Actual", y = "Predicted", title = "Confusion Matrix") +
  theme(plot.title = element_text(hjust = 0.5))

# Obtain classification statistics
accuracy <- sum(diag(cm)) / sum(cm)
precision <- diag(cm) / colSums(cm)
recall <- diag(cm) / rowSums(cm)
print(paste0("Accuracy: ", round(accuracy, 2)))
print(paste0("Precision: ", round(precision, 2)))
print(paste0("Recall: ", round(recall, 2)))

# Linear Regression

# Build the Multilinear regression model
model <- lm(Weekly_Sales ~ Store + Dept + IsHoliday + Type + Size + MarkDown2 + MarkDown3 + MarkDown4 + MarkDown5 + CPI + Unemployment  + Date_day + Date_week , data = train)

# Summarize the model
summary(model)

# Make predictions
predicted_values <- predict(model)


# Create a dataframe with actual and predicted values
actual_vs_predicted <- data.frame(Actual = train$Weekly_Sales, Predicted = predicted_values)

#Applying Backward Stepwise feature selection
model_reduced <- step(model, direction="backward")

# Summarize the selected model
summary(model_reduced)

#Applying Forward Stepwise feature selection
null=lm(Weekly_Sales~1, data=train) 
model_forw <- step(null, scope=list(lower=null, upper=model), direction="forward")
summary(model_forw)

#Applying Both Stepwise feature selection
model_both <- step(null, scope=list(lower=null, upper=model), direction="both")
summary(model_both)

# Make predictions on model using step wise feature selection
predicted_values_both <- predict(model_reduced)

# Create a dataframe with actual and predicted values
actual_vs_predicted_both <- data.frame(Actual = train$Weekly_Sales, Predicted = predicted_values_both)


# SARIMAX

# Merge the data frames
sales1 <- merge(sales, stores, by = "Store", all.x = TRUE)
train <- merge(sales1, features, by = c("Store", "Date", "IsHoliday"), all.x = TRUE)

# Load required libraries
library(dplyr)
library(lubridate)

# Convert 'Date' column to date datatype
train$Date <- as.Date(train$Date)

# Create week day, month, year and day fields in train data
train <- train %>%
  mutate(Date_dayofweek = wday(Date),
         Date_month = month(Date),
         Date_year = year(Date),
         Date_day = day(Date),
         Date_week = week(Date))


# Load required libraries
library(dplyr)
library(tidyr)
library(ggplot2)

# Group by at week year level
sales_by_week <- train %>%
  group_by(Date_year, Date_week) %>%
  summarise(Weekly_Sales = sum(Weekly_Sales)) %>%
  ungroup()

# Pivot the dataframe to create separate columns for each year
df_pivoted <- sales_by_week %>%
  pivot_wider(names_from = Date_year, values_from = Weekly_Sales, values_fill = 0)

# Load required libraries
library(ggplot2)
library(scales)

# Plot the pivoted dataframe
ggplot(data = df_pivoted, aes(x = Date_week)) +
  geom_line(aes(y = `2010`, color = "2010"), size = 1) +
  geom_line(aes(y = `2011`, color = "2011"), size = 1) +
  geom_line(aes(y = `2012`, color = "2012"), size = 1) +
  scale_color_manual(name = "Year", values = c("2010" = "#F8766D", "2011" = "#00BFC4", "2012" = "#7CAE00")) +
  labs(x = "Week", y = "Weekly Sales", title = "Weekly Sales by Year") +
  scale_y_continuous(labels = dollar_format()) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5, size = 16),
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        axis.text = element_text(size = 12),
        legend.position = "top",
        legend.title = element_text(size = 14),
        legend.text = element_text(size = 12))

# Group by at week year level
sales_by_week <- train %>%
  group_by(Date) %>%
  summarise(Weekly_Sales = sum(Weekly_Sales)) %>%
  ungroup()

# Convert the data to a time series object
ts_sales <- ts(sales_by_week$Weekly_Sales, start = c(2010, 5), frequency = 52)

# Decompose the time series into its components
decomp_sales <- decompose(ts_sales)

# Create a time series decomposition plot
plot(decomp_sales)

sales_ts <- ts(sales_by_week$Weekly_Sales, frequency = 52, start = c(2010, 5))

# Split the data into training and testing sets
n <- length(sales_ts)
train_size <- floor(0.8 * n)
train <- window(sales_ts, end = c(2012, 13))
test <- window(sales_ts, start = c(2012, 14))

# Fit the SARIMAX model on the training data
fit <- Arima(train, order = c(1,0,1), seasonal = list(order = c(1,0,1), period = 52))

# Forecast the test data using the SARIMAX model
forecast <- forecast(fit, h = length(test))

library(tseries)
library(tidyverse)

# Conduct an ADF test to check for stationarity
adf.test(sales_ts)
# Plot the ACF and PACF
ggtsdisplay(sales_ts, main = "ACF and PACF for Weekly Sales")






