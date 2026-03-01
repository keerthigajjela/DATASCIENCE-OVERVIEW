# Student Performance Analysis & Prediction (R)

library(tidyverse)
library(corrplot)
library(caret)
library(randomForest)

# Step 1) Data Collection - Loading dataset
student_data <- read.csv("student_performance_tp01.csv")

cat("Loaded dataset successfully.\n")
cat("Rows:", nrow(student_data), "Columns:", ncol(student_data), "\n\n")

# Step 2) Data Preprocessing
cat("Missing values per column:\n")
print(colSums(is.na(student_data)))

# Converting character columns to factors
student_data <- student_data %>%
  mutate(across(where(is.character), as.factor))

# Creating output folder
if (!dir.exists("outputs")) dir.create("outputs")

# Step 3) EDA Figure 1 (Histogram of G3)
p1 <- ggplot(student_data, aes(x = G3)) +
  geom_histogram(bins = 20, fill = "skyblue") +
  labs(
    title = "Figure 1: Distribution of Final Grade (G3)",
    x = "Final Grade (G3)",
    y = "Frequency"
  )

ggsave("outputs/Figure1_G3_Distribution.png", plot = p1, width = 6, height = 4, dpi = 300)

# Step 4) EDA Figure 2 (Studytime vs G3)
if ("studytime" %in% names(student_data)) {
  p2 <- ggplot(student_data, aes(x = studytime, y = G3)) +
    geom_point(color = "darkgreen") +
    geom_smooth(method = "lm", se = TRUE) +
    labs(
      title = "Figure 2: Study Time vs Final Grade (G3)",
      x = "Study Time",
      y = "Final Grade (G3)"
    )
  
  ggsave("outputs/Figure2_Studytime_vs_G3.png", plot = p2, width = 6, height = 4, dpi = 300)
} else {
  cat("Note: 'studytime' column not found, Figure 2 not created.\n")
}

# Step 5) Correlation Analysis (Square Heatmap)

numeric_vars <- student_data %>% select(where(is.numeric))

if (ncol(numeric_vars) >= 2) {
  cor_matrix <- cor(numeric_vars, use = "complete.obs")
  
  png("outputs/Figure3_Correlation_Heatmap.png", width = 1200, height = 900, res = 150)
  
  corrplot(
    cor_matrix,
    method = "color",
    type = "full",      
    addCoef.col = "black",
    tl.cex = 0.8,
    number.cex = 0.7
  )
  
  dev.off()
} else {
  cat("Not enough numeric columns.\n")
}
# Step 6) Train/Test Split
set.seed(123)
train_index <- createDataPartition(student_data$G3, p = 0.80, list = FALSE)
train_set <- student_data[train_index, ]
test_set  <- student_data[-train_index, ]

cat("\nTrain rows:", nrow(train_set), "Test rows:", nrow(test_set), "\n")

# Step 7) Model 1: Linear Regression
linear_model <- lm(G3 ~ ., data = train_set)
lm_out <- summary(linear_model)

pred_linear <- predict(linear_model, newdata = test_set)
rmse_linear <- RMSE(pred_linear, test_set$G3)
r2_linear   <- R2(pred_linear, test_set$G3)

sink("outputs/LinearRegression_Summary.txt")
cat("LINEAR REGRESSION SUMMARY\n")
cat("=========================\n\n")
print(lm_out)
cat("\nTest RMSE:", rmse_linear, "\n")
cat("Test R2:", r2_linear, "\n")
sink()

# Step 8) Model 2: Random Forest
set.seed(123)
rf_model <- randomForest(G3 ~ ., data = train_set, importance = TRUE)

pred_rf <- predict(rf_model, newdata = test_set)
rmse_rf <- RMSE(pred_rf, test_set$G3)
r2_rf   <- R2(pred_rf, test_set$G3)

sink("outputs/RandomForest_Summary.txt")
cat("RANDOM FOREST SUMMARY\n")
cat("=====================\n\n")
print(rf_model)
cat("\nTest RMSE:", rmse_rf, "\n")
cat("Test R2:", r2_rf, "\n")
sink()

# Step 9) Feature Importance 
imp_raw <- importance(rf_model)

if ("%IncMSE" %in% colnames(imp_raw)) {
  imp_df <- data.frame(
    Feature = rownames(imp_raw),
    Importance = imp_raw[, "%IncMSE"],
    stringsAsFactors = FALSE
  )
  y_label_name <- "Importance (%IncMSE)"
} else {
  imp_df <- data.frame(
    Feature = rownames(imp_raw),
    Importance = imp_raw[, "IncNodePurity"],
    stringsAsFactors = FALSE
  )
  y_label_name <- "Importance (IncNodePurity)"
}

imp_df_clean <- imp_df %>%
  arrange(desc(Importance))

top_n <- 10
imp_top <- imp_df_clean %>% slice_head(n = top_n)

p_clean <- ggplot(imp_top, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Figure 4: Top Feature Importance (Random Forest)",
    x = "Predictor",
    y = y_label_name
  )

ggsave("outputs/Figure4_RF_Feature_Importance_CLEAN.png",
       plot = p_clean, width = 7, height = 5, dpi = 300)

# Step 10) Table: Model Comparison
results_table <- data.frame(
  Model = c("Linear Regression", "Random Forest"),
  RMSE  = c(rmse_linear, rmse_rf),
  R2    = c(r2_linear, r2_rf)
)

write.csv(results_table, "outputs/Table1_Model_Comparison.csv", row.names = FALSE)

cat("\nModel comparison (saved to outputs/Table1_Model_Comparison.csv):\n")
print(results_table)

cat("Check outputs/ for:\n")
cat("- Figure1_G3_Distribution.png\n")
cat("- Figure2_Studytime_vs_G3.png\n")
cat("- Figure3_Correlation_Heatmap.png  (WITH numbers)\n")
cat("- Figure4_RF_Feature_Importance_CLEAN.png\n")
cat("- LinearRegression_Summary.txt\n")
cat("- RandomForest_Summary.txt\n")
cat("- Table1_Model_Comparison.csv\n")

