# ============================================================================
# 02_feature_selection.R
# Spearman correlation filtering and Boruta feature selection
# ============================================================================

library(tidyverse)
library(corrplot)
library(pheatmap)
library(RColorBrewer)
library(Boruta)
library(rsample)

# ============================================================================
# PART A: Spearman Correlation Filtering
# ============================================================================

# Load training set after imputation but before Spearman filtering
data <- read_csv("imputed_done.csv")
str(data)

# Keep only numeric columns
num_data <- data[, sapply(data, is.numeric)]

# Compute Spearman correlation matrix and p-values
cor_result <- cor(num_data, method = "spearman", use = "complete.obs")
p_result <- cor.mtest(num_data, method = "spearman", use = "complete.obs")$p

cat("===== Spearman Correlation Matrix =====\n")
print(round(cor_result, 3))

cat("\n===== P-value Matrix =====\n")
print(round(p_result, 3))

# Spearman correlation heatmap
pdf("spearman_heatmap.pdf", width = 16, height = 16)

corrplot(cor_result, method = "color",
         type = "upper",
         tl.cex = 0.9,
         tl.col = "black",
         tl.srt = 30,
         tl.pos = "lt",
         p.mat = p_result,
         sig.level = c(0.001, 0.01, 0.05),
         insig = "label_sig",
         pch.cex = 0.8,
         cl.cex = 0.8)

corrplot(cor_result, method = "number",
         type = "lower",
         tl.pos = "n",
         cl.pos = "n",
         number.cex = 0.6,
         add = TRUE)

dev.off()

# ============================================================================
# PART B: Boruta Feature Selection
# ============================================================================

# Load data after MICE imputation
merged_data <- read_csv("imputed.csv")
str(merged_data)

# Stratified 7:3 split
set.seed(123)
datasplit <- initial_split(merged_data, prop = 0.7, strata = EAO, breaks = 10)
traindata <- training(datasplit)
testdata  <- testing(datasplit)

# Prepare Boruta inputs
X <- traindata |>
  select(-EAO) |>
  as.data.frame()

y <- as.factor(traindata$EAO)

# Run Boruta
set.seed(123)
boruta_output <- Boruta(X, y, doTrace = 2, maxRuns = 200)

print(boruta_output)

# Get confirmed important features
selected_vars <- getSelectedAttributes(boruta_output, withTentative = FALSE)
print(selected_vars)

# Boruta importance plot
plot(boruta_output, cex.axis = 0.7, las = 2,
     xlab = "", main = "Variable Importance")

# Subset data to selected features
traindata <- traindata |>
  select(EAO, all_of(selected_vars))

testdata <- testdata |>
  select(EAO, all_of(selected_vars))

# Save feature-selected datasets
write_csv(traindata, "traindata_features.csv")
write_csv(testdata, "testdata_features.csv")

cat("Feature selection completed. Selected features:", paste(selected_vars, collapse = ", "), "\n")
