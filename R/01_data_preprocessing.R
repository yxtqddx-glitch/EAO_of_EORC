# ============================================================================
# 01_data_preprocessing.R
# Data cleaning and MICE multiple imputation
# ============================================================================

library(tidyverse)
library(naniar)
library(VIM)
library(mice)
library(readxl)

set.seed(321)

# ------------------------------------------------------------------------
# 1. Load raw data
# ------------------------------------------------------------------------
sadata <- read_excel("EORC_data.xlsx")
str(sadata)

# ------------------------------------------------------------------------
# 2. Convert categorical variables to factors
# ------------------------------------------------------------------------
factor_vars <- c(
  'sex', 'genetic_disease', 'smoke', 'alcohol_intake', 'allergy',
  'neoadjuvant_therapy', 'differentiation', 'MMR', 'Type_of_surgery',
  'Lymphovascular_invasion', 'Perineural_invasion', 'ASA', 'cT',
  'cN', 'cTNM', 'pT', 'pN', 'pTNM', 'futility'
)

sadata_for_imputation <- sadata |>
  mutate(across(all_of(factor_vars), as.factor))

num_vars <- setdiff(names(sadata_for_imputation), factor_vars)
sadata_for_imputation[num_vars] <- lapply(sadata_for_imputation[num_vars], as.numeric)

str(sadata_for_imputation)

# ------------------------------------------------------------------------
# 3. Remove variables with >20% missing values
# ------------------------------------------------------------------------
missing_prop <- colMeans(is.na(sadata_for_imputation))
vars_to_remove <- names(missing_prop[missing_prop > 0.20])
cat("Variables removed:", paste(vars_to_remove, collapse = ", "), "\n")

sadata_for_imputation <- sadata_for_imputation |>
  select(-all_of(vars_to_remove))

print(dim(sadata_for_imputation))
print(sum(is.na(sadata_for_imputation)))

# ------------------------------------------------------------------------
# 4. Summarise missing percentages
# ------------------------------------------------------------------------
missing_percentages <- sadata_for_imputation |>
  summarise(across(everything(), ~ mean(is.na(.)) * 100)) |>
  pivot_longer(everything(), names_to = "Variable", values_to = "Missing_Percentage") |>
  arrange(desc(Missing_Percentage))

write.csv(missing_percentages, "missing_percentages.csv", row.names = FALSE)

overall_missing <- mean(is.na(sadata_for_imputation)) * 100
cat(sprintf("Overall missing percentage: %.2f%%\n", overall_missing))

# ------------------------------------------------------------------------
# 5. MICE multiple imputation (m = 5, maxit = 50)
# ------------------------------------------------------------------------
columns_to_impute <- names(sadata_for_imputation)[colSums(is.na(sadata_for_imputation)) > 0]

# Set imputation methods
imp_methods <- make.method(sadata_for_imputation[, columns_to_impute])
# Numeric: predictive mean matching
imp_methods[sapply(sadata_for_imputation[, columns_to_impute], is.numeric)] <- "pmm"
# Binary factors: logistic regression
# Multi-class factors: polytomous regression
for (col in columns_to_impute) {
  if (is.factor(sadata_for_imputation[[col]])) {
    if (nlevels(sadata_for_imputation[[col]]) == 2) {
      imp_methods[col] <- "logreg"
    } else {
      imp_methods[col] <- "polyreg"
    }
  }
}

print(imp_methods)

set.seed(321)
imputed_data <- mice(
  sadata_for_imputation[, columns_to_impute],
  m = 5,
  maxit = 50,
  method = imp_methods,
  printFlag = TRUE
)

# ------------------------------------------------------------------------
# 6. Pool imputed datasets
#   - Categorical: mode across 5 imputations
#   - Continuous: median across 5 imputations
# ------------------------------------------------------------------------
get_mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

merged_data <- sadata_for_imputation

for (var in columns_to_impute) {
  all_imputations <- sapply(1:5, function(i) complete(imputed_data, i)[[var]])
  if (is.factor(sadata_for_imputation[[var]])) {
    merged_data[[var]] <- apply(all_imputations, 1, get_mode)
    merged_data[[var]] <- factor(merged_data[[var]], levels = levels(sadata_for_imputation[[var]]))
  } else {
    merged_data[[var]] <- apply(all_imputations, 1, median)
  }
}

str(merged_data)
print(names(merged_data))

# ------------------------------------------------------------------------
# 7. Save imputed data
# ------------------------------------------------------------------------
write_csv(merged_data, "imputed_done.csv")

# ------------------------------------------------------------------------
# 8. Diagnostic plots
# ------------------------------------------------------------------------
# Missing data pattern
tiff("missing_pattern.tiff", width = 12, height = 5, units = "in", res = 300)
aggr_plot <- aggr(
  sadata_for_imputation,
  col = c("#FFA07A", "#20B2AA"),
  numbers = TRUE,
  sortVars = TRUE,
  labels = names(sadata_for_imputation),
  cex.axis = 0.7,
  gap = 3,
  ylab = c("Missing data", "Pattern")
)
dev.off()

# Convergence plot
tiff("imputation_convergence.tiff", width = 12, height = 5, units = "in", res = 300)
plot(imputed_data)
dev.off()

# Density plot (compare imputed vs observed)
tiff("imputation_density.tiff", width = 12, height = 5, units = "in", res = 300)
densityplot(imputed_data)
dev.off()

cat("Data imputation completed. Imputed CSV and 3 diagnostic plots saved.\n")
