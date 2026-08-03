# ============================================================================
# 04_model_evaluation.R
# Internal validation (stratified 7:3 split), independent external validation,
# calibration, discrimination, decision curve analysis, and confusion matrices
# ============================================================================

library(tidymodels)
library(probably)
library(pROC)
library(dcurves)

# Source helper functions
source("tidyfuncs4cls2.R")

# ============================================================================
# 0. Load all model evaluation results
# ============================================================================
evalfiles <- list.files("cls2/", full.names = TRUE)
lapply(evalfiles, load, .GlobalEnv)

nmodels <- 12
cols4model <- rainbow(nmodels)

# ============================================================================
# 1. Internal Validation: Performance metrics on training and test sets
# ============================================================================

# --- Training set metrics ---
evaltrain <- bind_rows(
  lapply(list(predtrain_logistic, predtrain_dt,
              predtrain_lasso, predtrain_ridge, predtrain_enet,
              predtrain_knn, predtrain_lightgbm, predtrain_rf,
              predtrain_xgboost, predtrain_svm, predtrain_mlp,
              predtrain_stack),
         "[[", "metrics")
) |>
  mutate(model = forcats::as_factor(model))

# --- Test set metrics ---
evaltest <- bind_rows(
  lapply(list(predtest_logistic, predtest_dt,
              predtest_lasso, predtest_ridge, predtest_enet,
              predtest_knn, predtest_lightgbm, predtest_rf,
              predtest_xgboost, predtest_svm, predtest_mlp,
              predtest_stack),
         "[[", "metrics")
) |>
  mutate(model = forcats::as_factor(model))

# Wide-format tables
evaltrain2 <- evaltrain |>
  select(-.estimator) |>
  pivot_wider(names_from = .metric, values_from = .estimate)

evaltest2 <- evaltest |>
  select(-.estimator) |>
  pivot_wider(names_from = .metric, values_from = .estimate)

print("Training set metrics:")
print(evaltrain2)
print("Test set metrics:")
print(evaltest2)

# ============================================================================
# 2. ROC Curves: all models on test set
# ============================================================================
roctest <- bind_rows(
  lapply(list(predtest_logistic, predtest_dt,
              predtest_lasso, predtest_ridge, predtest_enet,
              predtest_knn, predtest_lightgbm, predtest_rf,
              predtest_xgboost, predtest_svm, predtest_mlp,
              predtest_stack),
         "[[", "rocresult")
) |>
  mutate(model = forcats::as_factor(model))

roctest |>
  mutate(modelauc = paste(model, curvelab),
         modelauc = forcats::as_factor(modelauc)) |>
  ggplot(aes(x = 1 - specificity, y = sensitivity, color = modelauc)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_color_manual(values = cols4model) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0),
                     breaks = seq(0, 1, by = 0.2)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0),
                     breaks = seq(0, 1, by = 0.2)) +
  labs(color = "", title = "ROCs on Test Set") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1, 0),
        legend.background = element_blank(),
        legend.key = element_blank(),
        text = element_text(family = "serif"))

# ============================================================================
# 3. Precision-Recall Curves: all models on test set
# ============================================================================
prtest <- bind_rows(
  lapply(list(predtest_logistic, predtest_dt,
              predtest_lasso, predtest_ridge, predtest_enet,
              predtest_knn, predtest_lightgbm, predtest_rf,
              predtest_xgboost, predtest_svm, predtest_mlp,
              predtest_stack),
         "[[", "prresult")
) |>
  mutate(model = forcats::as_factor(model))

prtest |>
  mutate(modelauc = paste(model, curvelab),
         modelauc = forcats::as_factor(modelauc)) |>
  ggplot(aes(x = recall, y = precision, color = modelauc)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_color_manual(values = cols4model) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0),
                     breaks = seq(0, 1, by = 0.2)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0),
                     breaks = seq(0, 1, by = 0.2)) +
  labs(color = "", title = "PR Curves on Test Set") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(0, 0),
        legend.background = element_blank(),
        legend.key = element_blank(),
        text = element_text(family = "serif"))

# ============================================================================
# 4. Calibration Curves
# ============================================================================

# Collect all test set predictions
predtest <- bind_rows(
  lapply(list(predtest_logistic, predtest_dt,
              predtest_lasso, predtest_ridge, predtest_enet,
              predtest_knn, predtest_lightgbm, predtest_rf,
              predtest_xgboost, predtest_svm, predtest_mlp,
              predtest_stack),
         "[[", "prediction")
) |>
  mutate(model = forcats::as_factor(model))

# Brier scores
bstest <- predtest |>
  group_by(model) |>
  yardstick::brier_class(.obs, .pred_no) |>
  mutate(
    meanpred = 0.35,
    meanobs = 0.8,
    text = paste0("Brier=", sprintf("%.4f", .estimate))
  )

# Calibration curves with confidence bands
predtest |>
  probably::cal_plot_breaks(
    .obs, .pred_yes,
    event_level = "second",
    num_breaks = 5,
    .by = model
  ) +
  geom_text(
    bstest,
    mapping = aes(x = meanpred, y = meanobs, label = text),
    color = "black"
  ) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0),
                     breaks = seq(0, 1, by = 0.2)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0),
                     breaks = seq(0, 1, by = 0.2)) +
  scale_color_manual(values = cols4model) +
  theme_bw() +
  theme(
    legend.position = "none",
    panel.grid = element_blank(),
    text = element_text(family = "serif")
  )

# Calibration curves (line format)
calitest <- bind_rows(
  lapply(list(predtest_logistic, predtest_dt,
              predtest_lasso, predtest_ridge, predtest_enet,
              predtest_knn, predtest_lightgbm, predtest_rf,
              predtest_xgboost, predtest_svm, predtest_mlp,
              predtest_stack),
         "[[", "caliresult")
) |>
  mutate(model = forcats::as_factor(model))

calitest |>
  left_join(bstest, by = "model") |>
  mutate(
    label = paste0(model, " Brier=", sprintf("%.4f", .estimate)),
    label = forcats::as_factor(label)
  ) |>
  ggplot(aes(x = predprobgroup, y = Fraction, color = label)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3, pch = 15) +
  geom_abline(linetype = "dashed") +
  scale_color_manual(values = cols4model) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0),
                     breaks = seq(0, 1, by = 0.2)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0),
                     breaks = seq(0, 1, by = 0.2)) +
  labs(color = "", x = "Bin Midpoint", y = "Event Rate",
       title = "Calibration Curves on Test Set") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1, 0),
        legend.background = element_blank(),
        legend.key = element_blank(),
        text = element_text(family = "serif"))

# ============================================================================
# 5. Decision Curve Analysis (DCA)
# ============================================================================

# Wide-format test predictions for DCA
predtest2 <- predtest |>
  select(-.pred_no) |>
  mutate(id = rep(1:nrow(predtest_logistic$prediction),
                  length(unique(predtest$model)))) |>
  pivot_wider(
    id_cols = c(id, .obs),
    names_from = model,
    values_from = .pred_yes
  ) |>
  select(id, .obs, sort(unique(predtest$model)))

testdca_obj <- dcurves::dca(
  as.formula(paste0(".obs ~ ",
    paste(colnames(predtest2)[3:ncol(predtest2)], collapse = " + "))),
  data = predtest2,
  thresholds = seq(0, 1, by = 0.01)
)

plot(testdca_obj, smooth = TRUE) +
  scale_color_manual(values = c("black", "grey", cols4model)) +
  labs(title = "DCA on Test Set") +
  theme(
    panel.grid = element_blank(),
    legend.position = "inside",
    legend.justification = c(1, 1),
    legend.background = element_blank(),
    legend.key = element_blank(),
    text = element_text(family = "serif")
  )

# ============================================================================
# 6. Performance Heatmap (test set)
# ============================================================================
evaltest |>
  filter(!(.metric %in% c("detection_prevalence"))) |>
  select(model, .metric, .estimate) |>
  pivot_wider(names_from = .metric, values_from = .estimate) |>
  mutate(model = reorder(model, roc_auc)) |>
  pivot_longer(cols = -1) |>
  group_by(name) |>
  mutate(valuescale = (value - min(value)) / (max(value) - min(value))) |>
  ungroup() |>
  ggplot(aes(x = name, y = model, fill = valuescale)) +
  geom_tile(color = "white", show.legend = FALSE) +
  geom_text(aes(label = sprintf("%.2f", value))) +
  scale_fill_gradient(low = "green", high = "red") +
  labs(x = "", y = "") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
    text = element_text(family = "serif")
  )

# ============================================================================
# 7. Cross-validation Performance Summary
# ============================================================================
evalcv <- bind_rows(
  lapply(list(evalcv_logistic, evalcv_dt,
              evalcv_lasso, evalcv_ridge, evalcv_enet,
              evalcv_knn, evalcv_lightgbm, evalcv_rf,
              evalcv_xgboost, evalcv_svm, evalcv_mlp),
         "[[", "evalcv")
) |>
  mutate(
    model = forcats::as_factor(model),
    modelperf = paste0(model, "(", sprintf("%.2f", mean),
                       "\u00b1", sprintf("%.2f", sd), ")")
  )

# CV ROC AUC comparison (mean ± SD)
evalcv |>
  filter(.metric == "roc_auc") |>
  group_by(model) |>
  sample_n(size = 1) |>
  ungroup() |>
  ggplot(aes(x = model, y = mean, color = model)) +
  geom_point(size = 2, show.legend = FALSE) +
  geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd),
                width = 0.1, linewidth = 1.2, show.legend = FALSE) +
  scale_y_continuous(limits = c(0.5, 1)) +
  scale_color_manual(values = cols4model) +
  labs(x = "", y = "CV ROC AUC") +
  theme_bw() +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor.y = element_blank(),
    text = element_text(family = "serif")
  )

# CV PR AUC comparison (mean ± SD)
evalcv |>
  filter(.metric == "pr_auc") |>
  group_by(model) |>
  sample_n(size = 1) |>
  ungroup() |>
  ggplot(aes(x = model, y = mean, color = model)) +
  geom_point(size = 2, show.legend = FALSE) +
  geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd),
                width = 0.1, linewidth = 1.2, show.legend = FALSE) +
  scale_y_continuous(limits = c(0.5, 1)) +
  scale_color_manual(values = cols4model) +
  labs(x = "", y = "CV PR AUC") +
  theme_bw() +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor.y = element_blank(),
    text = element_text(family = "serif")
  )

# ============================================================================
# 8. Independent External Validation
# ============================================================================
cat("\n=== External Validation ===\n")

newdata <- readxl::read_excel("newdata.xlsx")

# Convert categorical variables to factors
for (i in c(1, 7, 9, 10, 11, 12)) {
  newdata[[i]] <- factor(newdata[[i]])
}
newdata$Id <- NULL
newdata <- na.omit(newdata)

newdata$EAO <- factor(
  newdata$EAO,
  levels = c(yournegativelevel, yourpositivelevel)
)

# Predict with stacking model
predresult <- newdata |>
  bind_cols(predict(final_stack, new_data = newdata, type = "prob")) |>
  mutate(
    .pred_class = factor(
      ifelse(.pred_yes >= predtrain_stack$diycutoff,
             yourpositivelevel, yournegativelevel)
    )
  )

# External validation evaluation
prednew_stack <- eval4cls2(
  model = final_stack, dataset = newdata,
  yname = "EAO", modelname = "Stacking",
  datasetname = "external",
  cutoff = predtrain_stack$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)

prednew_stack$metrics
pROC::auc(prednew_stack$proc)
pROC::ci.auc(prednew_stack$proc)

cat("\n=== Model evaluation completed ===\n")
