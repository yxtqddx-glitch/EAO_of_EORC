# ============================================================================
# 06_figures.R
# Figure generation: model comparison heatmaps, calibration plots,
# SHAP dependence/waterfall plots, correlation matrices, and workflow diagrams
# ============================================================================

library(tidymodels)
library(probably)
library(pROC)
library(dcurves)
library(corrplot)
library(pheatmap)
library(RColorBrewer)
library(ggplot2)
library(ggrepel)
library(shapviz)

source("tidyfuncs4cls2.R")

# Load all evaluation results
evalfiles <- list.files("cls2/", full.names = TRUE)
lapply(evalfiles, load, .GlobalEnv)

nmodels <- 12
cols4model <- rainbow(nmodels)

# ============================================================================
# FIGURE 1: Model Performance Heatmap (Test Set)
# ============================================================================
cat("\n=== Figure: Test Set Performance Heatmap ===\n")

evaltest <- bind_rows(
  lapply(list(predtest_logistic, predtest_dt,
              predtest_lasso, predtest_ridge, predtest_enet,
              predtest_knn, predtest_lightgbm, predtest_rf,
              predtest_xgboost, predtest_svm, predtest_mlp,
              predtest_stack),
         "[[", "metrics")
) |>
  mutate(model = forcats::as_factor(model))

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
  labs(x = "", y = "", title = "Model Performance Heatmap (Test Set)") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
    text = element_text(family = "serif")
  )

ggsave("figures/heatmap_test_performance.pdf", width = 12, height = 8)

# ============================================================================
# FIGURE 2: Multi-model ROC Curves (Test Set)
# ============================================================================
cat("\n=== Figure: Multi-model ROC Curves ===\n")

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
  labs(color = "", title = "ROC Curves — Test Set") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1, 0),
        legend.background = element_blank(),
        legend.key = element_blank(),
        text = element_text(family = "serif"))

ggsave("figures/roc_test_all_models.pdf", width = 10, height = 8)

# ============================================================================
# FIGURE 3: Calibration Curves (Test Set)
# ============================================================================
cat("\n=== Figure: Calibration Curves ===\n")

predtest <- bind_rows(
  lapply(list(predtest_logistic, predtest_dt,
              predtest_lasso, predtest_ridge, predtest_enet,
              predtest_knn, predtest_lightgbm, predtest_rf,
              predtest_xgboost, predtest_svm, predtest_mlp,
              predtest_stack),
         "[[", "prediction")
) |>
  mutate(model = forcats::as_factor(model))

calitest <- bind_rows(
  lapply(list(predtest_logistic, predtest_dt,
              predtest_lasso, predtest_ridge, predtest_enet,
              predtest_knn, predtest_lightgbm, predtest_rf,
              predtest_xgboost, predtest_svm, predtest_mlp,
              predtest_stack),
         "[[", "caliresult")
) |>
  mutate(model = forcats::as_factor(model))

bstest <- predtest |>
  group_by(model) |>
  yardstick::brier_class(.obs, .pred_no) |>
  mutate(meanpred = 0.35, meanobs = 0.8,
         text = paste0("Brier=", sprintf("%.4f", .estimate)))

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
       title = "Calibration Curves — Test Set") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1, 0),
        legend.background = element_blank(),
        legend.key = element_blank(),
        text = element_text(family = "serif"))

ggsave("figures/calibration_test_all_models.pdf", width = 10, height = 8)

# ============================================================================
# FIGURE 4: Decision Curve Analysis (Test Set)
# ============================================================================
cat("\n=== Figure: Decision Curve Analysis ===\n")

predtest2 <- predtest |>
  select(-.pred_no) |>
  mutate(id = rep(1:nrow(predtest_logistic$prediction),
                  length(unique(predtest$model)))) |>
  pivot_wider(id_cols = c(id, .obs), names_from = model,
              values_from = .pred_yes) |>
  select(id, .obs, sort(unique(predtest$model)))

testdca_obj <- dcurves::dca(
  as.formula(paste0(".obs ~ ",
    paste(colnames(predtest2)[3:ncol(predtest2)], collapse = " + "))),
  data = predtest2,
  thresholds = seq(0, 1, by = 0.01)
)

pdf("figures/dca_test_all_models.pdf", width = 10, height = 8)
plot(testdca_obj, smooth = TRUE) +
  scale_color_manual(values = c("black", "grey", cols4model)) +
  labs(title = "Decision Curve Analysis — Test Set") +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1, 1),
        legend.background = element_blank(),
        legend.key = element_blank(),
        text = element_text(family = "serif"))
dev.off()

# ============================================================================
# FIGURE 5: SHAP Global Importance + Beeswarm
# ============================================================================
cat("\n=== Figure: SHAP Summary ===\n")

load("cls2shiny/shiny_stack_data.RData")

traindatax <- traindata_shiny |> select(-futility)
catvars <- getcategory(traindatax)
convars <- getcontinuous(traindatax)

shap_stack <- shap4cls2(
  finalmodel = final_stack_data,
  predfunc = function(model, newdata) {
    predict(model, newdata, type = "prob") |>
      select(ends_with("yes")) |> pull()
  },
  datax = traindatax,
  datay = traindata_shiny$futility,
  yname = "futility",
  flname = catvars,
  lxname = convars,
  plotname = c(catvars, convars)
)

pdf("figures/shap_global_importance.pdf", width = 8, height = 6)
print(shap_stack$shapvipplot)
dev.off()

pdf("figures/shap_beeswarm.pdf", width = 8, height = 6)
print(shap_stack$shapplot)
dev.off()

pdf("figures/shap_combined_importance_beeswarm.pdf", width = 12, height = 10)
shapplotplus(shap_stack,
  plotname = c(catvars, convars),
  bottomxstart = -0.3, bottomxstop = 0.4, topxrange = 0.15)
dev.off()

# ============================================================================
# FIGURE 6: SHAP Dependence Plots
# ============================================================================
cat("\n=== Figure: SHAP Dependence Plots ===\n")

# Continuous variables
pdf("figures/shap_dependence_continuous.pdf", width = 10, height = 12)
print(shap_stack$shapplotc_facet)
dev.off()

# Categorical variables
pdf("figures/shap_dependence_categorical.pdf", width = 10, height = 12)
print(shap_stack$shapplotd_facet)
dev.off()

# ============================================================================
# FIGURE 7: SHAP Waterfall (local explanation)
# ============================================================================
cat("\n=== Figure: SHAP Waterfall ===\n")

shap_sv <- shapviz::shapviz(shap_stack$shapley, X = traindatax)

pdf("figures/shap_waterfall_sample1.pdf", width = 8, height = 5)
shapviz::sv_waterfall(shap_sv, row_id = 1) +
  theme(text = element_text(family = "serif"))
dev.off()

pdf("figures/shap_force_sample1.pdf", width = 10, height = 4)
shapviz::sv_force(shap_sv, row_id = 1) +
  theme(text = element_text(family = "serif"))
dev.off()

# ============================================================================
# FIGURE 8: Spearman Correlation Matrix
# ============================================================================
cat("\n=== Figure: Correlation Matrix ===\n")

data <- read_csv("先分组的训练集-用于斯皮尔曼-剔除后.csv")
num_data <- data[, sapply(data, is.numeric)]

cor_result <- cor(num_data, method = "spearman", use = "complete.obs")
p_result <- cor.mtest(num_data, method = "spearman", use = "complete.obs")$p

pdf("figures/spearman_correlation_matrix.pdf", width = 14, height = 14)
corrplot(cor_result, method = "color",
         type = "upper", tl.cex = 0.8, tl.col = "black",
         tl.srt = 30, tl.pos = "lt",
         p.mat = p_result, sig.level = c(0.001, 0.01, 0.05),
         insig = "label_sig", pch.cex = 0.7, cl.cex = 0.8)
corrplot(cor_result, method = "number",
         type = "lower", tl.pos = "n", cl.pos = "n",
         number.cex = 0.5, add = TRUE)
dev.off()

# ============================================================================
# FIGURE 9: CV Performance Comparison (Error bars)
# ============================================================================
cat("\n=== Figure: CV Performance ===\n")

evalcv <- bind_rows(
  lapply(list(evalcv_logistic, evalcv_dt,
              evalcv_lasso, evalcv_ridge, evalcv_enet,
              evalcv_knn, evalcv_lightgbm, evalcv_rf,
              evalcv_xgboost, evalcv_svm, evalcv_mlp),
         "[[", "evalcv")
) |>
  mutate(model = forcats::as_factor(model))

pdf("figures/cv_roc_auc_comparison.pdf", width = 12, height = 6)
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
  theme(panel.grid.major.y = element_blank(),
        panel.grid.minor.y = element_blank(),
        text = element_text(family = "serif"))
dev.off()

cat("\n=== All figures saved to figures/ directory ===\n")
