# ============================================================================
# 05_shap_analysis.R
# SHAP (SHapley Additive exPlanations) analysis for global and local
# model interpretability
#
# Includes:
#   - Global SHAP feature importance (bar charts + beeswarm)
#   - SHAP dependence plots (continuous and categorical variables)
#   - Local SHAP waterfall plots for individual predictions
#   - SHAP interaction analysis (tree-based models)
# ============================================================================

library(tidymodels)
library(fastshap)
library(shapviz)
library(DALEXtra)
library(treeshap)

# Source helper functions
source("tidyfuncs4cls2.R")

# Load trained model and training data
load("cls2shiny/shiny_stack_data.RData")  # final_stack_data, traindata_shiny
traindata <- traindata_shiny
final_model <- final_stack_data

yourpositivelevel <- "yes"
yournegativelevel <- "no"

# Prepare feature data
traindatax <- traindata |>
  select(-EAO)

catvars <- getcategory(traindatax)
convars <- getcontinuous(traindatax)

# ============================================================================
# 1. SHAP for Stacking Ensemble (via fastshap)
# ============================================================================
cat("\n=== SHAP Analysis: Stacking Ensemble ===\n")

shapresult_stack <- shap4cls2(
  finalmodel = final_model,
  predfunc = function(model, newdata) {
    predict(model, newdata, type = "prob") |>
      select(ends_with(yourpositivelevel)) |>
      pull()
  },
  datax = traindatax,
  datay = traindata$EAO,
  yname = "EAO",
  flname = catvars,
  lxname = convars,
  plotname = c(catvars, convars)
)

# Global importance bar chart
print(shapresult_stack$shapvipplot)

# SHAP beeswarm plot
print(shapresult_stack$shapplot)

# SHAP dependence plots: categorical variables
shapresult_stack$shapplotd_facet
shapresult_stack$shapplotd_one

# SHAP dependence plots: continuous variables
shapresult_stack$shapplotc_facet
shapresult_stack$shapplotc_one
shapresult_stack$shapplotc_one2

# Unified SHAP plots (all variables in one figure)
shapresult_stack$shapvipplot_unity
shapresult_stack$shapplot_unity

# Combined importance bar + beeswarm
shapplotplus(shapresult_stack,
  plotname = c(catvars, convars),
  bottomxstart = -0.3,
  bottomxstop = 0.4,
  topxrange = 0.15
)

# ============================================================================
# 2. Local SHAP Waterfall Plots (individual predictions)
# ============================================================================
cat("\n=== Local SHAP Waterfall ===\n")

shap_stack <- shapviz::shapviz(
  shapresult_stack$shapley,
  X = traindatax
)

# Force plot for sample 1
shapviz::sv_force(shap_stack, row_id = 1) +
  theme(text = element_text(family = "serif"))

# Waterfall plot for sample 1
shapviz::sv_waterfall(shap_stack, row_id = 1) +
  theme(text = element_text(family = "serif"))

# ============================================================================
# 3. SHAP for Random Forest (via treeshap for interaction effects)
# ============================================================================
cat("\n=== SHAP Interaction: Random Forest ===\n")

load("cls2/evalresult_rf.RData")  # tune_rf, final_rf, ...

# Extract fitted RF engine
final_rf_engine <- final_rf |>
  extract_fit_engine()

# Unify RF model for treeshap
traindata_binary <- traindata |>
  mutate(EAO = factor(ifelse(EAO == yourpositivelevel, 1, 0)))

datarecipe_rf_bin <- recipe(EAO ~ ., traindata_binary)
set.seed(123)
rf_binary <- workflow() |>
  add_recipe(datarecipe_rf_bin) |>
  add_model(
    rand_forest(mode = "classification", engine = "randomForest")
  ) |>
  finalize_workflow(
    tune_rf |> select_by_one_std_err(metric = "roc_auc", desc(min_n))
  ) |>
  fit(traindata_binary) |>
  extract_fit_engine()

unify_rf <- treeshap::randomForest.unify(rf_binary, traindata_binary)

set.seed(123)
treeshap_rf <- treeshap::treeshap(
  unify_rf,
  as.data.frame(traindatax),
  interactions = TRUE
)

shapley_rf <- shapviz::shapviz(treeshap_rf, X = traindatax)

# Interaction matrix
shapviz::sv_interaction(shapley_rf, max_display = ncol(traindatax))

# Feature dependence with interaction coloring
shapviz::sv_dependence(
  shapley_rf,
  v = c("NLR", "pT"),
  color_var = c("Age", "Lymphovascular_invasion"),
  interactions = TRUE
)

# ============================================================================
# 4. FastSHAP for all individual base models
# ============================================================================

# Helper function for per-model SHAP
run_model_shap <- function(final_obj, model_name, traindatax, traindata) {
  cat(sprintf("\n--- SHAP for %s ---\n", model_name))

  shap_res <- shap4cls2(
    finalmodel = final_obj,
    predfunc = function(model, newdata) {
      predict(model, newdata, type = "prob") |>
        select(ends_with(yourpositivelevel)) |>
        pull()
    },
    datax = traindatax,
    datay = traindata$EAO,
    yname = "EAO",
    flname = catvars,
    lxname = convars,
    plotname = c(catvars, convars)
  )

  # Waterfall
  shap_sv <- shapviz::shapviz(shap_res$shapley, X = traindatax)
  print(shapviz::sv_waterfall(shap_sv, row_id = 1) +
    theme(text = element_text(family = "serif")))

  shap_res
}

# Run SHAP for key models
load("cls2/evalresult_logistic.RData")
shap_logistic <- run_model_shap(final_logistic, "Logistic", traindatax, traindata)

load("cls2/evalresult_rf.RData")
shap_rf <- run_model_shap(final_rf, "RF", traindatax, traindata)

load("cls2/evalresult_xgboost.RData")
shap_xgboost <- run_model_shap(final_xgboost, "XGBoost", traindatax, traindata)

load("cls2/evalresult_lightgbm.RData")
shap_lightgbm <- run_model_shap(final_lightgbm, "LightGBM", traindatax, traindata)

load("cls2/evalresult_svm.RData")
shap_svm <- run_model_shap(final_svm, "SVM", traindatax, traindata)

load("cls2/evalresult_mlp.RData")
shap_mlp <- run_model_shap(final_mlp, "MLP", traindatax, traindata)

cat("\n=== SHAP analysis completed ===\n")
