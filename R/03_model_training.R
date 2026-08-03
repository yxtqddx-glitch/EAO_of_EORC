# ============================================================================
# 03_model_training.R
# Training of 12 machine learning models with Bayesian hyperparameter
# optimization and 5-fold cross-validation
#
# Models:
#   1. Decision Tree (rpart)         7. KNN (kknn)
#   2. Random Forest (randomForest)  8. Logistic Regression (glm)
#   3. XGBoost (xgboost)             9. LASSO (glmnet)
#   4. LightGBM (lightgbm/bonsai)    10. Ridge Regression (glmnet)
#   5. SVM (kernlab)                 11. Elastic Net (glmnet)
#   6. MLP / Neural Network (nnet)   12. Stacking Ensemble (stacks)
# ============================================================================

library(tidymodels)
library(bonsai)
library(doParallel)
library(stacks)

# Source helper functions
source("tidyfuncs4cls2.R")

# Parallel backend
registerDoParallel(
  makePSOCKcluster(max(1, (parallel::detectCores(logical = FALSE)) - 1))
)

# ============================================================================
# 0. Load data
# ============================================================================
traindata <- read.csv("traindata_features.csv")
testdata  <- read.csv("testdata_features.csv")

# Convert categorical variables to factors
factor_vars <- c("EAO", "Neoadjuvant", "MMR",
                 "Lymphovascular_invasion", "pT", "pTNM")

traindata <- traindata |>
  mutate(across(all_of(factor_vars), as.factor))
testdata <- testdata |>
  mutate(across(all_of(factor_vars), as.factor))

# Ensure numeric variables are numeric
num_vars <- setdiff(names(traindata), factor_vars)
traindata[num_vars] <- lapply(traindata[num_vars], as.numeric)
testdata[num_vars]  <- lapply(testdata[num_vars], as.numeric)

skimr::skim(traindata)
str(traindata)

# Set positive/negative levels
yourpositivelevel <- "yes"
yournegativelevel <- "no"

traindata$EAO <- factor(traindata$EAO,
  levels = c(yournegativelevel, yourpositivelevel))
testdata$EAO <- factor(testdata$EAO,
  levels = c(yournegativelevel, yourpositivelevel))

# 5-fold cross-validation
set.seed(123)
folds <- vfold_cv(traindata, v = 5, strata = EAO)

# ============================================================================
# Helper: Bayesian tuning + evaluation for a model
# ============================================================================
tune_and_evaluate <- function(
    wk, model_name, param_update = NULL,
    hp_select = "roc_auc", hp_desc = "cost_complexity") {

  # Bayesian optimization
  if (!is.null(param_update)) {
    set.seed(123)
    tune_res <- wk |>
      tune_bayes(
        resamples = folds,
        initial = 10,
        iter = 50,
        param_info = param_update,
        metrics = metricset_cls2,
        control = control_bayes(
          save_pred = TRUE,
          verbose = TRUE,
          no_improve = 10,
          uncertain = 5,
          event_level = "second",
          parallel_over = "everything",
          save_workflow = TRUE
        )
      )
  } else {
    set.seed(123)
    tune_res <- wk |>
      tune_bayes(
        resamples = folds,
        initial = 10,
        iter = 50,
        metrics = metricset_cls2,
        control = control_bayes(
          save_pred = TRUE,
          verbose = TRUE,
          no_improve = 10,
          uncertain = 5,
          event_level = "second",
          parallel_over = "everything",
          save_workflow = TRUE
        )
      )
  }

  # Select best hyperparameters
  hpbest <- tune_res |>
    select_by_one_std_err(metric = hp_select, desc(!!sym(hp_desc)))

  # Fit final model
  set.seed(123)
  final_model <- wk |>
    finalize_workflow(hpbest) |>
    fit(traindata)

  # Training set evaluation
  predtrain <- eval4cls2(
    model = final_model, dataset = traindata,
    yname = "EAO", modelname = model_name,
    datasetname = "traindata", cutoff = "yueden",
    positivelevel = yourpositivelevel,
    negativelevel = yournegativelevel
  )

  # Test set evaluation
  predtest <- eval4cls2(
    model = final_model, dataset = testdata,
    yname = "EAO", modelname = model_name,
    datasetname = "testdata", cutoff = predtrain$diycutoff,
    positivelevel = yourpositivelevel,
    negativelevel = yournegativelevel
  )

  # CV evaluation
  evalcv <- bestcv4cls2(
    wkflow = wk, tuneresult = tune_res, hpbest = hpbest,
    yname = "EAO", modelname = model_name,
    v = 5, positivelevel = yourpositivelevel
  )

  list(
    tune = tune_res, final = final_model,
    predtrain = predtrain, predtest = predtest,
    evalcv = evalcv, hpbest = hpbest
  )
}

# ============================================================================
# 1. Decision Tree (rpart)
# ============================================================================
cat("\n=== Training Decision Tree ===\n")

datarecipe_dt <- recipe(EAO ~ ., traindata)

model_dt <- decision_tree(
  mode = "classification",
  engine = "rpart",
  tree_depth = tune(),
  min_n = tune(),
  cost_complexity = tune()
) |>
  set_args(model = TRUE)

wk_dt <- workflow() |>
  add_recipe(datarecipe_dt) |>
  add_model(model_dt)

result_dt <- tune_and_evaluate(wk_dt, "DT")

# DALEX variable importance
traindatax <- traindata |> select(-EAO)
explainer_dt <- DALEXtra::explain_tidymodels(
  result_dt$final,
  data = traindatax,
  y = ifelse(traindata$EAO == yourpositivelevel, 1, 0),
  type = "classification",
  label = "DT"
)
vip_dt <- viplot(explainer_dt, showN = 5)
vipdata_dt <- vip_dt$data

save(result_dt$tune, result_dt$predtrain, result_dt$predtest,
     result_dt$evalcv, vipdata_dt,
     file = "cls2/evalresult_dt.RData")

# ============================================================================
# 2. Random Forest (randomForest)
# ============================================================================
cat("\n=== Training Random Forest ===\n")

datarecipe_rf <- recipe(EAO ~ ., traindata)

model_rf <- rand_forest(
  mode = "classification",
  engine = "randomForest",
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) |>
  set_args(importance = TRUE)

wk_rf <- workflow() |>
  add_recipe(datarecipe_rf) |>
  add_model(model_rf)

param_rf <- model_rf |>
  extract_parameter_set_dials() |>
  update(
    mtry = mtry(c(2, 10)),
    trees = trees(c(100, 1000)),
    min_n = min_n(c(7, 55))
  )

result_rf <- tune_and_evaluate(
  wk_rf, "RF", param_update = param_rf,
  hp_desc = "min_n"
)

explainer_rf <- DALEXtra::explain_tidymodels(
  result_rf$final,
  data = traindatax,
  y = ifelse(traindata$EAO == yourpositivelevel, 1, 0),
  type = "classification",
  label = "RF"
)
vip_rf <- viplot(explainer_rf, showN = 9)
vipdata_rf <- vip_rf$data

save(result_rf$tune, result_rf$predtrain, result_rf$predtest,
     result_rf$evalcv, vipdata_rf,
     file = "cls2/evalresult_rf.RData")

# ============================================================================
# 3. XGBoost
# ============================================================================
cat("\n=== Training XGBoost ===\n")

datarecipe_xgboost <- recipe(EAO ~ ., traindata) |>
  step_dummy(all_nominal_predictors(), naming = new_dummy_names)

model_xgboost <- boost_tree(
  mode = "classification",
  engine = "xgboost",
  mtry = tune(),
  trees = 1000,
  min_n = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  stop_iter = 25
) |>
  set_args(validation = 0.2, event_level = "second")

wk_xgboost <- workflow() |>
  add_recipe(datarecipe_xgboost) |>
  add_model(model_xgboost)

param_xgboost <- model_xgboost |>
  extract_parameter_set_dials() |>
  update(mtry = mtry(c(2, 6)))

result_xgboost <- tune_and_evaluate(
  wk_xgboost, "Xgboost", param_update = param_xgboost,
  hp_desc = "min_n"
)

explainer_xgboost <- DALEXtra::explain_tidymodels(
  result_xgboost$final,
  data = traindatax,
  y = ifelse(traindata$EAO == yourpositivelevel, 1, 0),
  type = "classification",
  label = "Xgboost"
)
vip_xgboost <- viplot(explainer_xgboost, showN = 9)
vipdata_xgboost <- vip_xgboost$data

save(result_xgboost$tune, result_xgboost$predtrain, result_xgboost$predtest,
     result_xgboost$evalcv, vipdata_xgboost,
     file = "cls2/evalresult_xgboost.RData")

# ============================================================================
# 4. LightGBM (bonsai)
# ============================================================================
cat("\n=== Training LightGBM ===\n")

datarecipe_lightgbm <- recipe(EAO ~ ., traindata) |>
  step_dummy(all_nominal_predictors(), naming = new_dummy_names)

model_lightgbm <- boost_tree(
  mode = "classification",
  engine = "lightgbm",
  mtry = tune(),
  trees = 1000,
  min_n = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  stop_iter = 25
) |>
  set_args(validation = 0.2, event_level = "second")

wk_lightgbm <- workflow() |>
  add_recipe(datarecipe_lightgbm) |>
  add_model(model_lightgbm)

param_lightgbm <- model_lightgbm |>
  extract_parameter_set_dials() |>
  update(mtry = mtry(c(2, 6)))

result_lightgbm <- tune_and_evaluate(
  wk_lightgbm, "Lightgbm", param_update = param_lightgbm,
  hp_desc = "min_n"
)

explainer_lightgbm <- DALEXtra::explain_tidymodels(
  result_lightgbm$final,
  data = traindatax,
  y = ifelse(traindata$EAO == yourpositivelevel, 1, 0),
  type = "classification",
  label = "Lightgbm"
)
vip_lightgbm <- viplot(explainer_lightgbm, showN = 9)
vipdata_lightgbm <- vip_lightgbm$data

save(result_lightgbm$tune, result_lightgbm$predtrain, result_lightgbm$predtest,
     result_lightgbm$evalcv, vipdata_lightgbm,
     file = "cls2/evalresult_lightgbm.RData")

# ============================================================================
# 5. SVM (kernlab)
# ============================================================================
cat("\n=== Training SVM ===\n")

datarecipe_svm <- recipe(EAO ~ ., traindata) |>
  step_dummy(all_nominal_predictors(), naming = new_dummy_names) |>
  step_normalize(all_numeric_predictors())

model_svm <- svm_rbf(
  mode = "classification",
  engine = "kernlab",
  cost = tune(),
  rbf_sigma = tune()
)

wk_svm <- workflow() |>
  add_recipe(datarecipe_svm) |>
  add_model(model_svm)

result_svm <- tune_and_evaluate(
  wk_svm, "SVM", hp_desc = "cost"
)

explainer_svm <- DALEXtra::explain_tidymodels(
  result_svm$final,
  data = traindatax,
  y = ifelse(traindata$EAO == yourpositivelevel, 1, 0),
  type = "classification",
  label = "SVM"
)
vip_svm <- viplot(explainer_svm, showN = 9)
vipdata_svm <- vip_svm$data

save(result_svm$tune, result_svm$predtrain, result_svm$predtest,
     result_svm$evalcv, vipdata_svm,
     file = "cls2/evalresult_svm.RData")

# ============================================================================
# 6. MLP — Single Hidden Layer Neural Network (nnet)
# ============================================================================
cat("\n=== Training MLP ===\n")

datarecipe_mlp <- recipe(EAO ~ ., traindata) |>
  step_dummy(all_nominal_predictors(), naming = new_dummy_names) |>
  step_normalize(all_numeric_predictors())

model_mlp <- mlp(
  mode = "classification",
  engine = "nnet",
  hidden_units = tune(),
  penalty = tune(),
  epochs = tune()
)

wk_mlp <- workflow() |>
  add_recipe(datarecipe_mlp) |>
  add_model(model_mlp)

result_mlp <- tune_and_evaluate(
  wk_mlp, "MLP", hp_desc = "penalty"
)

explainer_mlp <- DALEXtra::explain_tidymodels(
  result_mlp$final,
  data = traindatax,
  y = ifelse(traindata$EAO == yourpositivelevel, 1, 0),
  type = "classification",
  label = "MLP"
)
vip_mlp <- viplot(explainer_mlp, showN = 9)
vipdata_mlp <- vip_mlp$data

save(result_mlp$tune, result_mlp$predtrain, result_mlp$predtest,
     result_mlp$evalcv, vipdata_mlp,
     file = "cls2/evalresult_mlp.RData")

# ============================================================================
# 7. KNN (kknn)
# ============================================================================
cat("\n=== Training KNN ===\n")

datarecipe_knn <- recipe(EAO ~ ., traindata) |>
  step_dummy(all_nominal_predictors(), naming = new_dummy_names) |>
  step_normalize(all_numeric_predictors())

model_knn <- nearest_neighbor(
  mode = "classification",
  engine = "kknn",
  neighbors = tune(),
  weight_func = tune(),
  dist_power = tune()
)

wk_knn <- workflow() |>
  add_recipe(datarecipe_knn) |>
  add_model(model_knn)

result_knn <- tune_and_evaluate(
  wk_knn, "KNN", hp_desc = "neighbors"
)

explainer_knn <- DALEXtra::explain_tidymodels(
  result_knn$final,
  data = traindatax,
  y = ifelse(traindata$EAO == yourpositivelevel, 1, 0),
  type = "classification",
  label = "KNN"
)
vip_knn <- viplot(explainer_knn, showN = 9)
vipdata_knn <- vip_knn$data

save(result_knn$tune, result_knn$predtrain, result_knn$predtest,
     result_knn$evalcv, vipdata_knn,
     file = "cls2/evalresult_knn.RData")

# ============================================================================
# 8. Logistic Regression (glm)
# ============================================================================
cat("\n=== Training Logistic Regression ===\n")

datarecipe_logistic <- recipe(EAO ~ ., traindata)

model_logistic <- logistic_reg(
  mode = "classification",
  engine = "glm"
)

wk_logistic <- workflow() |>
  add_recipe(datarecipe_logistic) |>
  add_model(model_logistic)

# Logistic regression has no hyperparameters to tune
set.seed(123)
final_logistic <- wk_logistic |>
  fit(traindata)

predtrain_logistic <- eval4cls2(
  model = final_logistic, dataset = traindata,
  yname = "EAO", modelname = "Logistic",
  datasetname = "traindata", cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)

predtest_logistic <- eval4cls2(
  model = final_logistic, dataset = testdata,
  yname = "EAO", modelname = "Logistic",
  datasetname = "testdata", cutoff = predtrain_logistic$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)

# CV evaluation for logistic regression
set.seed(123)
cv_logistic <- wk_logistic |>
  fit_resamples(
    folds,
    metrics = metricset_cls2,
    control = control_resamples(
      save_pred = TRUE, verbose = TRUE,
      event_level = "second",
      parallel_over = "everything",
      save_workflow = TRUE
    )
  )

evalcv_logistic <- list()
metrictemp <- metric_set(yardstick::roc_auc, yardstick::pr_auc)
evalcv_logistic$evalcv <- collect_predictions(cv_logistic) |>
  group_by(id) |>
  metrictemp(EAO, .pred_yes, event_level = "second") |>
  group_by(.metric) |>
  mutate(
    model = "Logistic",
    mean = mean(.estimate),
    sd = sd(.estimate) / sqrt(length(folds$splits))
  )

explainer_logistic <- DALEXtra::explain_tidymodels(
  final_logistic,
  data = traindatax,
  y = ifelse(traindata$EAO == yourpositivelevel, 1, 0),
  type = "classification",
  label = "Logistic"
)
vip_logistic <- viplot(explainer_logistic, showN = 9)
vipdata_logistic <- vip_logistic$data

save(predtrain_logistic, predtest_logistic, evalcv_logistic, vipdata_logistic,
     file = "cls2/evalresult_logistic.RData")

# ============================================================================
# 9. LASSO (glmnet)
# ============================================================================
cat("\n=== Training LASSO ===\n")

datarecipe_lasso <- recipe(EAO ~ ., traindata) |>
  step_dummy(all_nominal_predictors(), naming = new_dummy_names) |>
  step_normalize(all_numeric_predictors())

model_lasso <- logistic_reg(
  mode = "classification",
  engine = "glmnet",
  penalty = tune(),
  mixture = 1
)

wk_lasso <- workflow() |>
  add_recipe(datarecipe_lasso) |>
  add_model(model_lasso)

result_lasso <- tune_and_evaluate(
  wk_lasso, "LASSO", hp_desc = "penalty"
)

explainer_lasso <- DALEXtra::explain_tidymodels(
  result_lasso$final,
  data = traindatax,
  y = ifelse(traindata$EAO == yourpositivelevel, 1, 0),
  type = "classification",
  label = "LASSO"
)
vip_lasso <- viplot(explainer_lasso, showN = 9)
vipdata_lasso <- vip_lasso$data

save(result_lasso$tune, result_lasso$predtrain, result_lasso$predtest,
     result_lasso$evalcv, vipdata_lasso,
     file = "cls2/evalresult_lasso.RData")

# ============================================================================
# 10. Ridge Regression (glmnet)
# ============================================================================
cat("\n=== Training Ridge Regression ===\n")

datarecipe_ridge <- recipe(EAO ~ ., traindata) |>
  step_dummy(all_nominal_predictors(), naming = new_dummy_names) |>
  step_normalize(all_numeric_predictors())

model_ridge <- logistic_reg(
  mode = "classification",
  engine = "glmnet",
  penalty = tune(),
  mixture = 0
)

wk_ridge <- workflow() |>
  add_recipe(datarecipe_ridge) |>
  add_model(model_ridge)

result_ridge <- tune_and_evaluate(
  wk_ridge, "Ridge", hp_desc = "penalty"
)

explainer_ridge <- DALEXtra::explain_tidymodels(
  result_ridge$final,
  data = traindatax,
  y = ifelse(traindata$EAO == yourpositivelevel, 1, 0),
  type = "classification",
  label = "Ridge"
)
vip_ridge <- viplot(explainer_ridge, showN = 9)
vipdata_ridge <- vip_ridge$data

save(result_ridge$tune, result_ridge$predtrain, result_ridge$predtest,
     result_ridge$evalcv, vipdata_ridge,
     file = "cls2/evalresult_ridge.RData")

# ============================================================================
# 11. Elastic Net (glmnet)
# ============================================================================
cat("\n=== Training Elastic Net ===\n")

datarecipe_enet <- recipe(EAO ~ ., traindata) |>
  step_dummy(all_nominal_predictors(), naming = new_dummy_names) |>
  step_normalize(all_numeric_predictors())

model_enet <- logistic_reg(
  mode = "classification",
  engine = "glmnet",
  penalty = tune(),
  mixture = tune()
)

wk_enet <- workflow() |>
  add_recipe(datarecipe_enet) |>
  add_model(model_enet)

result_enet <- tune_and_evaluate(
  wk_enet, "ElasticNet", hp_desc = "penalty"
)

explainer_enet <- DALEXtra::explain_tidymodels(
  result_enet$final,
  data = traindatax,
  y = ifelse(traindata$EAO == yourpositivelevel, 1, 0),
  type = "classification",
  label = "ElasticNet"
)
vip_enet <- viplot(explainer_enet, showN = 9)
vipdata_enet <- vip_enet$data

save(result_enet$tune, result_enet$predtrain, result_enet$predtest,
     result_enet$evalcv, vipdata_enet,
     file = "cls2/evalresult_enet.RData")

cat("\n=== Models 1–11 training completed ===\n")

# ============================================================================
# 12. Stacking Ensemble (stacks)
# ============================================================================
cat("\n=== Building Stacking Ensemble ===\n")

# Load base model tuning results
load("cls2/evalresult_xgboost.RData")
load("cls2/evalresult_knn.RData")
load("cls2/evalresult_rf.RData")

models_stack <- stacks() |>
  add_candidates(tune_xgboost) |>
  add_candidates(tune_knn) |>
  add_candidates(tune_rf)

# Fit meta-learner with LASSO penalty
set.seed(123)
meta_stack <- blend_predictions(
  models_stack,
  penalty = 10^seq(-2, -0.5, length = 20),
  control = control_grid(
    save_pred = TRUE,
    verbose = TRUE,
    event_level = "second",
    parallel_over = "everything",
    save_workflow = TRUE
  )
)

# Fit final base models
set.seed(123)
final_stack <- fit_members(meta_stack)

# Training set evaluation
predtrain_stack <- eval4cls2(
  model = final_stack, dataset = traindata,
  yname = "EAO", modelname = "Stacking",
  datasetname = "traindata", cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)

# Test set evaluation
predtest_stack <- eval4cls2(
  model = final_stack, dataset = testdata,
  yname = "EAO", modelname = "Stacking",
  datasetname = "testdata", cutoff = predtrain_stack$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)

# Stacking variable importance via DALEX
explainer_stack <- DALEXtra::explain_tidymodels(
  final_stack,
  data = traindatax,
  y = ifelse(traindata$EAO == yourpositivelevel, 1, 0),
  type = "classification",
  label = "Stacking"
)
vip_stack <- viplot(explainer_stack, showN = 9)
vipdata_stack <- vip_stack$data

save(predtrain_stack, predtest_stack, vipdata_stack,
     file = "cls2/evalresult_stack.RData")

# Save model for Shiny deployment
final_stack_data <- final_stack
traindata_shiny <- traindata
save(final_stack_data, traindata_shiny,
     file = "cls2shiny/shiny_stack_data.RData")

cat("\n=== All 12 models trained and saved ===\n")
