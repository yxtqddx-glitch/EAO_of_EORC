# 适用于独热编码过程的新命名函数
new_dummy_names <- function (var, lvl, ordinal = FALSE) {
  args <- vctrs::vec_recycle_common(var, lvl)
  var <- args[[1]]
  lvl <- args[[2]]
  nms <- paste(var, lvl, sep = "_")
  nms
}
# 二分类模型评估指标
metricset_cls2 <- metric_set(
  yardstick::roc_auc, 
  yardstick::pr_auc, 
  yardstick::accuracy
)
# 分类型、连续型自变量名称
getcategory <- function(datax){
  if(sum(sapply(datax, is.factor)) == 0){
    catvars <- NULL
  } else{
    catvars <- 
      colnames(datax)[sapply(datax, is.factor)]
  }
  return(catvars)
}
getcontinuous <- function(datax){
  if(sum(sapply(datax, is.factor)) == 0){
    convars <- colnames(datax)
  } else if(sum(sapply(datax, is.factor)) == ncol(datax)){
    convars <- NULL
  } else{
    catvars <- 
      colnames(datax)[sapply(datax, is.factor)]
    convars <- setdiff(colnames(datax), catvars)
  }
  return(convars)
}
# 二分类模型预测评估函数
eval4cls2 <- function(model, dataset, yname, modelname, datasetname,
                      cutoff, positivelevel, negativelevel) {
  ylevels <- c(negativelevel, positivelevel)
  ylevels2 <- c(positivelevel, negativelevel)
  # 预测概率
  predresult <- model %>% 
    predict(new_data = dataset, type = "prob") %>% 
    dplyr::select(all_of(paste0(".pred_", ylevels))) %>% 
    mutate(.obs = dataset[[yname]],
           data = datasetname,
           model = modelname)
  # 预测概率分布
  predprobplot <- predresult %>%
    ggplot(aes(x = .data[[paste0(".pred_", positivelevel)]])) +
    geom_histogram(aes(y = after_stat(density),
                       fill = .data[[".obs"]]),
                   color = "black",
                   position = "dodge") +
    geom_density(aes(color = .data[[".obs"]]), 
                 linewidth = 1.2) +
    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
    labs(x = paste("Predicted Probability", 
                   yname, "=", positivelevel),
         fill = yname, 
         color = yname) +
    theme_bw() +
    theme(panel.grid = element_blank(),
          text = element_text(family = "serif"),
          legend.position = "inside",
          legend.justification = c(0.5,0.9),
          legend.background = element_blank(),
          legend.key = element_blank())
  # 阳性预测概率
  predprob4pos <- predresult %>%
    dplyr::select(paste0(".pred_", positivelevel)) %>%
    pull()
  # pROC
  proc <- pROC::roc(response = dataset[[yname]], 
                    predictor = predprob4pos,
                    levels = ylevels,
                    direction = "<", 
                    ci = T)
  # ROC
  rocaucresult <- predresult %>%
    roc_auc(.obs, 
            paste0(".pred_", positivelevel), 
            event_level = "second")
  rocresult <- predresult %>%
    roc_curve(.obs, 
              paste0(".pred_", positivelevel), 
              event_level = "second") %>%
    mutate(
      data = datasetname,
      model = modelname,
      rocauc = rocaucresult$.estimate,
      curvelab = paste0(
        "ROCAUC=", 
        sprintf("%.4f", rocauc),
        "(", sprintf("%.4f", as.numeric(pROC::ci.auc(proc))[1]), 
        "-", sprintf("%.4f", as.numeric(pROC::ci.auc(proc))[3]),
        ")"
      )
    )
  rocplot <- rocresult %>%
    ggplot(aes(x = 1-specificity, 
               y = sensitivity, 
               color = curvelab)) +
    geom_path(linewidth = 1) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                       breaks = seq(0, 1, by = 0.2),
                       labels = c(0, seq(0.2, 0.8, by = 0.2), 1)) +
    scale_y_continuous(expand = c(0,0), limits = c(0, 1),
                       breaks = seq(0, 1, by = 0.2),
                       labels = c(0, seq(0.2, 0.8, by = 0.2), 1)) +
    scale_color_manual(values = "blue") +
    labs(color = "") +
    theme_bw() +
    theme(panel.grid = element_blank(),
          legend.position = "inside",
          legend.justification = c(1,0),
          legend.background = element_blank(),
          legend.key = element_blank(), 
          plot.margin = ggplot2::margin(t = 10, b = 5, l = 5, r = 10),
          text = element_text(family = "serif"))
  # PR
  pr_boot <- list()
  for (i in 1:500) {
    set.seed(i)
    pri <- predresult %>%
      slice_sample(n = nrow(predresult), replace = T) %>%
      pr_auc(.obs, 
             paste0(".pred_", positivelevel), 
             event_level = "second")
    pr_boot[[i]] <- pri
  }
  pr_boot2 <- pr_boot %>%
    bind_rows() %>%
    pull(.estimate)
  praucresult <- predresult %>%
    pr_auc(.obs, 
           paste0(".pred_", positivelevel), 
           event_level = "second")
  prresult <- predresult %>%
    pr_curve(.obs, 
             paste0(".pred_", positivelevel), 
             event_level = "second") %>%
    mutate(
      data = datasetname,
      model = modelname,
      PRAUC = praucresult$.estimate,
      curvelab = paste0("PRAUC=", 
                        sprintf("%.4f", PRAUC),
                        "(", sprintf("%.4f", quantile(pr_boot2, 0.025)), 
                        "-", sprintf("%.4f", quantile(pr_boot2, 0.975)),
                        ")")
    )
  prplot <- prresult %>%
    ggplot(aes(x = recall, y = precision, color = curvelab)) +
    geom_path(linewidth = 1) +
    geom_abline(slope = -1, intercept = 1, linetype = "dashed") +
    scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                       breaks = seq(0, 1, by = 0.2),
                       labels = c(0, seq(0.2, 0.8, by = 0.2), 1)) +
    scale_y_continuous(expand = c(0,0), limits = c(0, 1),
                       breaks = seq(0, 1, by = 0.2),
                       labels = c(0, seq(0.2, 0.8, by = 0.2), 1)) +
    scale_color_manual(values = "red") +
    labs(color = "") +
    theme_bw() +
    theme(panel.grid = element_blank(),
          legend.position = "inside",
          legend.justification = c(0,0),
          legend.background = element_blank(),
          legend.key = element_blank(), 
          plot.margin = ggplot2::margin(t = 10, b = 5, l = 5, r = 10),
          text = element_text(family = "serif"))
  # KS-PLOT
  ksresult <- rocresult %>%
    mutate(p = .threshold,
           Pos = 1-sensitivity,
           Neg = specificity)
  ksresult2 <- ksresult %>%
    mutate(DIFF = Neg - Pos) %>%
    dplyr::select(p, Neg, Pos, DIFF) %>%
    slice_max(order_by = DIFF, n = 1) %>%
    slice_min(order_by = p, n = 1)
  ksplot <- ksresult %>%
    dplyr::select(p, Neg, Pos) %>%
    pivot_longer(cols = -1) %>%
    ggplot() +
    geom_line(aes(x = p, y = value, color = name), linewidth = 1) +
    geom_segment(x = ksresult2$p, xend = ksresult2$p, 
                 y = ksresult2$Pos, yend = ksresult2$Neg,
                 color = "black", linetype = "dashed", linewidth = 1) + 
    scale_x_continuous(expand = c(0, 0), limits = c(0, 1),
                       breaks = seq(0, 1, by = 0.2),
                       labels = c(0, seq(0.2, 0.8, by = 0.2), 1)) +
    scale_y_continuous(expand = c(0, 0), limits = c(0, 1),
                       breaks = seq(0, 1, by = 0.2),
                       labels = c(0, seq(0.2, 0.8, by = 0.2), 1)) +
    scale_color_manual(values = c("#F8766D", "#00BFC4"),
                       breaks = c("Neg", "Pos"),
                       labels = ylevels) +
    labs(color = yname, 
         y = "Cumulative Rate", 
         x = paste0("p=", sprintf("%.4f", ksresult2$p),
                    ", KS=", sprintf("%.4f", ksresult2$DIFF))) +
    theme_bw() +
    theme(legend.position = "inside",
          legend.justification = c(0,1),
          legend.background = element_blank(),
          legend.key = element_blank(),
          panel.grid = element_blank(),
          text = element_text(family = "serif"))
  # DCA
  dcaresult <- rmda::decision_curve(
    as.formula(paste0(".obs ~ .pred_", positivelevel)),
    data = predresult %>%
      mutate(.obs = ifelse(.obs == positivelevel, 1, 0)),
    fitted.risk = T
  ) 
  dcadata <- dcaresult$derived.data %>%
    mutate(
      model = ifelse(model %in% c("All", "None"), model, modelname)
    )
  rmda::plot_decision_curve(
    dcaresult,  
    curve.names = modelname,
    cost.benefit.axis = F,
    las = 1,
    confidence.intervals = F,
    family = "serif"
  )
  rmda::plot_clinical_impact(
    dcaresult,
    cost.benefit.axis = F,
    col = c("red", "blue"),
    las = 1,
    confidence.intervals = F,
    family = "serif"
  )
  # 校准曲线
  if (length(unique(predprob4pos)) <= 5) {
    predprob4pos_group <- predprob4pos
  } else {
    predprob4pos_group <- as.numeric(as.character(
      cut(predprob4pos, 
          breaks = seq(0, 1, by = 0.2), 
          labels = seq(1, 10, by = 2))
    )) / 10
  }
  caliresult <- predresult %>%
    mutate(predprobgroup = predprob4pos_group) %>%
    dplyr::group_by(predprobgroup) %>%
    dplyr::count(.obs) %>%
    mutate(N = sum(n),
           Fraction = n/N) %>%
    filter(.obs == positivelevel) %>%
    mutate(data = datasetname, model = modelname)
  caliplot <- caliresult %>%
    ggplot(aes(x = predprobgroup, y = Fraction)) +
    geom_line(color = "brown", linewidth = 1) +
    geom_point(color = "brown", size = 3, pch = 15) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                       breaks = seq(0, 1, by = 0.1),
                       labels = seq(0, 1, by = 0.1)) +
    scale_y_continuous(expand = c(0,0), limits = c(0, 1),
                       breaks = seq(0, 1, by = 0.1),
                       labels = seq(0, 1, by = 0.1)) +
    labs(x = "Bin Midpoint", y = "Event Rate") +
    theme_bw() +
    theme(panel.grid = element_blank())
  # 预测分类
  if (cutoff == "yueden") {
    yuedencf <- rocresult %>%
      mutate(yueden = sensitivity + specificity - 1) %>%
      slice_max(yueden) %>%
      slice_max(sensitivity) %>%
      slice_max(specificity) %>%
      slice_min(.threshold - 0.5) %>%
      # slice_min(0.5 - .threshold) %>%
      pull(.threshold)
  } else {
    yuedencf <- cutoff
  }
  predresult <- predresult %>%
    mutate(
      .pred_class = factor(
        ifelse(predprob4pos >= yuedencf, 
               positivelevel, 
               negativelevel),
        levels = ylevels
      )
    )
  # 混淆矩阵
  cmresult <- predresult %>%
    conf_mat(truth = .obs, estimate = .pred_class)
  cmplot <- cmresult$table %>%
    as.data.frame() %>%
    mutate(
      Prediction = factor(Prediction, levels = ylevels2),
      Truth = factor(Truth, levels = ylevels2)
    ) %>%
    cvms::plot_confusion_matrix(
      target_col = "Truth",
      prediction_col = "Prediction",
      counts_col = "Freq",
      digits = 2,
      palette = list(low="white", high="cyan"),
      tile_border_color = "black",
      darkness = 1
    )
  # 合并指标
  evalresult <- cmresult %>%
    summary(event_level = "second") %>%
    bind_rows(rocaucresult) %>%
    bind_rows(praucresult) %>%
    mutate(data = datasetname,
           model = modelname,
           diycutoff = yuedencf)
  # 返回结果list
  return(list(prediction = predresult,
              predprobplot = predprobplot,
              rocresult = rocresult,
              rocplot = rocplot,
              proc = proc,
              prresult = prresult,
              prplot = prplot,
              ksplot = ksplot,
              dcadata = dcadata,
              caliresult = caliresult,
              caliplot = caliplot,
              cmresult = cmresult,
              cmplot = cmplot,
              metrics = evalresult,
              diycutoff = yuedencf))
}
# 最优超参数交叉验证结果提取函数
bestcv4cls2 <- function(wkflow, tuneresult, hpbest, yname, 
                        modelname, v, positivelevel) {
  # 评估指标设定
  metrictemp <- metric_set(yardstick::roc_auc, yardstick::pr_auc)
  # 调优的超参数个数
  hplength <- wkflow %>%
    extract_parameter_set_dials() %>%
    pull(name) %>%
    length()
  # 交叉验证过程中验证集的预测结果-最优参数
  predcv <- tuneresult %>%
    collect_predictions() %>%
    inner_join(hpbest[, 1:hplength]) %>%
    dplyr::rename(".obs" = all_of(yname))
  # 交叉验证过程中验证集的预测结果评估-最优参数
  evalcv <- predcv %>%
    dplyr::group_by(id) %>%
    metrictemp(.obs, 
               paste0(".pred_", positivelevel), 
               event_level = "second") %>%
    mutate(model = modelname) %>%
    dplyr::group_by(.metric) %>%
    mutate(mean = mean(.estimate),
           sd = sd(.estimate)/sqrt(v))
  # 交叉验证过程中验证集的预测结果图示-最优参数
  cvroc <- predcv %>%
    dplyr::group_by(id) %>%
    roc_curve(.obs, 
              paste0(".pred_", positivelevel), 
              event_level = "second") %>%
    ungroup() %>%
    left_join(evalcv[, c(1,2,4)] %>% filter(.metric == "roc_auc"), 
              by = "id") %>%
    mutate(idAUC = paste(id, " ROC_AUC:", sprintf("%.4f", .estimate)),
           idAUC = forcats::as_factor(idAUC)) %>%
    ggplot(aes(x = 1-specificity, y = sensitivity, color = idAUC)) +
    geom_path(linewidth = 1) +
    geom_abline(linetype = "dashed") +
    scale_x_continuous(expand = c(0, 0), limits = c(0, 1),
                       breaks = seq(0, 1, by = 0.2),
                       labels = c(0, seq(0.2, 0.8, by = 0.2), 1)) +
    scale_y_continuous(expand = c(0, 0), limits = c(0, 1),
                       breaks = seq(0, 1, by = 0.2),
                       labels = c(0, seq(0.2, 0.8, by = 0.2), 1)) +
    labs(color = "") +
    theme_bw() +
    theme(panel.grid = element_blank(),
          legend.position = "inside",
          legend.justification = c(1,0),
          legend.background = element_blank(),
          legend.key = element_blank(), 
          text = element_text(family = "serif"))
  cvpr <- predcv %>%
    dplyr::group_by(id) %>%
    pr_curve(.obs, 
              paste0(".pred_", positivelevel), 
              event_level = "second") %>%
    ungroup() %>%
    left_join(evalcv[, c(1,2,4)] %>% filter(.metric == "pr_auc"), 
              by = "id") %>%
    mutate(idAUC = paste(id, " PR_AUC:", sprintf("%.4f", .estimate)),
           idAUC = forcats::as_factor(idAUC)) %>%
    ggplot(aes(x = recall, y = precision, color = idAUC)) +
    geom_path(linewidth = 1) +
    geom_abline(linetype = "dashed", intercept = 1, slope = -1) +
    scale_x_continuous(expand = c(0, 0), limits = c(0, 1),
                       breaks = seq(0, 1, by = 0.2),
                       labels = c(0, seq(0.2, 0.8, by = 0.2), 1)) +
    scale_y_continuous(expand = c(0, 0), limits = c(0, 1),
                       breaks = seq(0, 1, by = 0.2),
                       labels = c(0, seq(0.2, 0.8, by = 0.2), 1)) +
    labs(color = "") +
    theme_bw() +
    theme(panel.grid = element_blank(),
          legend.position = "inside",
          legend.justification = c(0,0),
          legend.background = element_blank(),
          legend.key = element_blank(), 
          text = element_text(family = "serif"))
  return(list(cvroc = cvroc,
              cvpr = cvpr,
              evalcv = evalcv))
}
# dalex变量重要性
viplot <- function(explainer, showN=10){
  set.seed(915)
  vip <- DALEX::model_parts(
    explainer,
    type = "ratio"
  )
  pi <- plot(vip, show_boxplots = FALSE, max_vars = showN) +
    labs(subtitle = NULL) +
    theme(text = element_text(family = "serif"))
  return(list(data = as.data.frame(vip),
              plot = pi))
}
# dalex偏依赖图
pdplot <- function(explainer, vars){
  set.seed(915)
  pdp <- DALEX::model_profile(
    explainer,
    variables = vars
  )
  plot(pdp) +
    labs(subtitle = NULL) +
    theme(text = element_text(family = "serif"))
}
# shap相关函数
shap4cls2 <- function(finalmodel, predfunc, datax, datay,
                      yname, flname, lxname, plotname) {
  # shap-explain对象
  set.seed(915)
  shap <- fastshap::explain(
    finalmodel, 
    X = as.data.frame(datax),
    nsim = 10,
    adjust = T,
    pred_wrapper = predfunc,
    shap_only = F
  )
  shap$shapley_values[is.na(shap$shapley_values)] <- 0
  # 分类变量
  data3d <- NULL
  data3d1 <- NULL
  data3d2 <- NULL
  data3d3 <- NULL
  shapimpd <- NULL
  shapplotd_facet <- NULL
  shapplotd_one <- NULL
  if(!is.null(flname)){
    data1d <- shap$shapley_values %>%
      as.data.frame() %>%
      dplyr::select(all_of(flname)) %>%  
      dplyr::mutate(id = 1:n()) %>%
      pivot_longer(cols = -ncol(.),
                   names_to = "feature",
                   values_to = "shap")
    data2d <- shap$feature_values  %>%
      dplyr::select(all_of(flname)) %>%   
      dplyr::mutate(id = 1:n()) %>%
      pivot_longer(cols = -ncol(.),
                   names_to = "feature")
    data3d <- data1d %>%
      left_join(data2d, by = c("id", "feature")) %>%
      left_join(data.frame(id = 1:length(datay), Y = datay),
                by = "id")
    data3d1 <- data3d %>%
      pivot_wider(names_from = c(feature, value),
                  values_from = shap, 
                  values_fill = 0) %>%
      pivot_longer(cols = -c(1,2),
                   names_to = "feature",
                   values_to = "shap")
    data3d2 <- data3d %>%
      mutate(shap = 1) %>%
      pivot_wider(names_from = c(feature, value), 
                  values_from = shap, 
                  values_fill = 0) %>%
      pivot_longer(cols = -c(1,2),
                   names_to = "feature",
                   values_to = "value")
    data3d3 <- data3d1 %>%
      left_join(data3d2, by = c("id", "Y", "feature"))
    shapimpd <- data1d %>%
      dplyr::group_by(feature) %>%
      dplyr::summarise(
        shap.abs.mean = mean(abs(shap), na.rm = T),
        .groups = "drop"
      ) %>%
      dplyr::arrange(shap.abs.mean) %>%
      dplyr::mutate(feature = forcats::as_factor(feature))
    shapplotd_facet <- data3d %>%
      na.omit() %>%
      left_join(shapimpd, by = c("feature")) %>%
      dplyr::arrange(-shap.abs.mean) %>%
      dplyr::mutate(feature = forcats::as_factor(feature)) %>%
      ggplot(aes(x = value, y = shap)) +
      geom_violin(fill = "lightgreen") +
      geom_point(aes(color = Y)) + 
      geom_hline(yintercept = 0, color = "grey10") +
      scale_color_viridis_d() +
      labs(x = "", y = "SHAP value", color = yname) +
      facet_wrap(~feature, scales = "free") +
      theme_bw() +
      theme(axis.text.x = element_text(angle = 30, hjust = 1),
            legend.position = "bottom",
            panel.grid.minor.x = element_blank(),
            panel.grid.major.y = element_blank(),
            panel.grid.minor.y = element_blank(),
            text = element_text(family = "serif"))
    library(ggh4x)
    shapplotd_one <- data3d %>%
      na.omit() %>%
      left_join(shapimpd, by = c("feature")) %>%
      dplyr::arrange(shap.abs.mean) %>%
      dplyr::mutate(feature = forcats::as_factor(feature)) %>%
      ggplot(aes(x = interaction(value, feature), y = shap)) +
      geom_boxplot(aes(fill = feature), show.legend = F) +
      geom_hline(yintercept = 0, color = "grey10") +
      scale_x_discrete(NULL, guide = "axis_nested") +
      geom_point(aes(color = Y)) +
      scale_colour_viridis_d() +
      labs(x = "", y = "SHAP value", colour = yname) + 
      coord_flip() +
      theme_bw() +
      theme(legend.position = "bottom",
            panel.grid.minor.x = element_blank(),
            panel.grid.major.x = element_blank(),
            text = element_text(family = "serif"),
            axis.text = element_text(size = 11))
  }
  # 连续变量
  data3c <- NULL
  shapimpc <- NULL
  shapplotc_facet <- NULL
  shapplotc_one <- NULL
  shapplotc_one2 <- NULL
  if(!is.null(lxname)){
    data1c <- shap$shapley_values %>%
      as.data.frame() %>%
      dplyr::select(all_of(lxname)) %>%
      dplyr::mutate(id = 1:n()) %>%
      pivot_longer(cols = -ncol(.),
                   names_to = "feature",
                   values_to = "shap")
    data2c <- shap$feature_values  %>%
      dplyr::select(all_of(lxname)) %>%  
      dplyr::mutate(id = 1:n()) %>%
      pivot_longer(cols = -ncol(.),
                   names_to = "feature")
    data3c <- data1c %>%
      left_join(data2c, by = c("id", "feature")) %>%
      left_join(data.frame(id = 1:length(datay), Y = datay),
                by = "id")
    shapimpc <- data1c %>%
      dplyr::group_by(feature) %>%
      dplyr::summarise(
        shap.abs.mean = mean(abs(shap), na.rm = T),
        .groups = "drop"
      ) %>%
      dplyr::arrange(shap.abs.mean) %>%
      dplyr::mutate(feature = forcats::as_factor(feature))
    shapplotc_facet <- data3c %>%
      na.omit() %>%
      left_join(shapimpc, by = c("feature")) %>%
      dplyr::arrange(-shap.abs.mean) %>%
      dplyr::mutate(feature = forcats::as_factor(feature)) %>%
      ggplot(aes(x = value, y = shap)) +
      geom_point(aes(color = Y)) +
      geom_smooth(method = "loess", 
                  formula = 'y ~ x', 
                  color = "red", 
                  se = F) +
      geom_hline(yintercept = 0, color = "grey10") +
      scale_color_viridis_d() +
      labs(x = "", color = yname, y = "SHAP value") + 
      facet_wrap(~feature, scales = "free") +
      theme_bw() +
      theme(legend.position = "bottom",
            panel.grid = element_blank(),
            text = element_text(family = "serif"))
    shapplotc_one <- data3c %>%
      na.omit() %>%
      left_join(shapimpc, by = c("feature")) %>%
      dplyr::arrange(shap.abs.mean) %>%
      dplyr::mutate(feature = forcats::as_factor(feature)) %>%
      dplyr::group_by(feature) %>%
      dplyr::mutate(
        value = (value - min(value)) / (max(value) - min(value))
      ) %>%
      dplyr::arrange(value) %>%
      dplyr::ungroup() %>%
      ggplot(aes(x = shap, y = feature, color = value)) +
      ggbeeswarm::geom_quasirandom(width = 0.1) +
      geom_vline(xintercept = 0) +
      scale_color_gradient2(
        low = "red", 
        mid = "darkmagenta",
        high = "dodgerblue", 
        midpoint = 0.5,
        breaks = c(0, 1), 
        labels = c("Low", "High"), 
        guide = guide_colorbar(barwidth = 0.5,
                               barheight = 10, 
                               ticks = F,
                               title.position = "right",
                               title.hjust = 0.5)
      ) +
      labs(x = "SHAP value", 
           color = "Feature value", 
           y = "") +
      theme_minimal() +
      theme(legend.title = element_text(angle = -90),
            panel.grid.minor.x = element_blank(),
            text = element_text(family = "serif"),
            axis.text = element_text(size = 11))
    data3c2 <- data3c %>%
      na.omit() %>%
      left_join(shapimpc, by = c("feature")) %>%
      dplyr::arrange(-shap.abs.mean) %>%
      dplyr::mutate(feature = forcats::as_factor(feature)) %>%
      dplyr::group_by(feature) %>%
      dplyr::mutate(
        value2 = 0.01 + 0.98 * (value - min(value)) / (max(value) - min(value))
      ) %>%
      ungroup() %>%
      dplyr::arrange(feature, value) %>%
      dplyr::mutate(
        value3 = value2 + rep(0:(length(lxname)-1), each = nrow(datax))
      )
    xbrks <- seq(0, length(lxname), 0.5)
    xlabs <- rep(0, length(xbrks))
    xlabs[seq(2, length(xbrks), by = 2)] <- 
      as.character(unique(data3c2$feature))
    xlabs[xlabs == 0] <- ""
    shapplotc_one2 <- data3c2 %>%
      ggplot(aes(x = value3, y = shap, color = feature)) +
      geom_point(show.legend = F) +
      geom_smooth(aes(group = feature), 
                  method = "loess",
                  formula = 'y ~ x',
                  color = "black",
                  se = F) +
      geom_vline(xintercept = seq(0, length(lxname), by = 1),
                 color = "grey50") +
      geom_hline(yintercept = 0, color = "grey50") +
      scale_x_continuous(breaks = xbrks, 
                         labels = xlabs,
                         expand = c(0,0)) +
      labs(x = "", y = "SHAP value") +
      theme_minimal() +
      theme(panel.grid= element_blank(),
            text = element_text(family = "serif"),
            axis.text = element_text(size = 11))
  }
  # shap变量重要性
  shapvip <- shap$shapley_values %>%
    as.data.frame() %>%
    dplyr::mutate(id = 1:n()) %>%
    pivot_longer(cols = -(ncol(.)), 
                 names_to = "feature",
                 values_to = "shap") %>%
    dplyr::group_by(feature) %>%
    dplyr::summarise(
      shap.abs.mean = mean(abs(shap), na.rm = T), 
      .groups = "drop"
    ) %>%
    dplyr::arrange(shap.abs.mean) %>%
    dplyr::mutate(feature = forcats::as_factor(feature))
  shapvipplot <- shapvip %>%
    dplyr::filter(feature %in% c(plotname)) %>%
    ggplot(aes(x = shap.abs.mean, 
               y = feature, 
               fill = shap.abs.mean)) +
    geom_col(show.legend = F, color = "grey") +
    scale_fill_distiller(palette = "YlOrRd", direction = 1) +
    labs(y = "", x = "mean(|SHAP value|)") +
    theme_minimal() +
    theme(text = element_text(family = "serif"),
          axis.text = element_text(size = 11))
  # 合并shap数据
  shapdata_unity <- data3d3 %>%
    bind_rows(data3c)
  # shap变量重要性
  shapvip_unity <- shapdata_unity %>%
    dplyr::group_by(feature) %>%
    dplyr::summarise(importance = mean(abs(shap)), 
                     .groups = "drop")
  # shap变量重要性图
  shapvipplot_unity <- shapvip_unity %>%
    mutate(feature = stats::reorder(feature, importance)) %>%
    ggplot(aes(x = importance, y = feature, fill = importance)) +
    geom_col(show.legend = F, color = "grey") +
    scale_fill_distiller(palette = "YlOrRd", direction = 1) +
    labs(fill = "", y = "", x = "mean(|SHAP value|)") +
    theme_minimal() +
    theme(text = element_text(family = "serif"),
          axis.text = element_text(size = 11))
  # shap依赖图
  shapplot_unity <- shapdata_unity %>%
    na.omit() %>%
    left_join(shapvip_unity, by = c("feature")) %>%
    dplyr::arrange(importance) %>%
    dplyr::mutate(feature = forcats::as_factor(feature)) %>%
    dplyr::group_by(feature) %>%
    dplyr::mutate(
      value = (value - min(value)) / (max(value) - min(value))
    ) %>%
    dplyr::arrange(value) %>%
    dplyr::ungroup() %>%
    ggplot(aes(x = shap, y = feature, color = value)) +
    ggbeeswarm::geom_quasirandom(width = 0.1) +
    geom_vline(xintercept = 0) +
    scale_color_gradient2(
      low = "red", 
      mid = "darkmagenta",
      high = "dodgerblue", 
      midpoint = 0.5,
      breaks = c(0, 1), 
      labels = c("Low", "High"), 
      guide = guide_colorbar(
        barwidth = 0.5,
        barheight = 10, 
        ticks = F,
        title.position = "right",
        title.hjust = 0.5
      )
    ) +
    labs(x = "SHAP value", color = "Feature value", y = "") + 
    tidytext::scale_y_reordered() +
    theme_minimal() +
    theme(legend.title = element_text(angle = -90),
          panel.grid.minor.x = element_blank(),
          text = element_text(family = "serif"),
          axis.text = element_text(size = 11))
  # 统一shap
  data1 <- shap$shapley_values %>%
    as.data.frame() %>%
    dplyr::select(all_of(plotname)) %>%  
    dplyr::mutate(id = 1:n()) %>%
    pivot_longer(cols = -ncol(.),
                 names_to = "feature",
                 values_to = "shap")
  data2 <- shap$feature_values  %>%
    data.matrix() %>%
    as.data.frame() %>%
    dplyr::select(all_of(plotname)) %>%  
    dplyr::mutate(id = 1:n()) %>%
    pivot_longer(cols = -ncol(.),
                 names_to = "feature")
  data3 <- data1 %>%
    left_join(data2, by = c("id", "feature")) %>%
    na.omit() %>%
    left_join(shapvip, by = c("feature")) %>%
    dplyr::arrange(shap.abs.mean) %>%
    dplyr::mutate(feature = forcats::as_factor(feature)) %>%
    dplyr::group_by(feature) %>%
    dplyr::mutate(
      value = (value - min(value)) / (max(value) - min(value))
    ) %>%
    dplyr::arrange(value) %>%
    dplyr::ungroup()
  shap1 <- data3 %>%
    dplyr::filter(feature %in% c(plotname)) %>%
    ggplot(aes(x = shap, y = feature, color = value)) +
    ggbeeswarm::geom_quasirandom(width = 0.1) +
    geom_vline(xintercept = 0) +
    scale_color_gradient2(
      low = "red", 
      mid = "darkmagenta",
      high = "dodgerblue", 
      midpoint = 0.5,
      breaks = c(0, 1), 
      labels = c("Low", "High"), 
      guide = guide_colorbar(
        barwidth = 0.5,
        barheight = 10, 
        ticks = F,
        title.position = "right",
        title.hjust = 0.5
      )
    ) +
    labs(x = "SHAP value",
         color = "Feature Value",
         y = "") +
    theme_minimal() +
    theme(legend.title = element_text(angle = -90),
          panel.grid.minor.x = element_blank(),
          text = element_text(family = "serif"),
          axis.text = element_text(size = 11))
  return(list(shapley = shap,
              shapdatad = data3d,
              shapplotd_one = shapplotd_one,
              shapplotd_facet = shapplotd_facet,
              shapdatac = data3c,
              shapplotc_one = shapplotc_one,
              shapplotc_one2 = shapplotc_one2,
              shapplotc_facet = shapplotc_facet,
              shapvip = shapvip,
              shapvipplot = shapvipplot,
              shapplot = shap1,
              shapdata_unity = shapdata_unity,
              shapvip_unity = shapvip_unity,
              shapvipplot_unity = shapvipplot_unity,
              shapplot_unity = shapplot_unity))
}
shapplotplus <- function(
    shapresult, 
    plotname,
    bottomxstart,
    bottomxstop,
    topxrange,
    xtick = 2
){
  shap <- shapresult$shapley
  shapviptemp <- shapresult$shapvip %>%
    dplyr::filter(feature %in% c(plotname))
  data1 <- shap$shapley_values %>%
    as.data.frame() %>%
    dplyr::select(all_of(plotname)) %>%  
    dplyr::mutate(id = 1:n()) %>%
    pivot_longer(cols = -ncol(.),
                 names_to = "feature",
                 values_to = "shap")
  data2 <- shap$feature_values  %>%
    data.matrix() %>%
    as.data.frame() %>%
    dplyr::select(all_of(plotname)) %>%  
    dplyr::mutate(id = 1:n()) %>%
    pivot_longer(cols = -ncol(.),
                 names_to = "feature")
  data3 <- data1 %>%
    left_join(data2, by = c("id", "feature")) %>%
    na.omit() %>%
    left_join(shapviptemp, by = c("feature")) %>%
    na.omit() %>%
    dplyr::arrange(shap.abs.mean) %>%
    dplyr::mutate(feature = forcats::as_factor(feature)) %>%
    dplyr::group_by(feature) %>%
    dplyr::mutate(
      value = (value - min(value)) / (max(value) - min(value))
    ) %>%
    dplyr::arrange(value) %>%
    dplyr::ungroup()
  sk <- topxrange/(bottomxstop-bottomxstart)
  data3$shap2 <- 
    sk * (data3$shap - bottomxstart)
  ggplot() +
    geom_col(
      data = shapviptemp,
      mapping = aes(x = shap.abs.mean, y = feature), 
      fill = "lightgreen", 
      alpha = 0.75
    ) +
    ggbeeswarm::geom_quasirandom(
      data = data3,
      mapping = aes(x = shap2, y = feature, color = value),
      width = 0.1,
      size = 1
    ) +
    geom_vline(xintercept = sk * (0 - bottomxstart)) +
    scale_color_gradient2(
      low = "red", 
      mid = "darkmagenta",
      high = "dodgerblue", 
      midpoint = 0.5,
      breaks = c(0, 1), 
      labels = c("Low", "High"),
      guide = guide_colorbar(barwidth = 0.5,
                             barheight = 10, 
                             title.position = "right",
                             title.hjust = 0.5,
                             ticks = F)
    ) +
    scale_x_continuous(
      expand = c(0,0),
      limits = c(0, topxrange),
      breaks = seq(0, topxrange, length = 5),
      labels = round(seq(bottomxstart, bottomxstop, length = 5), xtick),
      sec.axis = sec_axis(
        ~.,
        name = "mean(|SHAP value|)"
      )
    ) +
    labs(color = "feature value", y = "", x = "SHAP value") +
    theme_classic() +
    theme(legend.title = element_text(angle = -90),
          panel.grid.major.y = element_line(color = "grey"),
          text = element_text(family = "serif"),
          axis.text = element_text(size = 11))
  
}
# 单变量shap图示
sdplot <- function(shapresult, varname, yname, withmargin = T){
  if (varname %in% shapresult$shapdatad$feature) {
    ploti <- shapresult$shapdatad %>%
      dplyr::filter(feature == varname) %>% 
      na.omit() %>%
      ggplot(aes(x = value, y = shap)) +
      geom_violin(fill = "lightgreen") +
      geom_point(aes(color = Y)) + 
      geom_hline(yintercept = 0, color = "grey10") +
      scale_color_viridis_d() +
      labs(x = varname, color = yname, y = "SHAP value") + 
      theme_bw() + 
      theme(panel.grid.minor.x = element_blank(),
            panel.grid.major.y = element_blank(),
            panel.grid.minor.y = element_blank(),
            text = element_text(family = "serif"))
    if (withmargin) {
      ggExtra::ggMarginal(
        ploti + theme(legend.position = "bottom"),
        type = "histogram",
        margins = "y",
        fill = "skyblue"
      )
    } else {
      return(ploti)
    }
  } else {
    ploti <- shapresult$shapdatac %>%
      dplyr::filter(feature == varname) %>% 
      na.omit() %>%
      ggplot(aes(x = value, y = shap)) +
      geom_point(aes(color = Y)) +
      geom_smooth(color = "red", 
                  se = F,
                  method = 'loess',
                  formula = 'y ~ x') +
      geom_hline(yintercept = 0, color = "grey10") +
      scale_color_viridis_d() +
      labs(x = varname, color = yname, y = "SHAP value") + 
      theme_bw() + 
      theme(panel.grid = element_blank(),
            text = element_text(family = "serif"))
  }
  if (withmargin) {
    ggExtra::ggMarginal(
      ploti + theme(legend.position = "bottom"),
      type = "histogram",
      fill = "skyblue"
    )
  } else {
    return(ploti)
  }
}
predict_model.workflow <- function(x, newdata, type, ...) {
  res1 <- predict(x, new_data = newdata, type = "class")
  res2 <- predict(x, new_data = newdata, type = "prob")
  switch(
    type,
    raw = data.frame(Response = res1[[1]], 
                     stringsAsFactors = FALSE),
    prob = as.data.frame(res2, 
                         check.names = FALSE)
  )
}
model_type.workflow <- function(x, ...) 'classification'
predict_model.model_stack <- function(x, newdata, type, ...) {
  res1 <- predict(x, new_data = newdata, type = "class")
  res2 <- predict(x, new_data = newdata, type = "prob")
  switch(
    type,
    raw = data.frame(Response = res1[[1]], 
                     stringsAsFactors = FALSE),
    prob = as.data.frame(res2, 
                         check.names = FALSE)
  )
}
model_type.model_stack <- function(x, ...) 'classification'
