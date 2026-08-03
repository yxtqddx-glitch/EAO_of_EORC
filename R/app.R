# ============================================================================
# app.R
# Shiny web-based prediction calculator for surgical futility in
# early-onset rectal cancer
#
# Features:
#   - Real-time risk prediction using the stacking ensemble model
#   - Risk stratification (Low/Medium/High)
#   - Local SHAP force plot for individual prediction explanation
#   - Prediction history table
# ============================================================================

library(shiny)
library(DT)
library(tidymodels)
library(rpart)
library(bonsai)
library(mxjqshiny)
library(fastshap)
library(stacks)

# ============================================================================
# Load trained model and training data
# ============================================================================
set.seed(42)
load("shiny_stack_data.RData")  # final_stack_data, traindata_data

xatcol <- c(2:12)  # Column indices of features in traindata_data

catvars <- names(which(sapply(traindata_data[, xatcol], is.factor)))
convars <- setdiff(colnames(traindata_data)[xatcol], catvars)

xname_train <- c(convars, catvars)
xname_input <- c(convars, catvars)

# Risk stratification thresholds
risk_thresholds <- list(
  low    = 0.2069663,  # < low: Low Risk
  medium = 0.3         # < medium: Medium Risk; >= medium: High Risk
)

# ============================================================================
# UI
# ============================================================================
ui <- fluidPage(
  column(width = 1),
  column(
    width = 10,
    titlePanel("Calculator for Predicting EAO of Early Onset Rectal Cancer Surgery"),
    hr(),
    sidebarLayout(
      sidebarPanel(
        h2("New Observation to Predict"),
        fluidRow(
          map2(xname_train, xname_input, mxjqshiny1, data4x = traindata_data)
        ),
        fluidRow(
          actionButton("predict", "Predict", class = "btn-success btn-block")
        )
      ),
      mainPanel(
        h2("Prediction Result"),
        span(
          style = "font-size:30px;color:red;",
          textOutput("prediction")
        ),
        div(
          style = "font-size:24px;font-weight:bold;margin-top:10px;",
          htmlOutput("risk_stratification")
        ),
        hr(),
        plotOutput("shapley", height = "200px"),
        hr(),
        h2("Prediction History"),
        DTOutput("samples"),
        br(),
        hr(),
        h3("Disclaimer"),
        div(
          style = "background-color:#f8f9fa;padding:15px;border-radius:5px;border:1px solid #dee2e6;",
          tags$p(
            style = "color:#6c757d;font-size:14px;",
            "This predictive tool is developed based on a machine learning model and is intended for reference by clinicians only. It cannot replace professional medical judgment."
          ),
          tags$p(
            style = "color:#6c757d;font-size:14px;",
            "The prediction results are based on training with historical data, and actual clinical scenarios may vary."
          ),
          tags$p(
            style = "color:#6c757d;font-size:14px;",
            "The developer assumes no responsibility for any clinical decisions made using this tool."
          )
        )
      )
    )
  ),
  column(width = 1)
)

# ============================================================================
# Server
# ============================================================================
server <- function(input, output) {
  values <- reactiveValues(
    samples = data.frame(),
    tempdf  = data.frame()
  )

  get_risk_stratification <- function(prob) {
    if (prob < risk_thresholds$low) {
      return(list(
        level = "Low Risk",
        color = "green",
        description = "The risk of surgical futility is low."
      ))
    } else if (prob < risk_thresholds$medium) {
      return(list(
        level = "Medium Risk",
        color = "orange",
        description = "The risk of surgical futility is medium."
      ))
    } else {
      return(list(
        level = "High Risk",
        color = "red",
        description = "The risk of surgical futility is high."
      ))
    }
  }

  observeEvent(input$predict, {
    temps <- mxjqshiny2(input, xname_train, traindata_data, xatcol)
    lastN <- nrow(temps$new2)

    prediction <- predict(final_stack_data, temps$new2, type = "prob")[lastN, ]
    pred_prob <- prediction[[2]]

    risk_info <- get_risk_stratification(pred_prob)

    output$prediction <- renderText({
      paste("Predicted Prob(Futility=Yes) is",
            scales::percent(pred_prob, 0.01))
    })

    output$risk_stratification <- renderUI({
      HTML(paste(
        "Risk level: ",
        "<span style='color:", risk_info$color, ";font-weight:bold;'>",
        risk_info$level,
        "</span>",
        " - ", risk_info$description
      ))
    })

    # SHAP force plot for this prediction
    shapresult <- mxjqshiny4(final_stack_data, temps$new2, "yes")
    shapresult2 <- shapviz::shapviz(shapresult)
    output$shapley <- renderPlot(
      shapviz::sv_force(shapresult2)
    )

    # Save to history
    values$samples <- rbind(
      values$samples,
      cbind(temps$new, round(pred_prob, 4), risk_info$level)
    )

    values$tempdf <- values$samples |>
      select(any_of(xname_train), everything())

    colnames(values$tempdf)[1:length(xname_input)] <- xname_input
    if (ncol(values$tempdf) > length(xname_input)) {
      colnames(values$tempdf)[(length(xname_input) + 1):ncol(values$tempdf)] <-
        c("Pred_Prob", "Risk_Level")
    }
  })

  output$samples <- renderDT({
    values$tempdf
  }, options = list(scrollX = TRUE, pageLength = 3, dom = "tp"))
}

# ============================================================================
# Run app
# ============================================================================
shinyApp(ui = ui, server = server)
