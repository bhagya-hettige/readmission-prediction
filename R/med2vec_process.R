library(eeptools)
library(plyr)
# library(dplyr)
library(caret)
library(DMwR) # for smote implementation
library(purrr) # for functional programming (map)
library(pROC) # for AUC calculations
require(PRROC) # for AUC-PR calculations

library(doMC)
registerDoMC(cores = 32)

# Load data from csv
dataDirectory <- '/home/bhet0001/research/readmission/Med2Vec/proj/Med2Vec Model/'
claims <- read.csv(paste(dataDirectory, 'claims_codevectors.csv', sep = ''), header = TRUE)

df <- subset(claims, select = -c(1:2, 6:21))
df <- subset(df, select = -c(PATIENT_ID))
df_new <- df
df_new <- df_new[, order(names(df_new))]

#df_demo <- subset(df, select=c("PATIENT_ID", "SEX", "RACE", "COUNTRY", "SP_STATE_CODE"))

#df_demo <- df_demo[unique(df_demo$PATIENT_ID)]
#write.csv(df_demo, file = "/home/bhet0001/research/readmission/project/demo.csv")

# Distribution counts 
# temp <- unique(subset(df_new, select=c(PATIENT_ID, AGE)))
# table(df_new$AGE)

df_new[['READMITTED']] <- factor(df_new[['READMITTED']])
levels(df_new$READMITTED) <- c("no", "yes")

table(df_new$READMITTED)

write.csv(df_new, file = "/home/bhet0001/research/readmission/project/exp_2.csv")



#df_new$PATIENT_ID <- as.factor(df_new$PATIENT_ID)
#levels(df_new$PATIENT_ID) <- 1:length(levels(df_new$PATIENT_ID))
#df_new$PATIENT_ID <- as.numeric(df_new$PATIENT_ID)

set.seed(3033)
intrain <- createDataPartition(y = df_new$READMITTED, p = 0.75, list = FALSE)
training <- df_new[intrain,]
testing <- df_new[-intrain,]

write.csv(training, file = "/home/bhet0001/research/readmission/project/admission_claims_train.csv")
write.csv(testing, file = "/home/bhet0001/research/readmission/project/admission_claims_test.csv")

table(training$READMITTED)
table(testing$READMITTED)

ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 10, verboseIter = FALSE, summaryFunction = twoClassSummary, classProbs = TRUE)
# Linear SVM - svmLinear ()
# SMOTE
ctrl$sampling <- "smote"
set.seed(3033)
svmLinear <- train(READMITTED ~ .,
                   data = training,
                   method = "svmLinear",
                   verbose = FALSE,
                   metric = "ROC",
                   trControl = ctrl,
                   preProcess = c("center", "scale"),
                   na.action=na.exclude)

# Build custom AUC function to extract AUC
# from the caret model object
test_roc <- function(model, data) {
  roc(data$READMITTED,
      predict(model, data, type = "prob")[, "yes"])
}

svmLinear %>%
  test_roc(data = testing) %>%
  auc()

svmRadial
set.seed(3033)
svmPoly <- train(READMITTED ~ .,
                 data = training,
                 method = "svmPoly",
                 verbose = FALSE,
                 metric = "ROC",
                 trControl = ctrl,
                 preProcess = c("center", "scale"))

set.seed(3033)
svmRadial <- train(READMITTED ~ .,
                   data = training,
                   method = "svmRadial",
                   verbose = FALSE,
                   metric = "ROC",
                   trControl = ctrl,
                   preProcess = c("center", "scale"))

set.seed(3033)
logReg <- train(READMITTED ~.,
                data = training,
                family="binomial",
                trControl=ctrl,
                verbose = FALSE,
                metric = "ROC",
                preProcess = c("center", "scale"))

set.seed(3033)
rf <- train(READMITTED ~ .,
            data = training,
            method = "rf",
            verbose = FALSE,
            metric = "ROC",
            trControl = ctrl,
            preProcess = c("center", "scale"),
            na.action=na.exclude)

set.seed(3033)
xgbTree <- train(READMITTED ~ .,
                 data = training,
                 method = "xgbTree",
                 verbose = FALSE,
                 metric = "ROC",
                 trControl = ctrl,
                 preProcess = c("center", "scale"),
                 na.action=na.exclude)

# Examine results for test set
model_list <- list(svmLinear = svmLinear,
                   svmPoly = svmPoly,
                   svmRadial = svmRadial,
                   logReg = logReg,
                   rf = rf)

model_list_roc <- model_list %>%
  map(test_roc, data = testing)

model_list_roc %>%
  map(auc)

library(dplyr)
results_list_roc <- list(NA)
num_mod <- 1

for(the_roc in model_list_roc){
  
  results_list_roc[[num_mod]] <- 
    data_frame(tpr = the_roc$sensitivities,
               fpr = 1 - the_roc$specificities,
               model = names(model_list)[num_mod])
  
  num_mod <- num_mod + 1
  
}

results_df_roc <- bind_rows(results_list_roc)

# Plot ROC curve for all 5 models

custom_col <- c("#000000", "#009E73", "#0072B2", "#D55E00", "#CC79A7")

ggplot(aes(x = fpr,  y = tpr, group = model), data = results_df_roc) +
  geom_line(aes(color = model), size = 1) +
  scale_color_manual(values = custom_col) +
  geom_abline(intercept = 0, slope = 1, color = "gray", size = 1) +
  theme_bw(base_size = 18)

# AUC-ROC
probs <- predict(svmLinear, testing, type = "prob")[, "yes"]

fg <- probs[testing$READMITTED == "yes"]
bg <- probs[testing$READMITTED == "no"]

roc <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
plot(roc)

# AUC-PR
probs <- predict(svmLinear, testing, type = "prob")[, "yes"]

fg <- probs[testing$READMITTED == "yes"]
bg <- probs[testing$READMITTED == "no"]

pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
plot(pr)

# Function: evaluation metrics
## True positives (TP) - Correctly idd as success
## True negatives (TN) - Correctly idd as failure
## False positives (FP) - success incorrectly idd as failure
## False negatives (FN) - failure incorrectly idd as success
## Precision - P = TP/(TP+FP) how many idd actually success/failure
## Recall - R = TP/(TP+FN) how many of the successes correctly idd
## F-score - F = (2 * P * R)/(P + R) harm mean of precision and recall
prf <- function(predAct){
  ## predAct is two col dataframe of pred,act
  preds = predAct[,1]
  trues = predAct[,2]
  xTab <- table(preds, trues)
  clss <- as.character(sort(unique(preds)))
  r <- matrix(NA, ncol = 7, nrow = 1, 
              dimnames = list(c(),c('Acc',
                                    paste("P",clss[1],sep='_'), 
                                    paste("R",clss[1],sep='_'), 
                                    paste("F",clss[1],sep='_'), 
                                    paste("P",clss[2],sep='_'), 
                                    paste("R",clss[2],sep='_'), 
                                    paste("F",clss[2],sep='_'))))
  r[1,1] <- sum(xTab[1,1],xTab[2,2])/sum(xTab) # Accuracy
  r[1,2] <- xTab[1,1]/sum(xTab[,1]) # Miss Precision
  r[1,3] <- xTab[1,1]/sum(xTab[1,]) # Miss Recall
  r[1,4] <- (2*r[1,2]*r[1,3])/sum(r[1,2],r[1,3]) # Miss F
  r[1,5] <- xTab[2,2]/sum(xTab[2,]) # Hit Precision
  r[1,6] <- xTab[2,2]/sum(xTab[2,]) # Hit Recall
  r[1,7] <- (2*r[1,5]*r[1,6])/sum(r[1,5],r[1,6]) # Hit F
  r}

pred <- predict(svmLinear, testing)
act <- testing$READMITTED
predAct <- data.frame(pred, act)
prf(predAct)
