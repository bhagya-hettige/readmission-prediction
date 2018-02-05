library(eeptools)
library(plyr)
# library(dplyr)
library(caret)
library(DMwR) # for smote implementation
library(purrr) # for functional programming (map)
library(pROC) # for AUC calculations

library(doMC)
registerDoMC(cores = 32)

# Load data from csv
dataDirectory <- '/home/bhet0001/research/readmission/CMS/'
claims <- read.csv(paste(dataDirectory, 'DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv', sep = ''), header = TRUE)
# 3 patient summary csv are there, chronic disease conditions can change over time
patient_2008 <- read.csv(paste(dataDirectory, 'DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv', sep = ''), header = TRUE)
patient_2009 <- read.csv(paste(dataDirectory, 'DE1_0_2009_Beneficiary_Summary_File_Sample_1.csv', sep = ''), header = TRUE)
patient_2010 <- read.csv(paste(dataDirectory, 'DE1_0_2010_Beneficiary_Summary_File_Sample_1.csv', sep = ''), header = TRUE)
# pde <- read.csv(paste(dataDirectory, 'DE1_0_2008_to_2010_Prescription_Drug_Events_Sample_1.csv', sep = ''), header = TRUE)

head(as.integer(patient_2008$year))
patient_2008$year <- 2008
head(as.integer(patient_2009$year))
patient_2009$year <- 2009
head(as.integer(patient_2010$year))
patient_2010$year <- 2010

patients <- rbind(patient_2008, patient_2009, patient_2010)

rm(patient_2008, patient_2009, patient_2010)

# Format the date fields for admission and discharge
claims$CLM_ADMSN_DT <- as.Date(as.character(claims$CLM_ADMSN_DT), format = '%Y%m%d')
claims$NCH_BENE_DSCHRG_DT <- as.Date(as.character(claims$NCH_BENE_DSCHRG_DT), format = '%Y%m%d')
patients$BENE_BIRTH_DT <- as.Date(as.character(patients$BENE_BIRTH_DT), format = '%Y%m%d')

# Select first N records (for faster testing)
# claims <- claims[1:2000,]

# Sort the claim records by patient id and admission date respectively
claims <- claims[order(claims$DESYNPUF_ID, claims$CLM_ADMSN_DT),]

head(as.numeric(claims$year))
claims$year <- as.integer(format(claims$CLM_ADMSN_DT, "%Y"))

df1 <- subset(claims, select = c(DESYNPUF_ID, year, CLM_ADMSN_DT, NCH_BENE_DSCHRG_DT))

# Add column - length of stay
head(as.numeric(df1$L))
df1$L <- as.numeric(claims$NCH_BENE_DSCHRG_DT - claims$CLM_ADMSN_DT)

# Add columns - diagnosis codes
df1$ADMTNG_DGNS_CD <- claims$ADMTNG_ICD9_DGNS_CD
df1$PRMY_DIAG_CD <- claims$ICD9_DGNS_CD_1

# Map diagnosis codes into numeric numbers
df1$ADMTNG_DGNS_CD <- as.factor(df1$ADMTNG_DGNS_CD)
levels(df1$ADMTNG_DGNS_CD) <- 1:length(levels(df1$ADMTNG_DGNS_CD))
df1$ADMTNG_DGNS_CD <- as.numeric(df1$ADMTNG_DGNS_CD)

df1$PRMY_DIAG_CD <- as.factor(df1$PRMY_DIAG_CD)
levels(df1$PRMY_DIAG_CD) <- 1:length(levels(df1$PRMY_DIAG_CD))
df1$PRMY_DIAG_CD <- as.numeric(df1$PRMY_DIAG_CD)

# Add column - sex, race, address
# , BENE_ESRD_IND, SP_STATE_CODE, SP_ALZHDMTA, SP_CHF, SP_CHRNKIDN, SP_CNCR, SP_COPD, SP_DEPRESSN, SP_DIABETES, SP_ISCHMCHT, SP_OSTEOPRS, SP_RA_OA, SP_STRKETIA
df2 <- subset(patients, select = c(DESYNPUF_ID, BENE_BIRTH_DT, year, BENE_SEX_IDENT_CD, BENE_RACE_CD, BENE_COUNTY_CD, SP_STATE_CODE, BENE_ESRD_IND, SP_ALZHDMTA, SP_CHF, SP_CHRNKIDN, SP_CNCR, SP_COPD, SP_DEPRESSN, SP_DIABETES, SP_ISCHMCHT, SP_OSTEOPRS, SP_RA_OA, SP_STRKETIA))

df <- merge(df1, df2, by=c('DESYNPUF_ID', 'year'))

# rm(df1, df2)

# Add column - age
head(as.integer(df$AGE))
df$AGE <- floor(age_calc(df$BENE_BIRTH_DT, enddate = df$CLM_ADMSN_DT, units = "years", precise = TRUE))

# Add column - readmitted
head(as.integer(df$READMITTED))
n <- nrow(df)
for (row in 1:n) {
  df[row, 'READMITTED'] <- 0
  discharge <- df[row, 'NCH_BENE_DSCHRG_DT']
  
  if (row < n) {
    if(as.character(df[row, 'DESYNPUF_ID']) == as.character(df[row + 1, 'DESYNPUF_ID'])) {
      readmit <- df[row + 1, 'CLM_ADMSN_DT']
      # print(paste(discharge, readmit, readmit - discharge))
      if (readmit - discharge < 30) {
        df[row, 'READMITTED'] <- 1
      }
    }
  }
}
# rm(patients, claims, n, row, discharge, readmit)

df <- subset(df, select=-c(year, BENE_BIRTH_DT, CLM_ADMSN_DT, NCH_BENE_DSCHRG_DT))

# Rename some columns into meaningful names
df <- rename(df, c(DESYNPUF_ID = 'PATIENT_ID', BENE_SEX_IDENT_CD = 'SEX', BENE_RACE_CD = 'RACE', BENE_COUNTY_CD = 'COUNTRY'))
df$BENE_ESRD_IND <- as.integer(df$BENE_ESRD_IND)

df_new <- df[1:10000,]
df_new <- subset(df_new, select = -c(PATIENT_ID))
df_new <- df_new[, order(names(df_new))]

write.csv(df_new, file = "/home/bhet0001/research/readmission/project/exp_1.csv")


# Distribution counts 
# temp <- unique(subset(df_new, select=c(PATIENT_ID, AGE)))
# table(df_new$AGE)

df_new[['READMITTED']] <- factor(df_new[['READMITTED']])
levels(df_new$READMITTED) <- c("no", "yes")

table(df_new$READMITTED)

#df_new$PATIENT_ID <- as.factor(df_new$PATIENT_ID)
#levels(df_new$PATIENT_ID) <- 1:length(levels(df_new$PATIENT_ID))
#df_new$PATIENT_ID <- as.numeric(df_new$PATIENT_ID)

set.seed(3033)
intrain <- createDataPartition(y = df_new$READMITTED, p = 0.75, list = FALSE)
training <- df_new[intrain,]
testing <- df_new[-intrain,]

table(training$READMITTED)
table(testing$READMITTED)

ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 10, verboseIter = FALSE, summaryFunction = twoClassSummary, classProbs = TRUE)
# Linear SVM - svmLinear ()
# SMOTE
ctrl$sampling <- "smote"
set.seed(3033)
system.time(svmLinear <- train(READMITTED ~ .,
                               data = training,
                               method = "svmLinear",
                               verbose = FALSE,
                               metric = "ROC",
                               trControl = ctrl,
                               preProcess = c("center", "scale")))

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
            preProcess = c("center", "scale"))

set.seed(3033)
system.time(xgbTree <- train(READMITTED ~ .,
                 data = training,
                 method = "xgbTree",
                 verbose = FALSE,
                 metric = "ROC",
                 trControl = ctrl,
                 preProcess = c("center", "scale")))

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
