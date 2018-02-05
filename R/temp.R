library(eeptools)
library(plyr)
library(caret)

# Load the data from the csv
dataDirectory <- '/home/bhet0001/research/readmission/CMS/'
claims <- read.csv(paste(dataDirectory, 'DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv', sep = ''), header = TRUE)
patient_2008 <- read.csv(paste(dataDirectory, 'DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv', sep = ''), header = TRUE)
patient_2009 <- read.csv(paste(dataDirectory, 'DE1_0_2009_Beneficiary_Summary_File_Sample_1.csv', sep = ''), header = TRUE)
patient_2010 <- read.csv(paste(dataDirectory, 'DE1_0_2010_Beneficiary_Summary_File_Sample_1.csv', sep = ''), header = TRUE)

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

# Sort the claim records by the admission date
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
# , SP_STATE_CODE, SP_ALZHDMTA, SP_CHF, SP_CHRNKIDN, SP_CNCR, SP_COPD, SP_DEPRESSN, SP_DIABETES, SP_ISCHMCHT, SP_OSTEOPRS, SP_RA_OA, SP_STRKETIA
df2 <- subset(patients, select = c(DESYNPUF_ID, BENE_BIRTH_DT, year, BENE_SEX_IDENT_CD, BENE_RACE_CD, BENE_COUNTY_CD, SP_STATE_CODE, SP_ALZHDMTA, SP_CHF, SP_CHRNKIDN, SP_CNCR, SP_COPD, SP_DEPRESSN, SP_DIABETES, SP_ISCHMCHT, SP_OSTEOPRS, SP_RA_OA, SP_STRKETIA))

df <- merge(df1, df2, by=c('DESYNPUF_ID', 'year'))

# rm(df1, df2)

# Add column - age
head(as.integer(df$AGE))
df$AGE <- age_calc(df$BENE_BIRTH_DT, enddate = df$CLM_ADMSN_DT, units = "years", precise = TRUE)

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

df <- subset(df, select=-c(year, CLM_ADMSN_DT, NCH_BENE_DSCHRG_DT))

# Rename some columns into meaningful names
df <- rename(df, c('DESYNPUF_ID' = 'PATIENT_ID', 'BENE_SEX_IDENT_CD' = 'SEX', 'BENE_RACE_CD' = 'RACE', 'BENE_COUNTY_CD' = 'COUNTRY'))

df_new <- df[1:2000,]

df_new <- subset(df_new, select = -c(PATIENT_ID))
set.seed(3033)
intrain <- createDataPartition(y = df_new$READMITTED, p = 0.75, list = FALSE)
training <- df_new[intrain,]
testing <- df_new[-intrain,]

dim(training)
dim(testing)

training[['READMITTED']] = factor(training[['READMITTED']])
testing[['READMITTED']] = factor(testing[['READMITTED']])

trctrl1 <- trainControl(method = "repeatedcv", number = 10, repeats = 10, verboseIter = FALSE)
trctrl2 <- trainControl(method = "repeatedcv", number = 10, repeats = 10, verboseIter = FALSE, sampling = "down")
trctrl3 <- trainControl(method = "repeatedcv", number = 10, repeats = 10, verboseIter = FALSE, sampling = "up")
trctrl4 <- trainControl(method = "repeatedcv", number = 10, repeats = 10, verboseIter = FALSE, sampling = "rose")
trctrl5 <- trainControl(method = "repeatedcv", number = 10, repeats = 10, verboseIter = FALSE, sampling = "smote")

set.seed(3233)
# Linear SVM - svmLinear ()
# None
svm_Linear <- train(READMITTED ~., data = training, method="svmLinear",
                    trControl=trctrl1,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)

test_pred <- predict(svm_Linear, newdata = testing)
test_pred
confusionMatrix(test_pred, testing$READMITTED)

# Under
svm_Linear <- train(READMITTED ~., data = training, method="svmLinear",
                    trControl=trctrl2,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)

test_pred <- predict(svm_Linear, newdata = testing)
test_pred
confusionMatrix(test_pred, testing$READMITTED)

# Over
svm_Linear <- train(READMITTED ~., data = training, method="svmLinear",
                    trControl=trctrl3,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)

test_pred <- predict(svm_Linear, newdata = testing)
test_pred
confusionMatrix(test_pred, testing$READMITTED)

# ROSE
svm_Linear <- train(READMITTED ~., data = training, method="svmLinear",
                    trControl=trctrl4,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)

test_pred <- predict(svm_Linear, newdata = testing)
test_pred
confusionMatrix(test_pred, testing$READMITTED)

# SMOTE
svm_Linear <-train(READMITTED ~., data = training, method="svmLinear",
                    trControl=trctrl5,
                    preProcess = c("center", "scale"))

test_pred <- predict(svm_Linear, newdata = testing)
test_pred
confusionMatrix(test_pred, testing$READMITTED)

# SVM Radial
svm_Radial <- train(READMITTED ~., data = training, method="svmRadial",
                           trControl=trctrl5,
                           preProcess = c("center", "scale"))

test_pred <- predict(svm_Radial, newdata = testing)
test_pred
confusionMatrix(test_pred, testing$READMITTED)

# Poly SVM - svmPoly ()
# None
svm_Poly <- train(READMITTED ~., data = training, method="svmPoly",
                  trControl=trctrl1,
                  # preProcess = c("center", "scale"),
                  tuneLength = 10)

test_pred <- predict(svm_Poly, newdata = testing)
test_pred
confusionMatrix(test_pred, testing$READMITTED)

# Under
svm_Poly <- train(READMITTED ~., data = training, method="svmPoly",
                  trControl=trctrl2,
                  # preProcess = c("center", "scale"),
                  tuneLength = 10)

test_pred <- predict(svm_Poly, newdata = testing)
test_pred
confusionMatrix(test_pred, testing$READMITTED)

# Over
svm_Poly <- train(READMITTED ~., data = training, method="svmPoly",
                  trControl=trctrl3,
                  # preProcess = c("center", "scale"),
                  tuneLength = 10)

test_pred <- predict(svm_Poly, newdata = testing)
test_pred
confusionMatrix(test_pred, testing$READMITTED)

# ROSE
svm_Poly <- train(READMITTED ~., data = training, method="svmPoly",
                  trControl=trctrl4,
                  # preProcess = c("center", "scale"),
                  tuneLength = 10)

test_pred <- predict(svm_Poly, newdata = testing)
test_pred
confusionMatrix(test_pred, testing$READMITTED)

# SMOTE
svm_Poly <- train(READMITTED ~., data = training, method="svmPoly",
                  trControl=trctrl5,
                  # preProcess = c("center", "scale"),
                  tuneLength = 10)

test_pred <- predict(svm_Poly, newdata = testing)
test_pred
confusionMatrix(test_pred, testing$READMITTED)

# Logisric regression - glm|family="binary" ()
# None
log_reg <- train(READMITTED ~., data = training, method="glm", family="binary",
                 trControl=trctrl1,
                 preProcess = c("center", "scale"),
                 tuneLength = 10)

test_pred <- predict(log_reg, newdata = testing)
test_pred
confusionMatrix(test_pred, testing$READMITTED)

# Under
log_reg <- train(READMITTED ~., data = training, method="glm", family="binary",
                 trControl=trctrl2,
                 preProcess = c("center", "scale"),
                 tuneLength = 10)

test_pred <- predict(log_reg, newdata = testing)
test_pred
confusionMatrix(test_pred, testing$READMITTED)

# Over
log_reg <- train(READMITTED ~., data = training, method="glm", family="binary",
                 trControl=trctrl3,
                 preProcess = c("center", "scale"),
                 tuneLength = 10)

test_pred <- predict(log_reg, newdata = testing)
test_pred
confusionMatrix(test_pred, testing$READMITTED)

# ROSE
log_reg <- train(READMITTED ~., data = training, method="glm", family="binary",
                 trControl=trctr4,
                 preProcess = c("center", "scale"),
                 tuneLength = 10)

test_pred <- predict(log_reg, newdata = testing)
test_pred
confusionMatrix(test_pred, testing$READMITTED)

# SOMTE
log_reg <- train(READMITTED ~., data = training, method="glm", family="binary",
                 trControl=trctrl5,
                 preProcess = c("center", "scale"),
                 tuneLength = 10)

test_pred <- predict(log_reg, newdata = testing)
test_pred
confusionMatrix(test_pred, testing$READMITTED)

# Random Forest - ranger
# None
rf <- train(READMITTED ~., data = training, method="ranger",
            trControl=trctrl1,
            preProcess = c("center", "scale"),
            tuneLength = 10)

test_pred <- predict(rf, newdata = testing)
test_pred
confusionMatrix(test_pred, testing$READMITTED)

# Under
rf <- train(READMITTED ~., data = training, method="ranger",
            trControl=trctrl2,
            preProcess = c("center", "scale"),
            tuneLength = 10)

test_pred <- predict(rf, newdata = testing)
test_pred
confusionMatrix(test_pred, testing$READMITTED)

# Over
rf <- train(READMITTED ~., data = training, method="ranger",
            trControl=trctrl3,
            preProcess = c("center", "scale"),
            tuneLength = 10)

test_pred <- predict(rf, newdata = testing)
test_pred
confusionMatrix(test_pred, testing$READMITTED)

# ROSE
rf <- train(READMITTED ~., data = training, method="ranger",
            trControl=trctrl4,
            preProcess = c("center", "scale"),
            tuneLength = 10)

test_pred <- predict(rf, newdata = testing)
test_pred
confusionMatrix(test_pred, testing$READMITTED)

# SMOTE
rf <- train(READMITTED ~., data = training, method="ranger",
            trControl=trctrl5,
            preProcess = c("center", "scale"),
            tuneLength = 10)

test_pred <- predict(rf, newdata = testing)
test_pred
confusionMatrix(test_pred, testing$READMITTED)