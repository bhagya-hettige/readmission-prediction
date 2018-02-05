library(eeptools)
library(plyr)
library(caret)
library(DMwR) # for smote implementation

dataDirectory <- '/home/bhet0001/research/MIMIC/Data/'

# Load data from csv
patients <- read.csv(paste(dataDirectory, 'PATIENTS.csv', sep = ''), header = TRUE)
admissions <- read.csv(paste(dataDirectory, 'ADMISSIONS.csv', sep = ''), header = TRUE)
diagnoses <- read.csv(paste(dataDirectory, 'DIAGNOSES_ICD.csv', sep = ''), header = TRUE)
procedures <- read.csv(paste(dataDirectory, 'PROCEDURES_ICD.csv', sep = ''), header = TRUE)
drgcodes <- read.csv(paste(dataDirectory, 'DRGCODES.csv', sep = ''), header = TRUE)

# Drop ROW_ID column in the tables
patients <- subset(patients, select=-c(ROW_ID))
admissions <- subset(admissions, select=-c(ROW_ID))
diagnoses <- subset(diagnoses, select=-c(ROW_ID))
procedures <- subset(procedures, select=-c(ROW_ID))
drgcodes <- subset(drgcodes, select=-c(ROW_ID))

# Filter out ADMISSION_TYPE = NEWBORN
admissions <- subset(admissions, admissions$ADMISSION_TYPE != 'NEWBORN')

# Filter out patients who died during admission
admissions <- subset(admissions, admissions$DEATHTIME == "")

# -------------------------------------------------------------
# Filter out the heart failure patients
diagnoses <- diagnoses[order(diagnoses$HADM_ID, diagnoses$SEQ_NUM),]
d <- aggregate(ICD9_CODE ~ SUBJECT_ID + HADM_ID, diagnoses, paste, collapse = ",")
# IcD9 codes for HF - 402.01, 402.11, 402.91,
# 425.1, 425.4, 425.5, 425.7, 425.8, 425.9, 428.0, 428.1, 428.2,
# 428.21, 428.22, 428.23, 428.3, 428.31, 428.32, 428.33, 428.4,
# 428.41, 428.42, 428.43, 428.9
HF_CODES <- c("40201", "40211", "40291", "4251", "4254", "4255", "4257", "4258", "4259", "4280", "4281", "4282", "42821", "42822", "42823", "4283", "42831", "42832", "42833", "4284", "42841", "42842", "42843", "4289")
d <- d[grepl(paste(HF_CODES, collapse = "|"), d$ICD9_CODE),]
admissions <- admissions[admissions$HADM_ID %in% d$HADM_ID, ]
# -------------------------------------------------------------

# Merge admissions with patients
df <- merge(admissions, patients, by=c('SUBJECT_ID'))

# Sort by patient id and admission date
df <- df[order(df$SUBJECT_ID, df$ADMITTIME),]

# Date format
df$ADMITTIME <- as.Date(df$ADMITTIME)
df$DISCHTIME <- as.Date(df$DISCHTIME)
df$DOB <- as.Date(df$DOB)

# Add column - length of stay
df['L'] <- as.integer(df$DISCHTIME - df$ADMITTIME)

# Add column - emergency
df['E'] <- as.integer(ifelse(df$EDREGTIME != "", 1, 0))

# Add column - age
df['AGE'] <- floor(age_calc(df$DOB, enddate = df$ADMITTIME, units = "years", precise = TRUE))
# Clean - filter out AGE > 120
df <- subset(df, df$AGE <= 120)

# Map into numeric numbers
levels(df$ADMISSION_TYPE) <- 1:length(levels(df$ADMISSION_TYPE))
df$ADMISSION_TYPE <- as.integer(df$ADMISSION_TYPE)

levels(df$ADMISSION_LOCATION) <- 1:length(levels(df$ADMISSION_LOCATION))
df$ADMISSION_LOCATION <- as.integer(df$ADMISSION_LOCATION)

levels(df$DISCHARGE_LOCATION) <- 1:length(levels(df$DISCHARGE_LOCATION))
df$DISCHARGE_LOCATION <- as.integer(df$DISCHARGE_LOCATION)

levels(df$INSURANCE) <- 1:length(levels(df$INSURANCE))
df$INSURANCE <- as.integer(df$INSURANCE)

levels(df$LANGUAGE) <- 1:length(levels(df$LANGUAGE))
df$LANGUAGE <- as.integer(df$LANGUAGE)

levels(df$RELIGION) <- 1:length(levels(df$RELIGION))
df$RELIGION <- as.integer(df$RELIGION)

levels(df$MARITAL_STATUS) <- 1:length(levels(df$MARITAL_STATUS))
df$MARITAL_STATUS <- as.integer(df$MARITAL_STATUS)

levels(df$ETHNICITY) <- 1:length(levels(df$ETHNICITY))
df$ETHNICITY <- as.integer(df$ETHNICITY)

levels(df$GENDER) <- 1:length(levels(df$GENDER))
df$GENDER <- as.integer(df$GENDER)

# Add column - READMITTED class label
head(as.integer(df$READMITTED))
n <- nrow(df)
for (row in 1:n) {
  df[row, 'READMITTED'] <- 0
  discharge <- df[row, 'DISCHTIME']
  
  if (row < n) {
    if(as.character(df[row, 'SUBJECT_ID']) == as.character(df[row + 1, 'SUBJECT_ID'])) {
      readmit <- df[row + 1, 'ADMITTIME']
      # print(paste(discharge, readmit, readmit - discharge))
      if (readmit - discharge < 30) {
        df[row, 'READMITTED'] <- 1
      }
    }
  }
}

# When the dataset is too large
set.seed(3033)
intrain <- createDataPartition(y = df$READMITTED, p = 0.25, list = FALSE)
df_new <- df[intrain,]
df <- df_new

# -------------------Exp1-----------------------
# Primary medical codes
# Add primary diagnosis and procedure code
df['DIAG'] <- merge(df, subset(diagnoses, diagnoses$SEQ_NUM == 1), all.x = TRUE)$ICD9_CODE
df['PROCD'] <- merge(df, subset(procedures, procedures$SEQ_NUM == 1), all.x = TRUE)$ICD9_CODE

# Map diagnosis codes into numeric numbers
levels(df$DIAG) <- 1:length(levels(df$DIAG))
df$DIAG <- as.integer(df$DIAG)
df$DIAG <- ifelse(is.na(df$DIAG), 0, df$DIAG)

# Map procedure codes into numeric numbers
# levels(df$PROCD) <- 1:length(levels(df$PROCD))
# df$PROCD <- as.numeric(df$PROCD)
df$PROCD <- ifelse(is.na(df$PROCD), 0, df$PROCD)

#------------------------------------------------

# -------------------Exp2-----------------------
df <- subset(df, select = -c(DIAG, PROCD))

maxAdm <- max(df$HADM_ID, na.rm = TRUE)
diagnoses <- diagnoses[diagnoses$HADM_ID <= maxAdm,]
procedures <- procedures[procedures$HADM_ID <= maxAdm,]
drgcodes <- drgcodes[drgcodes$HADM_ID <= maxAdm,]

diagnoses$ICD9_CODE <- paste0('D', diagnoses$ICD9_CODE)
procedures$ICD9_CODE <- paste0('P', procedures$ICD9_CODE)
drgcodes$DRG_CODE <- paste0('DR', drgcodes$DRG_CODE)
codes <- c(NA)
codes <- append(codes, diagnoses$ICD9_CODE)
codes <- append(codes, procedures$ICD9_CODE)
codes <- append(codes, drgcodes$DRG_CODE)

codes <- unique(codes)
values <- c(1:length(codes))

look <- data.frame(codes, values)

# Construct list of diagnosis, procedure, and drug codes for each admission
diagnoses <- diagnoses[order(diagnoses$HADM_ID, diagnoses$SEQ_NUM),]

procedures <- procedures[order(procedures$HADM_ID, procedures$SEQ_NUM),]

diagnoses['DIAG'] <- look$values[match(diagnoses$ICD9_CODE, look$codes)] - 1
procedures['PROCD'] <- look$values[match(procedures$ICD9_CODE, look$codes)] - 1
drgcodes['DRG'] <- look$values[match(drgcodes$DRG_CODE, look$codes)] - 1

d <- aggregate(DIAG ~ SUBJECT_ID + HADM_ID, diagnoses, paste, collapse = ",")
p <- aggregate(PROCD ~ SUBJECT_ID + HADM_ID, procedures, paste, collapse = ",")
drg <- aggregate(DRG ~ SUBJECT_ID + HADM_ID, drgcodes, paste, collapse = ",")

df['DIAG'] <- merge(df, d, by=c('HADM_ID', 'SUBJECT_ID'), all.x = TRUE)$DIAG
df['PROCD'] <- merge(df, p, by=c('HADM_ID', 'SUBJECT_ID'), all.x = TRUE)$PROCD
df['DRG'] <- merge(df, drg, by=c('HADM_ID', 'SUBJECT_ID'), all.x = TRUE)$DRG

df['visit'] <- apply(df[c('DIAG', 'PROCD', 'DRG')], 1, function(x) paste0(x[!is.na(x)], collapse = ","))
df['visit'] <- paste0('[', df$visit, ']')

str <- "["
for (i in 1:nrow(df)) {
  if (i == 1) {
    str <- paste0(str, df[i, 'visit'])
  } else if (as.character(df[i, 'SUBJECT_ID']) == as.character(df[i-1, 'SUBJECT_ID'])) {
    str <- paste0(str, ",", df[i, 'visit'])
  } else {
    str <- paste0(str, ",[-1],", df[i, 'visit'])
  }
}
str <- paste0(str, "]")

fileConn <- file(paste0(dataDirectory, "exp/exp_2_input.txt"))
writeLines(c(str), fileConn)
close(fileConn)

# -----------------------------------------------------------

# ------------------------- Exp3 ----------------------------
fileConn <- file(paste0(dataDirectory, "exp/exp_3_input.txt"))
writeLines(c(str), fileConn)
close(fileConn)

id <- df$SUBJECT_ID

patients_df <- subset(df, select = c(SUBJECT_ID, ADMISSION_TYPE, ADMISSION_LOCATION, DISCHARGE_LOCATION, INSURANCE, LANGUAGE, RELIGION, MARITAL_STATUS, ETHNICITY, GENDER, L, E, AGE))
patients_df <- patients_df[, order(colnames(patients_df))]
patients_temp <- subset(patients_df, select=-c(SUBJECT_ID))
patients_df$str <- apply(patients_temp, 1, function(x) paste0(x, collapse = ","))
patients_df$str <- paste0("[", patients_df$str,"]")

str <- ""
for (i in 1:nrow(patients_df)) {
  if (i == 1) {
    str <- paste0(str, patients_df[i, 'str'])
  } else if (as.character(patients_df[i, 'SUBJECT_ID']) == as.character(patients_df[i-1, 'SUBJECT_ID'])) {
    str <- paste0(str, ",", patients_df[i, 'str'])
  } else {
    str <- paste0(str, ",[0,0,0,0,0,0,0,0,0,0,0,0],", patients_df[i, 'str'])
  }
}

fileConn <- file(paste0(dataDirectory, "/exp/exp_3_demo.txt"))
writeLines(c(str), fileConn)
close(fileConn)

# -----------------------------------------------------------

df3 <- subset(df, select = -c(SUBJECT_ID, HADM_ID, ADMITTIME, DISCHTIME, DEATHTIME, EDREGTIME, EDOUTTIME, DIAGNOSIS, HOSPITAL_EXPIRE_FLAG, HAS_CHARTEVENTS_DATA, DOB, DOD, DOD_HOSP, DOD_SSN, EXPIRE_FLAG))
write.csv(df3, file = paste0(dataDirectory, 'exp/', '/exp_1.csv'))

df3 <- subset(df, select = -c(SUBJECT_ID, HADM_ID, ADMITTIME, DISCHTIME, DEATHTIME, EDREGTIME, EDOUTTIME, DIAGNOSIS, HOSPITAL_EXPIRE_FLAG, HAS_CHARTEVENTS_DATA, DOB, DOD, DOD_HOSP, DOD_SSN, EXPIRE_FLAG, DIAG, PROCD, DRG))
write.csv(df3, file = paste0(dataDirectory, 'exp/', '/exp_2.csv'))

df3 <- subset(df, select = -c(SUBJECT_ID, HADM_ID, ADMITTIME, DISCHTIME, DEATHTIME, EDREGTIME, EDOUTTIME, DIAGNOSIS, HOSPITAL_EXPIRE_FLAG, HAS_CHARTEVENTS_DATA, DOB, DOD, DOD_HOSP, DOD_SSN, EXPIRE_FLAG, DIAG, PROCD, DRG))
write.csv(df3, file = paste0(dataDirectory, 'exp/', '/exp_3.csv'))
