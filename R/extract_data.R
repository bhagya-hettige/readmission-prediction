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

# Merge admissions with patients
df <- merge(admissions, patients, by=c('SUBJECT_ID'))

# Sort by patient id and admission date
df <- df[order(df$SUBJECT_ID, df$ADMITTIME),]

# Date format
df$ADMITTIME <- as.Date(df$ADMITTIME)
df$DISCHTIME <- as.Date(df$DISCHTIME)
df$DOB <- as.Date(df$DOB)

# Construct list of diagnosis, procedure, and drug codes for each admission
diagnoses <- diagnoses[order(diagnoses$HADM_ID, diagnoses$SEQ_NUM),]
diagnoses_list <- aggregate(ICD9_CODE ~ SUBJECT_ID + HADM_ID, diagnoses, paste, collapse = ",")
diagnoses_list <- rename(diagnoses_list, c(ICD9_CODE = 'DIAGNOSES'))

procedures <- procedures[order(procedures$HADM_ID, procedures$SEQ_NUM),]
procedures_list <- aggregate(ICD9_CODE ~ SUBJECT_ID + HADM_ID, procedures, paste, collapse = ",")
procedures_list <- rename(procedures_list, c(ICD9_CODE = 'PROCEDURES'))

drgcodes_list <- aggregate(DRG_CODE ~ SUBJECT_ID + HADM_ID, drgcodes, paste, collapse = ",") # TRY DRG_TYPE as well
#aggregate(DRG_CODE ~ SUBJECT_ID + HADM_ID, drgcodes, paste, collapse = ",")
#drgcodes_list['DRG_TYPE'] <- aggregate(DRG_TYPE ~ SUBJECT_ID + HADM_ID, drgcodes, paste, collapse = ",")

df <- merge(df, diagnoses_list, by=c('HADM_ID', 'SUBJECT_ID'))
df <- merge(df, procedures_list, by=c('HADM_ID', 'SUBJECT_ID'))
df <- merge(df, drgcodes_list, by=c('HADM_ID', 'SUBJECT_ID'))

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

df <- subset(df, select = -c(SUBJECT_ID, HADM_ID, ADMITTIME, DISCHTIME, DEATHTIME, EDREGTIME, EDOUTTIME, DIAGNOSIS, HOSPITAL_EXPIRE_FLAG, HAS_CHARTEVENTS_DATA, DOB, DOD, DOD_HOSP, DOD_SSN, EXPIRE_FLAG))

write.csv(df, file = paste0(dataDirectory, 'exp/', '/exp_1.csv'))

