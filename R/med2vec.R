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

df1 <- cbind(df1, claims[,21:36])

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




# Load data from csv
dataDirectory <- '/home/bhet0001/research/readmission/CMS/'
claims <- read.csv(paste(dataDirectory, 'DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv', sep = ''), header = TRUE)

claims <- claims[order(claims$DESYNPUF_ID, claims$CLM_ADMSN_DT),]
# claims <- claims[1:10000,]

claims <- df_new

codes <- c(NA)
codes <- append(codes, levels(claims$ICD9_DGNS_CD_1))
codes <- append(codes, levels(claims$ICD9_DGNS_CD_2))
codes <- append(codes, levels(claims$ICD9_DGNS_CD_3))
codes <- append(codes, levels(claims$ICD9_DGNS_CD_4))
codes <- append(codes, levels(claims$ICD9_DGNS_CD_5))
codes <- append(codes, levels(claims$ICD9_DGNS_CD_6))
codes <- append(codes, levels(claims$ICD9_DGNS_CD_7))
codes <- append(codes, levels(claims$ICD9_DGNS_CD_8))
codes <- append(codes, levels(claims$ICD9_DGNS_CD_9))
codes <- append(codes, levels(claims$ICD9_DGNS_CD_10))
codes <- append(codes, levels(claims$ICD9_PRCDR_CD_1))
codes <- append(codes, levels(claims$ICD9_PRCDR_CD_2))
codes <- append(codes, levels(claims$ICD9_PRCDR_CD_3))
codes <- append(codes, levels(claims$ICD9_PRCDR_CD_4))
codes <- append(codes, levels(claims$ICD9_PRCDR_CD_5))
codes <- append(codes, levels(claims$ICD9_PRCDR_CD_6))

codes <- unique(codes)
values <- c(1:length(codes))

look <- data.frame(codes, values)

write.csv(look, file = "/home/bhet0001/research/readmission/Med2Vec/proj/map.csv")

claims_1 <- claims[, order(names(claims))]
claims_1[4:19] <- lapply(claims[, 4:19], function(x) look$values[match(x, look$codes)])

id <- claims_1$PATIENT_ID
visit <- apply(claims_1[4:19], 1, function(x) paste0(x[x != 1 & x != 2 & !is.na(x)] - 2, collapse = ","))
visit <- paste0("[", visit,"]")

visits <- data.frame(id, visit)
visits$visit <- as.character(visits$visit)

visits <- visits[1:10000,]

str <- "["
for (i in 1:nrow(visits)) {
  if (i == 1) {
    str <- paste0(str, visits[i, 'visit'])
  } else if (as.character(visits[i, 'id']) == as.character(visits[i-1, 'id'])) {
    str <- paste0(str, ",", visits[i, 'visit'])
  } else {
    str <- paste0(str, ",[-1],", visits[i, 'visit'])
  }
}
str <- paste0(str, "]")

fileConn <- file("/home/bhet0001/research/readmission/Med2Vec/proj/output.txt")
writeLines(c(str), fileConn)
close(fileConn)

patients <- subset(claims, select = c(PATIENT_ID, L, SEX, RACE, COUNTRY, SP_STATE_CODE, BENE_ESRD_IND, AGE, SP_ALZHDMTA, SP_CHF, SP_CHRNKIDN, SP_CNCR, SP_COPD, SP_DEPRESSN, SP_DIABETES, SP_ISCHMCHT, SP_OSTEOPRS, SP_RA_OA, SP_STRKETIA))
patients <- patients[, order(colnames(patients))]
patients_temp <- subset(patients, select=-c(PATIENT_ID))
patients$str <- apply(patients_temp, 1, function(x) paste0(x, collapse = ","))
patients$str <- paste0("[", patients$str,"]")

str <- ""
for (i in 1:nrow(patients)) {
  if (i == 1) {
    str <- paste0(str, patients[i, 'str'])
  } else if (as.character(patients[i, 'PATIENT_ID']) == as.character(patients[i-1, 'PATIENT_ID'])) {
    str <- paste0(str, ",", patients[i, 'str'])
  } else {
    str <- paste0(str, ",[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],", patients[i, 'str'])
  }
}

fileConn <- file("/home/bhet0001/research/readmission/Med2Vec/proj/demo.txt")
writeLines(c(str), fileConn)
close(fileConn)

# For model use - recalculate vector values
write.csv(claims_1, file = "/home/bhet0001/research/readmission/Med2Vec/proj/Med2Vec Model/claims.csv")
