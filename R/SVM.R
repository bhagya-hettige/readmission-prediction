# Import Caret
library(caret)

# Load the data from the csv
dataDirectory <- '/home/bhet0001/research/readmission/project/'
df <- read.csv(paste0(dataDirectory, "train.csv"), sep = ',', header = FALSE)


