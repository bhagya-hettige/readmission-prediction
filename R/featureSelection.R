# Remove attributes with an absolute correlation of 0.75 or higher

sel_dataDirectory <- "/home/bhet0001/research/readmission/project/"
sel_df <- read.csv(paste(sel_dataDirectory, 'admissions.csv', sep = ''), header = TRUE)

sel_df$READMITTED <- as.factor(sel_df$READMITTED)
sel_df$BENE_ESRD_IND <- as.integer(sel_df$BENE_ESRD_IND)

sel_df <- sel_df[1:5000,]

# ensure the results are repeatable
set.seed(7)
# load the library
library(mlbench)
library(caret)
# load the data
# data(PimaIndiansDiabetes)
# calculate correlation matrix
correlationMatrix <- cor(sel_df[,1:ncol(sel_df)-1])
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)
# print indexes of highly correlated attributes
print(highlyCorrelated)

# ensure results are repeatable
set.seed(7)
# prepare training scheme
control <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
model <- train(READMITTED~., data=sel_df, method="lvq", preProcess="scale", trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)

# ensure the results are repeatable
set.seed(7)
# define the control using a random forest selection function
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
sel_df <- sel_df[1:100,]
results <- rfe(sel_df[,1:ncol(sel_df)-1], sel_df[,ncol(sel_df)], sizes=c(1:ncol(sel_df)-1), rfeControl=control)
# summarize the results
print(results)
# list the chosen features
predictors(results)
# plot the results
plot(results, type=c("g", "o"))
