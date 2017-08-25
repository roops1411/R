# R
library(data.table)
library(ade4)
library(rpart)
library(rpart.plot)
library(MASS)
library(caret)
library(neuralnet)
setwd("/home/kishore/Downloads")
flightdelays<-read.csv("Data_PredictingFlightDelays.csv",header = FALSE)
head(flightdelays)
colnames(flightdelays)
setnames(flightdelays,old=c("V1","V2","V3","V4","V5","V6","V7","V8"),new=c("Canceled","Month","DepartureTime","UniqueCarrier","SchedElapsedTime"
                                                                           ,"ArrDelay","DepDelay","Distance"))
unique(flightdelays$UniqueCarrier)
dim(flightdelays)
unique(flightdelays$Canceled)
write.csv(flightdelays,"fd.csv")

## NUll checks

unique(is.na(flightdelays$Canceled))
unique(is.na(flightdelays$Month))
unique(is.na(flightdelays$DepartureTime))
unique(is.na(flightdelays$UniqueCarrier))
unique(is.na(flightdelays$SchedElapsedTime))
unique(is.na(flightdelays$ArrDelay))
unique(is.na(flightdelays$DepDelay))
unique(is.na(flightdelays$Distance))



  

## Categorical variable checks 

unique(flightdelays$Canceled)
unique(flightdelays$Month)
unique(flightdelays$UniqueCarrier)
#unique(flightdelays$DepartureTime)


## Formatting time data to multiple buckets
library(stringr)
flightdelays$DepartureTime_hour<-substr(flightdelays$DepartureTime,1,2)
flightdelays$DepartureTime<-str_pad(flightdelays$DepartureTime, 4, pad = "0")


par(mfrow=c(3,2))
hist(as.numeric(flightdelays$DepartureTime_hour), main='FlightDelays: Departure hours', 
     col='darkgreen',breaks=24)

## Data distribution of arrivaldelay,depdelay,scheduledelapse time

hist(as.numeric(flightdelays$SchedElapsedTime), main='FlightDelays: Scheduledelapse time', 
     col='darkgreen',breaks=24)
hist(as.numeric(flightdelays$ArrDelay), main='FlightDelays: Arrival delay', 
     col='darkgreen',breaks=24)
hist(as.numeric(flightdelays$DepartureTime_hour), main='FlightDelays: Arrival delay', 
     col='darkgreen',breaks=24)
hist(as.numeric(flightdelays$Distance), main='FlightDelays: Distance', 
     col='darkgreen',breaks=12)

hist(flightdelays$Canceled, main='FlightDelays: Canceled', 
     col='darkgreen',breaks=2)

## One to c conversion 
## Splitting Data to categuorical and continuous/discrete variables
cat.col.names <-c("Month","UniqueCarrier","DepartureTime_hour")
cat_flightdelays<-flightdelays[,cat.col.names]
cat_flightdelays_conv<-acm.disjonctif(cat_flightdelays[c(1:3)])


cont.col.names <- c("SchedElapsedTime","ArrDelay","DepDelay","Distance")
cont_flightdelays<-flightdelays[,cont.col.names]
scaled.cont_flightdelays <- scale(cont_flightdelays)
scaled.cont_flightdelays


## Splitting Response variable

resp.col.names <- c("Canceled")
resp_flightdelays<-flightdelays[,resp.col.names]



## Binding categorical and normalized continuous supporting variables
pred.flightdelay.supp <- cbind(cat_flightdelays_conv,scaled.cont_flightdelays)
## Binding Independent variable and dependent variables
pred.flightdelay <- cbind(pred.flightdelay.supp,resp_flightdelays)


head(pred.flightdelay)


nrow(pred.flightdelay)
## Splitting data to train and test


## 75% of the sample size
sample_size <- floor(0.75 * nrow(pred.flightdelay))
sample_size



## set the seed to make your partition reproductible
set.seed(123)
train_ind <- sample(seq_len(nrow(pred.flightdelay)), size = sample_size)

train <- pred.flightdelay[train_ind, ]
test <- pred.flightdelay[-train_ind, ]

## Running Logistic regression

logistic_model <- glm(resp_flightdelays ~.,family=binomial(link='logit'),data=train)
summary(logistic_model)


anova(logistic_model, test="Chisq")
head(train)
test[,-which(names(test) %in% c("resp_flightdelays"))]
dim(test[,-which(names(test) %in% c("resp_flightdelays"))])
dim(train)
fitted.results <- predict(logistic_model,test[,-which(names(test) %in% c("resp_flightdelays"))],type='response')
fitted.results
xtab <- table(round(fitted.results,digits = 0), test$resp_flightdelays)
confusionMatrix(xtab)

## Decision Tree Classification
par(mfrow=c(1,1))
s_rpart<- rpart(resp_flightdelays~.,data=train,method="anova")
rpart.plot(s_rpart)
printcp(s_rpart)
s_rpart$cptable[which.min(s_rpart$cptable[,"xerror"]),"CP"]
prune(s_rpart,cp=0.01)
predicted <- predict(s_rpart,test[,-which(names(test) %in% c("resp_flightdelays"))])
#pred <- cbind(test$resp_flightdelays,round(predicted,digits = 0))
#t = as.data.frame(pred)
#head(t)
xtab <- table(round(predicted,digits = 0), test$resp_flightdelays)

confusionMatrix(xtab)



## Neural net
n <- names(train)
f <- as.formula(paste("resp_flightdelays ~", paste(n[!n %in% "resp_flightdelays"], collapse = " + ")))
f
nn <- neuralnet(f,data=train,hidden=c(15,8,2),linear.output=T)
plot(nn)

head(test[,1:40])
pr.nn <- compute(nn,test[,1:40])
nn.out<-abs(round(pr.nn$net.result))
a<-confusionMatrix(nn.out,test$resp_flightdelays)
a


write.csv(train,"train_flightdelay.csv")
write.csv(test,"test_flightdelay.csv")
