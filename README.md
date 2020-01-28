---
title: "STAT 444 Final Project - Real Estate Valuation Dataset"
author: "Shubham, Jin, Ruiqi"
date: "7/21/2019"
output: pdf_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

# Motivation and introduction of the problem:
House prices have always been a hot topic, and while the global housing markets have been steadily climbing up, we want to have an idea on what is affecting the housing price the most. In this project we chose to specifically focus on the Taiwan housing market, we studied and analayzed the real estate valuation data set, which consists of 6 key factors on the housing price in New Taipei City, by fitting three regularization models, as well as using smoothing spline, random forest and graident boosting to help guide us determing which factors play the most important role in Taiwan house prices, and later on to discuss if the conclusion we reach could be generalized and apply to a bigger region.

# Data:
We used a data set from the UCI Machine Learning Repository, it consists of 6 explanatory variables, and one response variable. The market historical data set of real estate valuation were collected from Sindian Dist., New Taipei City, Taiwan. The areal estate valuations is a regression problem.

## Attribute Information:

The inputs are as follows:

X1 - the transaction date (for example, 2013.250=2013 March, 2013.500=2013 June, etc.) 

X2 - the house age (unit: year) 

X3 - the distance to the nearest MRT station (unit: meter) 

X4 - the number of convenience stores in the living circle on foot (integer) 

X5 - the geographic coordinate, latitude. (unit: degree) 

X6 - the geographic coordinate, longitude. (unit: degree) 


The output is as follow:

Y - house price of unit area (10000 New Taiwan Dollar/Ping, where Ping is a local unit, 1 Ping = 3.3 meter squared) 

\newpage
## An overview of the Real Estate Valutaion data
```{r echo=FALSE, warning=FALSE}
# Import the housing data
library(readxl)
RE <- read_excel("RealEstate.xlsx")
summary(RE)
```

## Distributions of input variables
```{r, echo=FALSE}
par(mfrow=c(1,2))

# Transaction date
hist(x=RE$X1, main="Distriution of Transaction Dates", xlab="Transaction Dates", 
     ylab="Number of Houses", col="grey", las=1, cex.main=1)
abline(v=mean(RE$X1), col="red", lwd=2, lty=1)
abline(v=median(RE$X1), col="blue", lwd=2, lty=2)
legend("topleft", legend=c("Mean", "Median"),
       col=c("red","blue"), lty=1:2, lwd=2, cex = 0.75)

# House age
hist(x=RE$X2, main="Distriution of House Age", xlab="House Ages(year)", 
     ylab="Number of Houses", col="grey", las=1, cex.main=1)
abline(v=mean(RE$X2), col="red", lwd=2, lty=1)
abline(v=median(RE$X2), col="blue", lwd=2, lty=2)
legend("topright", legend=c("Mean", "Median"),
       col=c("red","blue"), lty=1:2, lwd=2, cex = 0.75)

par(mfrow=c(1,2))
# Distance to MRT station
hist(x=RE$X3, main="Distriution of Distance to MRT station", 
     xlab="Distance(meter)", ylab="Number of Houses", 
     col="grey", las=1, cex.main=1)
abline(v=mean(RE$X3), col="red", lwd=2, lty=1)
abline(v=median(RE$X3), col="blue", lwd=2, lty=2)
legend("topright", legend=c("Mean", "Median"),
       col=c("red","blue"), lty=1:2, lwd=2, cex = 0.75)

# Number of convenience stores in the living circle on foot
hist(x=RE$X4, main="Distriution of # of Convenience Stores", 
     xlab="Number of Convenience Stores", ylab="Number of Houses", 
     col="grey", las=1, cex.main=1)
abline(v=mean(RE$X4), col="red", lwd=2, lty=1)
abline(v=median(RE$X4), col="blue", lwd=2, lty=2)
legend("topright", legend=c("Mean", "Median"),
       col=c("red","blue"), lty=1:2, lwd=2, cex = 0.75)

par(mfrow=c(1,2))
# Geographic coordinate, latitude
hist(x=RE$X5, main="Distriution of Latitute", xlab="Latitude(degree)", 
     ylab="Number of Houses", col="grey", las=1, cex.main=1)
abline(v=mean(RE$X5), col="red", lwd=2, lty=1)
abline(v=median(RE$X5), col="blue", lwd=2, lty=2)
legend("topright", legend=c("Mean", "Median"),
       col=c("red","blue"), lty=1:2, lwd=2, cex = 0.75)

# Geographic coordinate, longitude
hist(x=RE$X6, main="Distriution of Longitude", xlab="Longitude(degree)", 
     ylab="Number of Houses", col="grey", las=1, cex.main=1)
abline(v=mean(RE$X6), col="red", lwd=2, lty=1)
abline(v=median(RE$X6), col="blue", lwd=2, lty=2)
legend("topleft", legend=c("Mean", "Median"),
       col=c("red","blue"), lty=1:2, lwd=2, cex = 0.75)
```

From the overveiw of the data, we see that the average house price per unit area is 379800 new Taiwan dollar/Ping which is approximately 4.8k cad/square meter, 450 cad/sqft. Most transactions happened in 2013 and the average house age is 18 years. Most houses have around 4 convenience stores nearby and is around 1km to the closet MRT station. One thing worth notice is according to the mean longtitude and mean latitude, we find that the majority of the houses are located around Zhonghe District, it lies south-west of New Taipei City, with a total area of 7.836 sq mi and over 410k population, which is a relatively high population density.

## Scatterplot matrix of all the attributes
```{r, echo=FALSE}
pairs(RE[,c(2:8)],pch=21, col=adjustcolor("black",0.5))
```

We notice an obvious pattern when X3 is plotted against X5 and X6, we will look more into the interaction effects between them later.

```{r ,echo=FALSE, warning=FALSE}
# Import the housing data
library(readxl)
RE <- read_excel("RealEstate.xlsx")
```
\newpage
# Regularization:
Here we will fit optimal LASSO, elastic net, and Ridge regression models. The data is randomly split 70/30 for training/test sets by random uniform selection. We begin with the optimal LASSO model for the training set. Next we fit the optimal LASSO model for the training set and compare how well the predictions of both models agree via scatterplot. We repeat this for the elastic net and ridge regression models and constrast each model fit. We analyze the predicative accuracy of each model using the MSPE of the training set.

```{r, echo=FALSE, warning=FALSE}
set.seed(444)
suppressMessages(library(glmnet))

N = nrow(RE)
RE$set <- ifelse(runif(n=N)>0.7, yes=2, no=1)

Train = RE[which(RE$set==1),]
Test  = RE[which(RE$set==2),]

y.1 <- Train$Y
y.2 <- Test$Y

x.1 <- as.matrix(Train[,2:7])
xs.1 = scale(x.1)
x.2 <- as.matrix(Test[,2:7])
xs.2 = scale(x.2)

```


##### Important points
- For all three regression models assumptions are the same as least squares regression except the assumption of normality does not have to be validated.
- These models all use a form of regression called regularization. The approach is to constrain or shrink the coefficent estimates, and reduce complex models to avoid the risk of overfitting. 
- This decreases the variance but is contingent on added bias. The goal is to find a bias-variance-tradeoff that minimizes the total error which we will determine using cross validation 

##### LASSO Regression:
- Able to perform model selection as it contrains certain coefficents to zero
- If variables are highly correlated, LASSO chooses one and shrinks the others to zero
- Tends to work well with smaller amount of significant variables

##### Ridge Regression:
- Ridge regression shrinks the value of coefficients but does not set them to zero, and thus does not perform variable selection
- Coefficents of correlated variables are similar
- Tends to work well for many large variables of similar values

##### Elastic Net Regression:
- Mixture of LASSO & Ridge
- Constrains coefficents more than ridge but less than LASSO
- Can perform model selection

```{r, echo=FALSE, warning=FALSE}
## Fit LASSO by glmnet(y=, x=). Gaussian is default
## Function produces series of fits for many values of lambda. 

# Training set data - Using scaled variables gives better coefficient path
lasso.1 <- glmnet(y=y.1, x= xs.1, family="gaussian")

# cv.glmnet() uses crossvalidation to estimate optimal lambda
cv.lasso.1 <- cv.glmnet(y=y.1, x= x.1, family="gaussian")

# Print out coefficients at minimum lambda 
lasso.1.min = coef(cv.lasso.1, s=cv.lasso.1$lambda.min) 
# Using the "+1SE rule" (see later) produces a sparser solution
lasso.1.1se = coef(cv.lasso.1, s=cv.lasso.1$lambda.1se) 

# Predict both sets using training data fit
predict.1.1 <- predict(cv.lasso.1, newx=x.1)
predict.1.2 <- predict(cv.lasso.1, newx=x.2)
MSPE.lasso <- mean((y.2 - predict.1.2)^2)
```

```{r,echo=FALSE}
## Do the same for the test set
lasso.2 <- glmnet(y=y.2, x= xs.2, family="gaussian")
cv.lasso.2 <- cv.glmnet(y=y.2, x= x.2, family="gaussian")

lasso.2.min = coef(cv.lasso.2, s=cv.lasso.1$lambda.min) 
lasso.2.1se = coef(cv.lasso.2, s=cv.lasso.1$lambda.1se) 

predict.2.1 <- predict(cv.lasso.2, newx=x.1)
predict.2.2 <- predict(cv.lasso.2, newx=x.2)
```

```{r,echo=FALSE}
## Same procedure for Elastic net except alpha=0.5
elasticnet.1 <- glmnet(y=y.1, x= xs.1, family="gaussian", alpha = 0.5)
elasticnet.2 <- glmnet(y=y.2, x= xs.2, family="gaussian", alpha = 0.5)

cv.elasticnet.1 <- cv.glmnet(y=y.1, x= x.1, family="gaussian", alpha=0.5)
cv.elasticnet.2 <- cv.glmnet(y=y.2, x= x.2, family="gaussian", alpha=0.5)

elasticnet.1.min = coef(cv.elasticnet.1, s=cv.lasso.1$lambda.min) 
elasticnet.1.1se = coef(cv.elasticnet.1, s=cv.lasso.1$lambda.1se) 
elasticnet.2.min = coef(cv.elasticnet.2, s=cv.lasso.1$lambda.min) 
elasticnet.2.1se = coef(cv.elasticnet.2, s=cv.lasso.1$lambda.1se) 

predict.3.1 <- predict(cv.elasticnet.1, newx=x.1)
predict.3.2 <- predict(cv.elasticnet.1, newx=x.2)
predict.4.1 <- predict(cv.elasticnet.2, newx=x.1)
predict.4.2 <- predict(cv.elasticnet.2, newx=x.2)

MSPE.elasticnet <- mean((y.2 - predict.3.2)^2)
```

```{r,echo=FALSE}
ridge.1 <- glmnet(y=y.1, x= xs.1, family="gaussian", alpha = 0)
ridge.2 <- glmnet(y=y.2, x= xs.2, family="gaussian", alpha = 0)

cv.ridge.1 <- cv.glmnet(y=y.1, x= x.1, family="gaussian", alpha=0)
cv.ridge.2 <- cv.glmnet(y=y.2, x= x.2, family="gaussian", alpha=0)

ridge.1.min = coef(cv.ridge.1, s=cv.lasso.1$lambda.min) 
ridge.1.1se = coef(cv.ridge.1, s=cv.lasso.1$lambda.1se) 
ridge.2.min = coef(cv.ridge.2, s=cv.lasso.1$lambda.min) 
ridge.2.1se = coef(cv.ridge.2, s=cv.lasso.1$lambda.1se) 

predict.5.1 <- predict(cv.ridge.1, newx=x.1)
predict.5.2 <- predict(cv.ridge.1, newx=x.2)
predict.6.1 <- predict(cv.ridge.2, newx=x.1)
predict.6.2 <- predict(cv.ridge.2, newx=x.2)

MSPE.Ridge <- mean((y.2 - predict.5.2)^2)
```

## LASSO (alpha=1) - Plot of Coefficent Paths with Scaled Paramaters & CV-MSPE:

```{r,echo=FALSE}
par(mfrow=c(1,2))
plot(lasso.1, main="Training Set")
plot(lasso.2, main="Test Set")

par(mfrow=c(1,2))
plot(cv.lasso.1)
abline(v=log(cv.lasso.1$lambda.min), col="darkblue",lty=2, lwd=2)
abline(v=log(cv.lasso.1$lambda.1se), col="darkgreen",lty=2, lwd=2)
legend("topleft",c("lambda.min","lambda.1se"),col=c("darkblue","darkgreen"),lty=2,lwd=2)

plot(cv.lasso.2)
abline(v=log(cv.lasso.2$lambda.min), col="darkblue",lty=2, lwd=2)
abline(v=log(cv.lasso.2$lambda.1se), col="darkgreen",lty=2, lwd=2)
legend("topleft",c("lambda.min","lambda.1se"),col=c("darkblue","darkgreen"),lty=2,lwd=2)
```

### Training set

- Coefficents at minimum lambda

```{r,echo=FALSE}
lasso.1.min
```

- Coefficents at optimal lambda using +1se rule (more sparse)

```{r,echo=FALSE}
lasso.1.1se
```

### Test set

- Coefficents at minimum lambda

```{r,echo=FALSE}
lasso.2.min
```

- Coefficents at optimal lambda using +1se rule (more sparse)

```{r,echo=FALSE}
lasso.2.1se
```


```{r, echo=FALSE, warning=FALSE}
# How well do the two models agree? Compare predicted values.  
plot(x=c(predict.1.1,predict.1.2), y=c(predict.2.1,predict.2.2), 
     main = "Comparison of predictions from the two models",
     xlab="Predictions using Training model", ylab="predictions using Testing model")
abline(a=0,b=1)
```

Comment: The plot of coefficients and CV-MSPE are very similar for both the training & test sets. The LASSO model using the +1SE rule to choose lambda seems like a better fit than lambda min since we are simplifying the model for small reduction in predicative accuracy. We can see this as the training model using +1SE has only 3 variables instead of 5 for the the lambda min model, and only sacrifices a small decrease in MSPE (shown in CV-MSPE plot). The training and test tests both conclude that X6 should be removed from the minimum lambda LASSO models. For the +1SE models, the training set removed variables X1,X2 and x6 whereas the test set removed variable X1,X6. From the scatterplot, we can see that the correlation between the predictors for both models is fairly linear which suggests that both models seem to agree pretty well.



## Elastic Net (alpha=0.5) - Plot of Coefficent Paths with Scaled Paramaters & CV-MSPE:

```{r,echo=FALSE}
par(mfrow=c(1,2))
plot(elasticnet.1, main="Training Set")
plot(elasticnet.2, main="Test Set")

par(mfrow=c(1,2))
plot(cv.elasticnet.1)
abline(v=log(cv.elasticnet.1$lambda.min), col="darkblue",lty=2, lwd=2)
abline(v=log(cv.elasticnet.1$lambda.1se), col="darkgreen",lty=2, lwd=2)
legend("topleft",c("lambda.min","lambda.1se"),col=c("darkblue","darkgreen"),lty=2,lwd=2)

plot(cv.elasticnet.2)
abline(v=log(cv.elasticnet.2$lambda.min), col="darkblue",lty=2, lwd=2)
abline(v=log(cv.elasticnet.2$lambda.1se), col="darkgreen",lty=2, lwd=2)
legend("topleft",c("lambda.min","lambda.1se"),col=c("darkblue","darkgreen"),lty=2,lwd=2)
```

### Training set

- Coefficents at minimum lambda

```{r,echo=FALSE}
elasticnet.1.min
```

- Coefficents at optimal lambda using +1se rule (more sparse)

```{r,echo=FALSE}
elasticnet.1.1se
```

### Test set

- Coefficents at minimum lambda

```{r,echo=FALSE}
elasticnet.2.min
```

- Coefficents at optimal lambda using +1se rule (more sparse)

```{r,echo=FALSE}
elasticnet.2.1se
```


```{r, echo=FALSE, warning=FALSE}
# How well do the two models agree? Compare predicted values.  
plot(x=c(predict.3.1,predict.3.2), y=c(predict.4.1,predict.4.2), 
     main = "Comparison of predictions from the two models",
     xlab="Predictions using Training model", ylab="predictions using Testing model")
abline(a=0,b=1)
```

Comment: The plot of coefficients and CV-MSPE are very similar for both the training & test sets. The Elastic Net model using the +1SE rule to choose lambda is a better fit than lambda min since we are simplifying the model for small reduction in predicative accuracy. The training and test tests for both lambda min and lambda +1SE models include all parameters, where the test set seems to have predictor variables that are more extreme (higher magnitude). From the scatterplot, we can see that the correlation between the predictors for both models is linear (more than LASSO) which suggests that both models agree well with one another.

## Ridge Regression (alpha=1) - Plot of Coefficent Paths with Scaled Paramaters & CV-MSPE:

```{r,echo=FALSE}
par(mfrow=c(1,2))
plot(ridge.1, main="Training Set")
plot(ridge.2, main="Test Set")

par(mfrow=c(1,2))
plot(cv.ridge.1)
abline(v=log(cv.ridge.1$lambda.min), col="darkblue",lty=2, lwd=2)
abline(v=log(cv.ridge.1$lambda.1se), col="darkgreen",lty=2, lwd=2)
legend("topleft",c("lambda.min","lambda.1se"),col=c("darkblue","darkgreen"),lty=2,lwd=2)

plot(cv.ridge.2)
abline(v=log(cv.ridge.2$lambda.min), col="darkblue",lty=2, lwd=2)
abline(v=log(cv.ridge.2$lambda.1se), col="darkgreen",lty=2, lwd=2)
legend("topleft",c("lambda.min","lambda.1se"),col=c("darkblue","darkgreen"),lty=2,lwd=2)
```


### Training set

- Coefficents at minimum lambda

```{r,echo=FALSE}
ridge.1.min
```

- Coefficents at optimal lambda using +1se rule (more sparse)

```{r,echo=FALSE}
ridge.1.1se
```


### Test set

- Coefficents at minimum lambda

```{r,echo=FALSE}
ridge.2.min
```

- Coefficents at optimal lambda using +1se rule (more sparse)

```{r,echo=FALSE}
ridge.2.1se
```



```{r, echo=FALSE, warning=FALSE}
# How well do the two models agree? Compare predicted values.  
plot(x=c(predict.5.1,predict.5.2), y=c(predict.6.1,predict.6.2), 
     main = "Comparison of predictions from the two models",
     xlab="Predictions using Training model", ylab="predictions using Testing model")
abline(a=0,b=1)
```

Comment: The plot of coefficients and CV-MSPE are very similar for both the training & test sets. The Ridge regression model using the +1SE rule to choose lambda is a better fit than lambda min since again, we are simplifying the model for small reduction in predicative accuracy. The training/test tests for both lambda min and lambda +1SE models include all parameters where the test set seems to have more extreme predictor coefficients (similar to Elastic Net). From the scatterplot, we can see that the correlation between the predictors for both models is very linear (more than Elastic Net & LASSO) which suggests that both models agree very well with each other. 

\newpage
## Comparison of LASSO, Elastic Net & Ridge

- Assessing Model accuracy with MSPE

```{r, echo=FALSE}
data.frame(
   Model = c("LASSO","Elastic_Net", "Ridge"), 
   Alpha = c(1,0.5,0),
   lambda.1se = c(cv.lasso.1$lambda.1se, cv.elasticnet.1$lambda.1se, cv.ridge.1$lambda.1se),
   Optimal_Model = c("Y ~ X3+X4+X5          ",
                     "Y ~ X1+X2+X3+X4+X5       ",
                     "Y ~ X1+X2+X3+X4+X5+X6    "),
   MSPE = c(MSPE.lasso, MSPE.elasticnet, MSPE.Ridge)
   )
```

### Comments
We can see that all three models are similar in terms of their prediction errors with LASSO having the highest MSPE (126.17), Ridge having the lowest MSPE (122.82). The LASSO model has 3 variables, Elastic Net has 5 and Ridge Regression contains the full model.

Overall, LASSO seems to provide the most optimal model as it works well with a smaller amount of significant variables, which is the case here as the full model consists of 6 total variables. We can see that the LASSO provides the most sparse/simple fit as it contains the least number variables (half the full model) in exchange for a small decrease in MSPE. Note that the LASSO eliminates unsignificant variables and handles multicollinearity by penalizing highly correlated variables by only keeping one. In this case, X1 (transaction data), X6 (age) was removed as they were considered to be insignificant and X6 (longitude) was removed as it is correlated with X5 (latitude).

The overall optimal model between the 3 methods is $$Y = -1841.9460 -0.0037X_3 + 0.4746X_4 + 75.3366X_5$$ which include the variables X3: distance to the nearest MRT station, X4: number of convenience stores in the living circle, and  X5: latitude coordinate

\newpage
# Smoothing method - Smoothing spline (5-fold CV)
```{r echo=FALSE, message=FALSE, warning=FALSE}
# Create Folds
library(caret)
y <- RE[8]
x <- RE[2:7]
folds <- createFolds(seq(1,nrow(y),1), k = 5, list = TRUE, returnTrain = FALSE)
```

## 1. No Interactions(GAM): 
### Shown below is the summary of the model and AIC and MSE of each fold
```{r, echo=FALSE}
library(mgcv)

# To keep track of the MSEs and AIC
spAIC1 = rep(0,5)
spmse1 = rep(0,5)

for (i in 1:5) {
  sm.spl1 <- gam(data=RE[-folds[[i]],],Y ~ s(X1)+s(X2)+s(X3)+s(X4)+s(X5)+s(X6))
  # AIC and MSE
  spAIC1[i] <- AIC(sm.spl1)
  pred <- predict(sm.spl1, newdata=RE[folds[[i]],])
  spmse1[i] <- mean((RE[folds[[i]],]$Y - pred)^2)
}

summary(sm.spl1)
spAIC1
spmse1
```

## 2. Interaction effects between the distance to the nearest MRT station(X3) and longitude of the house(X6):
### Shown below is the model and the AIC and MSE of each fold
```{r, echo=FALSE}
spAIC2 = rep(0,5)
spmse2 = rep(0,5)
for (i in 1:5) {
  sm.spl2 <- gam(data=RE[-folds[[i]],],
                 Y ~ s(X1)+s(X2)+s(X3)+s(X4)+s(X5)+s(X6)+s(X3,X6))
  # AIC and MSE
  spAIC2[i] <- AIC(sm.spl2)
  pred <- predict(sm.spl2, newdata=RE[folds[[i]],])
  spmse2[i] <- mean((RE[folds[[i]],]$Y - pred)^2)
}

summary(sm.spl2)
spAIC2
spmse2
```

From the summary of the model, we can tell that the interaction effect of the distance to the nearest station(X3) and the longtitude of the house is significant, which makes sense since wether there is a nearby MRT station or not highly depends on the location of the house.

## 3. Interaction effects between the distance to the nearest MRT station(X3), latitude and longitude of the
## house(X6):
```{r, echo=FALSE}
spAIC3 = rep(0,5)
spmse3 = rep(0,5)

for (i in 1:5) {
  sm.spl3 <- gam(data=RE[-folds[[i]],],
                 Y ~ s(X1)+s(X2)+s(X3)+s(X4)+s(X5)+s(X6)+ti(X3,X5,X6))
  # AIC and MSE
  spAIC3[i] <- AIC(sm.spl3)
  pred <- predict(sm.spl3, newdata=RE[folds[[i]],])
  spmse3[i] <- mean((RE[folds[[i]],]$Y - pred)^2)
}

summary(sm.spl3)
spAIC3
spmse3
```

From the summary of the model including the three way interaction between distance to the closest MRT station, the latitude and longtitude of the house, the interaction effect does not seem as important as the two way interaction effect.

## Compare the Residuals
### 1. No Interaction
```{r, echo=FALSE}
par(mfrow=c(2,2))
gam.check(sm.spl1)
```
### 2. Interaction effects between the distance to the nearest MRT station(X3) and longitude of the house(X6):
```{r, echo=FALSE}
par(mfrow=c(2,2))
gam.check(sm.spl2)
```

### 3. Interaction effects between the distance to the nearest MRT station(X3), latitude and longitude of the house(X6):
```{r, echo=FALSE}
par(mfrow=c(2,2))
gam.check(sm.spl3)
```

## Compare the AIC & MSE
```{r, echo=FALSE}
library(knitr)
val <- data.frame(MSE = integer(3), AIC = integer(3))
val$MSE <- c(mean(spmse1), mean(spmse2), mean(spmse3))
val$AIC <- c(mean(spAIC1), mean(spAIC2), mean(spAIC3))
rownames(val) <- c("model1", "model2", "model3")
kable(val)
```

While there seem to be no big difference between the residual plots of all three models, and since all the model have the same mean AIC, we pick the model with the smallest MSE, which is model2 that considers the interaction effects between the distance to the nearest MRT station(X3) and longitude of the house(X6).

\newpage
# Random Forest(m=p/3): 
### Shown below is the function call
```{r, echo=FALSE}
library(randomForest)
rfmse = rep(0,5)

for (i in 1:5) {
  rf = randomForest(Y ~ X1+X2+X3+X4+X5+X6, data=RE[-folds[[i]],], mtry=2, ntree=500)
  # Keep track of MSE
  pred <- predict(rf, newdata = RE[folds[[i]],])
  rfmse[i] <- mean((RE[folds[[i]],]$Y - pred)^2)
}

rf
```
## Plot OOB error vs. number of trees and histogram of tree sizes
```{r, echo=FALSE}
par(mfrow=c(1,2))
plot(rf, main="OOB error vs. ntree")
hist(treesize(rf), xlab="Tree Size", ylab="Number of Trees", 
     main="Histogram of Tree Sizes")
```

### The average MSE of 5 folds
```{r, echo=FALSE}
sprintf("The average MSE:%.3f",mean(rfmse))
```

The Random Forest model has a smaller overall MSE compare to the smoothing spline models.

### Variable Importance
```{r, echo=FALSE}
rfimportance <- rf$importance
kable(rfimportance)
varImpPlot(rf,type=2, main="Variable Importance", col="blue")
```

From the variable importance plot we can tell that distance to the nearest MRT station (X3) is the most important variable in the model, followed by the latitude(X5) and longtitude(X6) of the house, which means the geographical locaation of the house is more important compare to the house age(X2) and the number of convenience stores in the living circle of the house(X4). The transaction date(X1), has the smallest impact on the house price which makes sense since the dataset only recorded data over a period of less than one year.

\newpage
# Boosting - Gradient Boosting
### Again we use 5-fold CV
```{r echo=FALSE, message=FALSE, warning=FALSE}
library(gbm)
gbmmse = rep(0,5)
ctrl <- trainControl(method='cv', number=5)
boost <- train(data=RE, Y~X1+X2+X3+X4+X5+X6, method='gbm', 
               distribution='gaussian', trControl=ctrl, verbose=FALSE)
boost
```

We use tuneGrid to test out different combinations of tuning parameters:

- n.trees = 100, 150, 200

- interaction.depth = 5, 6, 7

- shrinkage = 0.01 0.05 0.1

- n.minobsinnode hold constant at 10

```{r, echo=FALSE}
grid <- expand.grid(n.trees=c(100, 150, 200),
                    interaction.depth=c(5, 6, 7),
                    shrinkage=c(0.01, 0.05, 0.1), n.minobsinnode=10)
boost.tune <- train(data=RE, Y~X1+X2+X3+X4+X5+X6, method='gbm', 
               distribution='gaussian', trControl=ctrl, 
               tuneGrid=grid, verbose=FALSE)
boost.tune
```

The optimal set of tuning parameters is:

- n.trees=150, interaction.depth = 6, shrinkage=0.05 and n.minobsinnode = 10

## Rerun the model with optimal tuning parameters
```{r, echo=FALSE}
for (i in 1:5) {
  gbm <- gbm(data=RE[-folds[[i]],], Y ~ X1+X2+X3+X4+X5+X6, 
                   distribution = "gaussian", n.trees = 150,
                   interaction.depth = 6, shrinkage = 0.05, n.minobsinnode = 10)

  # gbm prediction & RMSE
  pred <- predict(gbm, n.trees=150, newdata = RE[folds[[i]],])
  gbmmse[i] <- mean((RE[folds[[i]],]$Y - pred)^2)

  # gbm variable importance
  # gbm.importance <- summary(gbm.model)
  # gbm_cv_importance[,i] = gbm.importance[row.names(gbm_cv_importance),]$rel.inf
}
summary(gbm)
```

We see that while distance to the nearest MRT station (X3) remains to be the most influencial variable in the gradient boosting model, all the other varialbe importances are different from the random forest model. The second important variable becomes house age(X2), and followed by the latitude(X5) and longtitude(X6) of the house, number of convenience stores in the living circle of the house(X4) becomes the least important variable in this model.

## The overall MSE of 5 folds
```{r, echo=FALSE}
sprintf("The average MSE:%.3f",mean(gbmmse))
```

The overall MSE generated from the gradient boosting model is slightly bigger than the one from random forest model, but still slightly smaller than the smoothing spline models.

\newpage
# Statistical Conclusion 
## Model MSEs
To decide on which model is our best candidate, we compare the MSE of each model, note: for smoothing spline we used model2 which is the best one among the three models we fitted above.
```{r, echo=FALSE}
val <- data.frame(MSE = integer(6))
val$MSE <- c(mean(cv.lasso.1$cvm), mean(cv.elasticnet.1$cvm), mean(cv.ridge.1$cvm), mean(spmse2), mean(rfmse), mean(gbmmse))
rownames(val) <- c("LASSO", "Elasticnet", "Ridge Regression", "Smoothing Spline", "Random Forest", "Gradient Boosting")
kable(val)
```

From the MSE table above, among all the models, gradient boosting has the smallest MSE, therefore the gradient boosting model is the best candidate for fitting this dataset.

## Variable Importance
Since our motivation is to find which factors affect the housing price the most, we also want to compare the variable importance from the random forest and the gradient boosting model.

```{r, echo=FALSE}
par(mfrow=c(1,2))
varImpPlot(rf,type=2, main="RF Variable Importance", col="blue", cex.main=1)
summary(gbm)
```

By comparing the variable importance graphs, there is only a slight difference between two models, both models show that the distance to the nearest MRT station is the most crucial factor, and the latitude is the second most important, the house age and the longitude are right after, and the transaction date and the number of convenience stores around the house have the smallest impact on the house prices.


# Conclusion
Even though we chose gradient boosting model as the best candidate, it does not necessarily mean it is the true model for our dataset, but all the model fitting does provide a valuable insight on the question we are trying to answer, which is what factor amongst the six has the biggest influence on Taiwan house price. 
However since the dataset we have is relatively small with only 414 instances, and was only collected from one district in Taiwan, the conclusion we drew from it can somehow be biased, also the dataset only consists of data collected from mid 2012 to mid 2013, which is a really short period of time, the importance of all the factors may change in the following years.
Overall, the conclusion we reach applies to the New Taipei City and considering the population base is big enough, it can be applied to the whole Taiwan, but due to the bias that might exist, it cannot be generalized and applied to other cities or a bigger region, which lead us to the next part, what can we do in future work to strength the model.

# Future Work
Since we want a model that can give us less biased result and provide more accurate prediction, we need a bigger datset with more instances as well as taking more detailed and even financial factors into account, such as number of bedrooms in the house, the interest rate while purchaing the house, etc. And the data need to be collected over a longer period of time. In the process of model fitting, we can achieve a better result by putting more weight on the more important factors. We can also fit more models to see if any outperforms the gradient boosting model, for example the genralized linear model and logistic regression, etc.

# Contribution
Congxiao Jin: Introduction, Smoothing Method, Random Forest, Boosting, Conclusion

Shuby Sharma: Regularization-LASSO, Ridge Regression, Elastic Net Regression
