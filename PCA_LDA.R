
---
title: "Project -1"
author: "Neeru Bhardwaj"
date: "June 23, 2018"
output: html_document
---


```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Consider the digits data contained in the files data0,data1,....data9 (downloaded  from http://cis.jhu.edu/~sachin/digit/digit.html)
File format:
"Each file has 1000 training examples. Each training example is of size 28x28 pixels. The pixels are stored as unsigned chars (1 byte) and take values from 0 to 255. The first 28x28 bytes of the file correspond to the first training example, the next 28x28 bytes correspond to the next example and so on."
We have digits data and associated with every instance of a digit we have an image which is actually translated into a set of 784 variables which are greyscale values of the intensities at different pixel in the image.We have 10000 cases ,on each case we have row which are 784 pixel intensities for the hand drawn digit.We do digit classification using LDA. Two  methods are used here to estimate the error (misclassification rate) of a classifier i.e Resubstitution and Cross-validation. 

Consider the digit data and create four classification data sets each with different X-matrix.
Load the digits data and create the digit matrix after reading the binary data.


```{r}
fddata = "C:\\Users\\honey\\Desktop\\Digits-Data"
setwd(fddata)
fnmes = c("data0","data1.txt","data2","data3","data4","data5","data6","data7","data8","data9")
digits=NULL ;y=NULL
for(k in c(0:9)) {
  fi= fnmes[k+1]
  digits = c(digits,readBin(fi,n=28*28*1000,size=1,what=integer(),signed=FALSE))
  y = c(y,rep(k,1000)) }
digits=array(digits,c(28,28,1000,10))
xo=matrix(c(digits),byrow=T,ncol=784)
dim(xo)

```

## Part(i) Columns are intensities for 300 pixels with greatest variability(overall)

We will construct this matrix by applying standard Deviation on the digits data and filter out the data for the first 300 column which explains maximum variability in the data.


```{r} 
# X1
  sdi=apply(digits,c(1,2),sd)
order_sdi=sort.list(c(sdi),decreasing=T)
X1 <- xo[,order_sdi[1:300]]
dim(X1)
```

## Part(ii) Columns are obtained by applying PCA(on covariance) to the full set of 784 pixel .Retain the number of components required to explain 90\% of variance.

We construct this matrix by applying PCA on covariance of the digits data. We find out the percentage of variance explained and build a matrix by taking those components which explains 90 percentage of variance in the data.



```{r}
# X2
x2_pca<- prcomp(xo)
percentexp <- 100 * cumsum(x2_pca$sdev^2)/sum(x2_pca$sdev^2)
x2_comp<-percentexp <=90 #components with 90% of variance
X2<-matrix(x2_pca$x[,x2_comp],nrow=10000)
dim(X2)
```

### Part(iii) Columns are discriminant variables produced by LDA analysis of X1

The MASS package contains functions for performing lda() and qda().lda() takes dependent variable, or the variable to be predicted and the variables that will be used to model or predict dependent variable. Predict values are based on a model, we pass the model object to the predict() function. Scores computed as part of the predict() method of objects of class "lda" gives us the discriminant variables. It is returned as component x of the object produced by predict()


```{r}
# X3
library(MASS)
o = lda(X1,y)
oo=predict(o) 
X3=oo$x 
dim(X3)
```
### Part(iv) Columns are discriminant variables produced by LDA analysis of X2

Same method as above(matrix X4) is used for creating matrix X4

```{r}
# X4
library(MASS)
p = lda(X2,y)
pp=predict(p) 
X4=pp$x 
dim(X4)
```

### Using X1,X2,X3 and X4

### 1. Create 3X3 boxplot of the discriminant vriables found in (iii) and (iv)

LDA creates new variables (LDs) by maximizing variance between groups. These are still linear combinations of original variables, but rather than explain as much variance as possible with each sequential LD, instead they are drawn to maximize the DIFFERENCE between groups along that new variable. Rather than a similarity matrix, LDA uses a comparison matrix of between and within groups sum of squares and cross-products. 
We have 10 digits in the dataset which will give 9 discriminant variables as per min(j-1).Boxplot gives us sepration between the groups as a function of which disciriminant score.For j=1:9, Z[,j] gives 9 different ways we can classify the data set, using the scores. z1[,1] gives the first way of classifying the data. Likewise,each of these 9 plots gives us 9 ways of classifying digits(0-9).The one where the medians are visually most separated and have less overlap, defines most efficient way of classifying these digits.



```{r}
# Boxplot of Discriminant Variables found in X3 Matrix
  z1 = oo$x 
par(mfrow=c(3,3)) 
for(j in 1:9) { 
  boxplot(z1[,j]~y,
          main=as.character(j)) }
          
```


Among all the plots, Plot 1 gives us most separated medians thus this discriminant better separates the classes than others.


```{r}
# Boxplot of Discriminant Variables found in X4 Matrix
z2 = pp$x 
par(mfrow=c(3,3)) 
for(j in 1:9) { 
  boxplot(z2[,j]~y,
          main=as.character(j)) }

```

Among all the plots, Plot 1 gives us most separated medians thus this discriminant better separates the classes than others.

### Plot on same graph the ANOVA statistic evaluating separation between classes for each of the 9 discriminant variables produced by X3 and X4


Anova is obtained by plotting sdv values from lda againt numer of discriminant variables.svd- the singular values returned by lda() function , which give the ratio of the between- and within-group standard deviations on the linear discriminant variables. Their squares are the canonical F-statistics.Using this we will plot the separation between classes for each of the 9 discriminant variables produced by matrix X3 and X4.

```{r}
  matplot(1:9,cbind(o$svd^2,p$svd^2),pch=20:21,main="ANOVA",
          xlab="Discriminant Variables",ylab="ANOVA statistic")

legend("topright",c("Matrix X3","Matrix X4"),pch=20:21,cex=1)

```
 The plot shows that for both the matrix X1 and X2, the 9 discriminant variables separates the classes very well.


### Evaluate re-substitution estimates of misclassification by LDA using each of X1 to X4.

A simple estimate of the error rate can be obtained by trying out the classification procedure on the same data set that has been used to compute the classification functions. This method is commonly referred to as resubstitution. The proportion of misclassifications resulting from resubstitution is called the apparent error rate.


```{r}
#For X1
library(MASS)
res_X1 = lda(y~X1)
o_X1=predict(res_X1)
yhat1 = c(o_X1$class)-1
x1_Rmiss=print(table(y,yhat1))
#misclassification rate
1-sum(diag(x1_Rmiss))/sum(x1_Rmiss)

```


```{r}
#For X2
res_X2= lda(y~X2)
o_X2=predict(res_X2)
yhat2 = c(o_X2$class)-1
x2_Rmiss=print(table(y,yhat2))
#misclassification rate
1-sum(diag(x2_Rmiss))/sum(x2_Rmiss)
```


```{r}
#For X3
yhat3 = c(oo$class)-1
x3_Rmiss=print(table(y,yhat3))
# misclassification rate
1-sum(diag(x3_Rmiss))/sum(x3_Rmiss)

```


```{r}
#For X4
yhat4 = c(pp$class)-1
x4_Rmiss=print(table(y,yhat4))
1-sum(diag(x4_Rmiss))/sum(x4_Rmiss)
```


### Evaluate cross-validation estimates of misclassification by LDA using each of X1 to X4. 

Cross-validation is a n-fold cross-validation, where n is the number of training instances. That is, n classifiers are built for all possible (n-1) element subsets of the training set and then tested on the remaining single instance.This involves no random subsampling and makes maximum use of the data.

```{r}
#For X1
cv_X1 = lda(y~X1,CV=T);  
yhat_1 = c(cv_X1 $class)-1
x1_cv=print(table(y,yhat_1))
1-sum(diag(x1_cv))/sum(x1_cv)

```


```{r}
#For X2
cv_X2 = lda(y~X2,CV=T);  
yhat_2 = c(cv_X2 $class)-1
x2_cv=print(table(y,yhat_2))
1-sum(diag(x2_cv))/sum(x2_cv)

```


```{r}
#For X3
cv_X3 = lda(y~X3,CV=T);  
yhat_3 = c(cv_X3 $class)-1
x3_cv=print(table(y,yhat_3))
1-sum(diag(x3_cv))/sum(x3_cv)
```



```{r}
#For X4
cv_X4 = lda(y~X4,CV=T);  
yhat_4 = c(cv_X4 $class)-1
x4_cv=print(table(y,yhat_4))
1-sum(diag(x4_cv))/sum(x4_cv)

```


Comparison of missclassification rate for Resubtitution estimates and Cross-Validation estimates 

Misclassification in Resubtitution Estimates for matrices
X1 : 0.1146 
X2 : 0.1189
X3 : 0.1146 
X4 : 0.1189 

Misclassification in Cross-Validation Estimates for matrices 
X1 : 0.1393 
X2 : 0.1276
X3 : 0.115 
X4 : 0.12 

The error rate estimated by cross-validation is more than the resubstitution estimate of the error rate. While resubstitution  model may minimize the Mean Squared Error on the training data, it can be optimistic in its predictive error. Whereas in crossvalidation data,as stated above,n classifiers are built for all possible (n-1) element subsets of the training set and then tested on the remaining single instance.By doing this cross-validation loops through all the data and gets classification accuracy for each time, they are then averaged to give a number more representative of overall accuracy.

### For LDA and QDA, compute re-substition and cross-validation estimates of misclassification using X4.Report the results on a single page- 4 sets of labelled 10X10 tables

LDA assumes that the observations are drawn from a Gaussian distribution with a common covariance matrix across each class. QDA, on the other-hand, provides a non-linear quadratic decision boundary.LDA and QDA algorithm are both based on Bayes theorem.

1.LDA - Resubstition and Cross Validation

```{r}
library(MASS)
yhat_L1 = c(pp$class)-1
a1=print(table(y,yhat_L1))
1-sum(diag(a1))/sum(a1)

# Cross Validation
cv_X4 = lda(y~X4,CV=T);  
yhat4_L2 = c(cv_X4 $class)-1
a2=print(table(y,yhat4_L2))
1-sum(diag(a2))/sum(a2)
```

1.QDA - Resubstition and Cross Validation

```{r}
#Resubstition 
library(MASS)
q1= qda(X4,y)
a=predict(q1)
yhat4_Q1 = c(a$class)-1
a3=print(table(y,yhat4_Q1))
1-sum(diag(a3))/sum(a3)

# Cross Validation
library(MASS)
q2= qda(X4,y,CV=T)
yhat4_Q2 = c(q2$class)-1
a4=print(table(y,yhat4_Q2))
1-sum(diag(a4))/sum(a4)

```

Misclassification in LDA
Re-subsitution
X4 : 0.1189 
Cross-Validation
X4 : 0.12 

Misclassification in QDA
Re-subsitution
X4 : 0.0932 
Cross-Validation
X4 : 0.0952 


we get some improvement with the QDA model.This suggests that the quadratic form assumed by QDA may capture the true relationship more accurately than the linear forms assumed by LDA.QDA is recommended if the training set is very large, so that the variance of the classifier is not a major concern, or if the assumption of a common covariance matrix is clearly untenable

