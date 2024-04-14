# Credit-Card-Fraud-Detection-in-R
this project aimed to develop machine learning models to detect credit card fraud. 
Three models were evaluated: Logistic Regression, Naive Bayes Classifier, and Decision Tree.
The results showed that the Logistic Regression model achieved the highest AUC score of 0.983,
while the Naive Bayes Classifier and Decision Tree models achieved AUC scores of 0.877 and 0.856, respectively.

The high AUC score of the Logistic Regression model indicates that it is able to accurately distinguish
between fraudulent and non-fraudulent transactions with a high degree of confidence. Both the Naive Bayes Classifier
and Decision Tree models also demonstrated good performance, with AUC scores of 0.877 and 0.856.

However, it is important to note that there are some limitations to the project, including the use of only three modeling
techniques and the lack of different sampling techniques. These limitations could be addressed in
future work to improve the performance of the models.

Overall, the results of this project demonstrate the potential of machine learning models in detecting credit card fraud,
and highlight the importance of
continued research and development in this field. With the increasing prevalence of credit card fraud, it is essential to develop accurate
and efficient methods for detecting and preventing fraud. This project provides a starting point for further research in this area,
and offers insights into the potential of machine learning models for detecting credit card fraud.

---
title: "Credit Card Fraud Detection in R"
author: "Langat Erick"
date: "2023-06-18"
output:
  word_document: default
  pdf_document: default
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(
  message = FALSE,
  warning = FALSE,
  echo = TRUE)
```

# **Credit Card Fraud Detection in R**

### **Introduction**

Credit card fraud is a major concern for financial institutions and consumers alike. Fraudulent transactions not only result in financial loses for banks, but can damage their reputation and lose the trust of customers. The increase in online transactions over the past decade has made real-time detection and prevention of financial fraud even more important.

A dataset has been provided that contains credit card transactions over a period of 2 days from September 2013. The dataset is highly unbalanced - containing 492 fraudulent transactions out of 284,807 total. To maintain anonymity and confidentiality, most of the original features have been transformed into principle components. Only the features time, class and amount have been preserved in their original form.

This notebook represents an attempt to analyse the data within the dataset and recommend a model that could be used for a practical credit card fraud detection system.

**Project Outline**

1.  Importing libraries and dataset

2.  Data Exploration

3.  Data Manipulation

4.  Data Modelling

    -   4.1 Logistic Regression

    -   4.2 Naive Bayes Classifier

    -   4.3 Decision Tree

5.  Results and Limitations

6.  Conclusion

#### **Importing libraries and dataset**

```{r}
library(tidyverse)
# library(Hmisc)
library(caret)
library(ROSE)
library(ggplot2)
#library(gridExtra)
library(e1071)
library(corrplot)
library(broom)
library(partykit)
```

```{r}
# Load the dataset

data <- read.csv('C:/Users/JIT/Desktop/creditcard.csv',
                 stringsAsFactors = T)
head(data)
```

```{r}
str(data)
```

```{r}
# Convert Class column into Factors

data$Class <- as.factor(data$Class)
levels(data$Class) <- c("Legit", "Fraud")
```

#### **Data Exploration**

```{r}
summary(data)
```

The majority of the data has undergone PCA, limiting the amount of information that can be obtained from most features. However, the "Amount" and "Class" features could provide some insight of the data.

```{r}
ggplot(data, aes(x=Class, fill = Class)) +
        geom_bar(color = "black") + 
         ggtitle("Bar Distribution of Class") + 
         theme(plot.title = element_text(hjust = 0.5, face = "bold"))
```

```{r}
counts <- table(data$Class)
result <- data.frame(table(data$Class), round(prop.table(counts), 5))
noms <- c("Class", "Value", "none", "Proportion")

names(result) <- noms

print(result[c(1,2,4)])
```

-   **Legit** cases constitute 99.8% (284315) of the dataset.

-   **Fraud** cases constitute 0.2% (492)

As expected from the brief, analysis of the Class feature shows that the dataset is unbalanced and the vast majority of the cases are legitimate. Accuracy will not be an appropriate measure of performance here and AUC will be used instead.

```{r}
p <- ggplot(data, aes(x=Amount)) + geom_histogram(fill = "green",
                                color = "black", bins = 30) + 
       ggtitle("Histogram Distribution of Amount") +
        theme(plot.title = element_text(hjust = 0.5, face = "bold")) 

p
```

```{r}
# Function to find range of outliers in Amount

find_outlier_range <- function(x){
    outliers <- boxplot.stats(x)$out
    return(range(outliers))
}

find_outlier_range(data$Amount)
```

The histogram above shows that the vast majority of the transactions have low values. However, there are a non-neglible number of outliers with values ranging from 184.52 up to 25691.16 dollars. There is clearly a positive skew in the feature that will need to be considered during feature selection and modelling.

**Data Manipulation**

```{r}
# Check for missing values

colSums(is.na(data))
sum(is.na(data))
```

There are no missing values in the dataset.

```{r}
# Check for duplicate rows

sum(duplicated(data))

# Remove any duplicate rows

data <- distinct(data)
```

There are 1081 duplicate rows in the dataset which can be removed.

Also the Time feature can be removed as each value is unique and provides no predictive power.

```{r}
data$Time <- NULL
```

```{r}
# Standardise Amount feature

data$Amount <- scale(data$Amount)

summary(data$Amount)
```

```{r}
data2 <- data

data2$Class <- as.numeric(data2$Class)

corr <- cor(data2[], method = "pearson")

corrplot(corr)
```

As the data has previously gone through PCA, there is little to no correlation between most of the features. Correlation between the target variable and the PCA features varys with no particular features standing out.

#### **Data Modelling**

The dataset will be split for training and testing, with 80% of the data used for training and 20% used for testing. The seed will also be set to reproduce results.

```{r}
set.seed(123)

indices <- createDataPartition(data$Class, p=0.8, list = F)
trainData <- data[indices,]
testData <- data[-indices,]
```

We are going to be building using the following models:

-   Logistic Regression

-   Naive Bayes Classifier

-   Decision Tree

    **Logistic Regression**

    ```{r}
    model_lr <- glm(Class ~ ., data = trainData, family = "binomial")

    ```

```{r}
model_lr_prediction <- predict(model_lr, newdata = testData, type = 'response') 

roc.curve(testData$Class, model_lr_prediction, plotit = TRUE)
```

**Naive Bayes Classifier**

```{r}
NBmodCCF <- naiveBayes(Class ~ ., data = trainData, laplace = 1)
model_nb_prediction <- predict(NBmodCCF,
                               newdata = testData, 
                               type = "class")

roc.curve(testData$Class, model_nb_prediction, plotit = TRUE)
```

**Decision Tree**

```{r}
set.seed(123)

DTmodCCF <- ctree(Class ~ .,
                 data=trainData)
model_ctree_prediction <- predict(DTmodCCF, newdata = testData, type = "response")

roc.curve(testData$Class, model_ctree_prediction, plotit = TRUE)
```

#### **Results and Limitations**

The AUC performance results for the 3 models are as follows:

-   Logistic Regression: 0.983

-   Naive Bayes Classifier: 0.877

-   Decision Tree: 0.856

Although the naive bayes classifier and decision tree performed well, logistic regression is the clear winner in this group.

There are some limitations of the project that may be addressed in the future that may change the outcome. These would include:

-   Use of only 3 modelling techniques. If we add more modelling types in the future, there may be a model that performs even better than the LR model.

-   Lack of different sampling techniques. This is something I will learn more about and add to future projects.

### **Conclusion**

In conclusion, this project aimed to develop machine learning models to detect credit card fraud. Three models were evaluated: Logistic Regression, Naive Bayes Classifier, and Decision Tree. The results showed that the Logistic Regression model achieved the highest AUC score of 0.983, while the Naive Bayes Classifier and Decision Tree models achieved AUC scores of 0.877 and 0.856, respectively.

The high AUC score of the Logistic Regression model indicates that it is able to accurately distinguish between fraudulent and non-fraudulent transactions with a high degree of confidence. Both the Naive Bayes Classifier and Decision Tree models also demonstrated good performance, with AUC scores of 0.877 and 0.856.

However, it is important to note that there are some limitations to the project, including the use of only three modeling techniques and the lack of different sampling techniques. These limitations could be addressed in future work to improve the performance of the models.

Overall, the results of this project demonstrate the potential of machine learning models in detecting credit card fraud, and highlight the importance of continued research and development in this field. With the increasing prevalence of credit card fraud, it is essential to develop accurate and efficient methods for detecting and preventing fraud. This project provides a starting point for further research in this area, and offers insights into the potential of machine learning models for detecting credit card fraud.
