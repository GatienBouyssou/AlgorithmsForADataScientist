# Homework1

## Tasks

### Definition

Look at the TO DO 

Caroline : R algorithms for classification and regression : explanations, mesures, plots
Gatien   : scikit learn on classification (logistic regression + 1 to choose) 
Ankit    : 

Unassigned tasks : 

- complete Rmarkdown definitions, formulas 
- scikit learn for regression (simple and multiple regression with the dataset down)
- write the comparaison part when we have the results
- write the Validation part (part 4)

### Datasets 

#### Dataset for classification 

id <- "1GNbIhjdhuwPOBr0Qz82JMkdjUVBuSoZd"
tennis <- read.csv(sprintf("https://docs.google.com/uc?id=%s&export=download",id), header = T)

#### Dataset for regression 

id <- "1heRtzi8vBoBGMaM2-ivBQI5Ki3HgJTmO" # google file ID
data <- read.csv(sprintf("https://docs.google.com/uc?id=%s&export=download",  id), header = T)

### R : 

Add a classification method 
Either KNN, or decision trees or linear (or quadratic) discriminant analysis

Add two regression methods
Simple linear regression
Multiple Linear Regression 

### Scikit-learn 

Add a classification method
The same as for R

Add two regression method 
Same as R 
