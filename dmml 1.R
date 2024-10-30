################################################################################
## DMML1 - Assignment 
################################################################################

### Libraries ##################################################################
install.packages("mda")
install.packages("ROCR")
install.packages("highcharter")
install.packages("ggord")
install.packages("psych")
install.packages("klaR")
library(readr)
library(ggplot2)
library(dplyr)
library(corrplot)
library(tidyr)
library(caret)
library(GGally)
library(rpart)
library(rpart.plot)
library(class)
library(e1071)
library(gridExtra)
library(MASS)
library(mda)
library(ROCR)
library(pROC)
library(highcharter)
library(ggord)
library(psych)
library(klaR)

## Load the data ###############################################################
df <- read_csv("Desktop/df.csv")
str(df)
summary(df)

### Split the Data #############################################################

#  split to be 75%, 30% --------------------------------------------------------
n <- nrow(df)
ind1 <- sample(1:n, round(0.7 * n))  # 70% for training
ind2 <- setdiff(1:n, ind1)          # Remaining 30% for testing

train_data <- df[ind1, ]
test_data <- df[ind2, ]

train_data <- train_data[, -1]
test_data <- test_data[, -1]

dim(train_data)
dim(test_data)
str(test_data)
str(train_data) 

# Sum of Class equal to 0
sum(df$Class == 0)
sum(train_data$Class==0)
sum(test_data$Class==0)

# Sum of Class equal to 1
sum(df$Class == 1)
sum(train_data$Class==1)
sum(test_data$Class==1)

## EDA #########################################################################
summary(train_data)

# Create a vector of colors based on the Class variable
colors <- ifelse(train_data$Class == 0, "blue", "red")

# Set the color palette for the plot
palette <- c("blue", "red")

# Create the fancier pairs plot with colors, including all variables
ggpairs(train_data, aes(color = colors, shape=factor(Class)), columns = 2:12,
        lower = list(continuous = wrap("points", alpha = 0.5, size = 0.7)),
        upper = list(combo = wrap("box", alpha = 0.5, fill = "grey")),
        title = "Pairs Plot") +
  scale_color_manual(values = palette) +
  scale_shape_manual(values = c(0, 1)) +
  theme_bw() +
  theme(legend.position = "right")


# Distribution of the target variable
ggplot(train_data, aes(Class)) +
  geom_bar(fill = "blue") +
  labs(x = "Drug Use", y = "Count", title = "Distribution of Drug Use")

# Pairwise correlation plot
cor_matrix <- cor(train_data[, c("Age", "Education", "Country", "Ethnicity", "Nscore", "Escore", 
                         "Oscore", "Ascore", "Cscore", "Impulsive", "SS", "Class")])
corrplot(cor_matrix, type = "upper", tl.col = "black")

# Box plots of numerical variables
vars <- c("Age", "Education", "Nscore", "Escore", "Country", "Ethnicity",
              "Oscore", "Ascore", "Cscore", "Impulsive", "SS")
train_data %>%
  gather(key = "Variable", value = "Value", one_of(vars)) %>%
  ggplot(aes(x = Variable, y = Value)) +
  geom_boxplot(fill = "lightblue", color = "blue") +
  labs(x = "Variable", y = "Value", title = "Box Plots of  Variables")


## Box plots --------------------------------------------------------------------

plots <- list()  # Create an empty list to store the box plots

# Create a box plot for each variable and store it in the list
for (col in names(df)[-c(1, ncol(df))]) {
  plot <- ggplot(df, aes(x = factor(Class), y = .data[[col]], fill = factor(Class))) +
    geom_boxplot() +
    labs(x = "Class", y = col) +
    scale_fill_manual(values = c("blue", "red")) +
    ggtitle(paste("Box plot of", col, "by Class")) +
    theme_bw()
  
  plots[[col]] <- plot
}

# Arrange the box plots in a grid layout
grid.arrange(grobs = plots, ncol = 3)


## K - nearest neighbors #######################################################
# Create a vector of labels
train_label <- train_data$Class
test_label <- test_data$Class
sq.n <- sqrt(420)

# Use kNN to identify the test set categories
class.pred1 <- knn(train = train_data[-12], test = test_data[, -12], cl = as.factor(train_label), k = 1)
class.pred2 <- knn(train = train_data[-12], test = test_data[, -12], cl = as.factor(train_label), k = 10)
class.pred3 <- knn(train = train_data[-12], test = test_data[, -12], cl = as.factor(train_label), k = sq.n) 


# Confusion matrices
confusionMatrix(data = class.pred1, reference = as.factor(test_data$Class))   # 0.67
confusionMatrix(data = class.pred2, reference = as.factor(test_data$Class))   # 0.75
confusionMatrix(data = class.pred3, reference = as.factor(test_data$Class))   # 0.77


## Find the optimal K ----------------------------------------------------------
corr.class.rate<-numeric()
for(k in 1:25)
{
  pred.class<-knn(train_data[,-12], test_data[,-12], as.factor(train_label), k=k)
  corr.class.rate[k]<-sum((pred.class==as.factor(test_data$Class)))/length(pred.class)
}

plot_data <- data.frame(k = 1:25, rate = corr.class.rate)

hchart(plot_data, "line", hcaes(k, rate)) %>%
  hc_title(text = "Correct Classification Rates for the Test Data for a range of k") %>%
  hc_xAxis(title = list(text = "k")) %>%
  hc_yAxis(title = list(text = "Correct Classification Rate")) %>%
  hc_add_theme(hc_theme_google()) %>%
  hc_plotOptions(line = list(animation = FALSE))


corr.class.rate
knn.k2 <- which.max(corr.class.rate) # 21

predictions.k.train <- knn(train_data[,-12], train_data[,-12], as.factor(train_label), k=21) 
predictions.k.test <- knn(train_data[,-12], test_data[,-12], as.factor(train_label), k=21) 

# Create the confusion matrix --------------------------------------------------
cf.k.nearest.train <- confusionMatrix(data = predictions.k.train, reference = as.factor(train_data$Class))

##              Reference
## prediction     0       1
#           0   157      34
#           1    53     176

# Accuracy = 0.7929

# correct classification rate for Class=0:
157/(157+53) # 0.75

# correct classification rate for Class=1:
176/(176+34) # 0.84

cf.k.nearest.test <- confusionMatrix(data = predictions.k.test, reference = as.factor(test_data$Class))

##              Reference
## prediction     0      1
#           0    66     17
#           1    24     73

# accuracy = 0.7722

# correct classification rate for Class=0:
66/(66+24) # 0.73

# correct classification rate for Class=1:
73/(73+17) # 0.81

plot_predictions <- data.frame(
  test_data$Age,
  test_data$Education,
  test_data$Country,
  test_data$Ethnicity,
  test_data$Nscore,
  test_data$Escore,
  test_data$Oscore,
  test_data$Ascore,
  test_data$Cscore,
  test_data$Impulsive,
  test_data$SS,
  predicted= predictions.k.test
)

# remaining column names
colnames(plot_predictions) <- c("Age",
                                "Education",
                                "Country",
                                "Ethnicity",
                                "Nscore",
                                "Escore",
                                "Oscore",
                                "Ascore",
                                "Cscore",
                                "Impulsive",
                                "SS",
                                "predicted")
plot_predictions



p1 <- ggplot(plot_predictions, aes(Age,
                                   Education, 
                                   color = predicted,
                                   fill = predicted)) +
  geom_point(size = 2.5) +
  geom_text(aes(label = as.factor(test_label)), hjust = 1, vjust = 2) +
  ggtitle("Predicted relationship between Age and Education") +
  theme(plot.title = element_text(hjust = 0.5))


p2 <-  ggplot(plot_predictions, aes(Nscore,
                                    Oscore, 
                                    color = predicted,
                                    fill = predicted)) +
  geom_point(size = 2.5) +
  geom_text(aes(label = as.factor(test_label)), hjust = 1, vjust = 2) +
  ggtitle("Predicted relationship between Nscore and Oscore") +
  theme(plot.title = element_text(hjust = 0.5))

p3 <-  ggplot(plot_predictions, aes(Cscore,
                                    Oscore, 
                                    color = predicted,
                                    fill = predicted)) +
  geom_point(size = 2.5) +
  geom_text(aes(label = as.factor(test_label)), hjust = 1, vjust = 2) +
  ggtitle("Predicted relationship between Cscore and Oscore") +
  theme(plot.title = element_text(hjust = 0.5))

p4 <-  ggplot(plot_predictions, aes(Cscore,
                                    Impulsive, 
                                    color = predicted,
                                    fill = predicted)) +
  geom_point(size = 2.5) +
  geom_text(aes(label = as.factor(test_label)), hjust = 1, vjust = 2) +
  ggtitle("Predicted relationship between Cscore and Impulsive") +
  theme(plot.title = element_text(hjust = 0.5))

grid.arrange(p1,p2,p3,p4, ncol=2)
grid.arrange(p2,p3, ncol=2)


### K decision boundaries ------------------------------------------------------

# Fit k-NN model with k = 21
k <- 21
knn_model <- knn(train_data[, c("Nscore", "Oscore")], test_data[, c("Nscore", "Oscore")], cl = as.factor(train_label), k = k)

# Create a new data frame for visualization
plot_data <- data.frame(Nscore = test_data$Nscore, Oscore = test_data$Oscore, Class = test_data$Class, Predicted = knn_model)

# Plot the data points colored by the class labels and predicted labels
ggplot(plot_data, aes(x = Nscore, y = Oscore, color = factor(Class), shape = factor(Predicted))) +
  geom_point() +
  scale_color_manual(values = c("blue", "red"), labels = c("Never Used", "Used at some point")) +
  scale_shape_manual(values = c(16, 17), labels = c("Correct", "Misclassified")) +
  labs(x = "Nscore", y = "Oscore", title = "k-NN Classification (k = 21)") +
  theme_minimal()

# Add decision boundaries
resolution <- 100
x_grid <- seq(min(test_data$Nscore), max(test_data$Nscore), length.out = resolution)
y_grid <- seq(min(test_data$Oscore), max(test_data$Oscore), length.out = resolution)
grid <- expand.grid(Nscore = x_grid, Oscore = y_grid)
grid$Predicted <- knn(train_data[, c("Nscore", "Oscore")], grid, train_data$Class, k = k)
grid$Class <- NA

# Overlay decision boundaries on the plot
ggplot() +
  geom_point(data = plot_data, aes(x = Nscore, y = Oscore, color = factor(Class), shape = factor(Predicted))) +
  geom_tile(data = grid, aes(x = Nscore, y = Oscore, fill = factor(Predicted)), alpha = 0.3) +
  scale_color_manual(values = c("blue", "red"), labels = c("Never Used", "Used at some point")) +
  scale_shape_manual(values = c(16, 17), labels = c("Correct", "Misclassified")) +
  scale_fill_manual(values = c("lightblue", "pink"), labels = c("Never Used (Predicted)", "Used at some point (Predicted)")) +
  labs(x = "Nscore", y = "Oscore", title = "k-NN Classification (k = 21)") +
  theme_minimal()

## Decision Tree  ##############################################################

fit <- rpart(as.factor(Class)~., data = train_data, method = 'class')
rpart.plot(fit, extra = 106)

printcp(fit)

plotcp(fit)

predict_unseen <-predict(fit, test_data, type = 'class')          

table_mat <- table(as.factor(test_data$Class), predict_unseen)
table_mat
accuracy_Test <- sum(diag(table_mat)) / sum(table_mat) # Accuracy = 0.7556


## Tune the hyper parameters ---------------------------------------------------

control <- rpart.control(minsplit = 10,
                         minbucket = 2,
                         maxdepth = 4,
                         cp = 0.011)
tune_fit <- rpart(as.factor(Class)~.,
                  data = train_data,
                  method = 'class',
                  control = control)


predict_unseen <-predict(tune_fit, test_data, type = 'class')          

table_mat <- table(as.factor(test_data$Class), predict_unseen)
table_mat
accuracy_Test <- sum(diag(table_mat)) / sum(table_mat) # 0.7611

## Final Tree ------------------------------------------------------------------

fit.tree <- rpart(as.factor(Class)~.,
                  data = train_data,
                  method = 'class',
                  minsplit=10,
                  minbucket=2, 
                  maxdepth=4,
                  cp=0)

rpart.plot(fit.tree, extra = 106)

printcp(fit.tree)


# Make predictions on the train data ---------------------------------------------
predictions.train.tree <- predict(fit.tree, newdata = train_data, type = "class")

# Create the confusion matrix
confusion_matrix <- confusionMatrix(predictions.train.tree, as.factor(train_data$Class))

##              Reference
#  prediction     0       1
#           0   165      31
#           1    45     179

## Accuracy = 0.819
# correct classification rate for Class=0:
165/(165+45) # 0.79

# correct classification rate for Class=1:
179/(179+31) # 0.85


# Make predictions on the test data
predictions.test.tree <- predict(fit.tree, newdata = test_data, type = "class")

# Create the confusion matrix
confusion_matrix <- confusionMatrix(predictions.test.tree, as.factor(test_data$Class))

##              Reference
## prediction     0       1
#           0    66      19
#           1    24      71

## Accuracy = 0.7611

# correct classification rate for Class=0:
66/(66+24) # 0.73

# correct classification rate for Class=1:
71/(71+19) # 0.79

printcp(fit.tree)

plotcp(fit.tree)

## SVM #########################################################################

## Linear ## -------------------------------------------------------------------
svm.linear <- svm(as.factor(Class) ~ ., data = train_data, kernel = "linear")

# range of cost values 
cost_values <- c(0.01,10, by = 0.05)

# Perform cost parameter optimization using cross-validation
svm_tuned <- tune(svm, as.factor(Class) ~ .,
                  data = train_data,
                  kernel = "linear", 
                  ranges = list(cost = cost_values))

# Get the best cost value
best_cost <- svm_tuned$best.parameters$cost

# Train the SVM model with the best cost value
svm_model.linear <- svm(as.factor(Class) ~ .,
                        data = train_data, 
                        kernel = "linear", 
                        cost = best_cost) # cost 10

## Predictions on train data ---------------------------------------------------
svm_predictions.train.linear <- predict(svm_model.linear, newdata = train_data[,-12])

cf.linear.train <- confusionMatrix(svm_predictions.train.linear, as.factor(train_data$Class))

## Accuracy = 0.782


linear_error.train <- sum(svm_predictions.train.linear != as.factor(train_data$Class)) /
  length(as.factor(train_data$Class))
linear_error.train

## Plots of Train Data linear SVM ----------------------------------------------
plot(svm_model.linear, train_data, Nscore ~ Oscore)

plot(svm_model.linear, train_data, Age ~ Education)

plot(svm_model.linear, train_data, Cscore ~ Oscore)

## Predictions on test data ----------------------------------------------------
svm_predictions.test.linear <- predict(svm_model.linear, newdata = test_data[,-12])

cf.linear.test <- confusionMatrix(svm_predictions.test.linear, as.factor(test_data$Class))

##              Reference
## prediction     0       1
#           0    63      19
#           1    27      71
## Accuracy = 0.7444

linear_error.test <- sum(svm_predictions.test.linear != as.factor(test_data$Class)) / length(as.factor(test_data$Class))
linear_error.test

## Plots of Test Linear SVM ----------------------------------------------------
plot(svm_model.linear, test_data, Nscore ~ Oscore)

plot(svm_model.linear, test_data, Age ~ Education)

plot(svm_model.linear, test_data, Cscore ~ Oscore)


## Radial ## -------------------------------------------------------------------
gamma <- seq(0,1, by = 0.005)
cost <- c(0.01,10, by = 0.05)
parms <- expand.grid(cost=cost, gamma=gamma)  
acc_test <- numeric()
accuracy1 <- NULL
accuracy2 <- NULL

# - Identify best combintion of Gamma and Cost for SVM - #

for (i in 1:NROW(parms)) {
  learn_svm <- svm(as.factor(Class) ~ ., data = train_data, gamma = parms$gamma[i], cost = parms$cost[i])
  pre_svm <- predict(learn_svm, test_data[, -12])
  accuracy1 <- confusionMatrix(pre_svm, as.factor(test_data$Class))
  accuracy2[i] <- accuracy1$overall[1]
}

acc <- data.frame(p= seq(1,NROW(parms)), cnt = accuracy2)
opt_p <- subset(acc, cnt==max(cnt))[1,] 
sub <- paste("Optimal number of parameter is", opt_p$p, "(accuracy :", opt_p$cnt,") in SVM")

## Plot ------------------------------------------------------------------------
hchart(acc, 'line', hcaes(p, cnt)) %>%
  
  hc_title(text = "Accuracy With Varying Parameters (SVM)") %>%
  
  hc_subtitle(text = sub) %>%
  
  hc_add_theme(hc_theme_google()) %>%
  
  hc_xAxis(title = list(text = "Number of Parameters")) %>%
  
  hc_yAxis(title = list(text = "Accuracy"))

print(paste("Best Cost :",parms$cost[opt_p$p],", Best Gamma:",parms$gamma[opt_p$p]))

#"Best Cost : 0.01 , Best Gamma: 0.045" - outcome

# - Apply best gamma and cost to the SVM model ---------------------------------

svm_radial_model <-svm(as.factor(Class) ~.,
                       data=train_data,
                       type="C-classification",
                       kernel="radial",
                       scale = TRUE, 
                       cost = 0.01, 
                       gamma = 0.045)
summary(svm_radial_model)

## Predictions on train data ---------------------------------------------------
svm_predictions.train.radial <- predict(svm_radial_model, newdata = train_data[,-12])

cf.radial.train <- confusionMatrix(svm_predictions.train.radial, as.factor(train_data$Class))

##              Reference
## prediction     0       1
#           0   162      42
#           1    48     168
## Accuracy = 0.7857


radial_error.train <- sum(svm_predictions.train.radial != as.factor(train_data$Class)) /
  length(as.factor(train_data$Class))
radial_error.train

## Plots of Train Data ---------------------------------------------------------
plot(svm_radial_model, train_data, Nscore ~ Oscore)

plot(svm_radial_model, train_data, Age ~ Education)

plot(svm_radial_model, train_data, Cscore ~ Oscore)

## Predictions on test data ----------------------------------------------------

svm_predictions.test.radial <- predict(svm_radial_model, newdata = test_data[,-12])

cf.radial.test <- confusionMatrix(svm_predictions.test.radial, as.factor(test_data$Class))

##              Reference
## prediction     0       1
#           0    67      14
#           1    23      76
## Accuracy = 0.7944


radial_error.test <- sum(svm_predictions.test.radial != as.factor(test_data$Class)) /
  length(as.factor(test_data$Class))
radial_error.train


## Plots of Test Data ----------------------------------------------------------

plot(svm_radial_model, test_data, Nscore ~ Oscore)

plot(svm_radial_model, test_data, Age ~ Education)

plot(svm_radial_model, test_data, Cscore ~ Oscore)

## POLYNOMIAL SVM ## -----------------------------------------------------------
svfit=svm(as.factor(Class)~.,data=train_data,kernel="polynomial",gamma=1,cost=1)
svm.pred <- predict(svfit, test_data[,-12])
confusionMatrix(svm.pred, as.factor(test_data$Class))

poly.tune<-tune(svm,
                as.factor(Class) ~ .,
                data=train_data,
                type="C-classification",
                kernel="polynomial", 
                degree=2,
                ranges=list(cost = c(0.01,10, by = 0.05),
                            gamma = seq(0,1, by = 0.005),
                            coef0 = c(3,5,10)))

summary(poly.tune)$best.parameters  ## cost = 0.1, gamma = 0.01, coef0 = 10.

# - Apply best gamma and cost and coef to the SVM model ------------------------

svm_polynomial_model <-svm(as.factor(Class) ~.,
                       data=train_data,
                       type="C-classification",
                       kernel="polynomial",
                       scale = TRUE, 
                       cost = 0.01, 
                       gamma = 0.045,
                       coef0 = 10)

summary(svm_polynomial_model)

## Predictions on train data ---------------------------------------------------

svm_predictions.train.polynomial <- predict(svm_polynomial_model, newdata = train_data[,-12])

cf.polynomial.train <- confusionMatrix(svm_predictions.train.polynomial, as.factor(train_data$Class))


## Accuracy = 0.78

polynomial_error.train <- sum(svm_predictions.train.polynomial != as.factor(train_data$Class)) /
  length(as.factor(train_data$Class))
polynomial_error.train


## Plots of Train Data ---------------------------------------------------------
plot(svm_polynomial_model, train_data, Nscore ~ Oscore)

plot(svm_polynomial_model, train_data, Age ~ Education)

plot(svm_polynomial_model, train_data, Cscore ~ Oscore)

## Predictions on test data ----------------------------------------------------

svm_predictions.test.polynomial <- predict(svm_polynomial_model, newdata = test_data[,-12])

cf.polynomial.test <- confusionMatrix(svm_predictions.test.polynomial, as.factor(test_data$Class))

##              Reference
## prediction     0       1
#           0    63      18
#           1    27      72
## Accuracy = 0.75


polynomial_error.test <- sum(svm_predictions.test.polynomial != as.factor(test_data$Class)) /
  length(as.factor(test_data$Class))
polynomial_error.test


## Plots of Test Data ----------------------------------------------------------
plot(svm_polynomial_model, test_data, Nscore ~ Oscore)

plot(svm_polynomial_model, test_data, Age ~ Education)

plot(svm_polynomial_model, test_data, Cscore ~ Oscore)

## ERROR TRAIN COMPARISON ------------------------------------------------------
error_train<-matrix(c(linear_error.train,radial_error.train,polynomial_error.train),1,3)
colnames(error_train)<-c("Linear","Radial","Polynomial (quadr.)")
round(error_train,4)


## ERROR TEST COMPARISON -------------------------------------------------------
error_test <- matrix(c(linear_error.test, radial_error.test, polynomial_error.test),1,3)
colnames(error_test)<-c("Linear","Radial","Polynomial (quadr.)")
round(error_test,4)

##    Linear   Radial   Polynomial (quadr.)
##      0.25   0.2056                  0.25

## Model Based Classification ##################################################

## Fit the CVA model -----------------------------------------------------------
G <- length(unique(train_data$Class))
cva_model <- lda(as.factor(Class) ~ ., data = train_data,prior=rep(1/G,G))

# Make predictions on the test data
cva_predictions <- predict(cva_model, newdata = train_data)

xtab.valid.cva <- table(cva_predictions$class, as.factor(train_data$Class))
confusionMatrix(xtab.valid.cva)


##              Reference
#  prediction     0       1
#           0   167      37
#           1    43     173
## Accuracy = 0.8095

## Discriminant Analysis ##-----------------------------------------------------
pairs.panels(train_data[1:11],
            gap = 0,
            bg =c("purple", "yellow")[as.factor(train_data$Class)],
            pch=21)

## LDA MODEL -------------------------------------------------------------------
## Model
lda.model = lda(as.factor(Class) ~., data=train_data)
lda.model
plot(lda.model)

## Predicting training results -------------------------------------------------
predict_lda <- predict(lda.model)
xtab.train.lda<-table(predict_lda$class,as.factor(train_data$Class))
confusionMatrix(xtab.train.lda)

##               Reference
## prediction     0      1
#           0   167     37
#           1    43    173

## accuracy = 0.8095


## The below plot shows how the response class has been classified by the LDA classifier.
## The X-axis shows the value of line defined by the co-efficient of linear discriminant 
## for LDA model. The two groups are the groups for response classes.

predmodel.train.lda <- predict(lda.model)
ldahist(predmodel.train.lda$x[, 1], g = predmodel.train.lda$class)


## Prediction on test data -----------------------------------------------------
predict_lda_test <- predict(lda.model, newdata = test_data)
xtab_test_lda <- table(predict_lda_test$class, as.factor(test_data$Class))
confusion_matrix_lda <- confusionMatrix(xtab_test_lda)

##               Reference
## prediction     0      1
#           0    64     20
#           1    26     70

## accuracy = 0.7444

partimat(Class ~  Nscore + Escore + Oscore + 
           Ascore + Cscore,
         data= train_data %>% 
           mutate(Class= factor(Class)),
         method="lda")

partimat(Class ~  Nscore + Escore + Oscore +
           Ascore + Cscore,
         data= test_data %>% 
           mutate(Class= factor(Class)),
         method="lda")

## QDA Model -------------------------------------------------------------------
qda.model = qda(as.factor(Class)~., data=train_data)
qda.model

## Predicting training results -------------------------------------------------

predict_qda <- predict(qda.model)
xtab.train.qda<-table(predict_qda$class,as.factor(train_data$Class))
confusionMatrix(xtab.train.qda)


## Accuracy = 0.7995

## Prediction on test data
predict_qda_test <- predict(qda.model, newdata = test_data)
xtab_test_qda <- table(predict_qda_test$class, as.factor(test_data$Class))

confusion_matrix <- confusionMatrix(xtab_test_qda)

##               Reference
## prediction     0      1
#           0    62     28
#           1    28     62
## Accuracy = 0.6889 

dev.off()
partimat(Class ~  Nscore + Escore + Oscore + 
           Ascore + Cscore,
         data= train_data %>% 
           mutate(Class= factor(Class)),
         method="qda")

partimat(Class ~  Nscore + Escore + Oscore +
           Ascore + Cscore,
         data= test_data %>% 
           mutate(Class= factor(Class)),
         method="qda")

## Mixture discriminant Analysis -----------------------------------------------

# Fit  mda  model --------------------------------------------------------------
model.mda <- mda(Class~., data = train_data)
model.mda

plot(model.mda)

## Predictions on train data ---------------------------------------------------
predicted.classes <- model.mda %>% predict(train_data)

# Model accuracy
mean(predicted.classes == as.factor(train_data$Class))

## Prediction on Test data -----------------------------------------------------
predicted.classes <- model.mda %>% predict(test_data)

# Model accuracy
mean(predicted.classes == as.factor(test_data$Class))

## Model Comparisons ###########################################################
# Compare models
model_names <- c("k-Nearest Neighbors", "Decision Tree", "SVM (Radial)", "LDA")
accuracy <- c(cf.k.nearest.test$overall["Accuracy"], confusion_matrix$overall["Accuracy"],
              cf.radial.test$overall["Accuracy"], confusion_matrix_lda$overall["Accuracy"])
sensitivity <- c(cf.k.nearest.test$byClass["Sensitivity"], confusion_matrix$byClass["Sensitivity"],
                 cf.radial.test$byClass["Sensitivity"], confusion_matrix_lda$byClass["Sensitivity"])
specificity <- c(cf.k.nearest.test$byClass["Specificity"], confusion_matrix$byClass["Specificity"],
                 cf.radial.test$byClass["Specificity"], confusion_matrix_lda$byClass["Specificity"])
kappa <- c(cf.k.nearest.test$overall["Kappa"], confusion_matrix$overall["Kappa"],
           cf.radial.test$overall["Kappa"], confusion_matrix_lda$overall["Kappa"])

comparison_df <- data.frame(Model = model_names, Accuracy = accuracy, Sensitivity = sensitivity,
                            Specificity = specificity, Kappa = kappa)
comparison_df

## ROC CURVES ##################################################################
## KNN -------------------------------------------------------------------------
par(mfrow = c(2, 2))
k_value <-21
knn_model <- train(train_data, train_label, method = "knn", 
                   tuneGrid = data.frame(k = k_value))

test_predictions <- predict(knn_model, newdata = test_data)

knn_probs <- predict(knn_model, newdata = test_data)

# Create ROC curve and calculate AUC
knn_roc <- roc(as.factor(test_data$Class), knn_probs)

# Plot the ROC curve with AUC value
roc.knn <- plot(knn_roc, main = "ROC Curve - KNN", print.auc = TRUE,
     col = "blue", lwd = 2)

## Tree ------------------------------------------------------------------------
p1 <- predict(fit.tree, test_data, type = 'prob')
p1 <- p1[, 2]

# Create ROC curve
r <- multiclass.roc(as.factor(test_data$Class), p1, percent = TRUE)
roc <- r[['rocs']]
r1 <- roc[[1]]

# Plot the ROC curve
roc.tree <- plot.roc(r1,
                     print.auc = TRUE,
                     main = 'ROC Curve - Decision Tree',
                     col = "darkgreen",          
                     lwd = 2)  
## SVM -------------------------------------------------------------------------
svm_probs <- predict(svm_radial_model, newdata = test_data, probability = TRUE)

# Extract the probabilities of the positive class (Class 1)
svm_probs_class1 <- attr(svm_probs, "probabilities")[, "1"]

# Create ROC curve and calculate AUC
svm_roc <- roc(as.factor(test_data$Class), svm_probs_class1)

# Plot the ROC curve with AUC value
roc.svm <- plot(svm_roc, main = "ROC Curve - SVM Radial", print.auc = TRUE,
     col = "red", lwd = 2)

## LDA -------------------------------------------------------------------------
lda_probs <- predict(lda.model, newdata = test_data)$posterior[, "1"]
lda_roc <- roc(as.factor(test_data$Class), qda_probs)

roc.lda <- plot(qda_roc, main = "ROC Curve - LDA", print.auc = TRUE,
     col = "purple", lwd = 2)

