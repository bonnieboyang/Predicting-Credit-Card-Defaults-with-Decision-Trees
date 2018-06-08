
library(dplyr)
library(ggplot2)
library(GGally)
library(caret)

set.seed(100)

rm(list = ls())  # empty the environment

cc = read.delim("credit_card_default.tsv", head = TRUE, sep = '\t')

str(cc)

summary(cc)

# start to change the types of variables
to_factor = c('default_next_month', 'sex', 'education', 'marriage',
              'pay_sept', 'pay_aug', 'pay_july', 'pay_june', 'pay_may', 'pay_april')
cc[to_factor] = lapply(cc[to_factor], function(x) as.factor(x))

# explore the relationship between features and the target variable "default_next_month"
# create a funtion that can draw bar plot
bar_plot = function(data, x, facet, xlabel) {
  is_defaulter = facet
  data %>% ggplot(mapping = aes(x = x, fill = is_defaulter)) +
    geom_bar() +
    xlab(xlabel) 
}

# age and target variable
plot_age = bar_plot(cc, cc$age, cc$default_next_month, 'age')

# sex and target variable
plot_sex = bar_plot(cc, cc$sex, cc$default_next_month, 'sex')

#education and target variable
plot_education = bar_plot(cc, cc$education, cc$default_next_month, 'education')

# marriage and target variable
plot_marriage = bar_plot(cc, cc$marriage, cc$default_next_month, 'marriage')

# place multiple plots in a single page
library(gridExtra)
grid.arrange(plot_age, plot_sex, plot_education, plot_marriage)

# payment status and target variable
plot_pay_sept = bar_plot(cc, cc$pay_sept, cc$default_next_month, 'pay_sept')
plot_pay_aug = bar_plot(cc, cc$pay_aug, cc$default_next_month, 'pay_aug')
plot_pay_july = bar_plot(cc, cc$pay_july, cc$default_next_month, 'pay_july')
plot_pay_june = bar_plot(cc, cc$pay_june, cc$default_next_month, 'pay_june')
plot_pay_may = bar_plot(cc, cc$pay_may, cc$default_next_month, 'pay_may')
plot_pay_april = bar_plot(cc, cc$pay_april, cc$default_next_month, 'pay_april')
grid.arrange(plot_pay_sept, plot_pay_aug, plot_pay_july, plot_pay_june, plot_pay_may, plot_pay_april)

# household credit limit and target variable
is_defaulter = cc$default_next_month
cc %>% ggplot(mapping = aes(x = limit_bal, color = is_defaulter)) +
  geom_density()


# use ggpair to draw the scatterplot matrix for numerical features
to_plot = select(cc, 2, 13:24)
ggpairs(to_plot, 
        lower = list(continuous = wrap('smooth', colour = 'blue')))


# feature engineering
# based on limit_bal create a new feature is_low_credit
cc$is_low_credit = ifelse(cc$limit_bal <125000, 1, 0)
cc$is_low_credit = factor(cc$is_low_credit)
plot_credit = bar_plot(cc, cc$is_low_credit, cc$default_next_month, 'is_low_credit') # bar plot for new feature 

# based on age create a new feature CatAge
cc$CatAge = cut(cc$age, 
                breaks = c(20, 30, 40, 50, 60, 70, 80))
plot_CatAge = cc %>% ggplot(mapping = aes(x = CatAge, fill = is_defaulter))+
  geom_bar()

grid.arrange(plot_credit, plot_CatAge, nrow = 1)


# preprocess the data
#binarize the categorical features: sex, education, marriage, is_low_credit, CatAge
sex = model.matrix(~ sex -1, data = cc)
education = model.matrix(~ education -1, data = cc)
marriage = model.matrix(~ marriage -1, data = cc)
is_low_credit = model.matrix(~ is_low_credit -1, data = cc)
CatAge = model.matrix(~ CatAge -1, data = cc)
pay_sept = model.matrix( ~ pay_sept -1, data = cc)
pay_aug = model.matrix( ~ pay_aug -1, data = cc)
pay_june = model.matrix( ~ pay_june -1, data = cc)
pay_july = model.matrix( ~ pay_july -1, data = cc)
pay_may = model.matrix( ~ pay_may -1, data = cc)
pay_april = model.matrix( ~ pay_april -1, data = cc)

cc = cbind(cc, sex, education, marriage, 
           is_low_credit, CatAge,
           pay_sept, pay_aug, pay_june, pay_july, pay_may, pay_april)
cc = select(cc, -sex, -education, 
            -marriage, -is_low_credit, -limit_bal, -age, -CatAge,
            -pay_sept, -pay_aug, -pay_june, -pay_july, -pay_may, -pay_april)

# splitting data
in_train = createDataPartition( y = cc$default_next_month, p = 0.8, list = FALSE)
cc_train = cc[in_train, ]
cc_test = cc[-in_train, ]

str(cc_train)
str(cc_test)

# centering and scaling training data for analysis
Preprocess_steps = preProcess(select(cc_train, 2:13), method = c('center', 'scale'))

cc_train_proc = predict(Preprocess_steps, newdata = cc_train)
cc_test_proc = predict(Preprocess_steps, newdata = cc_test)
head(cc_train_proc)
head(cc_test_proc)

# checking for zero-variance features in training data, then remove them
nzv_train <- nearZeroVar(cc_train_proc, saveMetrics = TRUE) 
nzv_train

is_nzv_train = row.names(nzv_train[nzv_train$nzv == TRUE, ])
is_nzv_train

cc_train_proc = cc_train_proc[ , !(colnames(cc_train_proc) %in% is_nzv_train)]
str(cc_train_proc)

# identifying correlated predictors in training set, then remove them
cor_train = cor(cc_train_proc[, 2:13])
highly_cor = findCorrelation(cor_train, cutoff = .75)
cc_train_proc = cc_train_proc[ , -highly_cor]

# classification with logistic regression
# build logistic model
logistic_model = train(default_next_month ~ ., 
                       data = cc_train_proc, 
                       method ='glm',
                       family = binomial,
                       tuneLength = 20,
                       trControl = trainControl(method = 'cv', number = 10))

summary(logistic_model)
logistic_model$finalModel
# plot the variable importance plot
plot(varImp(logistic_model), main = 'varImp plot for logistic_model')

#test_data = head(cc_test_proc, 1)
#predict(logistic_model, newdata = test_data)

# test predictions for logistic model
logistic_predictions = predict(logistic_model, 
                               newdata = cc_test_proc)
confusionMatrix(logistic_predictions, 
                cc_test_proc$default_next_month)


# build Boosted Logistic regression model
boosted_model = train(default_next_month ~ .,
                    data = cc_train_proc,
                    method = 'LogitBoost',
                    family = binomial,
                    tuneGrid = expand.grid(nIter = 1:20),
                    trControl = trainControl(method = 'cv', number = 10))

summary(boosted_model)
boosted_model$finalModel
# plot the variable importance plot
plot(varImp(boosted_model), main = 'varImp plot for boosted_model')

# test predictions for boosted model
boosted_predictions = predict(boosted_model, 
                            newdata = cc_test_proc)
confusionMatrix(boosted_predictions, 
                cc_test_proc$default_next_month)

# build lasso model
trainX =select(cc_train_proc, -default_next_month)
trainY = cc_train_proc$default_next_month

lasso_model = train(trainX,
                    trainY,
                    method ='glmnet',
                    tuneGrid = expand.grid(
                         alpha = seq(.05, 1, length = 20),
                         lambda = c((1:5)/10)),
                    trControl = trainControl(method = 'cv', number = 10))

summary(lasso_model)
lasso_model$finalModel
# plot the variable importance plot
plot(varImp(lasso_model), main = 'varImp plot for lasso_model')


# test predictions for lasso model
lasso_predictions = predict(lasso_model, 
                               newdata = cc_test_proc)
confusionMatrix(lasso_predictions, 
                cc_test_proc$default_next_month)


# classification with decision tree
# prepare the data
set.seed(100)

rm(list = ls())  # empty the environment

cc = read.delim("credit_card_default.tsv", head = TRUE, sep = '\t')

str(cc)

summary(cc)

# start to change the types of variables
to_factor = c('default_next_month', 'sex', 'education', 'marriage',
              'pay_sept', 'pay_aug', 'pay_july', 'pay_june', 'pay_may', 'pay_april')
cc[to_factor] = lapply(cc[to_factor], function(x) as.factor(x))

# splitting data
in_train = createDataPartition( y = cc$default_next_month, p = 0.8, list = FALSE)
cc_train = cc[in_train, ]
cc_test = cc[-in_train, ]

str(cc_train)
str(cc_test)


# checking for zero-variance features in training data, then remove them
nzv_train <- nearZeroVar(cc_train, saveMetrics = TRUE) 
nzv_train

is_nzv_train = row.names(nzv_train[nzv_train$nzv == TRUE, ])
is_nzv_train

cc_train = cc_train[ , !(colnames(cc_train) %in% is_nzv_train)]
str(cc_train)

# keep all the features in train set except for the target variable
train_set = select(cc_train, -default_next_month) 

# grow one tree
tree_model = train(y = cc_train$default_next_month,
                   x = train_set,
                   method = 'rpart',
                   tuneLength = 20,
                   trControl = trainControl(method = 'cv', number = 10))

tree_model$finalModel

# plot the tree
library(rpart.plot)
rpart.plot(tree_model$finalModel)

# plot the variable importance plot
plot(varImp(tree_model), main = 'varImp plot for tree_model')

# test predictions for tree model
tree_predictions = predict(tree_model, newdata = cc_test)
confusionMatrix(tree_predictions, cc_test$default_next_month)


# build bagged CART
bagged_model = train(y = cc_train$default_next_month,
                     x = train_set,
                     method = 'treebag',
                     tuneLength = 20,
                     trControl = trainControl(method = 'cv', number = 10))

bagged_model$finalModel

# plot the variable importance plot
plot(varImp(bagged_model), main = 'varImp plot for bagged model')

# test predictions for bagged CART model
bagged_predictions = predict(bagged_model, newdata = cc_test)
confusionMatrix(bagged_predictions, cc_test$default_next_month)


#compare the models
new_results = resamples(list(#logistic_model = logistic_model,
                             #boosted_model = boosted_model,
                             #lasso_model = lasso_model,
                             tree_model = tree_model,
                             #bagged_model = bagged_model,
                             boostedTree_model = boostedTree_model))
                        
dotplot(new_results)

summary(new_results)

# build boostedTree_model
boostedTree_model = train(y = cc_train$default_next_month,
                     x = train_set,
                     method = 'blackboost')
                     #tuneLength = 20,
                     #trControl = trainControl(method = 'cv', number = 10))

boostedTree_model$finalModel

# plot the variable importance plot
plot(varImp(boostedTree_model), main = 'varImp plot for boostedTree model')

# test predictions for boosted tree model
boostedTree_predictions = predict(boostedTree_model, newdata = cc_test)
confusionMatrix(boostedTree_predictions, cc_test$default_next_month)

