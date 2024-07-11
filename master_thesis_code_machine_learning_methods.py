"""
This is the second part of the code for this analysis.
After the R-code file "master_thesis_code_preparation_logistic_regression.R", this one continues the analysis.

The path is automatically set to the location of the code file
Otherwise the working directory can be changed using "os.chdir(<path to the folder in which the  code is located>),
there must be an "/" at the end of the path
"""

# set working directory

import os
# check working directory
print(os.getcwd())

# get current path to the code file
path_abs = os.path.abspath(__file__)
directory_name = os.path.dirname(path_abs)

# directory can be set here, there must be an "/" at the end of the path
os.chdir(directory_name)

# check working directory
print(os.getcwd())



# define modes
# if mode_run == "hyperopt", hyperoptimization will run
# if mode_run == "optimized", hyperoptimization will not run and models will be loaded
mode_run = "optimized"
print("chosen mode: ", mode_run)

# if mode_figures == "not_keep", figures will be automatically closed after the code ran
# about 60 figures are printed througout the script, so changing mode_figures leads to having to close a lot of tabs
mode_figures = "no_keep"





# set up
# import packages
import time
import pickle
import json
from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc

from scikeras.wrappers import KerasClassifier
#from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt


# packages used for exploration
#import re
#import copy
#from sklearn.tree import plot_tree
#import shap
#from merf.merf import MERF
#from merf.viz import plot_merf_training_stats
#from catboost import CatBoostClassifier
#from sklearn.model_selection import train_test_split
#import catboost
#from catboost import CatBoostClassifier, Pool
#from sklearn.metrics import make_scorer
#from sklearn.feature_selection import SelectKBest



# load in data
data = pd.read_csv("data/data_model.csv")
train = pd.read_csv("data/train_model.csv")
train_downsampled = pd.read_csv("data/train_model_downsampled.csv")
train_upsampled = pd.read_csv("data/train_model_upsampled.csv")
train_rose = pd.read_csv("data/train_model_rose.csv")
valid = pd.read_csv("data/valid_model.csv")
test = pd.read_csv("data/test_model.csv")

data = data.drop(["Unnamed: 0"], axis = 1)
train = train.drop(["Unnamed: 0"], axis = 1)
train_downsampled = train_downsampled.drop(["Unnamed: 0"], axis = 1)
train_upsampled = train_upsampled.drop(["Unnamed: 0"], axis = 1)
train_rose = train_rose.drop(["Unnamed: 0"], axis = 1)
valid = valid.drop(["Unnamed: 0"], axis = 1)
test = test.drop(["Unnamed: 0"], axis = 1)

data.shape, train.shape, train_downsampled.shape, train_upsampled.shape, train_rose.shape, valid.shape, test.shape



# last steps of preparation for machine learning models
# check datatypes
print(train.dtypes)

# get feature matrix and outcome
X_train = train.drop(["decision"], axis = 1)
X_train_ds = train_downsampled.drop(["decision"], axis = 1)
X_train_us = train_upsampled.drop(["decision"], axis = 1)
X_train_rose = train_rose.drop(["decision"], axis = 1)
X_valid = valid.drop(["decision"], axis = 1)
X_test = test.drop(["decision"], axis = 1)

y_train = train["decision"]
y_train_ds = train_downsampled["decision"]
y_train_us = train_upsampled["decision"]
y_train_rose = train_rose["decision"]
y_valid = valid["decision"]
y_test = test["decision"]

X_train.shape, X_train_ds.shape, X_train_us.shape, X_train_rose.shape, X_valid.shape, X_test.shape, y_train.shape, y_train_ds.shape, y_train_us.shape, y_train_rose.shape, y_valid.shape, y_test.shape


# keep variables relevant for the analysis
X_train_cat_boost = X_train.copy(deep = True)
drop_var = ["rank_GB", "rank_CHO", "rank_CHEESE", "rank_WHEAT", "rank_TOM", "country", "av_income", "Generation", "lfdn",
           "choice", "final_comment"]
X_train = X_train.drop(drop_var, axis = 1)
X_train_cat_boost = X_train_cat_boost.drop(drop_var, axis = 1)
#X_train_rose = X_train_rose.drop(drop_var, axis = 1)  # rose already does not have these variables
X_train_ds = X_train_ds.drop(drop_var, axis = 1)
X_train_us = X_train_us.drop(drop_var, axis = 1)
X_valid = X_valid.drop(drop_var, axis = 1)
X_test = X_test.drop(drop_var, axis = 1)

# categorical features
cat_features = ["education", "occupation", "purchase_CHO", "purchase_PA", "purchase_GB", "gender", "Country_Name", 
                "av_income_factorized", "product_type", "environment"]    # meow i guess
num_features = [feature for feature in X_train.columns if feature not in cat_features]

cat_features, num_features


# create dataframe for variables that are supposed to be numeric here
X_train_num = X_train[num_features]
X_train_rose_num = X_train_rose[num_features]
X_train_ds_num = X_train_ds[num_features]
X_train_us_num = X_train_us[num_features]
X_valid_num = X_valid[num_features]
X_test_num = X_test[num_features]

# check datatypes
print(X_train_num.dtypes)


# change the datatypes of the numeric variables to float64
X_train_num = X_train_num.astype("float64")
X_train_rose_num = X_train_rose_num.astype("float64")
X_train_ds_num = X_train_ds_num.astype("float64")
X_train_us_num = X_train_us_num.astype("float64")
X_valid_num = X_valid_num.astype("float64")
X_test_num = X_test_num.astype("float64")

# check all datatypes
print(X_train_num.dtypes)


# check numeric variables
for var in X_train_num.columns:
    print(var, ":")
    print(X_train_num[var].unique())
    print()


# one-hot encode factor columns
X_train_fc = X_train[cat_features]
X_train_rose_fc = X_train_rose[cat_features]
X_train_ds_fc = X_train_ds[cat_features]
X_train_us_fc = X_train_us[cat_features]
X_valid_fc = X_valid[cat_features]
X_test_fc = X_test[cat_features]

X_train_fc.shape, X_valid_fc.shape, X_test_fc.shape

# check datatypes of variables that are supposed to be factor-variables
X_train_fc.dtypes

# adjust variable types
X_train_fc = X_train_fc.astype("object")
X_train_rose_fc = X_train_rose_fc.astype("object")
X_train_ds_fc = X_train_ds_fc.astype("object")
X_train_us_fc = X_train_us_fc.astype("object")
X_valid_fc = X_valid_fc.astype("object")
X_test_fc = X_test_fc.astype("object")

X_train_fc.dtypes


# one-hot encoding

# set up encoder
oh_1 = OneHotEncoder(sparse_output = False, handle_unknown = "error", drop = "first")

# fit one-hot encoder on whole data to be sure to get all options
oh_1.fit(data[cat_features])

# transform train and test data according to fitted encoder
X_train_fc = oh_1.transform(X_train_fc)
X_train_rose_fc = oh_1.transform(X_train_rose_fc)
X_train_ds_fc = oh_1.transform(X_train_ds_fc)
X_train_us_fc = oh_1.transform(X_train_us_fc)
X_valid_fc = oh_1.transform(X_valid_fc)
X_test_fc = oh_1.transform(X_test_fc)

X_train_fc.shape, X_valid_fc.shape, X_test_fc.shape


# name variables
cat_features_names = [f"{col}_{cat}" for i, col in enumerate(cat_features) for cat in oh_1.categories_[i][1:]]
cat_features_names[:10]


# add variable names to dataframe of categorical features
X_train_fc = pd.DataFrame(X_train_fc, columns = cat_features_names)
X_train_rose_fc = pd.DataFrame(X_train_rose_fc, columns = cat_features_names)
X_train_ds_fc = pd.DataFrame(X_train_ds_fc, columns = cat_features_names)
X_train_us_fc = pd.DataFrame(X_train_us_fc, columns = cat_features_names)
X_valid_fc = pd.DataFrame(X_valid_fc, columns = cat_features_names)
X_test_fc = pd.DataFrame(X_test_fc, columns = cat_features_names)

X_train_fc.shape, X_valid_fc.shape, X_test_fc.shape


# create dataframe for numeric variables
X_num_feature_names = X_train_num.columns

X_train_num = pd.DataFrame(X_train_num, columns = X_num_feature_names)
X_train_rose_num = pd.DataFrame(X_train_rose_num, columns = X_num_feature_names)
X_train_ds_num = pd.DataFrame(X_train_ds_num, columns = X_num_feature_names)
X_train_us_num = pd.DataFrame(X_train_us_num, columns = X_num_feature_names)
X_valid_num = pd.DataFrame(X_valid_num, columns = X_num_feature_names)
X_test_num = pd.DataFrame(X_test_num, columns = X_num_feature_names)

X_train_num.info()


# concat numerical and categorical dataframes
print("Expected number of columns: ", len(X_train_num.columns) + len(X_train_fc.columns))
X_train = pd.concat([X_train_num, X_train_fc], axis = 1).reindex(X_train_num.index)
X_train_rose = pd.concat([X_train_rose_num, X_train_rose_fc], axis = 1).reindex(X_train_rose_num.index)
X_train_ds = pd.concat([X_train_ds_num, X_train_ds_fc], axis = 1).reindex(X_train_ds_num.index)
X_train_us = pd.concat([X_train_us_num, X_train_us_fc], axis = 1).reindex(X_train_us_num.index)
X_valid = pd.concat([X_valid_num, X_valid_fc], axis = 1).reindex(X_valid_num.index)
X_test = pd.concat([X_test_num, X_test_fc], axis = 1).reindex(X_test_num.index)
print("Number of columns: ", len(X_train.columns))


# check outcome variable
print(y_train.value_counts())
print(y_train_rose.value_counts())
print(y_train_ds.value_counts())
print(y_train_us.value_counts())
print(y_valid.value_counts())
print(y_test.value_counts())



# feature importances
# mutual information
# determine feature importance using mutual information (works well for categorical variables as well)


info_gain = mutual_info_classif(X_train, y_train, discrete_features = True, random_state = 100)  
# always setting random_state (seed) for reproducibility

features = X_train

feature_scores = {}
for i in range(len(features.columns)):
    feature_scores[features.columns[i]] = info_gain[i]


# plot feature importances

sorted_features = sorted(feature_scores.items(), key = lambda x: x[1], reverse = False)

#for feature, score in sorted_features:
#    print("Feature: ", feature, "Score: ", score)

fig, ax = plt.subplots(figsize = (10, 30))
y_pos = np.arange(len(sorted_features))
ax.barh(y_pos, [score for feature, score in sorted_features], align = "center")
ax.set_yticks(y_pos)
ax.set_yticklabels([feature for feature, score in sorted_features])
ax.set_xlabel("Feature Importance")

for i, v in enumerate([score for feature, score in sorted_features]):
    ax.text(v + 0.01, i, str(round(v, 2)), color = "black", fontweight = "bold")
plt.show(block = False)  # dont stop execution of the code
#plt.pause(0.001)


# create diagram for the paper
sorted_features_comp = sorted_features[-10:]

fig, ax = plt.subplots(figsize = (10, 5))
y_pos = np.arange(len(sorted_features_comp))
ax.barh(y_pos, [score for feature, score in sorted_features_comp], align = "center")
ax.set_yticks(y_pos)
ax.set_yticklabels([feature for feature, score in sorted_features_comp])
ax.set_xlabel("Feature Importance")

for i, v in enumerate([score for feature, score in sorted_features_comp]):
    ax.text(v + 0.001, i, str(round(v, 2)), color = "black", fontweight = "bold")
plt.savefig("figures\\feature_selection.png")



# following code needed if data is supposed to be subset (feature selection)
# # subset data for most important features for faster computations

# best_features = SelectKBest(mutual_info_classif, k = 20)
# best_features.fit(X_train, y_train)
# best_feature_names = X_train.columns[best_features.get_support()]

# X_train = X_train[best_feature_names]
# X_train_rose = X_train_rose[best_feature_names]
# X_train_ds = X_train_ds[best_feature_names]
# X_train_us = X_train_us[best_feature_names]
# X_valid = X_valid[best_feature_names]
# X_test = X_test[best_feature_names]



# models

# logistic regression hyperparameter optimization
# using GridSearchCV for optimal resuts

# define parameters for grid search hyperparameter optimization
solvers = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
penalties = ["l1", "l2", "elasticnet", None]
c_values = [1.0, 0.1, 0.01, 0.001, 0.0001]
max_iter = [10, 50, 100, 200]
class_weights = ["balanced"]
class_weights.append(None)
l1_ratios = np.linspace(0, 1, 9).tolist()
l1_ratios.append(None)

# define grid
clf_random_grid = {"solver": solvers,
                   "penalty": penalties,
                   "C": c_values,
                   "max_iter": max_iter,
                   "class_weight": class_weights,
                   "l1_ratio": l1_ratios}

# check grid
print(clf_random_grid)


# create function that does hyperparameter-optimization, gets metrics, and saves model parameters
def log_reg_hyper_opt(X, y, grid, cv = 5, verbose = 2, scoring = "f1", iterations = 100, random_state = None,
                     file_name = "clf_temp", mode = None, n_jobs = -1):
    """function to automatically perform hyperparameter optimization
    and saving the best-performing model
    X: (X_train) features
    y: (y_train) outcome
    grid: parameter-grid with parameters for sklearns LogisticRegression-function to be tested
    cv: how many folds for cross-validation
    scoring: which metric to optimize for (options can be found in scoring parameter of sklearns RandomizedSearchCV-function)
    iterations: how many possible hyperparamter-combinations should be tested
    random_state: whether results should be replicable, possible values: integers
    verbose: how much information should be printed
    file_name: name of the file that saves the best model, does not require filetype
    mode: define method for hyperparameter-optimization; "all": GridSearchCV(), otherwise: RandomizedSearchCV()
    n_jobs: how many cpu-cores should be used for the computation; -1 to use all available cores
    
    returns best-performing model and value for the runtime of the hyperparameter optimization"""
    
    # initialize logistic regression and grid search
    clf_optim = LogisticRegression(random_state = random_state)
    if mode == "all":
        clf_gs = GridSearchCV(estimator = clf_optim, param_grid = grid, cv = cv, verbose = verbose, n_jobs = n_jobs, 
                              scoring = str(scoring), return_train_score = False)     
        print("mode: all")
    else:
        clf_gs = RandomizedSearchCV(estimator = clf_optim, param_distributions = grid, cv = cv, verbose = verbose, 
                                    n_jobs = n_jobs, scoring = str(scoring), n_iter = iterations, random_state = random_state,
                                    return_train_score = False)
        print("mode: random")
    
    
    # grid search
    # capture runtime
    start_time = time.time()
    clf_gs.fit(X, y)
    end_time = time.time()
    runtime = end_time - start_time
    print()  # this is just to make the printed text be more structured
    print("Run time logistic regression hyperparameter optimization: \n", np.round(runtime, 4) , "seconds")
    print("")
    print("Number of iterations: ", iterations, "; Number of observations: ", len(X))
    
    print()
    print("Optimal parameters:")
    print(clf_gs.best_params_)
    
    # define best model
    clf_best_model = clf_gs.best_estimator_
    
    # save best estimator
    # create folder for model if it does not exist already
    path = os.getcwd() + "\\models"
    path_model = path + "\\" + file_name + ".sav"
    Path(path).mkdir(parents = True, exist_ok = True)
    pickle.dump(clf_best_model, open(path_model, "wb"))
    print()
    print("Model parameters saved at: ", path_model)
    
    # return model and metrics
    return clf_best_model, runtime


# run hyperparameter optimization
# there will be warnings by construction as some of the combinations of hyperparameters do not work (e.g. some of the solvers
# do not work with some of the penalty-terms), these cases are skipped and the optimization continues, warnings are still
# printed
if mode_run == "optimized":
    clf_best_model = pickle.load(open("models\\clf_best_model.sav", "rb"))
else:
    clf_best_model, runtime_clf_hyper_opt = log_reg_hyper_opt(X_train, y_train, grid = clf_random_grid, cv = 5, verbose = 1, 
                                                              scoring = "f1", iterations = 1000, random_state = 100, 
                                                              file_name = "clf_best_model", mode = "all", n_jobs = -1)

# ROSE
if mode_run == "optimized":
    clf_rose_best_model = pickle.load(open("models\\clf_rose_best_model.sav", "rb"))
else:
    clf_rose_best_model, runtime_clf_rose_hyper_opt = log_reg_hyper_opt(X_train_rose, y_train_rose, grid = clf_random_grid, cv = 5,
                                                                        verbose = 1, scoring = "f1", iterations = 1000,
                                                                        random_state = 100, file_name = "clf_rose_best_model",
                                                                        mode = "all", n_jobs = -1)

# downsampled
if mode_run == "optimized":
    clf_ds_best_model = pickle.load(open("models\\clf_ds_best_model.sav", "rb"))
else:
    clf_ds_best_model, runtime_clf_ds_hyper_opt = log_reg_hyper_opt(X_train_ds, y_train_ds, grid = clf_random_grid, cv = 5,
                                                                    verbose = 1, scoring = "f1", iterations = 1000,
                                                                    random_state = 100, file_name = "clf_ds_best_model",
                                                                    mode = "all", n_jobs = -1)

# upsampled
if mode_run == "optimized":
    clf_us_best_model = pickle.load(open("models\\clf_us_best_model.sav", "rb"))
else:
    clf_us_best_model, runtime_clf_us_hyper_opt = log_reg_hyper_opt(X_train_us, y_train_us, grid = clf_random_grid, cv = 5,
                                                                    verbose = 1, scoring = "f1", iterations = 1000,
                                                                    random_state = 100, file_name = "clf_us_best_model",
                                                                    mode = "all", n_jobs = -1)


# create function that takes the test set and the model and outputs predictions and metrics
def log_reg_predictions(X, model):
    """function to get predictions and metrics from a trained model
    X: (X_test) test data
    model: trained model
    
    returns predictions and the runtime for the prediction of all observations in X"""
    
    # capture runtime
    start_time = time.time()
    predictions = model.predict(X)
    end_time = time.time()
    runtime = end_time - start_time
    print("Run time logistic regression predictions on the test set: \n", np.round(runtime, 4), " seconds")
    
    return predictions, runtime


# create function that takes the test set and the model and outputs predictions of probabilties of belonging to class "1"
def log_reg_predictions_proba(X, model):
    """function to get predictions of probabilities from a trained model
    X: (X_test) test data
    model: trained model
    
    returns predicted probabilities
    """
    
    predictions_probability = model.predict_proba(X)[:,1]
    
    return predictions_probability


# get predictions
clf_predictions, runtime_clf_predictions = log_reg_predictions(X_test, model = clf_best_model)
clf_predictions_proba = log_reg_predictions_proba(X_test, model = clf_best_model)

# ROSE, downsampled, and upsampled
clf_rose_predictions, runtime_clf_rose_predictions = log_reg_predictions(X_test, model = clf_rose_best_model)
clf_ds_predictions, runtime_clf_ds_predictions = log_reg_predictions(X_test, model = clf_ds_best_model)
clf_us_predictions, runtime_clf_us_predictions = log_reg_predictions(X_test, model = clf_us_best_model)

clf_rose_predictions_proba = log_reg_predictions_proba(X_test, model = clf_rose_best_model)
clf_ds_predictions_proba = log_reg_predictions_proba(X_test, model = clf_ds_best_model)
clf_us_predictions_proba = log_reg_predictions_proba(X_test, model = clf_us_best_model)


# write a function to evaluate the predictions

def evaluate_predictions(predictions, truth, predictions_proba):
    """function to get metrics regarding predictions and true values
    predictions: values fetched for the prediction of a trained model using test data
    truth: (y_test) true values for the test data
    predictions_proba: values fetched for the prediction of probabilites of trained model using test data"""
    
    # metrics
    accuracy = round(accuracy_score(truth, predictions) * 100, 2)
    balanced_accuracy = round(balanced_accuracy_score(truth, predictions) * 100, 2)
    f1 = round(f1_score(truth, predictions) * 100, 2)
    roc_auc = round(roc_auc_score(truth, predictions_proba) * 100, 2)
    pre_rec_auc = round(average_precision_score(truth, predictions_proba) * 100, 2)
    print("Metrics:")
    print("Accuracy-score: ", accuracy, "%")
    print("Balanced Accuracy-score: ", balanced_accuracy, "%")
    print("F1-score: ", f1, "%")
    print("ROC AUC-score: ", roc_auc, "%")
    print("Average precision score (Precision-Recall-curve AUC): ", pre_rec_auc, "%")
    
    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "roc_auc": roc_auc,
        "pre_rec_auc": pre_rec_auc
    }
    
    return metrics



def export_metrics_to_txt(metrics, file_name):
    """function that takes dictionary of metrics and exports them as .txt
    metrics: dictionary with metrics
    file_name: name of output file"""
    
    # define path to save the .txts
    path = os.getcwd() + "\\outputs\\metrics"
    path_txt = path + "\\" + file_name + ".txt"
    Path(path).mkdir(parents = True, exist_ok = True)
    
    with open(path_txt, "w") as f:
        f.write(json.dumps(metrics))
        
    print(f"The file named {file_name} was saved in {path_txt}.")


# evaluate predictions
metrics_clf = evaluate_predictions(clf_predictions, y_test, clf_predictions_proba)
metrics_clf_rose = evaluate_predictions(clf_rose_predictions, y_test, clf_rose_predictions_proba)
metrics_clf_ds = evaluate_predictions(clf_ds_predictions, y_test, clf_ds_predictions_proba)
metrics_clf_us = evaluate_predictions(clf_us_predictions, y_test, clf_us_predictions_proba)

# check saved metrics
print(metrics_clf)
print(metrics_clf_rose)
print(metrics_clf_ds)
print(metrics_clf_us)

# export metrics
export_metrics_to_txt(metrics_clf, "logistic_regression_metrics")
export_metrics_to_txt(metrics_clf_rose, "logistic_regression_rose_metrics")
export_metrics_to_txt(metrics_clf_ds, "logistic_regression_downsampled_metrics")
export_metrics_to_txt(metrics_clf_us, "logistic_regression_upsampled_metrics")


# function to create and save confusion matrices
def create_confusion_to_eliminate_confusion(predictions, truth, file_name = None, title = None):
    """plots and save a confusion matrix for the given data
    predictions: values fetched for the prediction of a trained model using test data
    truth: (y_test) true values for the test data
    file_name: name of the file that saves the confusion plot, does not require filetype
    title: title for the confusion matrix, if none is chose, the matrix will not have a title"""
    
    # file_name must be a string
    if file_name != None:
        if type(file_name) != str:
            print("file_name must have the type str")
            return
        else:
            pass 
        
        
    # define path to save the plot
    path = os.getcwd() + "\\figures\\confusion_matrices"
    path_conf_mat = path + "\\" + file_name + ".png"
    Path(path).mkdir(parents = True, exist_ok = True)
    
    # calculate confusion matrix
    conf_mat = confusion_matrix(truth, predictions)
    
    # plot confusion matrx
    categories = ["CN-label", "No CN-label"]
    plt.figure(figsize = plt.rcParams.get("figsize.figsize"))
    sns.heatmap(conf_mat, annot = True, fmt = "g", cmap = "Blues", xticklabels = categories,
               yticklabels = categories)
    plt.title(title, fontsize = 16)
    plt.xlabel("Prediction", fontsize = 14)
    plt.ylabel("Truth", fontsize = 14)
    plt.savefig(path_conf_mat)
    plt.show(block = False)  # dont stop execution of the code
    #plt.pause(0.001)
    print("Confusion matrix saved at: ", path_conf_mat)



# create and save confusion matrix
create_confusion_to_eliminate_confusion(clf_predictions, y_test, file_name = "clf_conf_mat")

# ROSE
create_confusion_to_eliminate_confusion(clf_rose_predictions, y_test, file_name = "clf_rose_conf_mat")

# downsampled
create_confusion_to_eliminate_confusion(clf_ds_predictions, y_test, file_name = "clf_ds_conf_mat")

# upsampled
create_confusion_to_eliminate_confusion(clf_us_predictions, y_test, file_name = "clf_us_conf_mat")


# function to create and save roc and precision-recall curves
def roc_precision_recall_curves(predictions, truth, test_features, predictions_proba, file_name_roc, file_name_precision_recall,
                               title_roc = None, title_precision_recall = None, model_name = "Model"):
    """creates roc- and precision-recall-plots and save them
    predictions: values fetched for the prediction of a trained model using test data
    truth: (y_test) true values for the test data
    test_features: (X_test) features-values used to get the predictions
    file_name_roc: name of the file that saves the roc-plot, does not require filetype
    file_name_precision_recall: name of the file that saves the precision-recall-plot, does not require filetype
    title_roc: title for the roc-plot, if none is chose, the matrix will not have a title
    title_precision_recall: title for the precision-recall-plot, if none is chose, the matrix will not have a title
    model_name: Name of the model used in printing of the results"""
        
    # file names must be strings
    if file_name_roc != None or file_name_precision_recall != None:
        if type(file_name_roc) != str or type(file_name_precision_recall) != str:
            print("file names must have the type str")
            return
        else:
            pass 
        
    # title names must be strings
    if title_roc != None or title_precision_recall != None:
        if type(title_roc) != str or type(title_precision_recall) != str:
            print("title names must have the type str")
            return
        else:
            pass
    
    # model_name must be a string
    if type(model_name) != str:
        print("model_name must have type str")
        return
    else:
        pass
    
    
    # define paths to save the plots
    path_roc = os.getcwd() + "\\figures\\roc_plots"
    path_roc_plot = path_roc + "\\" + file_name_roc + ".png"
    Path(path_roc).mkdir(parents = True, exist_ok = True)
    
    path_pre_rec = os.getcwd() + "\\figures\\precision_recall_plots"
    path_pre_rec_plot = path_pre_rec + "\\" + file_name_precision_recall +  ".png"
    Path(path_pre_rec).mkdir(parents = True, exist_ok = True)
    
    
    # create ROC-plot
    # values for curve that indicates naive guessing (based on the distribution of the outcome-variable)
    rg_probs = [0 for _ in range(len(test_features))]
    
    # scores
    rg_auc = roc_auc_score(truth, rg_probs)
    auc_roc = roc_auc_score(truth, predictions_proba)
    
    # print scores
    print("Naive guessing: ROC AUC = %.3f" % (rg_auc))
    print(f"{model_name} with parameter shrinkage: ROC AUC = %.3f" % (auc_roc))
    
    # plot ROC-curve
    # get false positive and true positive rates for ROC-curve
    fpr, tpr, _ = roc_curve(truth, predictions_proba)
    # for naive guessing
    rg_fpr, rg_tpr, _ = roc_curve(truth, rg_probs)
    
    # plot
    plt.figure()
    plt.plot(rg_fpr, rg_tpr, linestyle = "--", label = "Naive guessing")
    plt.plot(fpr, tpr, marker = ".", label = model_name)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title_roc)
    plt.legend()
    plt.savefig(path_roc_plot)
    plt.show(block = False)  # dont stop execution of the code
    #plt.pause(0.001)

    
    # create precision-recall-plot
    # precision-recall
    precision, recall, threshold = precision_recall_curve(truth, predictions_proba)
    
    # f1 (not racing-cars though)
    f1, auc_precision_recall = f1_score(truth, predictions), auc(recall, precision)
    
    # print scores
    print()
    print(f"{model_name}: f1 = %.3f; AUC Precision-Recall = %.3f" % (f1, auc_precision_recall))
    
    # plot precision-recall-curve
    random_guesses_precision_recall = len(truth[truth == 1]) / len(truth)
    plt.figure()
    plt.plot([0, 1], [random_guesses_precision_recall, random_guesses_precision_recall], linestyle = "--", 
             label = "Naive guesses")
    plt.plot(recall, precision, marker = ".", label = model_name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title_precision_recall)
    plt.legend()
    plt.savefig(path_pre_rec_plot)
    plt.show(block = False)  # dont stop execution of the code
    #plt.pause(0.001)
    
    return auc_roc, auc_precision_recall



# create and save roc- and precision-recall-plots
auc_roc_clf, auc_precision_recall_clf = roc_precision_recall_curves(clf_predictions, y_test, X_test, clf_predictions_proba, 
                                                                    file_name_roc = "clf_roc", 
                                                                    file_name_precision_recall = "clf_precision_recall", 
                                                                    model_name = "Logistic regression with parameter shrinkage")

# ROSE
auc_roc_clf_rose, auc_precision_recall_clf_rose = roc_precision_recall_curves(clf_rose_predictions, y_test, X_test, 
                                                                              clf_rose_predictions_proba, 
                                                                              file_name_roc = "clf_rose_roc", 
                                                                              file_name_precision_recall = "clf_rose_precision_recall",
                                                                              model_name = "Logistic regression with parameter shrinkage")

# downsampled
auc_roc_clf_ds, auc_precision_recall_clf_ds = roc_precision_recall_curves(clf_ds_predictions, y_test, X_test, 
                                                                          clf_ds_predictions_proba, 
                                                                          file_name_roc = "clf_ds_roc", 
                                                                          file_name_precision_recall = "clf_ds_precision_recall",
                                                                          model_name = "Logistic regression with parameter shrinkage")

# upsampeled
auc_roc_clf_us, auc_precision_recall_clf_us = roc_precision_recall_curves(clf_us_predictions, y_test, X_test, 
                                                                          clf_us_predictions_proba, 
                                                                          file_name_roc = "clf_us_roc", 
                                                                          file_name_precision_recall = "clf_us_precision_recall",
                                                                          model_name = "Logistic regression with parameter shrinkage")



# Random Forest

# grid-search parameter optimization

# number of trees
n_estimators = [100, 200, 500]
# splitting criterion
split_crit = ["gini", "entropy"]
# maximum tree depth
max_depth = [5, 10, 20]
# min sampels for split
min_samples_split = [10, 20, 50, 100]
# min nodes for leafs
min_samples_leaf = [5, 10, 20, 100]
## minimum fraction of obs in each leaf
#min_weight_fraction_leaf = [0.01, 0.02, 0.05]
# number of features for splits
max_features = ["sqrt", "log2", None]
# bootstrapping
bootstrap = [True, False]
# # maximum number of leaf nodes
# max_leaf_nodes = [20, 50, None]

# grid for random hyperparameter selection
rf_random_grid = {"n_estimators": n_estimators,
                  "criterion": split_crit,
                  "max_depth": max_depth,
                  "min_samples_split": min_samples_split,
                  "min_samples_leaf": min_samples_leaf,
                  "max_features": max_features,
                  "bootstrap": bootstrap}

print(rf_random_grid)


# create function that does hyperparameter-optimization, gets metrics, and saves model parameters
def random_forest_hyper_opt(X, y, grid, cv = 5, verbose = 2, scoring = "f1", iterations = 100, random_state = None,
                           file_name = "rf_temp", mode = None, n_jobs = -1):
    """function to automatically perform hyperparameter optimization and saving the best-performing model
    X: (X_train) features
    y: (y_train) outcome
    grid: parameter-grid with parameters for sklearns RandomForestClassifier-function to be tested
    cv: how many folds for cross-validation
    scoring: which metric to optimize for (options can be found in scoring parameter of sklearns RandomizedSearchCV-function)
    iterations: how many possible hyperparamter-combinations should be tested
    random_state: whether results should be replicable, possible values: integers
    verbose: how much information should be printed
    file_name: name of the file that saves the best model, does not require filetype
    mode: define method for hyperparameter-optimization; "all": GridSearchCV(), otherwise: RandomizedSearchCV()
    n_jobs: how many cpu-cores should be used for the computation; -1 to use all available cores
    
    returns best-performing model and value for the runtime of the hyperparameter optimization"""
    
    # initialize random forest and grid search
    rf_optim = RandomForestClassifier(random_state = random_state)
    if mode == "all":
        rf_gs = GridSearchCV(estimator = rf_optim, param_grid = grid, cv = cv, verbose = verbose, 
                             n_jobs = n_jobs, scoring = str(scoring), return_train_score = False)
        print("mode: all")
    else:
        rf_gs = RandomizedSearchCV(estimator = rf_optim, param_distributions = grid, cv = cv, verbose = verbose, 
                                   n_jobs = n_jobs, scoring = str(scoring), n_iter = iterations, random_state = random_state,
                                   return_train_score = False)
        print("mode: random")
        
    
    # grid search
    # capture runtime
    start_time = time.time()
    rf_gs.fit(X, y)
    end_time = time.time()
    runtime = end_time - start_time
    print()  # this is just to make the printed text be more structured
    print("Run time random forest hyperparameter optimization: \n", np.round(runtime, 4) , "seconds")
    print("Number of iterations: ", iterations, "; Number of observations: ", len(X))
    
    print()
    print("Optimal parameters:")
    print(rf_gs.best_params_)
    
    # define best model
    rf_best_model = rf_gs.best_estimator_
    
    # save best estimator
    # create folder for model if it does not exist already
    path = os.getcwd() + "\\models"
    path_model = path + "\\" + file_name + ".sav"
    Path(path).mkdir(parents = True, exist_ok = True)
    pickle.dump(rf_best_model, open(path_model, "wb"))
    print()
    print("Model parameters saved at: ", path_model)
    
    # return model and metrics
    return rf_best_model, runtime


# run hyperparameter optimization
if mode_run == "optimized":
    rf_best_model = pickle.load(open("models\\rf_best_model.sav", "rb"))
else:
    rf_best_model, runtime_rf_hyper_opt = random_forest_hyper_opt(X_train, y_train, grid = rf_random_grid, cv = 5, verbose = 2, 
                                                                scoring = "f1", iterations = 1000, random_state = 100, 
                                                                file_name = "rf_best_model", mode = "all", n_jobs = -1)

# ROSE
if mode_run == "optimized":
    rf_rose_best_model = pickle.load(open("models\\rf_rose_best_model.sav", "rb"))
else:
    rf_rose_best_model, runtime_rf_rose_hyper_opt = random_forest_hyper_opt(X_train_rose, y_train_rose, grid = rf_random_grid, 
                                                                            cv = 5, verbose = 2, scoring = "f1", iterations = 100, 
                                                                            random_state = 1000, file_name = "rf_rose_best_model", 
                                                                            mode = "all", n_jobs = -1)

# downsampled
if mode_run == "optimized":
    rf_ds_best_model = pickle.load(open("models\\rf_ds_best_model.sav", "rb"))
else:
    rf_ds_best_model, runtime_rf_ds_hyper_opt = random_forest_hyper_opt(X_train_ds, y_train_ds, grid = rf_random_grid, cv = 5, 
                                                                        verbose = 2, scoring = "f1", iterations = 100, 
                                                                        random_state = 1000, file_name = "rf_ds_best_model", 
                                                                        mode = "all", n_jobs = -1)

# upsampled
if mode_run == "optimized":
    rf_us_best_model = pickle.load(open("models\\rf_us_best_model.sav", "rb"))
else:
    rf_us_best_model, runtime_rf_us_hyper_opt = random_forest_hyper_opt(X_train_us, y_train_us, grid = rf_random_grid, cv = 5, 
                                                                        verbose = 2, scoring = "f1", iterations = 100, 
                                                                        random_state = 1000, file_name = "rf_us_best_model", 
                                                                        mode = "all", n_jobs = -1)


# create function that takes the test set and the model and outputs predictions and metrics
def random_forest_predictions(X, model):
    """function to get predictions and metrics from a trained model
    X: (X_test) test data
    model: trained model
    
    returns predictions and the runtime for the prediction of all observations in X"""
    
    # capture runtime
    start_time = time.time()
    predictions = model.predict(X)
    end_time = time.time()
    runtime = end_time - start_time
    print("Run time random forest predictions on the test set: \n", np.round(runtime, 4), " seconds")
    
    return predictions, runtime


# create function that takes the test set and the model and outputs predictions of probabilties of belonging to class "1"
def random_forest_predictions_proba(X, model):
    """function to get predictions of probabilities from a trained model
    X: (X_test) test data
    model: trained model
    
    returns predicted probabilities
    """
    
    predictions_probability = model.predict_proba(X)[:,1]
    
    return predictions_probability


# get predictions
rf_predictions, runtime_rf_predictions = random_forest_predictions(X_test, model = rf_best_model)
rf_predictions_proba = random_forest_predictions_proba(X_test, model = rf_best_model)

# ROSE, downsampled, upsampled
rf_rose_predictions, runtime_rf_rose_predictions = random_forest_predictions(X_test, model = rf_rose_best_model)
rf_ds_predictions, runtime_rf_ds_predictions = random_forest_predictions(X_test, model = rf_ds_best_model)
rf_us_predictions, runtime_rf_us_predictions = random_forest_predictions(X_test, model = rf_us_best_model)

rf_rose_predictions_proba = random_forest_predictions_proba(X_test, model = rf_rose_best_model)
rf_ds_predictions_proba = random_forest_predictions_proba(X_test, model = rf_ds_best_model)
rf_us_predictions_proba = random_forest_predictions_proba(X_test, model = rf_us_best_model)


# evaluate predictions
metrics_rf = evaluate_predictions(rf_predictions, y_test, rf_predictions_proba)
metrics_rf_rose = evaluate_predictions(rf_rose_predictions, y_test, rf_rose_predictions_proba)
metrics_rf_ds = evaluate_predictions(rf_ds_predictions, y_test, rf_ds_predictions_proba)
metrics_rf_us = evaluate_predictions(rf_us_predictions, y_test, rf_us_predictions_proba)

# check saved metrics
print(metrics_rf)
print(metrics_rf_rose)
print(metrics_rf_ds)
print(metrics_rf_us)

# export metrics
export_metrics_to_txt(metrics_rf, "random_forest_metrics")
export_metrics_to_txt(metrics_rf_rose, "random_forest_rose_metrics")
export_metrics_to_txt(metrics_rf_ds, "random_forest_downsampled_metrics")
export_metrics_to_txt(metrics_rf_us, "random_forest_upsampled_metrics")


# create and save confusion matrix
create_confusion_to_eliminate_confusion(rf_predictions, y_test, file_name = "rf_conf_mat")

# ROSE
create_confusion_to_eliminate_confusion(rf_rose_predictions, y_test, file_name = "rf_rose_conf_mat")

# downsampled
create_confusion_to_eliminate_confusion(rf_ds_predictions, y_test, file_name = "rf_ds_conf_mat")

# upsampled
create_confusion_to_eliminate_confusion(rf_us_predictions, y_test, file_name = "rf_us_conf_mat")


# create and save roc- and precision-recall-plots
auc_roc_rf, auc_precision_recall_rf = roc_precision_recall_curves(rf_predictions, y_test, X_test, rf_predictions_proba, 
                                                                    file_name_roc = "rf_roc", 
                                                                    file_name_precision_recall = "rf_precision_recall", 
                                                                    model_name = "random forest")

# ROSE
auc_roc_rf_rose, auc_precision_recall_rf_rose = roc_precision_recall_curves(rf_rose_predictions, y_test, X_test, 
                                                                            rf_rose_predictions_proba, 
                                                                            file_name_roc = "rf_rose_roc", 
                                                                            file_name_precision_recall = "rf_rose_precision_recall",
                                                                            model_name = "random forest")

# downsampled
auc_roc_rf_ds, auc_precision_recall_rf_ds = roc_precision_recall_curves(rf_ds_predictions, y_test, X_test, 
                                                                        rf_ds_predictions_proba, 
                                                                        file_name_roc = "rf_ds_roc", 
                                                                        file_name_precision_recall = "rf_ds_precision_recall",
                                                                        model_name = "random forest")

# upsampled
auc_roc_rf_us, auc_precision_recall_rf_us = roc_precision_recall_curves(rf_us_predictions, y_test, X_test, 
                                                                        rf_us_predictions_proba, 
                                                                        file_name_roc = "rf_us_roc", 
                                                                        file_name_precision_recall = "rf_us_precision_recall",
                                                                        model_name = "random forest")


## visualize one of the trees in the model (optional)
#rf_best_model_first_tree = rf_best_model.estimators_[0]
#
#plt.figure(figsize = (20, 20))
#plot_tree(rf_best_model_first_tree, feature_names = X_train.columns,
#         filled = True, rounded = True)
#plt.title("First decision tree from the random forest with the best performance", size = 16)
#plt.show(block = False)  # dont stop execution of the code
##plt.pause(0.001)



# Support Vector Machine

# grid-search parameter optimizationfrom sklearn import svm
# define hyperparameter values to test
svm_c = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
svm_gamma = [1.0, 0.1, 0.01, 0.001, 0.0001]
svm_gamma.append("scale")
svm_gamma.append("auto")
svm_kernel = ["rbf"]

# grid for random hyperparameter selection
svm_random_grid = {"C": svm_c,
                   "gamma": svm_gamma,
                   "kernel": svm_kernel}

print(svm_random_grid)


# create function that does hyperparameter-optimization, gets metrics, and saves model parameters
def svm_hyper_opt(X, y, grid, cv = 5, verbose = 2, scoring = "f1", iterations = 100, random_state = None,
                  file_name = "svm_temp", mode = None, n_jobs = -1):
    """function to automatically perform hyperparameter optimization and saving the best-performing model
    X: (X_train) features
    y: (y_train) outcome
    grid: parameter-grid with parameters for sklearns RandomForestClassifier-function to be tested
    cv: how many folds for cross-validation
    scoring: which metric to optimize for (options can be found in scoring parameter of sklearns RandomizedSearchCV-function)
    iterations: how many possible hyperparamter-combinations should be tested
    random_state: whether results should be replicable, possible values: integers
    verbose: how much information should be printed
    file_name: name of the file that saves the best model, does not require filetype
    mode: define method for hyperparameter-optimization; "all": GridSearchCV(), otherwise: RandomizedSearchCV()
    n_jobs: how many cpu-cores should be used for the computation; -1 to use all available cores
    
    returns best-performing model and value for the runtime of the hyperparameter optimization"""
    
    # initialize support vector machine and grid search
    svm_optim = svm.SVC(random_state = random_state, probability = True)
    if mode == "all":
        svm_gs = GridSearchCV(estimator = svm_optim, param_grid = grid, cv = cv, verbose = verbose,  n_jobs = n_jobs, 
                              scoring = str(scoring), return_train_score = False)
        print("mode: all")   
    else:
        svm_gs = RandomizedSearchCV(estimator = svm_optim, param_distributions = grid, n_iter = iterations, cv = cv, 
                                    verbose = verbose, random_state = random_state, n_jobs = n_jobs, scoring = scoring,
                                    return_train_score = False)
        print("mode: random")
        
    
    # grid search
    # capture runtime
    start_time = time.time()
    svm_gs.fit(X, y)
    end_time = time.time()
    runtime = end_time - start_time
    print()  # this is just to make the printed text be more structured
    print("Run time support vector machine hyperparameter optimization: \n", np.round(runtime, 4) , "seconds")
    print("Number of iterations: ", iterations, "; Number of observations: ", len(X))
    
    print()
    print("Optimal parameters:")
    print(svm_gs.best_params_)
    
    # define best model
    svm_best_model = svm_gs.best_estimator_
    
    # save best estimator
    # create folder for model if it does not exist already
    path = os.getcwd() + "\\models"
    path_model = path + "\\" + file_name + ".sav"
    Path(path).mkdir(parents = True, exist_ok = True)
    pickle.dump(svm_best_model, open(path_model, "wb"))
    print()
    print("Model parameters saved at: ", path_model)
    
    # return model and metrics
    return svm_best_model, runtime


# run hyperparameter optimization
if mode_run == "optimized":
    svm_best_model = pickle.load(open("models\\svm_best_model.sav", "rb"))
else:
    svm_best_model, runtime_svm_hyper_opt = svm_hyper_opt(X_train, y_train, grid = svm_random_grid, cv = 5, verbose = 2, 
                                                        scoring = "f1", iterations = 1000, random_state = 100, 
                                                        file_name = "svm_best_model", mode = "all", n_jobs = -1)

# ROSE
if mode_run == "optimized":
    svm_rose_best_model = pickle.load(open("models\\svm_rose_best_model.sav", "rb"))
else:
    svm_rose_best_model, runtime_svm_rose_hyper_opt = svm_hyper_opt(X_train_rose, y_train_rose, grid = svm_random_grid, cv = 5, 
                                                                    verbose = 2, scoring = "f1", iterations = 1000, 
                                                                    random_state = 100, file_name = "svm_rose_best_model", 
                                                                    mode = "all", n_jobs = -1)

# downsampled
if mode_run == "optimized":
    svm_ds_best_model = pickle.load(open("models\\svm_ds_best_model.sav", "rb"))
else:
    svm_ds_best_model, runtime_svm_ds_hyper_opt = svm_hyper_opt(X_train_ds, y_train_ds, grid = svm_random_grid, cv = 5, 
                                                                verbose = 2, scoring = "f1", iterations = 1000, 
                                                                random_state = 100, file_name = "svm_ds_best_model", 
                                                                mode = "all", n_jobs = -1)

# upsampled
if mode_run == "optimized":
    svm_us_best_model = pickle.load(open("models\\svm_us_best_model.sav", "rb"))
else:
    svm_us_best_model, runtime_svm_us_hyper_opt = svm_hyper_opt(X_train_us, y_train_us, grid = svm_random_grid, cv = 5, 
                                                                verbose = 2, scoring = "f1", iterations = 1000, 
                                                                random_state = 100, file_name = "svm_us_best_model", 
                                                                mode = "all", n_jobs = -1)


# create function that takes the test set and the model and outputs predictions and metrics
def support_vector_machine_predictions(X, model):
    """function to get predictions and metrics from a trained model
    X: (X_test) test data
    model: trained model
    
    returns predictions and the runtime for the prediction of all observations in X"""
    
    # capture runtime
    start_time = time.time()
    predictions = model.predict(X)
    end_time = time.time()
    runtime = end_time - start_time
    print("Run time support vector machine predictions on the test set: \n", np.round(runtime, 4), " seconds")
    
    return predictions, runtime

# create function that takes the test set and the model and outputs predictions of probabilties of belonging to class "1"
def support_vector_machine_predictions_proba(X, model):
    """function to get predictions of probabilities from a trained model
    X: (X_test) test data
    model: trained model
    
    returns predicted probabilities
    """
    
    predictions_probability = model.predict_proba(X)[:,1]
    
    return predictions_probability


# get predictions
svm_predictions, runtime_svm_predictions = support_vector_machine_predictions(X_test, model = svm_best_model)
svm_predictions_proba = support_vector_machine_predictions_proba(X_test, model = svm_best_model) 

# ROSE, downsampled, upsampled
svm_rose_predictions, runtime_svm_rose_predictions = support_vector_machine_predictions(X_test, model = svm_rose_best_model)
svm_ds_predictions, runtime_svm_ds_predictions = support_vector_machine_predictions(X_test, model = svm_ds_best_model)
svm_us_predictions, runtime_svm_us_predictions = support_vector_machine_predictions(X_test, model = svm_us_best_model)

svm_rose_predictions_proba = support_vector_machine_predictions_proba(X_test, model = svm_rose_best_model) 
svm_ds_predictions_proba = support_vector_machine_predictions_proba(X_test, model = svm_ds_best_model) 
svm_us_predictions_proba = support_vector_machine_predictions_proba(X_test, model = svm_us_best_model) 





# evaluate predictions
metrics_svm = evaluate_predictions(svm_predictions, y_test, svm_predictions_proba)
metrics_svm_rose = evaluate_predictions(svm_rose_predictions, y_test, svm_rose_predictions_proba)
metrics_svm_ds = evaluate_predictions(svm_ds_predictions, y_test, svm_ds_predictions_proba)
metrics_svm_us = evaluate_predictions(svm_us_predictions, y_test, svm_us_predictions_proba)

# check saved metrics
print(metrics_svm)
print(metrics_svm_rose)
print(metrics_svm_ds)
print(metrics_svm_us)

# export metrics
export_metrics_to_txt(metrics_svm, "support_vector_machine_metrics")
export_metrics_to_txt(metrics_svm_rose, "support_vector_machine_rose_metrics")
export_metrics_to_txt(metrics_svm_ds, "support_vector_machine_downsampled_metrics")
export_metrics_to_txt(metrics_svm_us, "support_vector_machine_upsampled_metrics")


# create and save confusion matrix
create_confusion_to_eliminate_confusion(svm_predictions, y_test, file_name = "svm_conf_mat")

# ROSE
create_confusion_to_eliminate_confusion(svm_rose_predictions, y_test, file_name = "svm_rose_conf_mat")

# downsampled
create_confusion_to_eliminate_confusion(svm_ds_predictions, y_test, file_name = "svm_ds_conf_mat")

# upsampled
create_confusion_to_eliminate_confusion(svm_us_predictions, y_test, file_name = "svm_us_conf_mat")


# create and save roc- and precision-recall-plots
auc_roc_svm, auc_precision_recall_svm = roc_precision_recall_curves(svm_predictions, y_test, X_test, svm_predictions_proba, 
                                                                    file_name_roc = "svm_roc", 
                                                                    file_name_precision_recall = "svm_precision_recall", 
                                                                    model_name = "support vector machine")

# ROSE
auc_roc_svm_rose, auc_precision_recall_svm_rose = roc_precision_recall_curves(svm_rose_predictions, y_test, X_test, 
                                                                              svm_rose_predictions_proba, 
                                                                              file_name_roc = "svm_rose_roc", 
                                                                              file_name_precision_recall = "svm_rose_precision_recall", 
                                                                              model_name = "support vector machine")

# downsampled
auc_roc_svm_ds, auc_precision_recall_svm_ds = roc_precision_recall_curves(svm_ds_predictions, y_test, X_test, 
                                                                          svm_ds_predictions_proba, 
                                                                          file_name_roc = "svm_ds_roc", 
                                                                          file_name_precision_recall = "svm_ds_precision_recall", 
                                                                          model_name = "support vector machine")

# upsampled
auc_roc_svm_us, auc_precision_recall_svm_us = roc_precision_recall_curves(svm_us_predictions, y_test, X_test, 
                                                                          svm_us_predictions_proba, 
                                                                          file_name_roc = "svm_us_roc", 
                                                                          file_name_precision_recall = "svm_us_precision_recall", 
                                                                          model_name = "support vector machine")



# Neural Network

# transform variables to numpy-arrays for the neural net
X_train_np = np.array(X_train)
X_train_rose_np = np.array(X_train_rose)
X_train_ds_np = np.array(X_train_ds)
X_train_us_np = np.array(X_train_us)
X_valid_np = np.array(X_valid)
X_test_np = np.array(X_test)
y_train_np = y_train.values
y_train_rose_np = y_train_rose.values
y_train_ds_np = y_train_ds.values
y_train_us_np = y_train_us.values
y_valid_np = y_valid.values
y_test_np = y_test.values

X_train_np.shape, y_train_np.shape


# function to build neural networks
def build_model(hyper_model):
    model = keras.Sequential()
    
    # first hidden layer (needs to be defined separately, because of the input shape)
    hyper_model_units = hyper_model.Int("units", min_value = 8, max_value = 64, step = 4)
    model.add(layers.Dense(units = hyper_model_units, activation = "relu", input_shape = (X_train_np.shape[1],)))
    
    # first regularization-layer
    hyper_model_dropout = hyper_model.Float("dropout", min_value = 0.1, max_value = 0.5, step = 0.1)
    model.add(layers.Dropout(rate = hyper_model_dropout))
    
    # more hidden layers and regularizaton-layers
    for i in range(hyper_model.Int("num_layers", 2, 10)):
        model.add(layers.Dense(units = hyper_model.Int(f"units_{i}", min_value = 8, max_value = 64), activation = "relu"))
        model.add(layers.Dropout(rate = hyper_model.Float(f"dropout_{i}", min_value = 0.1, max_value = 0.5, step = 0.1)))
        
    # output layer
    model.add(layers.Dense(1, activation = "sigmoid"))
    
    # hyperparameters
    hyper_model_learning_rate = hyper_model.Choice("learning_rate", values = [0.1, 0.01, 0.001, 0.0001])
    
    model.compile(optimizer = keras.optimizers.Adam(learning_rate = hyper_model_learning_rate),
                 loss = "binary_crossentropy",
                 metrics = ["accuracy",
                           tf.keras.metrics.Precision(name = "precision"),
                           tf.keras.metrics.Recall(name = "recall")])
    
    return model


# create function that does hyperparameter-optimization, gets metrics, and saves model parameters
def neural_net_opt(tuner, X_train, y_train, X_valid, y_valid, callbacks = None, epochs = 100):
    """function to automatically perform hyperparameter optimization and saving the best-performing model
    X_train: training data
    y_train: training data outcome
    X_valid: validation data
    y_valid: validation data outcome
    callbacks: callbacks used for e.g. regularization
    epochs: number of epochs
    file_name: name of the file that saves the best model, does not require filetype
    
    returns best-performing model and value for the runtime of the hyperparameter optimization"""
    
    
    # hyperparameter optimization
    start_time = time.time()
    tuner.search(X_train, y_train, callbacks = callbacks, epochs = 100, validation_data = (X_valid, y_valid))
    end_time = time.time()
    runtime = end_time - start_time
    print()
    print("Run time neural net hyperparameter optimization: \n", np.round(runtime, 4) , "seconds")
    
    # create model with best parameters
    best_hyper_model = tuner.get_best_hyperparameters(num_trials = 1)[0]
    best_model = tuner.hypermodel.build(best_hyper_model)
    print()
    print("Optimal model:")
    print(best_model.summary())
    
    # return model and metrics
    return best_model, runtime


# early-stopping regularization to speed up optimization
es = EarlyStopping(monitor = "val_recall",
                  mode = "max",
                  patience = 10,
                  restore_best_weights = True)

es_rose = EarlyStopping(monitor = "val_accuracy",
                  mode = "max",
                  patience = 10,
                  restore_best_weights = True)

es_ds = EarlyStopping(monitor = "val_accuracy",
                  mode = "max",
                  patience = 10,
                  restore_best_weights = True)

es_us = EarlyStopping(monitor = "val_accuracy",
                  mode = "max",
                  patience = 10,
                  restore_best_weights = True)


# define tuner for hyperparameter optimization
tuner = kt.RandomSearch(
    build_model,
    objective = kt.Objective("val_recall", direction = "max"),
    max_trials = 100,
    directory = os.getcwd() + "\\models\\neural_net",
    project_name = "hyperparameter_optimization",
    seed = 100)

# rose
tuner_rose = kt.RandomSearch(
    build_model,
    objective = kt.Objective("val_accuracy", direction = "max"),
    max_trials = 100,
    directory = os.getcwd() + "\\models\\neural_net",
    project_name = "hyperparameter_optimization_rose",
    seed = 100)

# downsampled
tuner_ds = kt.RandomSearch(
    build_model,
    objective = kt.Objective("val_accuracy", direction = "max"),
    max_trials = 100,
    directory = os.getcwd() + "\\models\\neural_net",
    project_name = "hyperparameter_optimization_downsampled",
    seed = 100)

# upsampled
tuner_us = kt.RandomSearch(
    build_model,
    objective = kt.Objective("val_accuracy", direction = "max"),
    max_trials = 100,
    directory = os.getcwd() + "\\models\\neural_net",
    project_name = "hyperparameter_optimization_upsampled",
    seed = 100)


# run optimization
neural_net_best_model, runtime_neural_net_hyper_opt = neural_net_opt(tuner = tuner, X_train = X_train_np, 
                                                                     y_train = y_train_np, X_valid = X_valid_np, 
                                                                     y_valid = y_valid_np, callbacks = [es], epochs = 100)

# rose
neural_net_best_model_rose, runtime_neural_net_hyper_opt_rose = neural_net_opt(tuner = tuner_rose, X_train = X_train_rose_np, 
                                                                               y_train = y_train_rose_np, X_valid = X_valid_np, 
                                                                               y_valid = y_valid_np, callbacks = [es_rose], 
                                                                               epochs = 100)

# downsampled
neural_net_best_model_ds, runtime_neural_net_hyper_opt_ds = neural_net_opt(tuner = tuner_ds, X_train = X_train_ds_np, 
                                                                           y_train = y_train_ds_np, X_valid = X_valid_np, 
                                                                           y_valid = y_valid_np, callbacks = [es_ds], 
                                                                           epochs = 100)

# upsampled
neural_net_best_model_us, runtime_neural_net_hyper_opt_us = neural_net_opt(tuner = tuner_us, X_train = X_train_us_np, 
                                                                           y_train = y_train_us_np, X_valid = X_valid_np, 
                                                                           y_valid = y_valid_np, callbacks = [es_us], 
                                                                           epochs = 100)


# derived optimal model architectures
neural_net_best_model.summary()
neural_net_best_model_rose.summary()
neural_net_best_model_ds.summary()
neural_net_best_model_us.summary()


# define paths to save the models
path = os.getcwd() + "\\models"
path_model = path + "\\" + "neural_net\\neural_net\\neural_net_best_model.weights.h5"
path_model_rose = path + "\\" + "neural_net\\neural_net_rose\\neural_net_rose_best_model.weights.h5"
path_model_ds = path + "\\" + "neural_net\\neural_net_ds\\neural_net_ds_best_model.weights.h5" 
path_model_us = path + "\\" + "neural_net\\neural_net_us\\neural_net_us_best_model.weights.h5" 
# create paths if they do not already exist
#Path(path).mkdir(parents = True, exist_ok = True)
#Path(path_model).mkdir(parents = True, exist_ok = True)
#Path(path_model_rose).mkdir(parents = True, exist_ok = True)
#Path(path_model_ds).mkdir(parents = True, exist_ok = True)
#Path(path_model_us).mkdir(parents = True, exist_ok = True)


# train or load model with best hyperparameters
if mode_run == "optimized":
    neural_net_best_model.load_weights(path_model)
else:
    history_best = neural_net_best_model.fit(X_train_np,
                                            y_train_np,
                                            callbacks = [es],
                                            epochs = 100,
                                            validation_data = (X_valid_np, y_valid_np))
    best_model_eval = neural_net_best_model.evaluate(X_test_np, y_test_np)
    neural_net_best_model.save_weights(path_model)
    best_model_eval

    # show metrics over training of model with best hyperparameters
    history_dict = history_best.history

    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]

    epochs = range(1, len(loss_values) + 1)
    
    plt.figure()
    plt.plot(epochs, loss_values, "bo", label = "Training loss")
    plt.plot(epochs, val_loss_values, "orange", label = "Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show(block = False)  # dont stop execution of the code
    #plt.pause(0.001)

    acc = history_best.history["accuracy"]
    val_acc = history_best.history["val_accuracy"]

    epochs = range(1, len(acc) + 1)
    
    plt.figure()
    plt.plot(epochs, acc, "bo", label = "Training accuracy")
    plt.plot(epochs, val_acc, "orange", label = "Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show(block = False)  # dont stop execution of the code
    #plt.pause(0.001)

    np.max(val_acc)

# ROSE
if mode_run == "optimized":
    neural_net_best_model_rose.load_weights(path_model_rose)
else:
    history_best_rose = neural_net_best_model_rose.fit(X_train_rose_np,
                                                    y_train_rose_np,
                                                    callbacks = [es_rose],
                                                    epochs = 100,
                                                    validation_data = (X_valid_np, y_valid_np))
    best_model_eval_rose = neural_net_best_model_rose.evaluate(X_test_np, y_test_np)
    neural_net_best_model_rose.save_weights(path_model_rose)
    best_model_eval_rose

    # show metrics over training of model with best hyperparameters
    history_dict = history_best_rose.history

    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]

    epochs = range(1, len(loss_values) + 1)
    
    plt.figure()
    plt.plot(epochs, loss_values, "bo", label = "Training loss")
    plt.plot(epochs, val_loss_values, "orange", label = "Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show(block = False)  # dont stop execution of the code
    #plt.pause(0.001)

    acc = history_best_rose.history["accuracy"]
    val_acc = history_best_rose.history["val_accuracy"]

    epochs = range(1, len(acc) + 1)
    
    plt.figure()
    plt.plot(epochs, acc, "bo", label = "Training accuracy")
    plt.plot(epochs, val_acc, "orange", label = "Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show(block = False)  # dont stop execution of the code
    #plt.pause(0.001)

    np.max(val_acc)

# downsampled
if mode_run == "optimized":
    neural_net_best_model_ds.load_weights(path_model_ds)
else:
    # train model with best hyperparameters
    history_best_ds = neural_net_best_model_ds.fit(X_train_ds_np,
                                                y_train_ds_np,
                                                callbacks = [es_ds],
                                                epochs = 100,
                                                validation_data = (X_valid_np, y_valid_np))
    best_model_eval_ds = neural_net_best_model_ds.evaluate(X_test_np, y_test_np)
    neural_net_best_model_ds.save_weights(path_model_ds)
    best_model_eval_ds

    # show metrics over training of model with best hyperparameters
    history_dict = history_best_ds.history

    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]

    epochs = range(1, len(loss_values) + 1)
    
    plt.figure()
    plt.plot(epochs, loss_values, "bo", label = "Training loss")
    plt.plot(epochs, val_loss_values, "orange", label = "Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show(block = False)  # dont stop execution of the code
    #plt.pause(0.001)

    acc = history_best_ds.history["accuracy"]
    val_acc = history_best_ds.history["val_accuracy"]

    epochs = range(1, len(acc) + 1)
    
    plt.figure()
    plt.plot(epochs, acc, "bo", label = "Training accuracy")
    plt.plot(epochs, val_acc, "orange", label = "Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show(block = False)  # dont stop execution of the code
    #plt.pause(0.001)

    np.max(val_acc)

# upsampled
if mode_run == "optimized":
    neural_net_best_model_us.load_weights(path_model_us)
else:
    # train model with best hyperparameters
    history_best_us = neural_net_best_model_us.fit(X_train_us_np,
                                                y_train_us_np,
                                                callbacks = [es_us],
                                                epochs = 100,
                                                validation_data = (X_valid_np, y_valid_np))
    best_model_eval_us = neural_net_best_model_us.evaluate(X_test_np, y_test_np)
    neural_net_best_model_us.save_weights(path_model_us)
    best_model_eval_us

    # show metrics over training of model with best hyperparameters
    history_dict = history_best_us.history

    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]

    epochs = range(1, len(loss_values) + 1)
    
    plt.figure()
    plt.plot(epochs, loss_values, "bo", label = "Training loss")
    plt.plot(epochs, val_loss_values, "orange", label = "Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show(block = False)  # dont stop execution of the code
    #plt.pause(0.001)

    acc = history_best_us.history["accuracy"]
    val_acc = history_best_us.history["val_accuracy"]

    epochs = range(1, len(acc) + 1)
    
    plt.figure()
    plt.plot(epochs, acc, "bo", label = "Training accuracy")
    plt.plot(epochs, val_acc, "orange", label = "Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show(block = False)  # dont stop execution of the code
    #plt.pause(0.001)

    np.max(val_acc)


# create function that takes the test set and the model and outputs predictions and metrics
def neural_net_predictions(X, model):
    """function to get predictions and metrics from a trained model
    X: (X_test) test data
    model: trained model
    
    returns predictions and the runtime for the prediction of all observations in X"""
    
    # capture runtime
    start_time = time.time()
    predictions = np.round(model.predict(X), 0)
    end_time = time.time()
    runtime = end_time - start_time
    
    print("Run time neural net predictions on the test set: \n", np.round(runtime, 4), " seconds")
    
    predictions_proba = np.round(model.predict(X), 0)
    
    return predictions, runtime, predictions_proba


# get predictions
nn_predictions, runtime_neural_net_predictions, nn_predictions_proba = neural_net_predictions(X_test_np, 
                                                                                              model = neural_net_best_model)

# rose
nn_rose_predictions, runtime_neural_net_rose_predictions, nn_rose_predictions_proba = neural_net_predictions(X_test_np, 
                                                                                                             model = neural_net_best_model_rose)

# downsampled
nn_ds_predictions, runtime_neural_net_ds_predictions, nn_ds_predictions_proba = neural_net_predictions(X_test_np, 
                                                                                                       model = neural_net_best_model_ds)

# upsampled
nn_us_predictions, runtime_neural_net_us_predictions, nn_us_predictions_proba = neural_net_predictions(X_test_np, 
                                                                                                       model = neural_net_best_model_us)


# evaluate predictions
metrics_nn = evaluate_predictions(nn_predictions, y_test_np, nn_predictions_proba)
metrics_nn_rose = evaluate_predictions(nn_rose_predictions, y_test_np, nn_rose_predictions_proba)
metrics_nn_ds = evaluate_predictions(nn_ds_predictions, y_test_np, nn_ds_predictions_proba)
metrics_nn_us = evaluate_predictions(nn_us_predictions, y_test_np, nn_us_predictions_proba)

# check saved metrics
print(metrics_nn)
print(metrics_nn_rose)
print(metrics_nn_ds)
print(metrics_nn_us)

# export metrics
export_metrics_to_txt(metrics_nn, "neural_network_metrics")
export_metrics_to_txt(metrics_nn_rose, "neural_network_rose_metrics")
export_metrics_to_txt(metrics_nn_ds, "neural_network_downsampled_metrics")
export_metrics_to_txt(metrics_nn_us, "neural_network_upsampled_metrics")


# create and save confusion matrix
create_confusion_to_eliminate_confusion(nn_predictions, y_test, file_name = "nn_conf_mat")
create_confusion_to_eliminate_confusion(nn_rose_predictions, y_test, file_name = "nn_rose_conf_mat")
create_confusion_to_eliminate_confusion(nn_ds_predictions, y_test, file_name = "nn_ds_conf_mat")
create_confusion_to_eliminate_confusion(nn_us_predictions, y_test, file_name = "nn_us_conf_mat")


# create and save roc- and precision-recall-plots
auc_roc_nn, auc_precision_recall_nn = roc_precision_recall_curves(nn_predictions, y_test_np, X_test_np, nn_predictions_proba,
                                                                 file_name_roc = "nn_roc", 
                                                                 file_name_precision_recall = "nn_precision_recall",
                                                                 model_name = "multi-layer perceptron")

# ROSE
auc_roc_nn_rose, auc_precision_recall_nn_rose = roc_precision_recall_curves(nn_rose_predictions, y_test_np, X_test_np, 
                                                                            nn_rose_predictions_proba,
                                                                            file_name_roc = "nn_rose_roc",
                                                                            file_name_precision_recall = "nn_rose_precision_recall",
                                                                            model_name = "multi-layer perceptron")

# downsampled
auc_roc_nn_ds, auc_precision_recall_nn_ds = roc_precision_recall_curves(nn_ds_predictions, y_test_np, X_test_np, 
                                                                        nn_ds_predictions_proba,
                                                                        file_name_roc = "nn_ds_roc", 
                                                                        file_name_precision_recall = "nn_ds_precision_recall",
                                                                        model_name = "multi-layer perceptron")

# upsampled
auc_roc_nn_us, auc_precision_recall_nn_us = roc_precision_recall_curves(nn_us_predictions, y_test_np, X_test_np, 
                                                                        nn_us_predictions_proba,
                                                                        file_name_roc = "nn_us_roc", 
                                                                        file_name_precision_recall = "nn_us_precision_recall",
                                                                        model_name = "multi-layer perceptron")



# Metrics for base Logistic Regression

# load predictions
log_reg_base_predictions = pd.read_csv("outputs//base_log_reg_predictions.csv")
log_reg_base_predictions_proba = pd.read_csv("outputs//base_log_reg_predictions_proba.csv")

# ROSE, downsampled, upsampled
log_reg_base_rose_predictions = pd.read_csv("outputs//base_log_reg_rose_predictions.csv")
log_reg_base_rose_predictions_proba = pd.read_csv("outputs//base_log_reg_rose_predictions_proba.csv")
log_reg_base_ds_predictions = pd.read_csv("outputs//base_log_reg_ds_predictions.csv")
log_reg_base_ds_predictions_proba = pd.read_csv("outputs//base_log_reg_ds_predictions_proba.csv")
log_reg_base_us_predictions = pd.read_csv("outputs//base_log_reg_us_predictions.csv")
log_reg_base_us_predictions_proba = pd.read_csv("outputs//base_log_reg_us_predictions_proba.csv")

# get data into the right format
log_reg_base_predictions = log_reg_base_predictions["x"].values
log_reg_base_predictions_proba = log_reg_base_predictions_proba["x"].values
log_reg_base_rose_predictions = log_reg_base_rose_predictions["x"].values
log_reg_base_rose_predictions_proba = log_reg_base_rose_predictions_proba["x"].values
log_reg_base_ds_predictions = log_reg_base_ds_predictions["x"].values
log_reg_base_ds_predictions_proba = log_reg_base_ds_predictions_proba["x"].values
log_reg_base_us_predictions = log_reg_base_us_predictions["x"].values
log_reg_base_us_predictions_proba = log_reg_base_us_predictions_proba["x"].values





# evaluate predictions
metrics_base = evaluate_predictions(log_reg_base_predictions, y_test, log_reg_base_predictions_proba)
metrics_base_rose = evaluate_predictions(log_reg_base_rose_predictions, y_test, log_reg_base_rose_predictions_proba)
metrics_base_ds = evaluate_predictions(log_reg_base_ds_predictions, y_test, log_reg_base_ds_predictions_proba)
metrics_base_us = evaluate_predictions(log_reg_base_us_predictions, y_test, log_reg_base_us_predictions_proba)

# check saved metrics
print(metrics_base)
print(metrics_base_rose)
print(metrics_base_ds)
print(metrics_base_us)

# export metrics
export_metrics_to_txt(metrics_base, "logistic_regression_base_metrics")
export_metrics_to_txt(metrics_base_rose, "logistic_regression_base_rose_metrics")
export_metrics_to_txt(metrics_base_ds, "logistic_regression_base_ds_metrics")
export_metrics_to_txt(metrics_base_us, "logistic_regression_base_us_metrics")


# create and save confusion matrix
create_confusion_to_eliminate_confusion(log_reg_base_predictions, y_test, file_name = "base_conf_mat")
create_confusion_to_eliminate_confusion(log_reg_base_rose_predictions, y_test, file_name = "base_rose_conf_mat")
create_confusion_to_eliminate_confusion(log_reg_base_ds_predictions, y_test, file_name = "base_ds_conf_mat")
create_confusion_to_eliminate_confusion(log_reg_base_us_predictions, y_test, file_name = "base_us_conf_mat")


# create and save roc- and precision-recall-plots
auc_roc_base, auc_precision_recall_base = roc_precision_recall_curves(log_reg_base_predictions, y_test, X_test, 
                                                                      log_reg_base_predictions_proba,
                                                                      file_name_roc = "base_roc", 
                                                                      file_name_precision_recall = "base_precision_recall",
                                                                      model_name = "logistic regression base model")

# ROSE
auc_roc_base_rose, auc_precision_recall_base_rose = roc_precision_recall_curves(log_reg_base_rose_predictions, y_test, X_test, 
                                                                      log_reg_base_rose_predictions_proba,
                                                                      file_name_roc = "base_rose_roc", 
                                                                      file_name_precision_recall = "base_rose_precision_recall",
                                                                      model_name = "logistic regression base model")

# downsampled
auc_roc_base_ds, auc_precision_recall_base_ds = roc_precision_recall_curves(log_reg_base_ds_predictions, y_test, X_test, 
                                                                      log_reg_base_ds_predictions_proba,
                                                                      file_name_roc = "base_ds_roc", 
                                                                      file_name_precision_recall = "base_ds_precision_recall",
                                                                      model_name = "logistic regression base model")

# upsampled
auc_roc_base_us, auc_precision_recall_base_us = roc_precision_recall_curves(log_reg_base_us_predictions, y_test, X_test, 
                                                                      log_reg_base_us_predictions_proba,
                                                                      file_name_roc = "base_us_roc", 
                                                                      file_name_precision_recall = "base_us_precision_recall",
                                                                      model_name = "logistic regression base model")



# closing or leaving the figures depending on mode_figures
if mode_figures == "no_keep":
    pass
else:
    plt.show()




