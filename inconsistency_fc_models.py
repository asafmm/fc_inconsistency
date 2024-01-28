import numpy as np
import pandas as pd
import scipy 
import random
import matplotlib.pyplot as plt
from numpy import matlib
from sklearn import preprocessing, metrics, model_selection, linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import matplotlib_inline
import matplotlib
import datetime

# fix random seed
random.seed(42)

def select_features(X, y, number_thresh = 0):
    '''
    Selects the top features (X) that are most correlated with y, in absolute value.
    
    Parameters
    ----------------------------
            X - pandas DataFrame
                feature matrix
            y - numpy float array
                target vairable
            number_thresh -  int
                number of features to select
            
    Returns
    ----------------------------
            top_features - string list
                list of the selected features 
    '''
    if number_thresh == 0:
        top_features = X.columns
    else:
        # select top features using number_thresh
        x_y_corr = []
        for i in range(X.shape[1]):
            r = np.corrcoef(X.iloc[:, i], y)
            x_y_corr.append(r[1, 0])
        x_y_corr = np.array(x_y_corr)
        abs_corr = np.abs(x_y_corr)
        top_indices = np.argsort(abs_corr)
        top_features = X.columns[top_indices[-1*number_thresh:]]
    return top_features

def train_model(X, y, model_type, k_cv=33, scale_x=1, feature_number=0, track_progress=1):
    '''
    Train and evaluate models with nested leave-one-out cross-validation
    
    Parameters
    ----------------------------
            X - pandas DataFrame
                feature matrix
            y - numpy float array
                target vairable
            model_type - {'RandomForest', 'Lasso'}
            k_cv -  int
                number of folds in the nested-cross validation for hyperparameter tuning
            scale_x - int (0 or 1)
                whether to scale the features or not
            feature_number - int
                number of features with the highest correlation with y to select
            track_progress - int (0 or 1)
                whether to print progress of the CV loop or not
            
    Returns
    ----------------------------
            coefs_list - list of pandas DataFrame 
                list of models' coefficients/feature importance dataframes
            y_hats - float array 
                predicted target values
            y_true - float array
                actual target values
            models - list 
                list of trained models
            scores - float array 
                list of models' MSEs
    '''
    models = []
    scores = []
    y = np.log(y)

    # hyperparameters
    n_estimators = np.linspace(60, 250, 10, dtype=int)
    n_estimators = np.insert(n_estimators, 0, 100)
    max_features = [1.0, 'sqrt'] # ["all features", "sqrt of features"]
    max_depth = np.linspace(10, 100, 5, dtype=int)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2]
    low_alphas = np.arange(10**-5, 1, 0.001)
    high_alphas = np.arange(1, 30, 2)
    alphas = np.concatenate([low_alphas, high_alphas])

    # model init
    if model_type == 'RandomForest':
        param_grid = {'n_estimators' : n_estimators,
                     'max_features' : max_features,
                     'max_depth' : max_depth,
                     'min_samples_split' : min_samples_split,
                     'min_samples_leaf' : min_samples_leaf}
        model = RandomForestRegressor(n_jobs = -1, random_state = 42)
    elif model_type == 'Lasso':
        model = linear_model.Lasso(max_iter=np.int64(5e5))
        param_grid = {'alpha': alphas}
    model_cv = GridSearchCV(model, param_grid=param_grid, n_jobs=-1, cv=k_cv, scoring='neg_mean_squared_error')

    # CV loop
    loo = model_selection.LeaveOneOut()
    i = 1
    y_hats = []
    y_true = []
    coefs_list = []
    for train_ind, test_ind in loo.split(X):
        if track_progress==1:
            current_time = datetime.datetime.strftime(datetime.datetime.now(), '%H:%M:%S %D')
            print(f'Fold: {i}/{len(y)}, Time: {current_time}')

        X_train, X_test = X.iloc[train_ind, :], X.iloc[test_ind, :]
        y_train, y_test = y.iloc[train_ind], y.iloc[test_ind]

        if feature_number!=0:
            selected_features = select_features(X_train, y_train, feature_number)
            X_train, X_test = X_train.loc[:, selected_features], X_test.loc[:, selected_features]

        features = X_train.columns

        if scale_x==1:
            X_train = pd.DataFrame(preprocessing.scale(X_train, axis=1), columns=X_train.columns)
            X_test = pd.DataFrame(preprocessing.scale(X_test, axis=1), columns=X_test.columns)
        else:
            pass

        mean_y = np.mean(y_train)
        std_y = np.std(y_train)
        y_train = (y_train - mean_y) / std_y
        # scale y_test with y_train
        y_test = (y_test - mean_y) / std_y
        
        model_cv.fit(X_train, y_train)
        best_model = model_cv.best_estimator_
        
        y_pred = best_model.predict(X_test)
        models.append((best_model, X_train))
        y_train_pred = best_model.predict(X_train)
        score = metrics.mean_squared_error(y_test, y_pred)
        scores.append(score)
        train_score = metrics.mean_squared_error(y_train, y_train_pred)
        if track_progress==1:
            print(f'Test MSE: {score:.4f}')
            print(f'Train MSE: {train_score:.4f}')
        y_hats.append(y_pred[0])
        y_true.append(y_test.values[0])
        
        features_std = np.std(X_train)
        coef_df = pd.DataFrame(features, columns = ['Feature'])
        coef_df = coef_df.set_index('Feature')

        if model_type=='RandomForest':
            coef_df['FeatureStd'] = features_std
            coefs = best_model.feature_importances_
            coef_df.loc[:, 'ScaledCoefficient'] = coefs * features_std
            coef_df['Coefficient'] = coefs
        elif model_type=='Lasso':
            coef_df.loc[:, 'FeatureStd'] = features_std
            coefs = best_model.coef_
            coef_df.loc[:, 'ScaledCoefficient'] = coefs * features_std
            coef_df.loc[:, 'Coefficient'] = coefs
        coef_df = coef_df.reset_index()
        # each model might have different sets of features, so we append all DataFrames to a list 
        coefs_list.append(coef_df)
        i += 1
    return coefs_list, y_hats, y_true, models, scores

def average_coefs(coef_df_list):
    '''
    Returns average coefficients of the prediction models.
    
    Parameters
    ----------------------------
            coefs_list - DataFrame list (or single DataFrame for Random Forest model)
                list of DataFrames containing coefficients and MSEs of the models 
    
    Returns
    ----------------------------
            avg_coefs - DataFrame
                DataFrame of average coefficient of all models
            score - float
                average score of models' predictions
    '''
    # if coef_df_list is a list, it contains coefficients 
    assert type(coef_df_list) == list
    number_of_models = len(coef_df_list)
    combined_coef_df = pd.concat(coef_df_list)
    # we don't use .mean() to account for the features that weren't selected in the feature selection, 
    # instead of mean() we divide by N models
    avg_coefs = combined_coef_df.groupby('Feature').sum().sort_values(by='Coefficient')
    avg_coefs = avg_coefs / number_of_models
    avg_coefs = avg_coefs.rename({'Coefficient':'Avg'}, axis=1)
    avg_coefs = avg_coefs.reset_index()
    # calculate std not by std() but by accounting for N models
    for f in avg_coefs.Feature:
        feature_coefficients = combined_coef_df.loc[combined_coef_df.Feature==f, 'Coefficient']
        feature_avg = avg_coefs.loc[avg_coefs.Feature==f, 'Avg'].values[0]
        feature_std = np.sqrt(np.sum((feature_coefficients - feature_avg)**2)/number_of_models)
        avg_coefs.loc[avg_coefs.Feature==f, 'Std'] = feature_std
        
    return avg_coefs

def correlation_permuation_test(y_pred, y_true):
    '''
    Test correlation between predicted and true target values using a permuation test
    
    Parameters
    ----------------------------
        y_pred - float array
            predicted target values
        y_true - float array
            true target values
        
    Returns
    ----------------------------
        r - float
            correlation between true and predicted target values
        pval - float
            p-value of the correlation, calculated using a permutation test
    '''
    permutations = 100000
    r, _ = scipy.stats.pearsonr(y_pred, y_true)
    y_pred_permute = y_pred.copy()
    corr_distribution = []

    for i in range(permutations):
        # for each permuation shuffle the predicted values between subjects, and calculate correlation with the true values 
        random.shuffle(y_pred_permute)
        this_corr, _ = scipy.stats.pearsonr(y_pred_permute, y_true)
        corr_distribution.append(this_corr)
    pval =  np.sum(r <= corr_distribution) / len(corr_distribution)
    return r, pval, corr_distribution

def plot_distribution(model_score, score_dist, p, score_str, bins):
    grey = '#606060'
    title_col = '#303030'
    fig, ax = plt.subplots(figsize = (8, 5))
    hist = plt.hist(score_dist, bins = bins, color='#AAB7B8')
    real_line = plt.vlines(x = model_score, ymin = 0, ymax = max(hist[0]), colors = '#CB4335', linestyles = '--')
    plt.legend([real_line], [f"p-val = {p}"], fontsize = 12, loc = 'upper left')
    plt.title(f'{score_str} Distribution', color=title_col, fontsize = 16)
    plt.xlabel(f'Permuted {score_str}', color=grey, fontsize = 14)
    plt.ylabel('Frequency', color=grey, fontsize = 14)
    plt.grid(visible=False, axis='both')
    plt.setp(ax.spines.values(), color=grey)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='both', direction='out', color = grey, labelcolor = grey, labelsize=12)

def print_results(y_true, y_pred, measure, model, plot_dist=0, yticks=[-1, 0, 1], save_path='', pre_calc=None):
    '''
    Plots results of the prediction models in a scatter plot with regression line and its p-value
    
    Parameters
    ----------------------------
        y_pred - float array
            predicted target values
        y_true - float array
            true target values
        measure - string
            name of target variable
        model - string
            name of model for figure title (Linear Regression / Random Forest)
        plot_dist - int (0 or 1)
            whether to plot the distribution of the permutation correlations or not
        yticks - float list
            y-ticks for the scatter plot
        save_path - string 
            path to save the figure
        pre_calc - float list (or None)
            pre-calculated correlation and p-value, if None, the function will calculate them 
        
                
    Returns
    ----------------------------
        r - float
            correlation between true and predicted target values
        pval - float
            p-value of the correlation, calculated using a permutation test
    '''
    if pre_calc is None:
        r, pval, corr_distribution = correlation_permuation_test(y_pred, y_true)
    else:
        r, pval = pre_calc
    
    black = 'black'
    scatter_col = '#3377B4'
    line_col = '#338FB4'
    title_col = 'black'

    matplotlib.rcParams['font.sans-serif'] = "Helvetica"
    matplotlib.rcParams['font.family'] = "sans-serif"
    FIGWIDTH, FIGLENGTH = 6, 4
    TITLESIZE = 16
    LABELSIZE = 14
    TICKSIZE = 12
    MARKERSIZE = 60
    MARKEREDGE = 1
    LINEWIDTH = 2
    AXISWIDTH = 1
    TICKWIDTH = 1
    TICKLENGTH = 3
    TEXT_X, TEXT_Y = 0.78, 0.03
    if save_path != '':
        FIGWIDTH, FIGLENGTH = 3*FIGWIDTH, 3*FIGLENGTH
        TITLESIZE = 3*TITLESIZE
        LABELSIZE = 3*LABELSIZE
        TICKSIZE = 3*TICKSIZE
        MARKERSIZE = 3*MARKERSIZE
        MARKEREDGE = 3*MARKEREDGE
        LINEWIDTH = 3*LINEWIDTH
        AXISWIDTH = 3*AXISWIDTH
        TICKWIDTH = 3*TICKWIDTH
        TICKLENGTH = 3*TICKLENGTH
        
    fig, ax = plt.subplots(figsize = (FIGWIDTH, FIGLENGTH))
    plt.scatter(y_true, y_pred, alpha=0.7, color=scatter_col, s=MARKERSIZE, linewidths=MARKEREDGE)
    if model!='':
        plt.title(model + ' Model', color=title_col, weight='bold', fontsize=TITLESIZE)
    plt.xlabel(f'Actual {measure} (Normalized)', color=black, fontsize=LABELSIZE)
    plt.ylabel(f'Predicted {measure}', color=black, fontsize=LABELSIZE)
    plt.ylim(bottom = -1.3, top = 1.3)
    plt.xlim(left = -2.5, right = 2.2)
    plt.yticks(yticks)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.grid(visible=False, axis='both')
    plt.setp(ax.spines.values(), color=black, linewidth=AXISWIDTH)
    ax.tick_params(axis='both', direction='out', color = black, labelcolor = black, labelsize=TICKSIZE, width=TICKWIDTH, length=TICKLENGTH)

    if pval>=0.0001:
        plt.text(TEXT_X, TEXT_Y, 'r = {:.4f}'.format(r) + '\np = {:.4f}'.format(pval), transform = ax.transAxes, fontsize=LABELSIZE, color=title_col, fontname='Helvetica', weight='bold')
    else:
        plt.text(TEXT_X, TEXT_Y, 'r = {:.4f}'.format(r) + '\np < 0.0001', transform = ax.transAxes, fontsize=LABELSIZE, color=title_col, fontname='Helvetica', weight='bold')
    slope, intercept = np.polyfit(y_true, y_pred, 1)
    _ = plt.plot(y_true, np.multiply(y_true, slope) + intercept, color = scatter_col, linewidth=LINEWIDTH)
    if plot_dist:
        plot_distribution(r, corr_distribution, pval, 'Correlation', 50)
    if 'svg' in save_path:
        plt.savefig(save_path, bbox_inches='tight', format='svg')
    elif save_path != '':
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    return r, pval

def save_results(y_hats, y_true, avg_coefs, dir_path, model_type):
    y_df = pd.DataFrame({'True':y_true, 'Predicted':y_hats})
    y_df.to_csv(dir_path + f'/{model_type}_results.csv')
    avg_coefs.to_csv(dir_path + f'/{model_type}_coef.csv')