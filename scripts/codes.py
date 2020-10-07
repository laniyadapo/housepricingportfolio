#Analysis packages
import pandas as pd
import sklearn as sk
import numpy as np
import scipy.stats as sp

#Visualization packages
import matplotlib.pyplot as plt
import matplotlib as matplot
from matplotlib.ticker import MaxNLocator
import seaborn as sns

#Preprocessing
from sklearn import preprocessing

#Models & Sklearn packages
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import cross_val_score as cvs
from sklearn.linear_model import LinearRegression as lr
from sklearn.ensemble import RandomForestRegressor as rfregr
from sklearn.ensemble import GradientBoostingRegressor as grbregr
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.svm import SVR


from tqdm import tqdm
import datetime

import warnings
warnings.filterwarnings('ignore')


class eda_process():
    def __init__(self, data):
        self.data = data
    
    def perc_missing_data(self, datadesc):
        '''Compute the number of missing values in each column of data'''
        self.datadesc = datadesc
        percent = (((self.data.isnull().sum() / len(self.data)) *100).round(decimals=2)).sort_values(ascending=False)
        print("Percentage of missing values in each column:")
        print(percent[percent > 0])
        plt.figure(figsize=(15, 4))
        plt.xticks(rotation='90')
        sns.barplot(x=percent[percent > 0].index, y=percent[percent > 0])
        plt.xlabel('Features', fontsize=15)
        plt.ylabel('Percent of missing values', fontsize=15)
        plt.title('Percent missing data by feature in ' + self.datadesc, fontsize=15)
        
    def deal_null_values(self, cols_cat, value):
        '''Replace null values with None or 0 as applicable'''
        self.value = value
        for col in cols_cat:
            self.data[col].replace(np.nan, value, inplace=True)
            
    def deal_null_values_freq(self, cols_cat):
        '''Replace Null values with mode of variable'''
        for col in cols_cat:
            mode_fill = self.data[col].mode()[0]
            self.data[col].replace(np.nan, mode_fill, inplace=True)
    
    def target_visual(self, target):
        '''Visual inspection of target data'''
        self.target = target
        value = (self.data / 1000)
        plt.figure(figsize = (14, 6))
        plt.subplot(1,2,1)
        sns.boxplot(value)
        plt.xlabel(self.target + "(X 1000)")
        plt.title('Box Plot')
        plt.subplot(1,2,2)
        sns.distplot(value, bins=20)
        plt.xlabel(self.target + "(X 1000)")
        plt.title('Distribution Plot')
        plt.show()
        
    def num_features_outliers(self):
        '''Find outliers in numerical features using bivariate analysis'''
        num_features = self.data.select_dtypes(exclude='object').drop('SalePrice', axis=1).copy()
        f = plt.figure(figsize=(12,20))
        for i in range(len(num_features.columns)):
            f.add_subplot(10, 4, i+1)
            sns.scatterplot(num_features.iloc[:,i], self.data.SalePrice)
        plt.tight_layout()
        plt.show()
    
    def heatmap_all(self, featuretype):
        '''Plot heatmap for all data features correlation'''
        self.featuretype = featuretype
        sns.set(font_scale=1.1)
        self.data = self.data.astype(float)
        correlation_train = self.data.corr()
        mask = np.triu(correlation_train.corr())
        plt.figure(figsize=(50, 70))
        plt.title('Correlation of ' + self.featuretype, size=40)
        sns.heatmap(correlation_train,
                    annot=True,
                    fmt='.1f',
                    cmap='coolwarm',
                    square=True,
                    mask=mask,
                    linewidths=1,
                    cbar=False)
        plt.show()
        
    def heatmap_num(self, datatype):
        '''Plot heatmap for numerical features correlation'''
        self.datatype = datatype
        sns.set(font_scale=1.1)
        correlation_train = self.data.corr()
        mask = np.triu(correlation_train.corr())
        plt.figure(figsize=(18, 20))
        plt.title('Correlation of ' + self.datatype, size=20)
        sns.heatmap(correlation_train,
                    annot=True,
                    fmt='.1f',
                    cmap='coolwarm',
                    square=True,
                    mask=mask,
                    linewidths=1,
                    cbar=False)
        plt.show()
    
    def normalize(self, cols):
        self.cols= cols
        for col in self.cols:
            self.data[col +'norm'] = preprocessing.Normalizer(norm='max').transform([self.data[col]])[0]
            
    def data_drop(self, train, test, cols):
        self.train= train
        self.test= test
        self.cols= cols
        for df in [self.train, self.test]:
            df.drop(columns = df[self.cols], inplace = True)
    
    def cat_features_plot(self, col):
        '''Plot to review relationship between independent variables and target'''
        self.col = col
        fig, axes = plt.subplots(10, 2, figsize=(25, 80))
        axes = axes.flatten()

        for i, j in zip(self.data.select_dtypes(include=['object']).columns, axes):
            sortd = self.data.groupby([i])[col].median().sort_values(ascending=False)
            sns.boxplot(x=i,
                        y=col,
                        data=self.data,
                        palette='plasma',
                        order=sortd.index,
                        ax=j)
            locator=MaxNLocator(prune='both', nbins=18)
            j.tick_params(labelrotation=45)
            j.yaxis.set_major_locator(locator)

            plt.tight_layout() 
    
    def heatmap(self, featuretype):
        '''Plot heatmap for categorical features'''
        self.featuretype = featuretype
        sns.set(font_scale=1.1)
        self.data = self.data.astype(float)
        correlation_train = self.data.corr()
        mask = np.triu(correlation_train.corr())
        plt.figure(figsize=(18, 20))
        plt.title('Correlation of ' + self.featuretype, size=20)
        sns.heatmap(correlation_train,
                    annot=True,
                    fmt='.1f',
                    cmap='coolwarm',
                    square=True,
                    mask=mask,
                    linewidths=1,
                    cbar=False)
        plt.show()
     
    def num_features_distplot(self, datadesc):
        '''Inspect all numerical feature for skewness'''
        self.datadesc = datadesc
        num_features = self.data.select_dtypes(exclude='object').copy()
        fig = plt.figure(figsize=(12,18))
        for i in range(len(num_features.columns)):
            fig.add_subplot(5,4,i+1)
            sns.distplot(num_features.iloc[:,i].dropna())
            plt.xlabel(num_features.columns[i])
        plt.tight_layout()
        fig.suptitle('Distribution Plot for ' + self.datadesc + ' Features', fontsize=15)
        fig.subplots_adjust(top=0.96)
        plt.show()
        
   
    def dist_plot(self, titles):
        '''Plotting the target variable to inspect for skewness'''
        self.titles = titles
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(15, 5)
        sns.distplot(self.data, ax=ax[0])
        sns.distplot((np.log(self.data.round(decimals=2))), ax=ax[1])
        ax[0].title.set_text('Distribution of '+ self.titles)
        ax[1].title.set_text('Distribution of Log-Transformed ' + self.titles)
        fig.show()
        print(self.titles + ' has a skew of ' + str(self.data.skew().round(decimals=2)) + 
        ' while the log-transformed ' + self.titles + ' improves the skew to ' + 
        str(np.log(self.data).skew().round(decimals=2)))
    
    def corr_list(self):
        '''Plotting heatmap to show correlation of features'''
        correlation= self.data.corr()
        correlation['SalePrice'].sort_values(ascending=False).head(20)
 
class model():
    def __init__(self, A, AB, C, CD):
        self.A = A
        self.AB = AB
        self.C = C
        self.CD = CD
         
   
    def model_run(self, model_list, title):
        self.model_list = model_list
        self.title = title
        self.model_preds = []
        self.model_names = []
      
        for model_name, model in (self.model_list):
            model_n = model
            model_n.fit(self.A, self.C)
            model_pred = model_n.predict(self.AB)
            model_pred_mae = mae(model_pred,self.CD)
            self.model_preds.append(model_pred_mae)
            self.model_names.append(model_name)
            output = "%s: %f" % (model_name, model_pred_mae)
            print(output)
        fig, ax = plt.subplots(figsize=(15,7))
        plt.style.use('dark_background')
        plt.title(self.title + ' Models Scores comparison', size=20)
        plt.ylabel('MAE_scores', fontsize=15, fontweight='bold')
        plt.xlabel('models_eval', fontsize=15, fontweight='bold')
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        ax.bar(self.model_names, self.model_preds)
        plt.show()
            
    def tunedmodel_run(self, model_list):
        self.model_list = model_list
      
        for model_name, model in (self.model_list):
            model_n = model
            model_n.fit(self.A, self.C)
            model_pred = model_n.predict(self.AB)
            model_pred_mae = mae((np.exp(model_pred)),(np.exp(self.CD)))
            
            output = "%s: %f" % (model_name, model_pred_mae)
            print(output)
            
    def models_scores_plot(self, scores, names, title):
        self.title = title
        self.scores = scores
        self.names = names
        fig, ax = plt.subplots(figsize=(15,7))
        plt.style.use('dark_background')
        plt.title(self.title + ' Models Scores comparison', size=20)
        plt.ylabel('MAE_scores', fontsize=15, fontweight='bold')
        plt.xlabel('models_eval', fontsize=15, fontweight='bold')
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        ax.bar(self.names, self.scores)
        plt.show()
    
    def cross_fold_val(self, model_list):
        self.model_list = model_list
        self.avg_scores = []
        self.std_dev = []
        self.model_names = []
        for model_name, model in tqdm(self.model_list):
            score = cvs(model, self.A, self.C, cv=5, scoring='neg_mean_absolute_error')
            scores = abs(score) # MAE scoring is negative in cross_val_score
            avg_score = np.mean(scores)
            std = np.std(scores)
            self.avg_scores.append(avg_score)
            self.std_dev.append(std) 
            self.model_names.append(model_name)
            output = "%s: %f (%f)" % (model_name, avg_score, std)
            print(output)
        fig, ax = plt.subplots(figsize=(15,7))
        plt.style.use('dark_background')
        plt.title(' Models with Cross Validation Scores comparison', size=20)
        plt.ylabel('Avg_scores', fontsize=15, fontweight='bold')
        plt.xlabel('model_list', fontsize=15, fontweight='bold')
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        ax.bar(self.model_names, self.avg_scores)
        plt.show()
        
    def feat_importance(self, model):
        self.model = model
        if hasattr(bestModel, 'feature_importances_'):
            importances = bestModel.feature_importances_

        else:
            # for linear models which don't have feature_importances_
            importances = [0]*len(train_df1_X.columns)

        feature_importances = pd.DataFrame({'feature':train_df1_X.columns, 'importance':importances})
        feature_importances.sort_values(by='importance', ascending=False, inplace=True)
    
        # set index to 'feature'
        feature_importances.set_index('feature', inplace=True, drop=True)
    
        # create plot
        feature_importances[0:25].plot.bar(figsize=(20,10))
        plt.show()
    