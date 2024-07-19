# for demonstration purpose, tabular data cleaning, feature selection, pass them through ML model to do classification , contents related to actual analysis are removed, generic code that can be adapted to any conditions
# install packages
!pip install shap
!pip install seaborn
# import packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import glob
import os

# load label
df_tau = pd.read_csv('/home/sagemaker-user/xxx.csv')
label = df_tau[['RID','tau_status']]

# decide which table to use for prediction
df_category = pd.read_csv('/home/sagemaker-user/yyy.csv')
df_category = df_category[df_category.Type.isin([......])]

# Load each CSV file into a unique DataFrame
for file in df_category.Name:
    # Extract the unique part of the filename to use as the key
    key = file#os.path.basename(file).split('.')[0]
    # Load the CSV file into a DataFrame
    df = pd.read_csv('/home/sagemaker-user/zzz/adnimerge_'+file+'.csv', on_bad_lines='skip')

    # Store the DataFrame in the dictionary
    dataframes[key] = (df, df_category.Type.iloc[np.argwhere(df_category.Name == key)[0][0]])

# combine data from different category
types = list(np.unique(df_category.Type))
u_types = np.copy(types)
for i in range(len(u_types)):
    types.append('no_' + u_types[i]) 
    for j in range(i+1,len(u_types)):
        types.append(u_types[i] + '_' + u_types[j])
        for k in range(j+1,len(u_types)):
            # print(i,j,k)
            types.append(u_types[i] + '_' + u_types[j] + '_' + u_types[k])
types.append('all')
merged_dfs = {}
for c in types:
    merged_dfs[c] = None
    print(c)

for key, df in dataframes.items():
    print(key)
    df[0].loc[(df[0].VISCODE=='sc')|(df[0].VISCODE=='scmri'),'VISCODE'] = 'bl'
    if merged_dfs[df[1]] is None:
        merged_dfs[df[1]] = df[0]
    else:
        # merged_df = pd.merge(merged_df, df, on=['RID','VISCODE'],
        #                      how='outer',suffixes=('', '_dup'))
        merged_dfs[df[1]] = pd.merge(merged_dfs[df[1]], df[0], on=['RID','VISCODE'],
                             how='outer',suffixes=('', key))
    if merged_dfs['all'] is None:
        merged_dfs['all'] = df[0]
    else:
        merged_dfs['all'] = pd.merge(merged_dfs['all'], df[0], on=['RID','VISCODE'],
                             how='outer',suffixes=('', key))
        
    for i in range(len(u_types)):
        nm = 'no_' + u_types[i]
        if df[1] !=u_types[i]:
            if merged_dfs[nm] is None:
                merged_dfs[nm] = df[0]
            else:
                merged_dfs[nm] = pd.merge(merged_dfs[nm], df[0], on=['RID','VISCODE'],
                                 how='outer',suffixes=('', key))
        for j in range(i+1,len(u_types)):
            nm = u_types[i] + '_' + u_types[j]
            if df[1] in u_types[[i,j]]:
                if merged_dfs[nm] is None:
                    merged_dfs[nm] = df[0]
                else:
                    merged_dfs[nm] = pd.merge(merged_dfs[nm], df[0], on=['RID','VISCODE'],
                                 how='outer',suffixes=('', key))
            for k in range(j+1,len(u_types)):
                # print(i,j,k)
                nm = u_types[i] + '_' + u_types[j] + '_' + u_types[k]
                if df[1] in u_types[[i,j,k]]:
                    if merged_dfs[nm] is None:
                        merged_dfs[nm] = df[0]
                    else:
                        merged_dfs[nm] = pd.merge(merged_dfs[nm], df[0], on=['RID','VISCODE'],
                                     how='outer',suffixes=('', key))
    

for c in types:
    merged_dfs[c] = merged_dfs[c].drop_duplicates().reset_index(drop=True)
    print(merged_dfs[c].shape)

# data cleaning
import re
from numpy import nanmean
df_merge_taus = {}
df_merge_tau_cleaneds = {}
df_merge_tau_bls = {}
df_merge_tau_bl_preNAfill_cleaneds = {}
df_merge_tau_bl_cleaneds = {}

nancounts = {}
for c in types:
    df_merge_taus[c] = merged_dfs[c][(merged_dfs[c].RID.isin(label.RID))]
    df_merge_taus[c] = df_merge_taus[c].merge(label[['RID','tau_status']].drop_duplicates(), on='RID', how='left')

    df_merge_tau_cleaneds[c] = df_merge_taus[c]
    # drop nonsense parameter
    ps = ['APTIME*','LONIUI[D+][_*]*','RI[D+][_*]*','Unnamed: 0*','ORIGPROT*','VISCODE*','EXAMDATE*','STUDYI[D+][_*]*','SITEI[D+][_*]*',
         'USERDATE*','COLPROT*','SUBJECT*','VI[D+][_*]*','VISITNUM*','DIAGNOSIS*','DATE*']
    for co in df_merge_tau_cleaneds[c].columns:
        rm = 0
        for p in ps:
            if re.match(p,co):
                rm = 1
                m = p
        if rm == 1:
            df_merge_tau_cleaneds[c] = df_merge_tau_cleaneds[c].drop(co,axis = 1)

    df_merge_tau_bls[c] = df_merge_taus[c][df_merge_taus[c].VISCODE.isin(['bl','sc','scmri'])]
    df_merge_tau_bl_preNAfill_cleaneds[c] = df_merge_tau_bls[c]
   
    for co in df_merge_tau_bl_preNAfill_cleaneds[c].columns:
        rm = 0
        for p in ps:
            if re.match(p,co):
                rm = 1
                m = p
        if rm == 1:
            df_merge_tau_bl_preNAfill_cleaneds[c] = df_merge_tau_bl_preNAfill_cleaneds[c].drop(co,axis = 1)
            # print(m,c, df_merge_tau_bl_cleaneds[c].shape)
    # print(df_merge_tau_cleaneds[c].shape)
    nancounts[c] = df_merge_tau_bl_preNAfill_cleaneds[c].notna().sum().sort_values(ascending=False)
    
    print(c)
    print(merged_dfs[c].shape)
    print(df_merge_taus[c].shape)
    print(df_merge_tau_bls[c].shape)

    df_merge_tau_bls[c] = df_merge_tau_bls[c].drop_duplicates(keep='first').\
        sort_values(by=['RID', 'VISCODE'],ascending=True).reset_index(drop=True)
    df_merge_tau_bls[c] = df_merge_tau_bls[c].drop_duplicates(subset=['RID', 'VISCODE'], keep='first').reset_index(drop=True)
    
    print(df_merge_tau_bls[c].shape)
    df_merge_tau_bl_cleaneds[c] = df_merge_tau_bls[c] 
    for co in df_merge_tau_bl_cleaneds[c].columns:
        rm = 0
        for p in ps:
            if re.match(p,co):
                rm = 1
                m = p
        if rm == 1:
            df_merge_tau_bl_cleaneds[c] = df_merge_tau_bl_cleaneds[c].drop(co,axis = 1)
    print(df_merge_tau_bl_cleaneds[c].shape)

# make a for loop, select top 10 predictors in each category
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import shap
results = {}
for i in range(len(u_types)):
    which = u_types[i]
    missingness_thresh = 0.9
    
    X = df_merge_tau_bl_cleaneds[which][nancounts[which][nancounts[which]>len(df_merge_tau_bl_cleaneds[which])*missingness_thresh].index]
    X = pd.get_dummies(X)

    y = X.tau_status.to_numpy()
    X = X.drop('tau_status',axis=1)
    feat_names = X.columns.to_list()
    X = X.to_numpy()
    
    # Initialize the classifier
    classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    
    # Initialize StratifiedKFold
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)#42)
    
    # List to store AUC scores for each fold
    auc_scores = []
    # List to store SHAP values for each fold
    shap_values_mat = np.zeros([X.shape[0],X.shape[1]])
    
    # Perform 10-fold cross-validation
    for train_index, test_index in tqdm(kfold.split(X, y), total=kfold.get_n_splits(), desc="k-fold"):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        # Train the classifier
        classifier.fit(X_train, y_train)
    
        # Predict probabilities
        y_pred_prob = classifier.predict_proba(X_test)
    
        # Calculate AUC for each class and average them
        auc = roc_auc_score(y_test, y_pred_prob[:,1])
        auc_scores.append(auc)
    
        # Calculate SHAP values
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_test)
        shap_values_mat[test_index,:] = shap_values
    
    # Calculate the mean AUC score across all folds
    mean_auc = np.mean(auc_scores)
    print(f'{which}: Mean AUC score across 10 folds: {mean_auc}')

    tmp = np.mean(np.abs(shap_values_mat),axis=0)

    idx = np.argsort(-tmp)

    results[which] = np.array(feat_names)[idx[:10]]

for i in range(len(u_types)):
    nm = 'no_' + u_types[i]
    for t in u_types:
        if t != u_types[i]:
            if nm not in results.keys():
                results[nm] = np.copy(results[t])
            else:
                results[nm] = np.concatenate((results[nm],results[t]))
    for j in range(i+1,len(u_types)):
        nm = u_types[i] + '_' + u_types[j]
        results[nm] = np.concatenate((results[u_types[i]],results[u_types[j]]))
        for k in range(j+1,len(u_types)):
            # print(i,j,k)
            nm = u_types[i] + '_' + u_types[j] + '_' + u_types[k]
            results[nm] = np.concatenate((results[u_types[i]],results[u_types[j]],results[u_types[k]]))

nm = 'all'
for t in u_types:
    if nm not in results.keys():
        results[nm] = np.copy(results[t])
    else:
        results[nm] = np.concatenate((results[nm],results[t]))


# fit model with the top 10 predictors
from sklearn.metrics import roc_curve
# use top 10 predictors of each category
output_dir = './tauPrediction'
os.makedirs(output_dir,exist_ok=True)
results_top = {'predictor name':[],'dimension post-dummy':[],'AUC':[]}
predictor_top = {}
for i in [0]:#range(len(types)):
    which = 'no_structural'#types[i]
    # which = types[i]
    results_top['predictor name'].append(which)
    missingness_thresh = 0.9 
    
    X = df_merge_tau_bl_cleaneds[which]#[nancounts[which][nancounts[which]>len(df_merge_tau_bl_cleaneds[which])*missingness_thresh].index]
    X = pd.get_dummies(X)

    y = X.tau_status.to_numpy()
    X = X.drop('tau_status',axis=1)
    
    # X = X[results_top10AfterRemoveTop10[which]]
    X = X[results[which]]
    feat_names = X.columns.to_list()
    results_top['dimension post-dummy'].append(X.shape)
    X = X.to_numpy()
    
    # Initialize the classifier
    classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    
    # Initialize StratifiedKFold
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)#42)
    
    # List to store AUC scores for each fold
    auc_scores = []
    y_tests = []
    y_pred_probs = []
    fprs = []
    tprs = []
    # List to store SHAP values for each fold
    shap_values_mat = np.zeros([X.shape[0],X.shape[1]])
    
    # Perform 10-fold cross-validation
    for train_index, test_index in tqdm(kfold.split(X, y), total=kfold.get_n_splits(), desc="k-fold"):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        # Train the classifier
        classifier.fit(X_train, y_train)
    
        # Predict probabilities
        y_pred_prob = classifier.predict_proba(X_test)
    
        # Calculate AUC for each class and average them
        auc = roc_auc_score(y_test, y_pred_prob[:,1])
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])
        auc_scores.append(auc)
        fprs.append(fpr)
        tprs.append(tpr)
        y_tests.append(y_test)
        y_pred_probs.append(y_pred_prob[:,1])
    
        # Calculate SHAP values
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_test)
        shap_values_mat[test_index,:] = shap_values
    # Calculate the mean AUC score across all folds
    mean_auc = np.mean(auc_scores)
    results_top['AUC'].append(np.round(mean_auc,4))
    print(f'{which}: Mean AUC score across 10 folds: {mean_auc}')
    
    tmp = np.mean(np.abs(shap_values_mat),axis=0)

    idx = np.argsort(-tmp)

    predictor_top[which] = np.array(feat_names)[idx[:50]]


# plot AUC 

import matplotlib.pyplot as plt

from sklearn.metrics import RocCurveDisplay,auc

mean_fpr = np.linspace(0, 1, 100)
interp_tprs = []
fig, ax = plt.subplots(figsize=(6, 6))
for i in range(len(tprs)):
    interp_tpr = np.interp(mean_fpr, fprs[i], tprs[i])
    interp_tpr[0] = 0.0
    interp_tprs.append(interp_tpr)

    ax.plot(
    fprs[i],
    tprs[i],
    lw=2,
    alpha=0.2,
    )
plt.xlim([0,1])
plt.ylim([0,1])
_ = ax.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title="Prediction of high Tau burden",
)
mean_tpr = np.array(interp_tprs).mean(axis = 0)

mean_auc = auc(mean_fpr, mean_tpr)

ax.plot(
mean_fpr,
mean_tpr,
lw=2,
alpha=0.9,
color = 'black',
label = 'mean ROC (AUC = '+ str(np.round(mean_auc,2))+')'
)
ax.plot([0,1],[0,1],color = 'gray', linestyle='dashed')

std_tpr = np.std(np.array(interp_tprs), axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

plt.legend()

# model explanation
feat_names_new = np.copy(feat_names)
for i in range(len(feat_names_new)):
    if feat_names_new[i] in nm_nofill.keys():
        feat_names_new[i] = nm_nofill[feat_names_new[i]]
# fn = [n[:10] for n in feat_names]
shap.summary_plot(shap_values_mat, X, feature_names=feat_names,show=False,max_display=10)
shap_values_mat.shape