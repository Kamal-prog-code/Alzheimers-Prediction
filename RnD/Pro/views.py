from django.shortcuts import render,redirect
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns
from .models import *
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler
from .forms import MediaForm
from tkinter import *
from django.core.files import File
import pathlib3x as pathlib

def fresh():
    df = pd.read_csv('media/AD.csv')
    my_dir_to_delete = pathlib.Path('static/imgs')
    my_dir_to_delete.rmtree(ignore_errors=True)
    pathlib.Path('static/imgs').mkdir(parents=True, exist_ok=True)
    return df

def data_cleaning():
    df = fresh()
    df = df.drop(['Subject ID','MRI ID', 'Visit', 'Hand'], axis=1)
    duplicate_rows_df = df[df.duplicated()]
    print("number of duplicate rows: ", duplicate_rows_df.shape)
    df = df.drop_duplicates()
    df['Group'] = df['Group'].replace(['Converted'], ['Demented'])
    return df
    
def ad(request):
    df = data_cleaning()
    for i in range(df['M/F'].size):
        if df['M/F'][i]=='M':
            df['M/F'][i]= 0
        elif df['M/F'][i]=='F':
            df['M/F'][i] = 1 
    df['Group'] = df['Group'].replace(['Demented', 'Nondemented'], [1,0]) 
    sns.set_style('whitegrid')
    sns.countplot(x='Group',data=df,palette='rainbow')
    plt.savefig('static/imgs/snsimage.png')
    sns.heatmap(df.isnull(),yticklabels=False,cmap='viridis')
    plt.savefig('static/imgs/snsimage1.png')
    
    x = df['EDUC']
    y = df['SES']

    ses_not_null_index = y[~y.isnull()].index
    x = x[ses_not_null_index]
    y = y[ses_not_null_index]
    plt.plot(x,y,'o')
    plt.savefig('static/imgs/x1.png')
    z = np.polyfit(x, y, 1) #y = mx^1 +c
    p = np.poly1d(z) # slope, intercept
    plt.plot(x, y, 'go')
    plt.plot(x, p(x), "b--")
    plt.xlabel('Education Level(EDUC)')
    plt.ylabel('Social Economic Status(SES)')
    plt.savefig('static/imgs/SesEduc.png')

    df["SES"].fillna(df.groupby("EDUC")["SES"].transform("median"), inplace=True)
    
    df=df.dropna(axis=0, how='any')
    fit = feature(df)
    dfscores = pd.DataFrame(fit.scores_)
    X=df.iloc[:,1:11]
    dfcolumns = pd.DataFrame(X.columns)
    featureScores = pd.concat([dfcolumns,dfscores],axis = 1)
    featureScores.columns = ['Specs','Scores']
    featureScores.nlargest(5,'Scores').plot(kind='bar')
    plt.savefig('static/imgs/featureScores.png')
    return render(request,'ad.html')

def feature(df):
    X=df.iloc[:,1:11]
    Y=df.iloc[:,0]
    bestfeatures = SelectKBest(score_func=chi2, k=5)
    fit = bestfeatures.fit(X,Y)
    return fit

def home(request):
    form = MediaForm()    
    if request.method == 'POST':
        form = MediaForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('home')
    context={'form':form}
    return render(request,'home.html',context)

def visualize_data(request):
    return render(request,'visualize.html')