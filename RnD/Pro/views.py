from django.shortcuts import render,redirect
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler
from .forms import MediaForm

def data_cleaning():
    df = pd.read_csv('media/AD.csv')
    df = df.drop(['Subject ID','MRI ID', 'Visit', 'Hand'], axis=1)
    duplicate_rows_df = df[df.duplicated()]
    print("number of duplicate rows: ", duplicate_rows_df.shape)
    df = df.drop_duplicates()
    return df

def ad(request):
    df = data_cleaning()
    return render(request,'ad.html')

def visualize_data(request):
    df1 = data_cleaning()
    df = data_cleaning()
    sns.set_style('whitegrid')
    sns.countplot(x='Group',data=df1,palette='rainbow')
    plt.savefig('media/snsimage.png')
    df['Group'] = df['Group'].replace(['Converted'], ['Demented'])
    sns.set_style('whitegrid')
    sns.countplot(x='Group',data=df,palette='rainbow')
    plt.savefig('media/snsimage1.png')
    return render(request,'visualize.html')

def home(request):
    form = MediaForm()    
    if request.method == 'POST':
        form = MediaForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('home')
    context={'form':form}
    return render(request,'home.html',context)
