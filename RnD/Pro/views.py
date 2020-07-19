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
from .forms import ClientForm
from django.core.files import File
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from .forms import ClientForm,MediaForm
from sklearn.linear_model import LogisticRegression

def data_cleaning():
    df = pd.read_csv('media/AD.csv')
    df = df.drop(['Subject ID','MRI ID', 'Visit', 'Hand','MR Delay'], axis=1)
    duplicate_rows_df = df[df.duplicated()]
    print("number of duplicate rows: ", duplicate_rows_df.shape)
    df = df.drop_duplicates()
    df['Group'] = df['Group'].replace(['Converted'], ['Demented'])
    return df
    
def serverside():
    df = data_cleaning()
    for i in range(df['M/F'].size):
        if df['M/F'][i]=='M':
            df['M/F'][i]= 0
        elif df['M/F'][i]=='F':
            df['M/F'][i] = 1 
    df['Group'] = df['Group'].replace(['Demented', 'Nondemented'], [1,0]) 

    df["SES"].fillna(df.groupby("EDUC")["SES"].transform("median"), inplace=True)
    
    df=df.dropna(axis=0, how='any')
    fit = bfeature(df)
    dfscores = pd.DataFrame(fit.scores_)
    X=df.iloc[:,1:11]
    dfcolumns = pd.DataFrame(X.columns)
    featureScores = pd.concat([dfcolumns,dfscores],axis = 1)
    featureScores.columns = ['Specs','Scores']
    fss=featureScores.nlargest(5,'Scores')
    fssi=fss.reset_index()
    fs=fssi.Specs
    return [fs,df]

def ad(request):
    if request.method == 'POST':
        ssl = serverside()
        list_pred = []
        for i in ssl[0]:
            x = request.POST.get(i)
            list_pred.append(float(x))
        list_pred = np.array(list_pred)    
        temp=[]
        temp.append(np.array(list_pred))
        listT = train(ssl[1],ssl[0],temp)
        answer = LogReg(listT)
        if answer[0] == 1:
            context = "You are predicted to be a demented person"
        if answer[0] == 0:
            context = "Your brain status is fine"
        return render(request,'ad.html',{'context':context})
    else:
        return render(request,'ad.html')    

def train(df,fs,list_test):
    X=df[fs].values
    Y=df['Group'].values
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25)
    #Feature_scaling
    sc_X=StandardScaler()
    X_train=sc_X.fit_transform(X_train)
    X_test=sc_X.transform(list_test) 
    lisT = [X_train,X_test,Y_train]
    return lisT
    

def bfeature(df):
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
            return redirect('index')
    context={'form':form}
    return render(request,'home.html',context)

def visualize_data(request):
    return render(request,'visualize.html')

def LogReg(listT):
    classifier =LogisticRegression()
    classifier.fit(listT[0],listT[2])    
    y_pred=classifier.predict(listT[1])
    return y_pred

def signup(request):
    if request.method == 'POST':
        cli = ClientForm(request.POST)
        if cli.is_valid():
            user = User.objects.create_user(username=cli.cleaned_data['User_Name'],
                                            password=cli.cleaned_data['Password'],
                                            email=cli.cleaned_data['Email'])

            user.save()
            cli.save()
            return redirect('signin')
    else:
        cli = ClientForm()
    return render(request,'signup.html',{'form':cli})

def signin(request):
    if request.method == 'POST':
        user = User()
        username = request.POST['user']
        password = request.POST['pass']
        user = authenticate(username=username, password=password)
        context = {'user':request.user}
        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            return render(request,'login.html',context)
    else:
        return render(request,'login.html')

def index(request):
    return render(request,'index.html')