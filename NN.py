'''
This program aims to train neuron network in order to estimate the snow 
water  equivalent (swe) over land using microwave brightness temperatures
as input.  It also allows you to calculate performance indices and 
visualize model sensitivity analysis figures. 

======= Important note ===========
You have to work under the google colab environment, otherwise you 
have to adapt the program for your own environment. 
==================================

Created  in Feb, Mar, Avr, Mai, Jun and Jul 2021
@auteur: hamichaabani@gmail.com (CHAABANI Hamid)
'''

# Import libraries 
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt 
from sklearn import metrics
import pandas as pd
import numpy as np 
import os
import io 
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import regularizers
import random
from keras.layers import Dropout 
import sklearn
import sklearn.model_selection
from google.colab import drive
import seaborn as sns
import scipy.stats
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from scipy.stats import pearsonr              
import netCDF4 as nc        
from netCDF4 import Dataset as NetCDFFile                          
import xarray as xr
from sklearn import *
drive.mount('/content/drive')
path = "/content/drive/MyDrive/data"  




# Import data into pandas form 
df1 = pd.read_csv('/content/drive/MyDrive/data/amsr_era5_daymean_20181101.csv',  header=None, delim_whitespace=True, low_memory=False) #Import csv file into dataframe df2      
df1.columns=['lon','lat','rsn', 'sd', 'tb6v','tb6h','tb7v','tb7h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89v','tb89h'] # Rename columns 

df2 = pd.read_csv('/content/drive/MyDrive/data/amsr_era5_daymean_20181110.csv',  header=None, delim_whitespace=True, low_memory=False) #Import csv file into dataframe df2      
df2.columns=['lon','lat', 'rsn', 'sd', 'tb6v','tb6h','tb7v','tb7h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89v','tb89h'] # Rename columns 

df3 = pd.read_csv('/content/drive/MyDrive/data/amsr_era5_daymean_20181130.csv',  header=None, delim_whitespace=True, low_memory=False) #Import csv file into dataframe df2      
df3.columns=['lon','lat','rsn', 'sd', 'tb6v','tb6h','tb7v','tb7h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89v','tb89h'] # Rename columns 

df4 = pd.read_csv('/content/drive/MyDrive/data/amsr_era5_daymean_20181201.csv',  header=None, delim_whitespace=True, low_memory=False) #Import csv file into dataframe df2      
df4.columns=['lon','lat', 'rsn', 'sd', 'tb6v','tb6h','tb7v','tb7h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89v','tb89h'] # Rename columns 

df5 = pd.read_csv('/content/drive/MyDrive/data/amsr_era5_daymean_20181210.csv',  header=None, delim_whitespace=True, low_memory=False) #Import csv file into dataframe df2      
df5.columns=['lon','lat', 'rsn', 'sd', 'tb6v','tb6h','tb7v','tb7h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89v','tb89h'] # Rename columns 

df6 = pd.read_csv('/content/drive/MyDrive/data/amsr_era5_daymean_20181230.csv',  header=None, delim_whitespace=True, low_memory=False) #Import csv file into dataframe df2      
df6.columns=['lon','lat','rsn', 'sd', 'tb6v','tb6h','tb7v','tb7h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89v','tb89h'] # Rename columns 

df7 = pd.read_csv('/content/drive/MyDrive/data/amsr_era5_daymean_20190101.csv',  header=None, delim_whitespace=True, low_memory=False) #Import csv file into dataframe df2      
df7.columns=['lon','lat','rsn', 'sd', 'tb6v','tb6h','tb7v','tb7h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89v','tb89h'] # Rename columns 

df8 = pd.read_csv('/content/drive/MyDrive/data/amsr_era5_daymean_20190110.csv',  header=None, delim_whitespace=True, low_memory=False) #Import csv file into dataframe df2      
df8.columns=['lon','lat','rsn', 'sd', 'tb6v','tb6h','tb7v','tb7h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89v','tb89h'] # Rename columns 

df9 = pd.read_csv('/content/drive/MyDrive/data/amsr_era5_daymean_20190130.csv',  header=None, delim_whitespace=True, low_memory=False) #Import csv file into dataframe df2      
df9.columns=['lon','lat','rsn', 'sd', 'tb6v','tb6h','tb7v','tb7h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89v','tb89h'] # Rename columns 

df10 = pd.read_csv('/content/drive/MyDrive/data/amsr_era5_daymean_20190201.csv',  header=None, delim_whitespace=True, low_memory=False) #Import csv file into dataframe df2      
df10.columns=['lon','lat', 'rsn','sd', 'tb6v','tb6h','tb7v','tb7h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89v','tb89h'] # Rename columns 

df11 = pd.read_csv('/content/drive/MyDrive/data/amsr_era5_daymean_20190210.csv',  header=None, delim_whitespace=True, low_memory=False) #Import csv file into dataframe df2      
df11.columns=['lon','lat', 'rsn', 'sd', 'tb6v','tb6h','tb7v','tb7h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89v','tb89h'] # Rename columns 

df12 = pd.read_csv('/content/drive/MyDrive/data/amsr_era5_daymean_20190220.csv',  header=None, delim_whitespace=True, low_memory=False) #Import csv file into dataframe df2      
df12.columns=['lon','lat', 'rsn', 'sd', 'tb6v','tb6h','tb7v','tb7h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89v','tb89h'] # Rename columns 

df13 = pd.read_csv('/content/drive/MyDrive/data/amsr_era5_daymean_20190228.csv',  header=None, delim_whitespace=True, low_memory=False) #Import csv file into dataframe df2      
df13.columns=['lon','lat', 'rsn', 'sd', 'tb6v','tb6h','tb7v','tb7h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89v','tb89h'] # Rename columns 

df14 = pd.read_csv('/content/drive/MyDrive/data/amsr_era5_daymean_20190301.csv',  header=None, delim_whitespace=True, low_memory=False) #Import csv file into dataframe df2      
df14.columns=['lon','lat', 'rsn', 'sd', 'tb6v','tb6h','tb7v','tb7h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89v','tb89h'] # Rename columns 

df15 = pd.read_csv('/content/drive/MyDrive/data/amsr_era5_daymean_20190310.csv',  header=None, delim_whitespace=True, low_memory=False) #Import csv file into dataframe df2      
df15.columns=['lon','lat', 'rsn', 'sd', 'tb6v','tb6h','tb7v','tb7h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89v','tb89h'] # Rename columns 

df16 = pd.read_csv('/content/drive/MyDrive/data/amsr_era5_daymean_20190330.csv',  header=None, delim_whitespace=True, low_memory=False) #Import csv file into dataframe df2      
df16.columns=['lon','lat', 'rsn', 'sd', 'tb6v','tb6h','tb7v','tb7h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89v','tb89h'] # Rename columns 

df17 = pd.read_csv('/content/drive/MyDrive/data/amsr_era5_daymean_20181120.csv',  header=None, delim_whitespace=True, low_memory=False) #Import csv file into dataframe df2      
df17.columns=['lon','lat', 'rsn', 'sd', 'tb6v','tb6h','tb7v','tb7h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89v','tb89h'] # Rename columns 

df18 = pd.read_csv('/content/drive/MyDrive/data/amsr_era5_daymean_20181220.csv',  header=None, delim_whitespace=True, low_memory=False) #Import csv file into dataframe df2      
df18.columns=['lon','lat', 'rsn', 'sd', 'tb6v','tb6h','tb7v','tb7h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89v','tb89h'] # Rename columns 

df19 = pd.read_csv('/content/drive/MyDrive/data/amsr_era5_daymean_20190120.csv',  header=None, delim_whitespace=True, low_memory=False) #Import csv file into dataframe df2      
df19.columns=['lon','lat', 'rsn', 'sd', 'tb6v','tb6h','tb7v','tb7h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89v','tb89h'] # Rename columns 

df20 = pd.read_csv('/content/drive/MyDrive/data/amsr_era5_daymean_20190220.csv',  header=None, delim_whitespace=True, low_memory=False) #Import csv file into dataframe df2      
df20.columns=['lon','lat', 'rsn', 'sd', 'tb6v','tb6h','tb7v','tb7h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89v','tb89h'] # Rename columns 

df21 = pd.read_csv('/content/drive/MyDrive/data/amsr_era5_daymean_20190320.csv',  header=None, delim_whitespace=True, low_memory=False) #Import csv file into dataframe df2      
df21.columns=['lon','lat', 'rsn', 'sd', 'tb6v','tb6h','tb7v','tb7h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89v','tb89h'] # Rename columns 

# Cancatenate dataframes by lines
df=pd.concat([ df2, df3, df4, df5, df6, df8, df9, df10, df11, df13, df14, df15, df16, df17, df18, df19, df20])
# df1, df7, df21 are conserved for the test 

# Rename the columns
df.columns=['lon','lat', 'rsn', 'swe', 'tb6v','tb6h','tb7v','tb7h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89v','tb89h'] # Rename columns 

# We can drop some useless columns
# df.drop(columns=['TB_6vA', 'TB_6hA','TB_7vA','TB_7hA','TB_89vA','TB_89hA'],inplace=True)

# Drop rows which ERA5 swe is very small or some important snow accumulations
df.drop(df[df['swe'] < 0.001].index, inplace=True)                       
df.drop(df[df['swe'] > 0.2].index, inplace=True)

# Drop rows which one of the Tbs is small than 340, in other words, the erroneous values of Tbs
df.drop(df[df['tb6v'] > 340].index, inplace=True)
df.drop(df[df['tb7v'] > 340].index, inplace=True)
df.drop(df[df['tb10v'] > 340].index, inplace=True)
df.drop(df[df['tb18v'] > 340].index, inplace=True)
df.drop(df[df['tb23v'] > 340].index, inplace=True)
df.drop(df[df['tb36v'] > 340].index, inplace=True)
df.drop(df[df['tb89v'] > 340].index, inplace=True)
df.drop(df[df['tb6h'] > 340].index, inplace=True) 
df.drop(df[df['tb7h'] > 340].index, inplace=True)
df.drop(df[df['tb10h'] > 340].index, inplace=True)
df.drop(df[df['tb18h'] > 340].index, inplace=True)
df.drop(df[df['tb23h'] > 340].index, inplace=True)
df.drop(df[df['tb36h'] > 340].index, inplace=True)
df.drop(df[df['tb89h'] > 340].index, inplace=True)


# Exponential transformation of target 
# df['swe'] = np.exp(df['swe'])

# Split the database into test and training set
train, test = sklearn.model_selection.train_test_split(df, train_size = 0.8)  


x_train = train[['tb6v','tb6h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89v', 'tb89h']]
x_test = test[['tb6v','tb6h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89v', 'tb89h']]

y_train = train['swe']
y_test = test['swe']



# Data normalization with mean and standard deviation
# Function for input normalization
def normalize(x_train, x_test):
  mu = np.mean(x_train, axis=0)
  std = np.std(x_train, axis=0)
  x_train_normalized = (x_train - mu) / std
  x_test_normalized = (x_test - mu) / std
  return x_train_normalized, x_test_normalized
        
x_train, x_test = normalize(x_train, x_test)
                                                                                
# Function for target normalization                                                                            
def normaliz(y_train, y_test):
  mu = np.mean(y_train, axis=0)
  std = np.std(y_train, axis=0)
  x_train_normalized = (y_train - mu) / std
  x_test_normalized = (y_test - mu) / std
  return x_train_normalized, x_test_normalized

#y_train, y_test = normaliz(y_train) 




"""
# Normalization with extremes (min, max)
# Auxiliary function 
def mean_norm(df_input):
    return df_input.apply(lambda x: (x-x.mean())/ x.std(), axis=0)

train[['tb6v','tb6h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89v', 'tb89h', 'swe']] = mean_norm(train[['tb6v','tb6h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89v', 'tb89h', 'swe']])
test[['tb6v','tb6h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89v', 'tb89h', 'swe']] = mean_norm(test[['tb6v','tb6h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89v', 'tb89h', 'swe']])

# Normalization with the extremes values
def mean_norm(df_input):
    return df_input.apply(lambda x: (x-min(x))/ (max(x)-min(x)), axis=0)

df[['tb6v','tb6h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89v', 'tb89h']] = mean_norm(df[['tb6v','tb6h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89v', 'tb89h']])
"""




# Build the model
def get_model(shape):
  model=keras.models.Sequential()
  model.add(keras.layers.Input(shape, name="input layer"))
  
  model.add(keras.layers.Dense(32, activation='relu', name="Dense_n1")) # bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5), kernel_regularizer=regularizers.l1_l2(l1=1e-5),
  #model.add(Dropout(0.6))      # he dropout layer in DL helps reduce overfitting by introducing regularization and generalization capabilities into the model
  model.add(keras.layers.Dense(64, activation='relu', name="Dense_n2")) # bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5), kernel_regularizer=regularizers.l1_l2(l1=1e-5),
  #model.add(Dropout(0.6))
  model.add(keras.layers.Dense(32, activation='relu', name="Dense_n3")) # bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5), kernel_regularizer=regularizers.l1_l2(l1=1e-5),

  model.compile(optimizer='adam',loss='mae', metrics=['mae', 'mse'])     

  return model    
                                                      	
# Train the model                
model=get_model((12,)) # 12 is the number of input 
print(model.summary()) # Summary of model caracteristics   
history=model.fit(x_train, y_train, epochs=100, batch_size = 1, verbose=1, validation_data=(x_test, y_test))
print(history)    


df22=pd.DataFrame(data=history.history)
print(df22)


min(df22['mae']), min(df22['mse'])
min(df22['val_mae']), min(df22['val_mse'])



# Progression of the loss function during training on the training data set and the test data set
x=[i for i in range(1, 501)]
loss=list(df22['loss']) 
val_loss=list(df22['val_loss']) 
fig = plt.figure(figsize=(12,8)) 
plt.style.use('seaborn-deep')         # seaborn-colorblind
p1=plt.plot(x,loss, label='LOSS')
p2=plt.plot(x,val_loss, label='VAL_LOSS')
plt.title("La progression de la fonction de perte en fonction des itérations")  # Problemes avec accents (plot_directive) !
plt.xlabel("Les itérations", size = 16,)
plt.ylabel("La fonction perte", size = 16)
plt.legend()



# Progression of the MAE function during training on the training data set and the test data set
x=[i for i in range(1, 501)]
mae=list(df22['mae']) 
val_mae=list(df22['val_mae']) 
fig = plt.figure(figsize=(12,8)) 
plt.style.use('seaborn-deep')         # seaborn-colorblind
p1=plt.plot(x,mae, label='MAE')
p2=plt.plot(x,val_mae, label='VAL_MAE')
plt.title("La progression de l'erreur absolue moyenne en fonction des itérations")  # Problemes avec accents (plot_directive) !
plt.xlabel("Les itérations", size = 16,)
plt.ylabel("L'erreur absolue moyenne", size = 16)
plt.legend()



# Progression of the MSE function during training on the training data set and the test data set
x=[i for i in range(1, 501)]
mse=list(df22['mse']) 
val_mse=list(df22['val_mse']) 
fig = plt.figure(figsize=(12,8)) 
plt.style.use('seaborn-deep')         # seaborn-colorblind
p1=plt.plot(x,mse, label='MSE')
p2=plt.plot(x,val_mse, label='VAL_MSE')
plt.title("La progression de l'erreur quadratique moyenne en fonction des itérations")  # Problemes avec accents (plot_directive) !
plt.xlabel("Les itérations", size = 16,)
plt.ylabel("L'erreur quadratique moyenne", size = 16)
plt.legend()



# Plot of predicted values in function of ERA5 values of SWE in testing data set
x_train['y_pred'] = model.predict(x_train[['tb6v','tb6h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89h','tb89v']])
x_test['y_pred'] = model.predict(x_test[['tb6v','tb6h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89h','tb89v']])
predicted_swe = list(x_test.y_pred) 
swe = test.swe
                          
#df = pd.DataFrame(data=swe, columns='swe') 
plt.figure(figsize=(22, 12))
idxs = [i for i in range(1, len(swe)+1)]
plt.plot(idxs,predicted_swe,label='Valeurs prédites',lw=0.7  )
plt.plot(idxs,swe,label='Valeurs réelles',lw=0.9 )
plt.xlabel('Index',fontsize=18)
plt.ylabel('SWE',fontsize=16)
plt.legend(fontsize=14)
ax = plt.gca()
ax.set_title(' Variation des valeurs réelles et prévues de la l\'équivalent en eau de la neige sur terre, hémisphère nord', fontsize=16)
plt.setp(ax.get_xticklabels(),fontsize=14)
plt.setp(ax.get_yticklabels(), fontsize=14)  

plt.figure(figsize=(18, 9))
sns.regplot(x=swe, y=predicted_swe, fit_reg=True)
plt.plot(swe, swe)
plt.xlabel("Les valeurs de SWE issues de ERA5")
plt.ylabel("Les valeurs prédites par le modèle")
plt.title("Le nuage des points des valeurs prédites de SWE en fonction des valeurs issues de ERA5")
plt.show()



# Scatter plot of predicted and ERA5 values of swe
predicted_swe=list(model.predict(x_test[['tb6v','tb6h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89h','tb89v']]))
swe=list(test['swe'])    
#df = pd.DataFrame(data=swe, columns='swe') 
plt.figure(figsize=(22, 12))
idxs = [i for i in range(1, len(swe)+1)]      
plt.plot(idxs , swe, 'o', label = 'SWE de ERA5')
plt.plot(idxs, predicted_swe, 'o', label = 'SWE prédit par le modèle')
plt.legend()
plt.xlabel('Index'); plt.ylabel('Valeurs journalières de swe'); plt.title('valeurs observées et valeurs prédites de swe-fichier test')




# Smooth density
x_test['predicted_swe'] = model.predict(x_test[['tb6v','tb6h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89h','tb89v']])
x_test['swe'] = list(y_test)
x_test['SWE - SWE_Prédit']= x_test['swe'] - x_test['predicted_swe']
df=x_test
#print(max(df['SWE - SWE_Prédit']), min(df['SWE - SWE_Prédit']), max(df['swe']), min(df['swe']))
sns.set_theme(style="white")
g = sns.JointGrid(data=df, x="swe", y="SWE - SWE_Prédit", space=0)
g.plot_joint(sns.kdeplot,
             fill=True, clip=((0.1, 1.225), (-0.333, 0.219)),                          
             thresh=0, levels=100, cmap="ocean")
g.plot_marginals(sns.histplot, color="#2734b0", alpha=1, bins=25)
plt.xlabel("SWE")
plt.ylabel("SWE - SWE predicted")                                                                  
plt.show() 



# Scores calculated in testing data set training data set
x_test['predicted_swe'] = model.predict(x_test[['tb6v','tb6h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89h','tb89v']])
#train['swe'] = list(train['swe'])
x_test['swe'] = test['swe']
df = x_test     
print(x_test)
print(x_train)

"""
predicted_swe = list(model.predict(x_train[['tb6v','tb6h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89h','tb89v']]))
swe = list(train['swe']) 
"""
print('La valeur moyenne de swe est \n', df.swe.mean()) # multioutput='variance_weighted'))
print("=========================================")
print("=========================================")
print('Le coefficient de détermination est \n', r2_score(df.swe, df.predicted_swe)) # multioutput='variance_weighted'))
print("=========================================")
print("=========================================")
print('L\'erreur quadratique moyenne est \n', mean_squared_error(df.swe, df.predicted_swe, squared=False))
print("=========================================")
print("=========================================")
print('L\'erreur absolue moyenne est \n', mean_absolute_error(df.swe, df.predicted_swe))
print("=========================================")
print("=========================================")
print('La corrélation de pearson est \n', pearsonr(df.swe, df.predicted_swe))

"""
ybar = np.sum(df.swe)/len(df.swe)          # or sum(y)/len(y)
sstot = np.sum((df.predicted_swe-df.swe)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
ssreg = np.sum((df.swe - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
r2 =  1 - ssreg / sstot
print(r2)
"""



# Build data to plot error distribution
path = "/content/drive/MyDrive/data" 

# Import data 
df1 = pd.read_csv('/content/drive/MyDrive/data/amsr_era5_daymean_20190320.csv',  header=None, delim_whitespace=True, low_memory=False) #Import csv file into dataframe df2      
df1.columns=['lon','lat','rsn', 'sd', 'tb6v','tb6h','tb7v','tb7h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89v','tb89h'] # Rename columns 

df1.drop(df1[df1['sd'] < 0.001].index, inplace=True)                       
df1.drop(df1[df1['sd'] > 0.2].index, inplace=True) 

def mean_norm(df_input):
    return df_input.apply(lambda x: (x-x.mean())/ x.std(), axis=0)

aux=df1[['tb6v','tb6h','tb10v','tb10h','tb18v','tb18h','tb36v','tb36h']]
df1[['tb6v','tb6h', 'tb10v','tb10h','tb18v','tb18h','tb36v','tb36h']] = mean_norm(aux)

df1['predicted_swe'] = model.predict(df1[['tb6v','tb6h','tb10v','tb10h','tb18v','tb18h','tb36v','tb36h']])
df1['erreur']= np.abs(df1['predicted_swe'] - df1['sd'])

df = df1[['lon', 'lat', 'erreur', 'sd']]
df['lon'] = 10*df['lon']  
df['lat'] = 10*df['lat']

aux1 = list(df['lon'])
aux2 = list(df['lat'])

for i in range(len(aux1)):
	if aux1[i] % 10 == 2:
		aux1[i] = aux1[i] + 3
	if aux1[i] % 10 == 8:
		aux1[i] = aux1[i] - 0.5
		
for i in range(len(aux2)):
	if aux2[i] % 10 == 2:
		aux2[i] = aux2[i] + 3
	if aux2[i] % 10 == 8:
		aux2[i] = aux2[i] - 0.5
df['lon'] = aux1
df['lon'] = df['lon'] / 10

df['lat'] = aux2
df['lat'] = df['lat'] / 10    

# These coming lines allow to put data on a regular grid to be 
# able to put the final df in netcdf format  
lat1=np.around(np.linspace(90, 30, 241), decimals=2)                                                                
lon1=np.around(np.linspace(180, -180, 1441), decimals=2)  

T1=[]                                                    
for i in range(241):
	for j in range(1441):
		T1.append((lat1[i],lon1[j]))
latt=[]                                                                         
lonn=[]              
for e in df.lat:
	latt.append(e)
for e in df.lon:
	lonn.append(e)
latt=np.float64(latt)
lonn=np.float64(lonn)

n=len(df.lat)
T2=[]
for i in range(n):
	T2.append((latt[i], lonn[i]))
	
	
set_diff = set(T1).difference(set(T2))
list_diff = list(set_diff)

m=len(list_diff)

L1=[]
for e in list_diff:
	L1.append(e[0])
L2=[]
for e in list_diff:
	L2.append(e[1])
L3=[]
for i in range(m):
	L3.append(-99999) 
	
 
L1=np.transpose(L1)
L2=np.transpose(L2)
L3=np.transpose(L3)

mat=[L2, L1, L3, L3]
mat=np.transpose(mat)
df10= pd.DataFrame(mat, columns = ['lon','lat','sd', 'erreur'])
df3=pd.concat([df10, df], sort=False)
df=df3.sort_values(['lat', 'lon'] , axis=0, ascending=False, inplace=False, na_position='last').reset_index()
#df['time']=1
df=df[[ 'lon','lat','erreur']]
df.columns=[ 'longitude', 'latitude', 'erreur']
#df.loc[df.tb36h==-99999.00,'tb36h']=0.00 

# ======================= !!!!!!!!!!!!!!!!!!!! =========================
df = df.set_index(['latitude', 'longitude'])
df = df[~df.index.duplicated(keep='first')]
xr.Dataset(df.to_xarray()).to_netcdf('error_20_03_M8.nc4')   

# End 
