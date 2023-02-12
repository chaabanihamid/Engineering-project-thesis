"""
#!/usr/bin/python3 

This program takes as input a text file (csv or txt) of the satellite 
data and reanalyses, and gives as output a netcdf of some selected
variables put on a regular grid. The output serves as input for
plot prgramm plot_data.py

Created  in Feb, Mar, Avr, Mai, Jun and Jul 2021
@author: hamichaabani@gmail.com (CHAABANI Hamid)  
"""
# Import libraries         
import numpy as np               
import netCDF4 as nc        
from netCDF4 import Dataset as NetCDFFile                          
from datetime import datetime
import pandas as pd 
import xarray as xr
from Reading_AMSR import Read_AMSR2_L1R_daily 
from sklearn import *
from sklearn.metrics import mean_squared_error, r2_score 

# !!!!! data (*.h5) are stored in /Users/hamid/2018 or 2019/ and month
     
# Import csv file into pd forme
# !!!!!! choose the parameters below carefully !!!!!!!!!!!

df = pd.read_csv('amsr_era5_daymean_20181101.csv',  header=None, delim_whitespace=True, low_memory=False) #Import csv file into dataframe df2      
df.columns=['lon','lat','rsn', 'sd', 'tb6v','tb6h','tb7v','tb7h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89v','tb89h'] # Rename columns 

df.drop(df[df['sd'] < 0.001].index, inplace=True) 
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
lat1=np.around(np.linspace(90, 30, 241), decimals=1)                                                                
lon1=np.around(np.linspace(180, -180, 1441), decimals=1)                              
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

n=len(df.sd)
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

mat=[L2, L1, L3, L3, L3, L3, L3, L3, L3, L3, L3, L3, L3, L3, L3, L3, L3, L3]
mat=np.transpose(mat)
df1= pd.DataFrame(mat, columns = ['lon','lat','rsn','swe','tb6v','tb6h','tb7v','tb7h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89v','tb89h'])
df3=pd.concat([df1, df], sort=False)
df=df.sort_values(['lat', 'lon'] , axis=0, ascending=False, inplace=False, na_position='last')
df['time']=1
df.loc[df.tb36h==-99999.00,'tb36h']=0.00


# Add historical algo (Chang)
df['Chang']=np.around(1.59*(df['tb18h'] - df['tb36h'])*0.01, decimals=3)

# ======================= !!!!!!!!!!!!!!!!!!!! =========================
df = df.set_index(['time', 'lat', 'lon'])
df = df[~df.index.duplicated(keep='first')]
xr.Dataset(df.to_xarray()).to_netcdf('/home/hamid/dataAMSR.nc4')                     































# ==================== !!!!!!!!!!!!!!!!! ===============================
# Don't remove these lines, they may be very useful 


# =========  Merge the tows dataframes by columns 'lat' and 'lon' ======
# ======================================================================    
#df=df2.merge(df1, on=['lat','lon']) 
"""
df=df5.sort_values(['lat','long'] , axis=0, ascending=False, inplace=False, na_position='last')
df = df.set_index(['time', 'lat', 'long'])
xr.Dataset(df.to_xarray()).to_netcdf('/home/hamid/dataAMSR.nc4')
"""
# ========== Some useful commands ======================================
# ======================================================================


"""
df5['TB_36hA'].replace(to_replace=-99999.00, value=0)
df=df5.sort_values(['latitude','longitude'] , axis=0, ascending=False, inplace=False, na_position='last')
df = df.set_index(['time', 'latitude', 'longitude'])
"""
