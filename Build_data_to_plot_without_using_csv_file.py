"""
#!/usr/bin/python3 
This program uses the function of the Reading_AMSR.py program and allows
to retrieve AMSR data that are poorly organized and to put them on a 
regular grid in average over a period chosen by the user and put them 
in a netCDF format.
NB: This program allows to retrieve all variables 

Created  in Feb, Mar, Avr, Mai, Jun and Jul 2021
@author: hamichaabani@gmail.com (CHAABANI Hamid)  
"""
# Import python moduls           
import numpy as np               
import netCDF4 as nc        
from netCDF4 import Dataset as NetCDFFile                          
from datetime import datetime
import pandas as pd 
import xarray as xr
from Reading_AMSR import Read_AMSR2_L1R_daily 

#data (*.h5) are stored in /Users/hamid/2018 or 2019/ and month
# !!!!!! choose the parameters below with care !!!!!!!!!!!     
year = '2018'                                          
month = '11'                                                                                                                      
day_start = 1                                                                                                                         
day_stop = 1                                                                                                                                                                                     
directory='/media/hamid/Elements/Projet_PFE/Data/AMSR'                                                                                                   
                                                                                                                                             
# Read AMSR2 data 
tab = Read_AMSR2_L1R_daily(year, month, day_start, day_stop, directory, lat_min=30)
latt=np.linspace(90, 30, 601) 
lonn=np.linspace(-180, 180, 3601)                                                               

# Around longitudes and latitudes values
tab[1,:]=np.around(tab[1,:], decimals=1)                   
tab[2,:]=np.around(tab[2,:], decimals=1)                      

data=np.transpose(tab)
df1 = pd.DataFrame(data, columns = ['time','latitude','longitude','flag_res6A','flag_res10A','flag_res23A','flag_res36A','TB_6vA','TB_6hA','TB_7vA','TB_7hA','TB_10vA','TB_10hA','TB_18vA','TB_18hA','TB_23vA','TB_23hA','TB_36vA','TB_36hA','TB_89vA','TB_89hA'])
df1=df1.groupby(['latitude', 'longitude']).mean().reset_index()  # Groupby two columns and return the mean of the remaining column and get original variables.
df1 = df1[['time','latitude','longitude','flag_res6A','flag_res10A','flag_res23A','flag_res36A','TB_6vA','TB_6hA','TB_7vA','TB_7hA','TB_10vA','TB_10hA','TB_18vA','TB_18hA','TB_23vA','TB_23hA','TB_36vA','TB_36hA','TB_89vA','TB_89hA']] # Rearrange Columns
 

df2 = pd.read_csv('data1.csv',  header=None, delim_whitespace=True, low_memory=False) #Import csv file into dataframe df2      
df2.columns=['time2', 'longitude', 'latitude', 'SD'] # Rename columns                                           
df2.drop(df2[df2['SD']==-32767].index, inplace=True) # Drop lines with indefined value in SD 
df2['latitude']=np.around(df2['latitude'], decimals=1)
df2['longitude']=np.around(df2['longitude'], decimals=1)
df2['SD']=np.around(df2['SD'], decimals=3) 
df2.drop(df2.tail(1).index,inplace=True)


df3=df2.merge(df1, on=['longitude', 'latitude'])  
df3.drop(df3[df3['SD']<0.001].index, inplace=True)
df3.drop(columns=['time2'],inplace=True)
df3['time']=62                                                                                                                                                                                                                                                               

lat1=np.around(np.linspace(90, 30, 601), decimals=1)                                                                
lon1=np.around(np.linspace(180, -180, 3601), decimals=1)                              
T1=[]                                                    
for i in range(601):
	for j in range(3601):
		T1.append((lat1[i],lon1[j]))
latt=[]                                                                         
lonn=[]              
for e in df3.latitude:
	latt.append(e)
for e in df3.longitude:
	lonn.append(e)
latt=np.float64(latt)
lonn=np.float64(lonn)

n=len(df3.SD)
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
	
time=[]     
for i in range(m):
	time.append(62)   
	 
L1=np.transpose(L1)
L2=np.transpose(L2)
L3=np.transpose(L3)

mat=[L2, L1,time, L3, L3, L3, L3, L3, L3, L3, L3, L3, L3, L3, L3, L3, L3, L3, L3, L3, L3, L3]
mat=np.transpose(mat)
df4= pd.DataFrame(mat, columns = ['longitude','latitude','time','SD', 'flag_res6A','flag_res10A','flag_res23A','flag_res36A','TB_6vA','TB_6hA','TB_7vA','TB_7hA','TB_10vA','TB_10hA','TB_18vA','TB_18hA','TB_23vA','TB_23hA','TB_36vA','TB_36hA','TB_89vA','TB_89hA'])
df5=pd.concat([df3, df4], sort=False)
df=df5[['time', 'latitude', 'longitude', 'SD', 'flag_res6A','flag_res10A','flag_res23A','flag_res36A','TB_6vA','TB_6hA','TB_7vA','TB_7hA','TB_10vA','TB_10hA','TB_18vA','TB_18hA','TB_23vA','TB_23hA','TB_36vA','TB_36hA','TB_89vA','TB_89hA']]
df=df.sort_values(['latitude', 'longitude'] , axis=0, ascending=False, inplace=False, na_position='last')

df.loc[df.TB_36hA==-99999.00,'TB_36hA']=0.00


# Add historical algo (Chang)
df['Chang']=np.around(1.59*(df['TB_18hA'] - df['TB_36hA']*0.01), decimals=3)

df = df.set_index(['time', 'latitude', 'longitude'])
df = df[~df.index.duplicated(keep='first')]
xr.Dataset(df.to_xarray()).to_netcdf('/home/hamid/dataAMSR.nc4')                     








# =========  Merge the tows dataframes by columns 'lat' and 'lon' =======
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
