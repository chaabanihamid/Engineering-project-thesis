"""
#!/usr/bin/python

This program aims to prepare data for Machine Learning from raw data 
Created  in Feb, Mar, Avr, Mai, Jun and Jul 2021
@author:  hamichaabani@gmail.com 
"""
import cdsapi
import getpass 
from Reading_AMSR_hourly import Read_AMSR2_L1R_hourly,_zeros_on_str 
from scipy.interpolate import griddata
from ftplib import FTP
import os
import time
import csv
import numpy as np
from pandas import *
import pandas as pd
import netCDF4 as nc
from netCDF4 import Dataset
import subprocess
import datetime as dt
import calendar
q_era5="0" 
q_amsr="0" 
q_era5 = input("###############################################################\n"
          "AMSR2 and ERA5 DATA PROCESSING FOR MACHINE LEARNING\n"
          " Work directory: $HOME/pfe2021 "
          " Data directory: data are stored in $HOME/pfe2021/year/month "
          " This script uses Reading_AMSR_hourly.py "
          "@author: Noureddine Semane EHTP 26 March 2021\n"
          "###############################################################\n\n"
          "Download ERA5-LAND data? 1 for yes and 0 for no  ")
q_amsr = input("###############################################################\n"
          "Download AMSR2? 1 for yes and 0 for no  ")
if "1" == q_amsr:
    login_jaxa = input("Enter login ftp.gportal.jaxa.jp\n")
    passwd=getpass.getpass(prompt='Enter Password: ftp.gportal.jaxa.jp ', stream=None) 
years = ['2013','2014','2015','2016','2017','2018','2019','2020']
#years = ['2018']
months = ['01','02','03','04','05','11','12']
#months = ['12']
hours = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
for k in range(len(years)):
    year=years[k]
    for i in range(len(months)):
        month=months[i]
        numberOfDays = calendar.monthrange(int(year), int(month))[1]
        for day in list(range(1, numberOfDays + 1)):
            d = '{:0>2}'.format(str(day))
            amsr_era5_daymean='amsr_era5_daymean_'+year+month+d+'.csv'
            for j in range(len(hours)):
                hour=hours[j]
                h = '{:0>2}'.format(str(hour))
                directory=os.path.expanduser("~")+'/pfe2021'
                print('Work directory', directory)
                Dir=os.path.expanduser("~")+'/pfe2021/'+year+'/'+month+'/'
                print('Data directory', Dir)
                os.system('mkdir -p %s'%Dir)
                era5=Dir+'snow_ecmwf'+year+month+d+h+'.nc'
                amsr_era5='amsr_era5_'+year+month+d+h+'.txt'
                if "1" == q_era5:
                    c = cdsapi.Client()
                    c.retrieve(             
                          'reanalysis-era5-land',
                          {
                            'product_type': 'reanalysis',
                            'format': 'netcdf',
                            'variable': [
                            'snow_depth', 'snow_depth_water_equivalent',
                             ],
                            'day': d,
                            'month': month,
                            'year': year,
                            'time': h+':00',
                            'area': [
                             90.00, -180.00, 30.00,
                             180.00,
                             ],
                          },
                          era5)
                if "1" == q_amsr:
                    path='standard/GCOM-W/GCOM-W.AMSR2/L1R/2/'+year+'/'+month+'/'
                    os.system("echo 'lcd '%s > %s" % (Dir,'batchfile.sftp'))
                    os.system("echo 'cd '%s >> %s" % (path,'batchfile.sftp'))
                    os.system("echo 'mget GW1AM2_'%s%s%s%s'*D*.h5'>> %s" % (year,month,d,h,'batchfile.sftp'))
                    os.system("echo 'bye' >> %s" %'batchfile.sftp')
                    os.system("sshpass -p %s sftp -oPort=2051 %s@ftp.gportal.jaxa.jp < %s" % (passwd, login_jaxa,'batchfile.sftp'))
                tab= Read_AMSR2_L1R_hourly(year,month,day,hour,directory,lat_min=30.00)
# data coordinates and values
                if not tab.any():
                   continue
                x=np.round(tab[2,:],1)
                y=np.round(tab[1,:],1)
                z1=tab[7]
                z2=tab[8]
                z3=tab[9]
                z4=tab[10]
                z5=tab[11]
                z6=tab[12]
                z7=tab[13]
                z8=tab[14]
                z9=tab[15]
                z10=tab[16]
                z11=tab[17]
                z12=tab[18]
                z13=tab[19]
                z14=tab[20]
                xy=np.array(list(zip(x,y,z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z14)))
                df=pd.DataFrame(xy,columns=['lon','lat','tb6v','tb6h','tb7v','tb7h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89v','tb89h'])
                xy=df.groupby(['lon','lat']).mean().reset_index().values
                df=pd.DataFrame(xy,columns=['lon','lat','tb6v','tb6h','tb7v','tb7h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89v','tb89h'])
                df=np.round(df,2)
                df['lon']=np.round(df['lon'],1)
                df['lat']=np.round(df['lat'],1)
                day=_zeros_on_str(day)
                hour=_zeros_on_str(hour)
                Date='%04d-%02d-%02d' % (int(year), int(month), int(day))
                print('Date Hour',Date,hour)
                day_of_the_year= (dt.date(int(year), int(month), int(day)) - dt.date(int(year),1,1)).days + 1
                os.system('cdo -selvar,sde %s %s' % (era5, 'era5_sde'))
                os.system('cdo -selvar,sd %s %s' % (era5, 'era5_sd'))
                
                os.system('cdo -outputtab,lon,lat,value %s > %s' % ('era5_sde','era5_sde.txt'))
                os.system('cdo -outputtab,value %s > %s' % ('era5_sd','era5_sd.txt'))
                
                os.system('paste %s %s > %s' % ('era5_sde.txt','era5_sd.txt','tmp1.txt'))
                
                os.system("sed '1d' %s > %s" % ('tmp1.txt','tmp2.txt'))
                os.system('echo lon lat sde swe > %s' % 'tmp3.txt')
                os.system("awk '{if ($3> 0) print $1, $2, $3, $4}' %s >> %s" % ('tmp2.txt','tmp3.txt'))
                df_era5=pd.read_csv('tmp3.txt', sep=' ', header='infer', index_col=None)
                df_era5=np.round(df_era5,3)
                df_era5['lon']=np.round(df_era5['lon'],1)
                df_era5['lat']=np.round(df_era5['lat'],1)
                # Calling merge() function 
                df = pd.merge(df_era5, df, how='inner', on=['lon', 'lat'])
                df.to_csv('tmp4.txt',sep = ' ', index = False)
                os.system("sed '1d' %s > %s" % ('tmp4.txt',amsr_era5))
                os.system('rm -f tmp*.txt era5_sd*.txt batchfile.sftp')
                #os.system("rm -f %sGW1AM2_%s%s%s%s*.h5" % (Dir,year,month,d,h))
                #os.system('rm -f %s ' %era5 )
                
            os.system('echo lon lat sde swe tb6v tb6h tb7v tb7h tb10v tb10h tb18v tb18h tb23v tb23h tb36v tb36h tb89v tb89h > %s' % 'tmp5.txt')
            os.system("cat amsr_era5_%s%s%s*.txt >> %s" % (year,month,d,'tmp5.txt'))
            os.system("rm -f  amsr_era5_%s%s%s*.txt" % (year,month,d))
            df=pd.read_csv('tmp5.txt', sep=' ', header='infer', index_col=None)
            if df.empty:
                print('DataFrame is empty!')
                os.system('rm -f tmp5.txt')
                continue
            xy=df.groupby(['lon','lat']).mean().reset_index().values
            df=pd.DataFrame(xy,columns=['lon','lat','sde','swe','tb6v','tb6h','tb7v','tb7h','tb10v','tb10h','tb18v','tb18h','tb23v','tb23h','tb36v','tb36h','tb89v','tb89h'])
            df.iloc[:,4:17]=np.round(df.iloc[:,4:17],2)                                                    
            df['sde']=np.round(df['sde'],3)                             
            df['swe']=np.round(df['swe'],3)                                                                                             
            df.to_csv('tmp6.txt',sep = ' ', index = False)                                             
            os.system("sed '1d' %s > %s" % ('tmp6.txt',amsr_era5_daymean))                              
            os.system('rm -f tmp5.txt tmp6.txt')
            
            
