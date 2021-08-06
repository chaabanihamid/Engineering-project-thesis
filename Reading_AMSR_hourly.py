"""
#!/usr/bin/env python3
This program aims à read the AMSR Data and extract data from HDF files

Created  in Feb, Mar, Avr, Mai, Jun and Jul 2021
@author: hamichaabani@gmail.com (CHAABANI Hamid) 
"""

# Import libraries
import numpy as np
from tqdm import tqdm
import os, h5py
 
#%% Useful func
def _zeros_on_str(day):
    if len(str(day))==1:
        day = '0'+str(day)
    else:
        day = str(day)
    return day

def _list_directory(path, motcle1, motcle2):
    """
    Browse a folder tree to recover a list of files

    Input:
        path        : (str) path of the directory
        motcle1     : (str) keyword wanted
        motcle2     : (str) keyword wanted
    Return:
        list_files  : (list) list of files containing keywords
    """
#    list_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            #ici tu mets ta condition à la place de "READ" genre "SIC0"
            if motcle1 in file and motcle2 in file :
                yield os.path.join(root, file)
#%% Reading Func

def Read_AMSR2_L1R_hourly(year,month,day,hour,directory,**kwargs):
    """
    Read ASMR2 hdf5 data (TB) available at:
            https://gportal.jaxa.jp/gpr/information/download
        Data stocked at: /
            standard/GCOM-W/GCOM-W.AMSR2/L1R/2
            
    Warning : If you want to delete all incorrect data,
              add 'tb_max=340' in kwargs, 340K being the maximum of the
              dynamical range of AMSR2.

    Arguments:
        years       : (str) year of data acquisition
        month       : (str) month of data acquisition
        day         : (str) day of data acquisition
        directory   : (str) path of the directory where the data is stored
    Return:
        tab         : (array) data with time, lat, lon, land_flag, TBh and TBv
    """
    folder = directory + '/' + year + '/' + month + '/'
    print('folder',folder)

    time_utcA = []
    latB = []
    lonB = []
    flag_res6A = []
    flag_res10A = []
    flag_res23A = []
    flag_res36A = []
    TB_6vA = []
    TB_6hA = []
    TB_7vA = []
    TB_7hA = []
    TB_10vA = []
    TB_10hA = []
    TB_18vA = []
    TB_18hA = []
    TB_23vA = []
    TB_23hA = []
    TB_36vA = []
    TB_36hA = []
    TB_89vA = []
    TB_89hA = []
    
    #days = np.linspace(day_start,day_stop,day_stop-day_start+1,dtype=int)
    #for day in tqdm(days):
    day = _zeros_on_str(day)
    hour = _zeros_on_str(hour)
    for fname in _list_directory(folder,year+month+day+hour,'.h5'):
    
    #        if fname[-3:] == '.h5':
             print('fname',fname)
             try:
                 file = h5py.File(fname, 'r')
             
                     
                 time_utc = file['/Scan Time'][()]
        
                 # time1=(time+9)/3600/24; # nombre de leap second depuis 1993/01/01 à  ajouter
                 # time2=date.toordinal(date(1993,1,1))+366 #LE 366 semble vraiment inutile meme faux...
                 # time_utc=time1+time2;
        
                 latA = np.transpose(file['Latitude of Observation Point for 89A'][()])
                 lonA = np.transpose(file['Longitude of Observation Point for 89A'][()])
                 qualityA=np.transpose(file['Pixel Data Quality 6 to 36'][()])

        
                 lat = latA[1::2][:] #seul les indices pairs sont utiles pour les freq<89GHz
                 lon = lonA[1::2][:]
                 quality=qualityA[1::2][:]
                 time_utc_rep = np.kron(np.ones((np.shape(lat)[0],1)),time_utc)
                 
                 
                 rs1=lat.shape[0]
                 rs2=lat.shape[1] 
                 scanquality_raw = file['/Scan Data Quality'][()] 
                 #scan quality is not read correctly: need to convert the 512 int8 into 128
                 #float single
                 scanquality_rawl = np.reshape(scanquality_raw,(rs2,128,4),order='C')

                 scan1=np.zeros((128,rs2))
                 for uu in range(0,127):
                     for yy in range(0,rs2-1):
                         #recompute the single float value based on the 4 int8 value
                         temp = np.array(scanquality_rawl[yy,uu,:],dtype=np.int8) 
                         scan1[uu,yy] = temp.view(dtype=np.float32)
                         
                 scanquality=scan1[2,:] # only the 3rd line is of interest. No doc found. I suppose that it is the same that for pixel data quality 0=good measurements
                 scanquality=np.tile(scanquality,[rs1])

        
                 flag_6_36 = file['/Land_Ocean Flag 6 to 36'][()]
                 flag_res6 = np.squeeze(flag_6_36[0,:,:])
                 flag_res10 = np.squeeze(flag_6_36[1,:,:])
                 flag_res23 = np.squeeze(flag_6_36[2,:,:])
                 flag_res36 = np.squeeze(flag_6_36[3,:,:])
        
                 TB_6v = file['/Brightness Temperature (res06,6.9GHz,V)'][()] * 0.01
                 TB_6h = file['/Brightness Temperature (res06,6.9GHz,H)'][()] * 0.01
        
                 TB_7v = file['/Brightness Temperature (res06,7.3GHz,V)'][()] * 0.01
                 TB_7h = file['/Brightness Temperature (res06,7.3GHz,H)'][()] * 0.01
        
                 TB_10v = file['/Brightness Temperature (res10,10.7GHz,V)'][()] * 0.01
                 TB_10h = file['/Brightness Temperature (res10,10.7GHz,H)'][()] * 0.01
        
                 TB_18v = file['/Brightness Temperature (res23,18.7GHz,V)'][()] * 0.01
                 TB_18h = file['/Brightness Temperature (res23,18.7GHz,H)'][()] * 0.01
        
                 TB_23v = file['/Brightness Temperature (res23,23.8GHz,V)'][()] * 0.01
                 TB_23h = file['/Brightness Temperature (res23,23.8GHz,H)'][()] * 0.01
        
                 TB_36v = file['/Brightness Temperature (res36,36.5GHz,V)'][()] * 0.01
                 TB_36h = file['/Brightness Temperature (res36,36.5GHz,H)'][()] * 0.01
        
                 TB_89v = file['/Brightness Temperature (res36,89.0GHz,V)'][()] * 0.01
                 TB_89h = file['/Brightness Temperature (res36,89.0GHz,H)'][()] * 0.01
        
                 time_utcl = np.reshape(time_utc_rep,np.size(lat), order='F')
        
                 latl = np.reshape(lat,(np.size(lat)),order='F')
                 lonl = np.reshape(lon,(np.size(lat)),order='F')
        
                 flag_res6l = np.reshape(flag_res6,np.size(lat))
                 flag_res10l = np.reshape(flag_res10,np.size(lat))
                 flag_res23l = np.reshape(flag_res23,np.size(lat))
                 flag_res36l = np.reshape(flag_res36,np.size(lat))
                 quality_l=np.reshape(quality,np.size(lat),order='F')
                 scanquality_l = np.reshape(scanquality,np.size(lat))
        
                 time_utcl = np.delete(time_utcl,np.where((quality_l!=0)|(scanquality_l!=0)))
                 latl = np.delete(latl,np.where((quality_l!=0)|(scanquality_l!=0)))
                 lonl = np.delete(lonl,np.where((quality_l!=0)|(scanquality_l!=0)))
                 flag_res6l =  np.delete(flag_res6l,np.where((quality_l!=0)|(scanquality_l!=0)))
                 flag_res10l = np.delete(flag_res10l,np.where((quality_l!=0)|(scanquality_l!=0)))
                 flag_res23l = np.delete(flag_res23l,np.where((quality_l!=0)|(scanquality_l!=0)))
                 flag_res36l = np.delete(flag_res36l,np.where((quality_l!=0)|(scanquality_l!=0)))
        
                 TB_6v = np.reshape(TB_6v,(np.size(TB_6v)))
                 TB_6h = np.reshape(TB_6h,(np.size(TB_6h)))
                 TB_6v = np.delete(TB_6v,np.where((quality_l!=0)|(scanquality_l!=0)))
                 TB_6h = np.delete(TB_6h,np.where((quality_l!=0)|(scanquality_l!=0)))
        
                 TB_7v = np.reshape(TB_7v,(np.size(TB_7v)))
                 TB_7h = np.reshape(TB_7h,(np.size(TB_7h)))
                 TB_7v = np.delete(TB_7v,np.where((quality_l!=0)|(scanquality_l!=0)))
                 TB_7h = np.delete(TB_7h,np.where((quality_l!=0)|(scanquality_l!=0)))
        
                 TB_10v = np.reshape(TB_10v,(np.size(TB_10v)))
                 TB_10h = np.reshape(TB_10h,(np.size(TB_10h)))
                 TB_10v = np.delete(TB_10v,np.where((quality_l!=0)|(scanquality_l!=0)))
                 TB_10h = np.delete(TB_10h,np.where((quality_l!=0)|(scanquality_l!=0)))
        
                 TB_18v = np.reshape(TB_18v,(np.size(TB_18v)))
                 TB_18h = np.reshape(TB_18h,(np.size(TB_18h)))
                 TB_18v = np.delete(TB_18v,np.where((quality_l!=0)|(scanquality_l!=0)))
                 TB_18h = np.delete(TB_18h,np.where((quality_l!=0)|(scanquality_l!=0)))
        
                 TB_23v = np.reshape(TB_23v,(np.size(TB_23v)))
                 TB_23h = np.reshape(TB_23h,(np.size(TB_23h)))
                 TB_23v = np.delete(TB_23v,np.where((quality_l!=0)|(scanquality_l!=0)))
                 TB_23h = np.delete(TB_23h,np.where((quality_l!=0)|(scanquality_l!=0)))
        
                 TB_36v = np.reshape(TB_36v,(np.size(TB_36v)))
                 TB_36h = np.reshape(TB_36h,(np.size(TB_36h)))
                 TB_36v = np.delete(TB_36v,np.where((quality_l!=0)|(scanquality_l!=0)))
                 TB_36h = np.delete(TB_36h,np.where((quality_l!=0)|(scanquality_l!=0)))
        
                 TB_89v = np.reshape(TB_89v,(np.size(TB_89v)))
                 TB_89h = np.reshape(TB_89h,(np.size(TB_89h)))
                 TB_89v = np.delete(TB_89v,np.where((quality_l!=0)|(scanquality_l!=0)))
                 TB_89h = np.delete(TB_89h,np.where((quality_l!=0)|(scanquality_l!=0)))
        
                 #Concatenation pour le fichier de sortie
                 if kwargs=={}:
                     time_utcA = np.concatenate((time_utcA,time_utcl))
        
                     latB = np.concatenate((latB,latl))
                     lonB = np.concatenate((lonB,lonl))
            
                     flag_res6A = np.concatenate((flag_res6A,flag_res6l))
                     flag_res10A = np.concatenate((flag_res10A,flag_res10l))
                     flag_res23A = np.concatenate((flag_res23A,flag_res23l))
                     flag_res36A = np.concatenate((flag_res36A,flag_res36l))
            
                     TB_6vA = np.concatenate((TB_6vA,TB_6v))
                     TB_6hA = np.concatenate((TB_6hA,TB_6h))
            
                     TB_7vA = np.concatenate((TB_7vA,TB_7v))
                     TB_7hA = np.concatenate((TB_7hA,TB_7h))
            
                     TB_10vA = np.concatenate((TB_10vA,TB_10v))
                     TB_10hA = np.concatenate((TB_10hA,TB_10h))
            
                     TB_18vA = np.concatenate((TB_18vA,TB_18v))
                     TB_18hA = np.concatenate((TB_18hA,TB_18h))
            
                     TB_23vA = np.concatenate((TB_23vA,TB_23v))
                     TB_23hA = np.concatenate((TB_23hA,TB_23h))
            
                     TB_36vA = np.concatenate((TB_36vA,TB_36v))
                     TB_36hA = np.concatenate((TB_36hA,TB_36h))
            
                     TB_89vA = np.concatenate((TB_89vA,TB_89v))
                     TB_89hA = np.concatenate((TB_89hA,TB_89h))
                 else:
            
                    conditions = []
                    if 'lat_min' in kwargs.keys():
                        conditions.append(np.squeeze([latl>kwargs['lat_min']]))
                    if 'lat_max' in kwargs.keys():
                        conditions.append(np.squeeze([latl<kwargs['lat_max']]))
                    if 'lon_min' in kwargs.keys():
                        conditions.append(np.squeeze([lonl>kwargs['lon_min']]))
                    if 'lon_max' in kwargs.keys():
                        conditions.append(np.squeeze([lonl<kwargs['lon_max']]))
                    if 'land' in kwargs.keys():
                        conditions.append(np.squeeze([flag_res10l>=kwargs['land']]))
                    if 'ocean' in kwargs.keys():
                        conditions.append(np.squeeze([flag_res10l<=kwargs['ocean']]))
                    if 'tb_max' in kwargs.keys():
                        conditions.append(np.squeeze([TB_6v<=kwargs['tb_max']]))
                        conditions.append(np.squeeze([TB_6h<=kwargs['tb_max']]))
                        conditions.append(np.squeeze([TB_7v<=kwargs['tb_max']]))
                        conditions.append(np.squeeze([TB_7h<=kwargs['tb_max']]))
                        conditions.append(np.squeeze([TB_10v<=kwargs['tb_max']]))
                        conditions.append(np.squeeze([TB_10h<=kwargs['tb_max']]))
                        conditions.append(np.squeeze([TB_18v<=kwargs['tb_max']]))
                        conditions.append(np.squeeze([TB_18h<=kwargs['tb_max']]))
                        conditions.append(np.squeeze([TB_23v<=kwargs['tb_max']]))
                        conditions.append(np.squeeze([TB_23h<=kwargs['tb_max']]))
                        conditions.append(np.squeeze([TB_36v<=kwargs['tb_max']]))
                        conditions.append(np.squeeze([TB_36h<=kwargs['tb_max']]))
                        conditions.append(np.squeeze([TB_89v<=kwargs['tb_max']]))
                        conditions.append(np.squeeze([TB_89h<=kwargs['tb_max']]))
        
                        
                    condition = conditions[0]
                    for i in conditions[1:]:
                        condition = condition & i
                        
                    time_utcA = np.concatenate((time_utcA,time_utcl[condition]))
            
                    latB = np.concatenate((latB,latl[condition]))
                    lonB = np.concatenate((lonB,lonl[condition]))
                    flag_res6A = np.concatenate((flag_res6A,flag_res6l[condition]))
                    flag_res10A = np.concatenate((flag_res10A,flag_res10l[condition]))
                    flag_res23A = np.concatenate((flag_res23A,flag_res23l[condition]))
                    flag_res36A = np.concatenate((flag_res36A,flag_res36l[condition]))
            
                    TB_6vA = np.concatenate((TB_6vA,TB_6v[condition]))
                    TB_6hA = np.concatenate((TB_6hA,TB_6h[condition]))
            
                    TB_7vA = np.concatenate((TB_7vA,TB_7v[condition]))
                    TB_7hA = np.concatenate((TB_7hA,TB_7h[condition]))
            
                    TB_10vA = np.concatenate((TB_10vA,TB_10v[condition]))
                    TB_10hA = np.concatenate((TB_10hA,TB_10h[condition]))
            
                    TB_18vA = np.concatenate((TB_18vA,TB_18v[condition]))
                    TB_18hA = np.concatenate((TB_18hA,TB_18h[condition]))
            
                    TB_23vA = np.concatenate((TB_23vA,TB_23v[condition]))
                    TB_23hA = np.concatenate((TB_23hA,TB_23h[condition]))
            
                    TB_36vA = np.concatenate((TB_36vA,TB_36v[condition]))
                    TB_36hA = np.concatenate((TB_36hA,TB_36h[condition]))
            
                    TB_89vA = np.concatenate((TB_89vA,TB_89v[condition]))
                    TB_89hA = np.concatenate((TB_89hA,TB_89h[condition]))
                 #FILTER STEP
         
        
                    
        
                 file.close()
             except OSError :
                 pass
    # data_amsr = data_amsr[:,data_amsr[1]>45]
    # data_amsr = data_amsr[:, data_amsr[4]==0]      
    tab = np.array([time_utcA,latB,lonB,
                    flag_res6A,flag_res10A,flag_res23A,flag_res36A,
                    TB_6vA,TB_6hA,
                    TB_7vA,TB_7hA,
                    TB_10vA,TB_10hA,
                    TB_18vA,TB_18hA,
                    TB_23vA,TB_23hA,
                    TB_36vA,TB_36hA,
                    TB_89vA,TB_89hA])

    return tab



