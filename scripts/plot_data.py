"""
#!/usr/bin/python3 

This program allows the user to visualize any Gridded data type stored 
in netcdf file, in a map format with different projections. You have just
to enter the input file path at the beginning of the program and the map
title at the end of the program. 

Created  in Feb, Mar, Avr, Mai, Jun and Jul 2021
@author:  hamichaabani@gmail.com (CHAABANI Hamid) 

"""
 
# Import libraries          
import numpy as np               
import netCDF4 as nc
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap          
from netCDF4 import Dataset as NetCDFFile                          
from datetime import datetime
import pandas as pd 
import sys

# ====================== Plot data =====================================
# ======================================================================
# Import input file 
nc = NetCDFFile('dataAMSR5.nc4')# Set the path and input file name 

# Extract latitudes and longitudes 
lats = nc.variables['latitude'][:]
lons = nc.variables['longitude'][:]


L=[]
for v in nc.variables:
	L.append(v)


#Choose variable to map
V = input('Enter the variable name to plot, 0 for variables list: ')
while "0" == V:
	for v in L:
		print(v)
	V = input("Give variable name ?|  0 for the list of variables: ")
if V in L:
	V = nc.variables[V][:] # Set variable to visualize
	data=V                                                                                                                                                                       
	
	plt.figure(figsize=(12,8))                                                                             
	date = datetime(2018,11,1) # date to plot. 
	grid_lon, grid_lat = np.meshgrid(lons, lats) # Regularly spaced 2D grid
	
	# Choose the projection 
	nsphere = Basemap(projection='npstere', llcrnrlon=-180, urcrnrlon=180.,\
	llcrnrlat=lats.min(),urcrnrlat=lats.max(),boundinglat=30.,\
	lon_0=180., resolution='c') # Resolution can take l, i, h, f , etc 
	cyl = Basemap(projection='cyl', llcrnrlon=-180,\
    urcrnrlon=180.,llcrnrlat=30.,urcrnrlat=lats.max(), \
    resolution='c')   
    # For more information about Basemap() funtion, look at the website :
    # https://matplotlib.org/basemap/api/basemap_api.html 
    
	list_P=['nsphere', 'cyl']
	P = input('Choose projection, 0 for projections list: ')
	while "0" == P:
		for p in list_P:
			print(p)
		P = input("Give projection name ?|  0 for projections list: ")
	if P=='nsphere':
		m=nsphere
	elif P=='cyl':                                                                                                                   
		m=cyl 
	else:
		sys.exit("Projection name is incorrect.")
	x, y = m(grid_lon, grid_lat)                                                                        
 
    #cs = m.pcolormesh(x,y,data[0,:,:],shading='flat',cmap=plt.cm.gist_stern_r)
	clevs = [150, 210,230,240, 250, 260,270, 280, 300] # Define the map legend # 
	cs= m.contourf(x,y,data[0,:,:],clevs,cmap=plt.cm.gist_stern_r)
 
	m.drawcoastlines()                                                                                                                                                                                                                         
	m.drawparallels([30, 60, 80],labels=[1,0,1,0])
	m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1])
 
	#plt.colorbar(cs,orientation='vertical', shrink=0.5)
	cb = m.colorbar(cs,"right", size="5%", pad="2%")
	
	# Set unit of measure
	cb.set_label('k (kelvin)')  
	
	# Set the name of the variable to plot
	plt.title(' Tb89h pour le 1er mars 2019' )
	plt.savefig('e'+'.png') # Set the output file name
	plt.show() 
else:
    sys.exit("Variablc name is incorrect.")
 #End                                                                                                                
