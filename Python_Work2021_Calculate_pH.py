
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Old script I wrote to load ocean data (GLODAP) to calculate a parameter from other parameters
# pH (from DIC, Talk, T and S)
# Note: Data paths are for calculating on a server

# Created by Lydi on Feb 15 2021
# Last edited by Lydi on Apr 06 2021
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# import packages
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import xarray as xr
import PyCO2SYS as pyco2  # https://pyco2sys.readthedocs.io/en/latest/
import numpy as np
import matplotlib.pyplot as plt
import cartopy.feature
import cartopy.crs as ccrs
from matplotlib import cm
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Load GLOPDAP data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
data_path = '/data/averdy/datasets/GLODAP/v2/GLODAPv2.2016b_MappedClimatologies/'

# DIC
filename = 'GLODAPv2.2016b.TCO2.nc'
DIC_file = xr.open_dataset(data_path + filename)
DIC = DIC_file.variables['TCO2']  # unit: micro mol kg-1 (data from 1972-2013, normalized to year 2002)
np.shape(DIC)  # dimensions: depth, lat, lon
depth = np.array(DIC_file.variables['Depth'])
lon = np.array(DIC_file.variables['lon'])  # deg. east (from 20.5 to 379.5)
lat = np.array(DIC_file.variables['lat'])  # deg. north
# from -180 to 180
lon[lon > 180] = lon[lon > 180] - 360
sort_ind = lon.argsort()
lon.sort()
DIC = DIC[:, :, sort_ind]
# DIC_error = DIC_file.variables['TCO2_error']
# DIC_error = DIC_error[:, :, sort_ind]


# Alkalinity
filename = 'GLODAPv2.2016b.TAlk.nc'
Talk_file = xr.open_dataset(data_path + filename)
TAlk = Talk_file.variables['TAlk']  # unit: micro mol kg-1 (data from 1972-2013, normalized to year 2002)
np.shape(TAlk)  # dimensions: depth, lat, lon
TAlk = TAlk[:, :, sort_ind]
# TAlk_error = Talk_file.variables['TAlk_error']
# TAlk_error = TAlk_error[:, :, sort_ind]


# Salinity
filename = 'GLODAPv2.2016b.salinity.nc'
Salt_file = xr.open_dataset(data_path + filename)
Salt = Salt_file.variables['salinity']
np.shape(Salt)  # dimensions: depth, lat, lon
Salt = Salt[:, :, sort_ind]
# Salt_error = Salt_file.variables['salinity_error']
# Salt_error = Salt_error[:, :, sort_ind]


# Temperature
filename = 'GLODAPv2.2016b.temperature.nc'
Temp_file = xr.open_dataset(data_path + filename)
Temp = Temp_file.variables['temperature']
np.shape(Temp)  # dimensions: depth, lat, lon
Temp = Temp[:, :, sort_ind]
# Temp_error = Temp_file.variables['temperature_error']
# Temp_error = Temp_error[:, :, sort_ind]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Calculate pH with CO2sys
# I used the same constants as Yui Takeshits uses for the SOCCOM float data, see
# https://docs.google.com/document/d/1VwmbLHaJnnHNeJlPUhb1ptEKAJSMu6SKFO-31GzgO5E/edit
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
calc_pH = np.empty((33, 180, 360))
calc_pH[:, :, :] = np.nan

for i in range(len(depth)):
    pyco2_kws = {}

    # Define the known marine carbonate system parameters
    pyco2_kws["par1"] = TAlk[i, :, :]  # Alk measured in the lab in μmol/kg-sw
    pyco2_kws["par2"] = DIC[i, :, :]  # DIC measured in the lab in μmol/kg-sw
    pyco2_kws["par1_type"] = 1  # tell PyCO2SYS: "par1 is a TAlk value"
    pyco2_kws["par2_type"] = 2  # tell PyCO2SYS: "par2 is a DIC value"

    # Define the seawater conditions and add them to the dict
    pyco2_kws["salinity"] = Salt[i, :, :]  # practical salinity
    pyco2_kws["temperature"] = Temp[i, :, :]  # lab temperature (input conditions) in °C
    pyco2_kws["pressure"] = depth[i]  # pressure of the obs

    # Define PyCO2SYS settings and add them to the dict
    pyco2_kws["opt_pH_scale"] = 1  # pH input is on the Total scale
    pyco2_kws["opt_k_carbonic"] = 10  # use carbonate equilibrium constants of LDK00 (Lueker et al. 2000)
    pyco2_kws["opt_k_bisulfate"] = 1  # use bisulfate dissociation constant of D90a (Dickson et al., 1990)
    pyco2_kws["opt_total_borate"] = 2  # use borate:salinity of LKB10 (Lee et al., 2010)"
    pyco2_kws["opt_k_fluoride"] = 2  # use hydrogen fluoride dissociation of PF78 (Perez and Fraga 1987)

    # Now calculate everything with PyCO2SYS
    results = pyco2.sys(**pyco2_kws)
    calc_pH[i, :, :] = results["pH"]

# check
print("pH range: ", np.nanmin(calc_pH), np.nanmax(calc_pH))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Plot map of calculated pH, check: compare with plots in Lauvset et al., 2020 (Fig. 6b)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
for i in range(len(depth)):
    figure_path = 'Figures/'
    fig = plt.figure(figsize=(12, 9))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_title('Calculated pH (GLODAP); depth: '+str(int(depth[i]))+'m', fontsize=22)
    p1 = plt.pcolor(lon, lat, np.squeeze(calc_pH[i, :, :]), transform=ccrs.PlateCarree(), cmap='RdYlBu')
    ax_cb = plt.axes([0.92, 0.25, 0.015, 0.5])
    cb = plt.colorbar(p1, cax=ax_cb, orientation='vertical', format='%.2f')
    cb.ax.set_ylabel('pH', fontsize=18)
    ax.add_feature(cartopy.feature.LAND, zorder=2, color=[0.8, 0.8, 0.8])
    ax.coastlines(resolution='50m', zorder=2)
    mean_pH = np.nanmean(np.nanmean(np.squeeze(calc_pH[i, :, :])))
    plt.clim([mean_pH-0.1, mean_pH+0.1])
    ax.gridlines()
    ax.set_global()
    plt.savefig(figure_path+'pH_'+str(int(depth[i]))+'m.png', dpi=100)


# at 20m with similar specs as Lauvset
i = 2
cmap = cm.get_cmap('RdYlBu', 17)  # less colours

fig = plt.figure(figsize=(12, 9))
ax = plt.axes(projection=ccrs.Robinson())
ax.set_title('Calculated pH (GLODAP); depth: '+str(int(depth[i]))+'m', fontsize=22)
p1 = plt.pcolor(lon, lat, np.squeeze(calc_pH[i, :, :]), transform=ccrs.PlateCarree(), cmap=cmap)
ax_cb = plt.axes([0.92, 0.25, 0.015, 0.5])
cb = plt.colorbar(p1, cax=ax_cb, orientation='vertical', cmap=cmap, format='%.2f')
cb.ax.set_ylabel('pH', fontsize=18)
ax.add_feature(cartopy.feature.LAND, zorder=2, color=[0.8, 0.8, 0.8])
ax.coastlines(resolution='50m', zorder=2)
plt.clim([8.0, 8.2])
ax.gridlines()
ax.set_global()
plt.savefig(figure_path+'Lauvset_pH_'+str(int(depth[i]))+'m.png', dpi=100)


# then copy the plots from server to local machine
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
