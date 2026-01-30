
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Old script I wrote to use a dataset of ocean carbon I created (MOBO-DIC),
# then integrate the total carbon in the upper 1500m,
# then plot the changes of that over time

# Created by Lydi in October 20205
# Last edited by Lydi Oct 14, 2025
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# import packages
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import xarray as xr
from scipy.io import loadmat
import cftime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gsw
import glob
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Load DIC data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Open the dataset
file_path = "/Users/lydi/Desktop/MOBO/Data/MPI_MOBO-DIC_2004-2019_v2.nc"
ds = xr.open_dataset(file_path, decode_times=False)

# Decode time
dates = [cftime.DatetimeGregorian(2004 + (int(m)//12), ((int(m) % 12) + 1), 1)
         for m in ds['juld'].values]
dates_shifted = []
for d in dates:
    if d.month == 1:
        new_year = d.year - 1
        new_month = 12
    else:
        new_year = d.year
        new_month = d.month - 1
    dates_shifted.append(cftime.DatetimeGregorian(new_year, new_month, 1))
ds = ds.assign_coords(juld=('juld', dates_shifted))

# Annual mean
annual_mean = ds.groupby('juld.year').mean(dim='juld')
annual_mean['DIC'] = xr.where(annual_mean['DIC'] < -999, np.nan, annual_mean['DIC'])

# mean
meanDIC = annual_mean['DIC'].mean(dim='year')

# Depth
depths = annual_mean['depth'].values  # (28,)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Load the data of the area of each grid cell and the bathymetry
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
file_path = "/Users/lydi/Desktop/MOBO/Data/area.mat"
area_size = loadmat(file_path)

file_path = "/Users/lydi/Desktop/MOBO/Data/etopo2.nc"
etopo = xr.open_dataset(file_path, decode_times=False)

# put on 1 deg-grid
etopo_1deg = etopo.interp(lat=ds['lat'], lon=ds['lon'])
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Load Argo T/S data for density
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Until end of 2018
file_path_T = "/Users/lydi/Desktop/MOBO/Data/Argo/RG_ArgoClim_Temperature_2019.nc"
file_path_S = "/Users/lydi/Desktop/MOBO/Data/Argo/RG_ArgoClim_Salinity_2019.nc"
argo_T = xr.open_dataset(file_path_T, decode_times=False)
argo_S = xr.open_dataset(file_path_S, decode_times=False)
argo_T['TIME'] = [pd.Timestamp("2004-01-01") + pd.DateOffset(months=int(m)) for m in argo_T['TIME'].values]
argo_S['TIME'] = [pd.Timestamp("2004-01-01") + pd.DateOffset(months=int(m)) for m in argo_S['TIME'].values]

# Annual mean
annual_meanT = argo_T.groupby('TIME.year').mean(dim='TIME')
annual_meanS = argo_S.groupby('TIME.year').mean(dim='TIME')
Temp = annual_meanT['ARGO_TEMPERATURE_MEAN'] + annual_meanT['ARGO_TEMPERATURE_ANOMALY']
Salt = annual_meanS['ARGO_SALINITY_MEAN'] + annual_meanS['ARGO_SALINITY_ANOMALY']
Pres = argo_T['PRESSURE']
Lon = argo_T['LONGITUDE']
Lat = argo_T['LATITUDE']

# 2019
file_path_all = "/Users/lydi/Desktop/MOBO/Data/Argo/"
files = sorted(glob.glob(f"{file_path_all}/RG_ArgoClim_2019*.nc"))
datasets = [xr.open_dataset(f, decode_times=False) for f in files]
temp19 = [ds["ARGO_TEMPERATURE_ANOMALY"] for ds in datasets]
salt19 = [ds["ARGO_SALINITY_ANOMALY"] for ds in datasets]
# Annual mean
temp19 = np.mean(np.squeeze(np.stack(temp19, axis=0)), axis=0)
salt19 = np.mean(np.squeeze(np.stack(salt19, axis=0)), axis=0)
# Absolute values, not anomalies
temp19_abs = temp19 + annual_meanT['ARGO_TEMPERATURE_MEAN'][1, :, :, :]
salt19_abs = salt19 + annual_meanS['ARGO_SALINITY_MEAN'][1, :, :, :]

# Add 2019 data
temp19_da = xr.DataArray(
    temp19_abs,
    dims=["PRESSURE", "LATITUDE", "LONGITUDE"],
    coords={
        "PRESSURE": Temp["PRESSURE"],
        "LATITUDE": Temp["LATITUDE"],
        "LONGITUDE": Temp["LONGITUDE"],
    },
    name=Temp.name if Temp.name else "TEMP"
)
temp19_da = temp19_da.expand_dims(dim={"year": [2019]})
Temp = xr.concat([Temp, temp19_da], dim="year")

salt19_da = xr.DataArray(
    salt19_abs,
    dims=["PRESSURE", "LATITUDE", "LONGITUDE"],
    coords={
        "PRESSURE": Temp["PRESSURE"],
        "LATITUDE": Temp["LATITUDE"],
        "LONGITUDE": Temp["LONGITUDE"],
    },
    name=Salt.name if Salt.name else "TEMP"
)
salt19_da = salt19_da.expand_dims(dim={"year": [2019]})
Salt = xr.concat([Salt, salt19_da], dim="year")

# Calculate density
CT = gsw.CT_from_t(Salt, Temp, Pres)  # Conservative temp
Rho = gsw.rho(Salt, CT, Pres)  # (kg/m^3)

# Lat from -90 to 90
new_lat = np.arange(-89.5, 90, Lat.values[1] - Lat.values[0])
Rho = Rho.reindex(LATITUDE=new_lat)

# Lon from -180 to 180
new_lon = np.where(Lon >180, Lon - 360, Lon)  # subtract 360 if >180
new_lon = np.where(new_lon < -180, new_lon + 360, new_lon)  # add 360 if <180
Rho = Rho.assign_coords(LONGITUDE=new_lon)
sort_idx = np.argsort(Rho['LONGITUDE'].values)
Rho = Rho.isel(LONGITUDE=sort_idx)
Rho = Rho.assign_coords(LONGITUDE=Rho['LONGITUDE'].values[sort_idx])

# Same depth levels as MOBO
Rho = Rho.interp(PRESSURE=depths)

# test plot
mask_slice = Rho[0, 27, :, :]  # shape (180, 360)
plt.figure(figsize=(10, 5))
plt.imshow(mask_slice, origin='lower', extent=[-180, 180, -90, 90],
           cmap='Blues', interpolation='none')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Upscale for masked regions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Etopo mask
topo = etopo_1deg['topo'].values  # (180, 360)
topo = np.where(topo > 0, np.nan, -topo)  # now positive = ocean depth in meters

etopo_mask = np.zeros((28, 180, 360), dtype=int)
for i in range(0, 28):
    etopo_mask[i, :, :] = np.where((topo >= depths[i]), 1, 0)

# Calculate the global mean DIC for each year and depth level
DIC_mean_year_depth = annual_mean['DIC'].mean(dim=['lat', 'lon'])

# Expand DIC
DIC_expanded = DIC_mean_year_depth.values[:, :, np.newaxis, np.newaxis]
mask_expanded = etopo_mask[np.newaxis, :, :, :]
upscale_masked_DIC = np.where(mask_expanded == 1, DIC_expanded, np.nan)

# Add upscaled values to MOBO-DIC
upscaled_DIC_1 = np.where(np.isnan(annual_mean['DIC']), upscale_masked_DIC, annual_mean['DIC'])

# Same for Rho
Rho_mean_year_depth = Rho.mean(dim=['LATITUDE', 'LONGITUDE'])
Rho_expanded = Rho_mean_year_depth.values[:, :, np.newaxis, np.newaxis]
mask_expanded = etopo_mask[np.newaxis, :, :, :]
upscale_masked_Rho = np.where(mask_expanded == 1, Rho_expanded, np.nan)
Rho = np.where(np.isnan(Rho), upscale_masked_Rho, Rho)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Test plot
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mask_slice = upscaled_DIC_1[0, 27, :, :]  # shape (180, 360)

plt.figure(figsize=(10, 5))
plt.imshow(mask_slice, origin='lower', extent=[-180, 180, -90, 90],
           cmap='Blues', interpolation='none')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Upscale for between 1500 and 4000m
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
depths_deep = np.array([1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000])
etopo_mask = np.zeros((10, 180, 360), dtype=int)
for i in range(0, 10):
    etopo_mask[i, :, :] = np.where((topo >= depths_deep[i]), 1, 0)

# Get the DIC values at 1500, divide by 2
DIC_1500 = upscaled_DIC_1[:, np.squeeze(np.where(depths == 1500)), :, :]  # shape: (year, lat, lon)
DIC_1500 = DIC_1500/2

# Apply to all values between 1500 and 4000m
DIC_deep = np.full((16, 10, 180, 360), np.nan)
for i in range(10):
    DIC_deep[:, i, :, :] = DIC_1500 * etopo_mask[i, :, :]

# Combine
upscaled_DIC_full = np.concatenate((upscaled_DIC_1, DIC_deep), axis=1)
depth = np.append(annual_mean['depth'], depths_deep)

# Rho deep
last_level = Rho[:, -1, :, :]     # shape: (16, 180, 360)
last_repeated = np.repeat(last_level[:, np.newaxis, :, :], 10, axis=1)
Rho = np.concatenate([Rho, last_repeated], axis=1)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Test plots
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mask_slice = DIC_deep[0, 9, :, :]  # shape (180, 360)
plt.figure(figsize=(10, 5))
plt.imshow(mask_slice, origin='lower', extent=[-180, 180, -90, 90],
           cmap='Blues', interpolation='none')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


mask_slice = upscaled_DIC_full[0, 36, :, :]  # shape (180, 360)
plt.figure(figsize=(10, 5))
plt.imshow(mask_slice, origin='lower', extent=[-180, 180, -90, 90],
           cmap='Blues', interpolation='none')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Get the volume of covered cells (m3)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dz = np.diff(depth)
dz = np.append(dz, dz[-1])
volume = dz[:, np.newaxis, np.newaxis] * area_size['area'][np.newaxis, :, :]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Integrate the upper 1500m
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# rho = 1025  # kg/m³
DIC_molm3 = upscaled_DIC_full * 1e-6 * Rho
DIC_mol = DIC_molm3 * volume[np.newaxis, :, :, :]
DIC_global_mol = np.nansum(DIC_mol, axis=(1, 2, 3))
molC_to_gC = 12.01
g_to_Pg = 1e-15
DIC_global_PgC = DIC_global_mol * molC_to_gC * g_to_Pg
annual_change = np.diff(DIC_global_PgC)

print('Annual mean change 2004 - 2020:', round(np.nanmean(annual_change), 2))
print('Annual mean change 2004 - 2009', round(np.nanmean(annual_change[0:6]), 2))
print('Annual mean change 2010s', round(np.nanmean(annual_change[7:]), 2))


np.savetxt("MOBO_Integrated_annual.txt", DIC_global_PgC)
np.savetxt("MOBO_Integrated_annual_change.txt", annual_change)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Plot the annual changes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
years = np.arange(2004, 2004 + len(DIC_global_PgC))
years_diff = years[1:]

# Create scatter plot
plt.figure(figsize=(10,6))
plt.scatter(years_diff, annual_change, color='#7570b3', s=120, edgecolor='k', alpha=0.9, zorder=3)
plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.5, zorder=1)
plt.ylabel('Change in DIC (Pg C)', fontsize=14, fontweight='bold')
plt.title('Annual Global Inventory Change in MOBO-DIC (upper 4000 m)', fontsize=16, fontweight='bold')
plt.xticks(years_diff, rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()
plt.savefig("MOBO_Integrated_annual_change.png", dpi=300, bbox_inches='tight')


plt.figure(figsize=(10,6))
plt.scatter(years, DIC_global_PgC, color='#7570b3', s=120, edgecolor='k', alpha=0.9, zorder=3)
plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.5, zorder=1)
plt.ylabel('DIC (Pg C)', fontsize=14, fontweight='bold')
plt.title('Annual Global Inventory MOBO-DIC (upper 4000 m)', fontsize=16, fontweight='bold')
plt.xticks(years_diff, rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()
plt.savefig("MOBO_Integrated_annual.png", dpi=300, bbox_inches='tight')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

