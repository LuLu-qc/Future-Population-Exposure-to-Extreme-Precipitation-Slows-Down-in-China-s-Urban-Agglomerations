# Copyright (c) 2025 [Lu Tang]. All rights reserved.
# Permission is granted to use, copy, modify, and distribute this code
# for non-commercial purposes only, provided that the original copyright
# notice and this permission notice appear in all copies.
# Commercial use is prohibited without prior written permission.

##Taylor Calculate and draw
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib import rcParams
import skill_metrics as sm
import geopandas as gpd
import rioxarray  # 用于裁剪操作

rcParams["figure.figsize"] = [8.0, 6.4]
rcParams['lines.linewidth'] = 1
rcParams.update({'font.size': 12})

shapes = {
    'PRD': gpd.read_file('/data19/wrk/2023tanglu/2023年_CTAmap_1.12版/城市群/zhusanjiao/zhusanjiao.shp'),
    # 'YRD': gpd.read_file('/data19/wrk/2023tanglu/2023年_CTAmap_1.12版/城市群/changsanjiao_3/Export_Output_csj.shp'),
    # 'BTH': gpd.read_file('/data19/wrk/2023tanglu/2023年_CTAmap_1.12版/城市群/Jingjinji_3/jjj.shp'),
    # 'CC':  gpd.read_file('/data19/wrk/2023tanglu/2023年_CTAmap_1.12版/城市群/chengyu/chengyu.shp')
}

base_dir = '/data19/wrk/2023tanglu/NEX-GDDP/mask(CN05.1)/'
obs_file = '/data19/wrk/2023tanglu/NEX-GDDP/CN05.1-1990-2020/mask_CN05.1_orin_1961-2020.nc'
file_names = [f for f in os.listdir(base_dir) if f.startswith('mask_pr_day_') and f.endswith('_history_1961-2014.nc')]

obs_data1 = xr.open_dataset(obs_file)
obs_data1 = obs_data1.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
obs_data1 = obs_data1.rio.write_crs("EPSG:4326", inplace=True)

desired_order = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CESM2', 'CMCC-CM2-SR5', 'CMCC-ESM2',
                 'CNRM-CM6-1', 'CNRM-ESM2-1', 'EC-Earth3', 'EC-Earth3-Veg-LR', 'FGOALS-g3', 'GFDL-ESM4', 'GISS-E2-1-G',
                 'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR', 'KIOST-ESM', 'MIROC6', 'MIROC-ES2L', 'MPI-ESM1-2-HR',
                 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NESM3', 'NorESM2-LM', 'NorESM2-MM', 'TaiESM1']

for name, shape in shapes.items():
    print(f"Processing {name} region")

    geometry = shape.geometry[0]

    obs_clipped = obs_data1.rio.clip([geometry], shape.crs, drop=True)
    obs_data = obs_clipped.sel(time=slice('1995-01-01', '2014-12-31'))
    fre = xr.where(obs_data >= 50, 1, 0)
    fre = fre.groupby('time.year').sum(dim='time')
    obs_data = fre['pre'].values.flatten()

    valid_mask = np.isfinite(obs_data)
    obs_data = obs_data[valid_mask]

    model_data = {}
    for file_name in file_names:
        model_name = file_name.split('_')[3]
        model_name = model_name.replace('mask_pr_day_', '')  # 从文件名中去掉前缀
        model_name = model_name.replace('_history_1961-2014.nc', '')  # 去掉后缀

        print(f"Checking file: {file_name} - Extracted model name: {model_name}")

        if model_name in desired_order:
            model_path = os.path.join(base_dir, file_name)
            ds = xr.open_dataset(model_path)
            ds = ds.sel(time=slice('1995-01-01', '2014-12-31'))
            ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
            ds = ds.rio.write_crs("EPSG:4326", inplace=True)
            model_clipped = ds.rio.clip([geometry], shape.crs, drop=True)
            thes = xr.open_dataarray(
                '/data19/wrk/2023tanglu/NEX-GDDP/CN05.1-1990-2020/theshold_CN05.1-1990-2020-pdfR50th.nc')
            model_pr = xr.where(model_clipped >= thes, 1, 0)
            model_pr = model_pr.groupby('time.year').sum(dim='time')['pr'].values.flatten()
            model_pr = model_pr[valid_mask]

            # 确保模型数据存在
            if model_name not in model_data:
                model_data[model_name] = model_pr
                print(f"Added model data for: {model_name}")
            else:
                print(f"Warning: {model_name} appears multiple times. Skipping duplicate.")

    if not model_data:
        raise ValueError(f"No model data matched the shape of the observation data for region {name}.")


    ordered_model_data = {model: model_data[model] for model in desired_order if model in model_data}

    missing_models = set(desired_order) - set(ordered_model_data.keys())
    if missing_models:
        print(f"Missing models: {missing_models}")

    ccoef = []
    std_dev = []
    rmse = []
    for model_pr in ordered_model_data.values():
        ccoef.append(np.corrcoef(obs_data, model_pr)[0, 1])
        std_dev.append(np.std(model_pr))
        rmse.append(np.sqrt(np.mean((obs_data - model_pr) ** 2)))

    sorted_model_names = list(ordered_model_data.keys())
    label = ['Non-Dimensional Observation'] + sorted_model_names
    sm.taylor_diagram(np.array(std_dev), np.array(rmse), np.array(ccoef), markerLabel=label,
                      markerLabelColor='k', markerColor='r', markerLegend='on',
                      tickRMS=range(0, 10, 2), tickRMSangle=110.0,
                      colRMS='#EE6AA7', styleRMS=':', widthRMS=2.0,
                      titleRMS='on', tickSTD=range(0, 10, 2),
                      axismax=10.0, colSTD='#1E90FF', styleSTD='-.',
                      widthSTD=1.0, titleSTD='on', colCOR='k', styleCOR='--',
                      widthCOR=1.0, titleCOR='on', markerSize=8, alpha=0.0)

    plt.show()

###Ranking
import numpy as np
import pandas as pd
#prd

sorted_model_names = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CESM2', 'CMCC-CM2-SR5',
                      'CMCC-ESM2', 'CNRM-CM6-1', 'CNRM-ESM2-1', 'EC-Earth3', 'EC-Earth3-Veg-LR', 'FGOALS-g3',
                      'GFDL-ESM4', 'GISS-E2-1-G', 'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR', 'KIOST-ESM',
                      'MIROC6', 'MIROC-ES2L', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NESM3',
                      'NorESM2-LM', 'NorESM2-MM', 'TaiESM1']

std_dev = []
rmse = []
ccoef = []

df = pd.DataFrame({
    'Model': sorted_model_names,
    'STD_DEV': std_dev,
    'RMSE': rmse,
    'CCOEF': ccoef
})

df['Rank_STD_DEV'] = df['STD_DEV'].rank(method='min')
df['Rank_RMSE'] = df['RMSE'].rank(method='min')
df['Rank_CCOEF'] = df['CCOEF'].rank(ascending=False, method='min')  # 相关系数越高越好，排序方式与前两个不同
df['Total_Rank'] = df['Rank_STD_DEV'] + df['Rank_RMSE'] + df['Rank_CCOEF']
df_sorted = df.sort_values(by='Total_Rank').reset_index(drop=True)
print(df_sorted[['Model', 'Rank_STD_DEV', 'Rank_RMSE', 'Rank_CCOEF', 'Total_Rank']])




##optimization
import xarray as xr
import os
import pandas as pd
import numpy as np
import geopandas as gpd


shapes = {
    'PRD': gpd.read_file('/data19/wrk/2023tanglu/2023年_CTAmap_1.12版/城市群/zhusanjiao/zhusanjiao.shp'),
}

base_dir = '/data19/wrk/2023tanglu/NEX-GDDP/mask(CN05.1)/'
obs_file = '/data19/wrk/2023tanglu/NEX-GDDP/CN05.1-1990-2020/mask_CN05.1_orin_1961-2020.nc'
file_names = [f for f in os.listdir(base_dir) if f.startswith('mask_pr_day_') and f.endswith('_history_1961-2014.nc')]

desired_order = ['BCC-CSM2-MR', 'INM-CM4-8', 'NorESM2-LM', 'GFDL-ESM4', 'IPSL-CM6A-LR', 'GISS-E2-1-G', 'CNRM-ESM2-1',
                 'FGOALS-g3', 'MPI-ESM1-2-HR',
                 'KIOST-ESM']

obs_data1 = xr.open_dataset(obs_file)
obs_data1 = obs_data1.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
obs_data1 = obs_data1.rio.write_crs("EPSG:4326", inplace=True)

for name, shape in shapes.items():
    print(f"Processing {name} region")

    geometry = shape.geometry[0]

    obs_clipped = obs_data1.rio.clip([geometry], shape.crs, drop=True)
    obs_data = obs_clipped.sel(time=slice('1995-01-01', '2014-12-31'))
    fre = xr.where(obs_data >= 50, 1, 0)
    fre = fre.groupby('time.year').sum(dim='time')
    obs_data = fre['pre'].values.flatten()

    valid_mask = np.isfinite(obs_data)
    obs_data = obs_data[valid_mask]

    model_data = {}
    for file_name in file_names:

        model_name = file_name.split('_')[3]
        model_name = model_name.replace('mask_pr_day_', '')  # 从文件名中去掉前缀
        model_name = model_name.replace('_history_1961-2014.nc', '')  # 去掉后缀

        if model_name in desired_order:
            model_path = os.path.join(base_dir, file_name)
            ds = xr.open_dataset(model_path)
            ds = ds.sel(time=slice('1995-01-01', '2014-12-31'))
            ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
            ds = ds.rio.write_crs("EPSG:4326", inplace=True)
            model_clipped = ds.rio.clip([geometry], shape.crs, drop=True)
            thes = xr.open_dataarray(
                '/data19/wrk/2023tanglu/NEX-GDDP/CN05.1-1990-2020/theshold_CN05.1-1990-2020-pdfR50th.nc')
            model_pr = xr.where(model_clipped >= thes, 1, 0)
            model_pr = model_pr.groupby('time.year').sum(dim='time')['pr'].values.flatten()
            model_pr = model_pr[valid_mask]

            if model_pr.shape == obs_data.shape:
                model_data[model_name] = model_pr
            else:
                print(f"Warning: {file_name} data shape mismatch. Skipping this model.")

    if not model_data:
        raise ValueError(f"No model data matched the shape of the observation data for region {name}.")

    ccoef_list = []
    std_dev_list = []
    rmse_list = []

    combined_data = None
    num_models = 0

    for model_name in desired_order:
        if model_name in model_data:
            num_models += 1

            if combined_data is None:
                combined_data = model_data[model_name]
            else:
                combined_data += model_data[model_name]

            recursive_mean = combined_data / num_models

            ccoef = np.corrcoef(obs_data, recursive_mean)[0, 1]
            std_dev = np.std(recursive_mean)
            rmse = np.sqrt(np.mean((obs_data - recursive_mean) ** 2))

            ccoef_list.append(ccoef)
            std_dev_list.append(std_dev)
            rmse_list.append(rmse)

            print(f"Recursive mean after {num_models} models: ccoef={ccoef}, std_dev={std_dev}, rmse={rmse}")

    print(f"Final recursive statistics for {name} region:")
    print("Correlation Coefficients:", ccoef_list)
    print("Standard Deviations:", std_dev_list)
    print("RMSE:", rmse_list)

import matplotlib.pyplot as plt

x_axis = list(range(1, len(ccoef_list) + 1))

fig, ax = plt.subplots(3, 1, figsize=(8, 8))

ax[0].plot(x_axis, ccoef_list, marker='o', color='b', linestyle='-')
ax[0].set_title('Correlation Coefficient vs. Number of Models')
ax[0].set_xlabel('Number of Models')
ax[0].set_ylabel('Correlation Coefficient')
ax[0].grid(True)

ax[1].plot(x_axis, std_dev_list, marker='o', color='g', linestyle='-')
ax[1].set_title('Standard Deviation vs. Number of Models')
ax[1].set_xlabel('Number of Models')
ax[1].set_ylabel('Standard Deviation')
ax[1].grid(True)

ax[2].plot(x_axis, rmse_list, marker='o', color='r', linestyle='-')
ax[2].set_title('Root Mean Square Error vs. Number of Models')
ax[2].set_xlabel('Number of Models')
ax[2].set_ylabel('Root Mean Square Error')
ax[2].grid(True)

plt.tight_layout()

plt.show()
