# Copyright (c) 2025 [Lu Tang]. All rights reserved.
# Permission is granted to use, copy, modify, and distribute this code
# for non-commercial purposes only, provided that the original copyright
# notice and this permission notice appear in all copies.
# Commercial use is prohibited without prior written permission.



##The observation threshold is converted to percentiles

import xarray as xr
import numpy as np

file_path = '/data19/wrk/2023tanglu/NEX-GDDP/CN05.1-1990-2020/mask_CN05.1_orin_1961-2020.nc'
ds = xr.open_dataset(file_path)
ds= ds.sel(time=slice('1990-01-01','2020-12-31'))
tp = ds['pre']

value_to_check = 50

def calculate_percentile(arr, value):
    valid_arr = arr[~np.isnan(arr)]
    if len(valid_arr) == 0:
        return np.nan
    sorted_arr = np.sort(valid_arr)
    percentile = np.searchsorted(sorted_arr, value, side='left') / len(sorted_arr) * 100
    return percentile

percentiles = xr.apply_ufunc(
    calculate_percentile,
    tp,
    input_core_dims=[['time']],
    kwargs={'value': value_to_check},
    vectorize=True,
    dask='parallelized',
    output_dtypes=[float]
)

percentiles


##Calculate the corresponding pattern threshold using the calculated percentiles


import xarray as xr
import numpy as np
import os


percentile_data_path = '/data19/wrk/2023tanglu/NEX-GDDP/CN05.1-1990-2020/CN05.1-1990-2020-pdfR50th.nc'
percentile_ds = xr.open_dataset(percentile_data_path)
percentile_values = percentile_ds['pre']  # 替换为实际百分位数变量名


base_dir = '/data19/wrk/2023tanglu/NEX-GDDP/mask(CN05.1)/'
file_names = [f for f in os.listdir(base_dir) if f.startswith('mask_pr_day_') and f.endswith('_history_1961-2014.nc')]
desired_order = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CESM2', 'CMCC-CM2-SR5', 'CMCC-ESM2',
    'CNRM-CM6-1', 'CNRM-ESM2-1', 'EC-Earth3', 'EC-Earth3-Veg-LR', 'FGOALS-g3', 'GFDL-ESM4', 'GISS-E2-1-G',
    'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR', 'KIOST-ESM', 'MIROC6', 'MIROC-ES2L', 'MPI-ESM1-2-HR',
    'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NESM3', 'NorESM2-LM', 'NorESM2-MM', 'TaiESM1']


model_percentile_values = {}
for file_name in file_names:
    model_name = file_name.split('_')[3]
    model_name = model_name.replace('mask_pr_day_', '')
    model_name = model_name.replace('_history_1961-2014.nc', '')

    if model_name in desired_order:
        model_path = os.path.join(base_dir, file_name)
        ds = xr.open_dataset(model_path)


        model_time_series = ds['pr']


        def calc_percentile_value(data, p_val):
            if np.isnan(p_val):
                return np.nan
            p_val = np.clip(p_val, 0, 100)
            return np.percentile(data, p_val, axis=0)


        actual_percentile_values = xr.apply_ufunc(
            calc_percentile_value,
            model_time_series,
            percentile_values,
            input_core_dims=[['time'], []],
            vectorize=True,
            dask='parallelized'
        )


        model_percentile_values[model_name] = actual_percentile_values

        print(f"Processed model: {model_name} - Percentile values extracted and stored.")