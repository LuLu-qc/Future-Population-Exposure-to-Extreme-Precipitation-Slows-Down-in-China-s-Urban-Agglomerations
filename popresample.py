# Copyright (c) 2025 [Lu Tang]. All rights reserved.
# Permission is granted to use, copy, modify, and distribute this code
# for non-commercial purposes only, provided that the original copyright
# notice and this permission notice appear in all copies.
# Commercial use is prohibited without prior written permission.


import xarray as xr
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer
import matplotlib.pyplot as plt

# 输入文件（全球人口数据）
tiff_file = '/data19/wrk/2023tanglu/NEX-GDDP/popnew/Worldpop/chn_ppp_2000_UNadj.tif'

# 定义目标网格（中国及周边）
min_lat, max_lat, lat_step = 14.75, 55.25, 0.25
min_lon, max_lon, lon_step = 69.75, 140.25, 0.25

# 网格中心点（确保匹配格点中心）
new_lat = min_lat + lat_step / 2 + np.arange(int((max_lat - min_lat) / lat_step + 1)) * lat_step
new_lon = min_lon + lon_step / 2 + np.arange(int((max_lon - min_lon) / lon_step + 1)) * lon_step

# 初始化新网格
new_population = np.zeros((len(new_lat), len(new_lon)), dtype=np.float32)

def process_file(tiff_file):
    with rasterio.open(tiff_file) as dataset:
        src_crs = dataset.crs
        dst_crs = "EPSG:4326"
        to_proj = Transformer.from_crs(dst_crs, src_crs, always_xy=True)
        from_proj = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

        # 将经纬度边界转换为源投影坐标并略微扩展范围
        left, bottom = to_proj.transform(min_lon - 1, min_lat - 1)
        right, top = to_proj.transform(max_lon + 1, max_lat + 1)

        # 构造裁剪窗口
        window = from_bounds(left, bottom, right, top, transform=dataset.transform)

        # 读取数据（裁剪）
        data = dataset.read(1, window=window).astype(float)
        no_data_value = dataset.nodata
        if no_data_value is not None:
            data[data == no_data_value] = np.nan
        data = np.nan_to_num(data, nan=0)

        # 获取裁剪窗口左上角坐标
        top_left_x, top_left_y = dataset.transform * (window.col_off, window.row_off)

        # 构造裁剪区域的投影坐标网格
        height, width = data.shape
        x_coords = top_left_x + np.arange(width) * dataset.transform.a
        y_coords = top_left_y + np.arange(height) * dataset.transform.e  # transform.e 是负的

        # 构造网格并转换为经纬度
        xx, yy = np.meshgrid(x_coords, y_coords)
        lon_flat, lat_flat = from_proj.transform(xx.flatten(), yy.flatten())

        # 对齐到目标网格索引
        lat_idx = np.digitize(lat_flat, new_lat) - 1
        lon_idx = np.digitize(lon_flat, new_lon) - 1

        # 合法索引掩码
        valid_mask = (
            (lat_idx >= 0) & (lat_idx < len(new_lat)) &
            (lon_idx >= 0) & (lon_idx < len(new_lon))
        )

        # 累加数据到新网格
        np.add.at(new_population, (lat_idx[valid_mask], lon_idx[valid_mask]), data.flatten()[valid_mask])

# 执行处理
process_file(tiff_file)

# 转为 xarray 并输出结果
new_population = np.round(new_population).astype(int)
new_population_da = xr.DataArray(new_population, coords=[("lat", new_lat), ("lon", new_lon)])
print(new_population_da)
