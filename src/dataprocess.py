import geopandas as gpd
from functools import partial
from shapely.geometry import Point
import numpy as np
import pandas as pd
import sys
from multiprocessing import Pool
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
from setproctitle import setproctitle
setproctitle('dataprocess@Yuzihan')
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--city', type=str, choices=['CHI', 'NYC'], default='NYC')
parser.add_argument('--slot', type=int, default=6, help='slot length in hours')

if __name__ == '__main__':
    args = parser.parse_args()
    time_range = ['2019-1-01 00:00:00', '2021-12-31 23:59:59']
    slot_len = args.slot * 3600  # 6h
    if args.city == 'CHI':
        category = ['THEFT', 'BATTERY', 'CRIMINAL DAMAGE', 'ASSAULT'] # 'DECEPTIVE PRACTICE', 'OTHER OFFENSE', 'NARCOTICS']
    elif args.city == 'NYC':
        category = ['PETIT LARCENY', 'HARRASSMENT 2', 'ASSAULT 3 & RELATED OFFENSES', 'CRIMINAL MISCHIEF & RELATED OF'] # 'GRAND LARCENY', 'DANGEROUS DRUGS', 'OFF. AGNST PUB ORD SENSBLTY &', 'FELONY ASSAULT', 'ROBBERY', 'BURGLARY', 'MISCELLANEOUS PENAL LAW']
    else:
        raise ValueError('Unknown city')

    time_range = [int(pd.to_datetime(date).timestamp()) for date in time_range]
    category = dict(zip(category, range(len(category))))

    if args.city == 'CHI':
        region_data = gpd.read_file('./dataset/chi_ctdata_census/chi_tract_with_census.shx')
    elif args.city == 'NYC':
        region_data = gpd.read_file('./dataset/nyc_cbdata_census/ny_nta_with_census.shx')
        target_crs = 'EPSG:4326'
        region_data = region_data.to_crs(target_crs)
    else:
        raise ValueError('Unknown city')

    from shapely.geometry import Polygon
    import pyproj
    # 创建投影坐标系，这里使用了WGS84经纬度坐标系
    crs_wgs84 = pyproj.CRS('EPSG:4326')
    # 创建一个等积投影坐标系，这里使用了Equal Earth等积投影
    crs_equal_earth = pyproj.CRS.from_string('+proj=eqearth')
    # 创建投影转换器
    transformer = pyproj.Transformer.from_crs(crs_wgs84, crs_equal_earth, always_xy=True)
    # 定义转换函数
    def transform_coordinates(x, y):
        return transformer.transform(x, y)
    areas = []
    for polygon in region_data['geometry']:
        # # 对Polygon对象的每个坐标点进行转换
        # transformed_coordinates = [transform_coordinates(x, y) for x, y in polygon.exterior.coords]
        # # 创建新的Polygon对象，使用转换后的坐标
        # transformed_polygon = Polygon(transformed_coordinates)
        # # 计算转换后的区域面积
        # areas.append(transformed_polygon.area)
        areas.append(polygon.area)

    # 打印结果
    print("实际地理面积（平方米）：", sum(areas), np.mean(areas))

    if args.city == 'CHI':
        data = pd.read_csv('./dataset/chi_crime.csv', usecols=['Date', 'Primary Type', 'Latitude', 'Longitude'])
        data['Timestamp'] = pd.to_datetime(data['Date'], format="%m/%d/%Y %I:%M:%S %p").apply(lambda x: int(x.timestamp()))
        mask = data['Primary Type'].isin(category) & (data['Timestamp'] >= time_range[0]) & (data['Timestamp'] <= time_range[1])
        data = data[mask]
        data['Crime Index'] = data['Primary Type'].apply(lambda x: category[x])
    elif args.city == 'NYC':
        data = pd.read_csv('./dataset/nyc_crime.csv', usecols=['CMPLNT_FR_DT', 'CMPLNT_FR_TM', 'OFNS_DESC', 'Latitude', 'Longitude'])
        error = data['CMPLNT_FR_DT'].apply(lambda x: str(x)[-4:-2] == '10') # filt those "1022 year"
        data = data[~error]
        date = pd.to_datetime(data['CMPLNT_FR_DT'] + ' ' + data['CMPLNT_FR_TM'], format="%m/%d/%Y %H:%M:%S")
        data = data[~date.isna()]  # filt those 'nan year'
        data['Timestamp'] = pd.to_datetime(data['CMPLNT_FR_DT'] + ' ' + data['CMPLNT_FR_TM'], format="%m/%d/%Y %H:%M:%S").apply(lambda x: int(x.timestamp()))
        mask = data['OFNS_DESC'].isin(category) & (data['Timestamp'] >= time_range[0]) & (data['Timestamp'] <= time_range[1])
        data = data[mask]
        data['Crime Index'] = data['OFNS_DESC'].apply(lambda x: category[x])
    else:
        raise ValueError('Unknown city')

    def get_occurence_matrix(time_range, category, slot_len, data, region_data):
        inc = np.zeros((int(time_range[1] - time_range[0] + 1) // slot_len, len(region_data), len(category)), dtype=np.uint8)
        for _, datum in tqdm(data.iterrows(), desc='Constructing occurence matrix', total=len(data), disable=True):
            for s_idx, region in enumerate(region_data['geometry']):
                if region.contains(Point(datum['Longitude'], datum['Latitude'])):
                    t_idx = (datum['Timestamp'] - time_range[0]) // slot_len
                    c_idx = datum['Crime Index']
                    inc[t_idx, s_idx, c_idx] += 1
                    break
        print(np.sum(inc))
        return inc

    inc = np.zeros((int(time_range[1] - time_range[0] + 1) // slot_len, len(region_data), len(category)), dtype=np.uint8)
    pool = Pool(32)
    pbar = tqdm(total=len(data) // 2048)
    update = lambda *args: pbar.update()
    func = partial(get_occurence_matrix, time_range, category, slot_len, region_data=region_data)
    for i in range(0, len(data), 2048):
        inc += pool.apply_async(func, (data[i:i + 2048],), callback=update).get()
    pool.close()
    pool.join()
    np.save(f'{args.city}.npy', inc)
    # inc = np.load(f'./baseline/STC-GNN/data/{args.city}-incidents-{args.slot}h.npz')['incident'].astype(np.uint8)

    A = np.zeros((len(region_data), len(region_data)), dtype=np.int)
    for idx, x in tqdm(enumerate(region_data['geometry']), desc='Constructing adjacency matrix'):
        for idy, y in enumerate(region_data['geometry']):
            if x.touches(y):
                A[idx, idy] += 1

    mask = np.nonzero(~np.any(inc, axis=(0, 2)))
    x = inc.reshape(-1, inc.shape[-1])
    threshold = np.mean(x, axis=0)
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    c_cor = np.einsum('Nx,Ny->xy', x, x) / x.shape[0]
    c_cor = c_cor - np.diag(np.diag(c_cor))

    if args.city == 'CHI':
        # ['H', 'W', 'B', 'ASIAN', 'NHPI', 'SOR']
        feature = ['TOT', 'H', 'W', 'B', 'AIAN', 'ASIAN', 'NHPI', 'SOR', 'MULTI', 'DI']
        region_data['H_P'] = region_data['H'] / region_data['TOT'] * 100
        region_data['W_P'] = region_data['W'] / region_data['TOT'] * 100
        region_data['B_P'] = region_data['B'] / region_data['TOT'] * 100
        region_data['ASIAN_P'] = region_data['ASIAN'] / region_data['TOT'] * 100
        region_data['NHPI_P'] = region_data['NHPI'] / region_data['TOT'] * 100
        region_data['SOR_P'] = region_data['SOR'] / region_data['TOT'] * 100
        feature = region_data[['TOT', 'H', 'H_P', 'W', 'W_P', 'B', 'B_P', 'ASIAN', 'ASIAN_P', 'NHPI', 'NHPI_P', 'SOR', 'SOR_P']].to_numpy()
        feature_name = ['TOT', 'Hsp_20', 'Hsp_20P', 'WNH_20', 'WNH_20P', 'BNH_20', 'BNH_20P', 'ANH_20', 'ANH_20P', 'NH2pl_20', 'NH2pl_20P', 'ONH_20', 'ONH_20P']
    elif args.city == 'NYC':
        # ['Hsp_20', 'WNH_20', 'BNH_20', 'ANH_20', 'NH2pl_20', 'ONH_20']
        TOTs = [
            region_data['Hsp_20'] / region_data['Hsp_20P'] * 100, 
            region_data['BNH_20'] / region_data['BNH_20P'] * 100,
            region_data['ANH_20'] / region_data['ANH_20P'] * 100,
            region_data['NH2pl_20'] / region_data['NH2pl_20P'] * 100,
            region_data['ONH_20'] / region_data['ONH_20P'] * 100,
        ]
        TOTs = np.stack([TOT.to_numpy() for TOT in TOTs], axis=0)
        region_data['TOT'] = (np.where(np.isnan(TOTs), 0., TOTs).sum(axis=0) / (np.where(np.isnan(TOTs), 0., 1.).sum(axis=0) + 1e-6)).round(0).astype(np.int64)
        region_data['WNH_20'] = (region_data['TOT'] * region_data['WNH_20P'] / 100).astype(np.int64)
        feature = region_data[['TOT', 'Hsp_20', 'Hsp_20P', 'WNH_20', 'WNH_20P', 'BNH_20', 'BNH_20P', 'ANH_20', 'ANH_20P', 'NH2pl_20', 'NH2pl_20P', 'ONH_20', 'ONH_20P']].to_numpy()
        feature_name = ['TOT', 'Hsp_20', 'Hsp_20P', 'WNH_20', 'WNH_20P', 'BNH_20', 'BNH_20P', 'ANH_20', 'ANH_20P', 'NH2pl_20', 'NH2pl_20P', 'ONH_20', 'ONH_20P']
    else:
        raise ValueError('Unknown city')

    np.savez(
        f'./dataset/{args.city}.npz', 
        incident = inc,
        mask = mask,
        threshold = threshold,
        s_adj = A,
        c_cor = c_cor,
        feature = feature,
        category_name = category,
        feature_name = feature_name,
    )
