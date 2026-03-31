#!/usr/bin/env python
"""
DMSP SSUSI极光成像数据处理
论文: Wang et al. (2023) - Figure 2复现

SSUSI数据说明:
- LBHS: 140-150 nm (短波LBH)
- LBHL: 165-180 nm (长波LBH)
- 这些波段对极光粒子沉降敏感

数据来源: https://ssusi.jhuapl.edu/data_products
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from datetime import datetime
import urllib.request
import requests

# DMSP卫星列表 (SSUSI搭载)
SATELLITES = ['F16', 'F17', 'F18', 'F19']

# SSUSI数据URL模板
SSUSI_BASE_URL = "https://ssusi.jhuapl.edu/data_products"

def download_ssusi_auroral_edr(satellite, year, day_of_year, output_dir='data/ssusi'):
    """
    下载SSUSI Auroral EDR数据

    Auroral EDR包含:
    - 电子能量通量 (Q)
    - 平均能量 (E0)
    - 极光边界 (equatorward和poleward)
    - 极光弧识别
    - 半球功率
    """
    os.makedirs(output_dir, exist_ok=True)

    # 构建URL路径
    # 示例: https://ssusi.jhuapl.edu/F17_AUR/2015/100/
    aur_dir = f"{satellite}_AUR"
    url_path = f"{SSUSI_BASE_URL}/{aur_dir}/{year}/{day_of_year:03d}"

    print(f"尝试获取SSUSI数据目录: {url_path}")

    try:
        response = requests.get(url_path, timeout=30)
        if response.status_code == 200:
            print(f"找到数据目录")
            # 解析页面找到NetCDF文件
            # 实际下载需要手动或使用其他方法
            return url_path
        else:
            print(f"未找到数据: {response.status_code}")
            return None
    except Exception as e:
        print(f"请求失败: {e}")
        return None

def read_ssusi_netcdf(filepath):
    """
    读取SSUSI NetCDF文件并提取关键数据

    关键变量:
    - Latitude, Longitude: 地理坐标
    - Magnetic_Local_Time: 磁地方时
    - Magnetic_Latitude: 磁纬度
    - LBHS_Radiance: 短波LBH辐射
    - LBHL_Radiance: 长波LBH辐射
    - Auroral_Energy_Flux: 极光能量通量
    - Auroral_Mean_Energy: 平均能量
    - Equatorward_Boundary: 向赤道边界
    - Poleward_Boundary: 向极边界
    """
    try:
        dataset = nc.Dataset(filepath)

        print("SSUSI文件变量:")
        for var in dataset.variables.keys():
            print(f"  - {var}")

        data = {}

        # 尝试提取常见变量
        for key in ['Latitude', 'Longitude', 'Magnetic_Local_Time',
                    'Magnetic_Latitude', 'LBHS_Radiance', 'LBHL_Radiance']:
            if key in dataset.variables:
                data[key] = dataset.variables[key][:]

        dataset.close()
        return data

    except Exception as e:
        print(f"读取文件失败: {e}")
        return None

def plot_ssusi_auroral_image(data, output_file='output/ssusi_auroral.png'):
    """
    绘制SSUSI极光图像

    目标: 复现论文Figure 2的极光形态
    - LBHS波段辐射图
    - 磁地方时-磁纬度投影
    - 识别HCA (Horse-Collar Aurora)特征
    """
    if data is None:
        print("无数据可绘制")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # LBHS极光图
    if 'LBHS_Radiance' in data and 'Magnetic_Local_Time' in data:
        mlt = data['Magnetic_Local_Time']
        mlat = data['Magnetic_Latitude']
        lbhs = data['LBHS_Radiance']

        # 绘制极光辐射
        ax1 = axes[0]
        # 注意: 数据可能有特殊结构，需要适配
        if hasattr(lbhs, 'shape') and len(lbhs.shape) >= 2:
            im = ax1.pcolormesh(mlt, mlat, lbhs, shading='auto', cmap='hot')
            ax1.set_xlabel('MLT (hours)')
            ax1.set_ylabel('MLAT (degrees)')
            ax1.set_title('LBHS Auroral Radiance (140-150 nm)')
            plt.colorbar(im, ax=ax1, label='Radiance (R)')
        else:
            ax1.text(0.5, 0.5, '数据格式需要调整', ha='center', va='center')

    # LBHL极光图
    if 'LBHL_Radiance' in data:
        ax2 = axes[1]
        lbhl = data['LBHL_Radiance']
        if hasattr(lbhl, 'shape') and len(lbhl.shape) >= 2:
            im2 = ax2.pcolormesh(mlt, mlat, lbhl, shading='auto', cmap='hot')
            ax2.set_xlabel('MLT (hours)')
            ax2.set_ylabel('MLAT (degrees)')
            ax2.set_title('LBHL Auroral Radiance (165-180 nm)')
            plt.colorbar(im2, ax=ax2, label='Radiance (R)')
        else:
            ax2.text(0.5, 0.5, '数据格式需要调整', ha='center', va='center')

    plt.suptitle('DMSP SSUSI Auroral Imaging - 2015 April 10', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"图像已保存: {output_file}")
    plt.close()

def identify_hca_features(data):
    """
    识别HCA (Horse-Collar Aurora) 特征

    HCA特征:
    1. 两条跨极盖极光弧
    2. 位于晨昏两侧
    3. Poleward edges清晰可见
    4. 在北向IMF条件下形成

    方法:
    - 分析极光辐射的空间分布
    - 提取高辐射区域的边界
    - 识别晨昏两侧的弧状结构
    """
    print("=" * 60)
    print("HCA特征识别方法:")
    print("=" * 60)
    print("""
    识别步骤:
    1. 提取极光边界 (Equatorward/Poleward)
    2. 分析极盖区极光分布
    3. 识别晨昏两侧的极光弧
    4. 计算极盖区宽度变化

    论文中的HCA特征:
    - 两条明亮弧在MLT ~06 和 ~18
    - 极盖区显著收缩
    - Poleward edges对应OCB (开放-闭合边界)
    """)

    # 实际识别需要具体数据
    return None

def ssusi_data_info():
    """打印SSUSI数据获取指南"""
    print("=" * 60)
    print("DMSP SSUSI数据获取指南")
    print("=" * 60)
    print("""
    1. JHU/APL官方数据源:
       https://ssusi.jhuapl.edu/data_products

    2. 选择数据类型:
       - Auroral EDR (F16_AUR, F17_AUR, F18_AUR)
       - 包含极光边界、能量通量、平均能量

    3. 数据格式: NetCDF

    4. 时间选择:
       - 2015年4月10日 (DOY = 100)
       - 卫星: F16, F17, F18 (论文中使用)

    5. 关键波段:
       - LBHS: 140-150 nm (对O2吸收敏感)
       - LBHL: 165-180 nm (O2吸收较少)
       - 两波段比值可推断沉降粒子能量

    6. 投影方式:
       - 磁地方时 (MLT) - 磁纬度 (MLAT)
       - 需要使用AACGM或IGRF坐标转换
    """)
    print("=" * 60)

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("DMSP SSUSI极光成像处理")
    print("论文: Wang et al. (2023)")
    print("=" * 60 + "\n")

    # 创建输出目录
    os.makedirs('output', exist_ok=True)

    # 显示数据获取指南
    ssusi_data_info()

    # 2015年4月10日 = DOY 100
    year = 2015
    doy = 100

    # 尝试获取数据目录
    for sat in ['F17', 'F18', 'F16']:
        result = download_ssusi_auroral_edr(sat, year, doy)
        if result:
            print(f"\n{sat} 数据位置: {result}")

    # HCA识别说明
    identify_hca_features(None)

    print("\n" + "=" * 60)
    print("下一步: 手动下载SSUSI NetCDF文件后运行read_ssusi_netcdf()")
    print("=" * 60)

if __name__ == '__main__':
    main()