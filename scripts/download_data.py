#!/usr/bin/env python
"""
数据下载脚本 - 2015年4月10日磁层收缩事件
论文: Wang et al. (2023) - Unusual shrinkage and reshaping of Earth's magnetosphere
"""

import os
from datetime import datetime

# 设置数据下载目录
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# 事件时间范围: 2015年4月10日
START_TIME = '2015-04-10 00:00:00'
END_TIME = '2015-04-10 23:59:59'

def download_omni_data():
    """下载OMNI太阳风和IMF数据"""
    from pyspedas.projects.omni import load

    print("=" * 60)
    print("下载OMNI太阳风数据...")
    print(f"时间范围: {START_TIME} 到 {END_TIME}")
    print("=" * 60)

    # 下载OMNI数据 (1分钟分辨率)
    omni_vars = load(
        trange=[START_TIME, END_TIME],
        datatype='1min',
        level='hro2',
        downloadonly=True
    )

    print(f"OMNI数据下载完成")
    return omni_vars

def download_dmsp_data():
    """下载DMSP卫星粒子数据 (SSJ5) - 需要从Madrigal获取"""
    print("=" * 60)
    print("DMSP SSJ5粒子数据下载...")
    print("=" * 60)
    print("注意: DMSP数据需要从Madrigal数据库获取")
    print("网址: https://cedar.openmadrigal.org")
    print("论文中使用: DMSP F16, F17, F18")
    print("时间: 2015-04-10")
    print("=" * 60)
    # pyspedas不支持DMSP，需要手动从Madrigal下载或使用其他工具

def download_themis_data():
    """下载THEMIS卫星数据"""
    from pyspedas.projects.themis import fgm

    print("=" * 60)
    print("下载THEMIS卫星数据...")
    print("=" * 60)

    try:
        themis_vars = fgm(
            trange=[START_TIME, END_TIME],
            probe='a',
            level='l2',
            downloadonly=True
        )
        print(f"THEMIS数据下载完成")
    except Exception as e:
        print(f"警告: THEMIS数据下载失败 - {e}")

def download_superdarn_data():
    """下载SuperDARN雷达数据 (使用pyDARNio)"""
    print("=" * 60)
    print("SuperDARN数据需要从vt.superdarn.org手动下载")
    print("论文中使用的时间: 2015-04-10")
    print("雷达站点: 查看论文补充材料获取具体站点")
    print("=" * 60)
    # pyDARNio主要用于读取本地数据，不直接下载
    # SuperDARN数据通常需要从 http://vt.superdarn.org 获取

def verify_data():
    """验证下载的数据"""
    import os

    print("=" * 60)
    print("验证数据完整性...")
    print("=" * 60)

    # 检查多个可能的数据目录
    data_dirs = [DATA_DIR, 'omni_data', 'themis_data']
    data_files = []

    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            for root, dirs, files in os.walk(data_dir):
                for f in files:
                    if f.endswith('.cdf') or f.endswith('.nc') or f.endswith('.dat'):
                        data_files.append(os.path.join(root, f))

    print(f"找到 {len(data_files)} 个数据文件")
    for f in data_files[:10]:  # 显示前10个
        print(f"  - {os.path.basename(f)}")

    if len(data_files) > 10:
        print(f"  ... 还有 {len(data_files) - 10} 个文件")

    return data_files

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("磁层收缩论文数据下载脚本")
    print("论文: Wang et al. (2023)")
    print("事件日期: 2015年4月10日")
    print("=" * 60 + "\n")

    # 下载各类数据
    omni_result = download_omni_data()
    dmsp_result = download_dmsp_data()
    themis_result = download_themis_data()
    superdarn_info = download_superdarn_data()

    # 验证数据
    data_files = verify_data()

    print("\n" + "=" * 60)
    print("数据下载完成!")
    print("=" * 60)

    return data_files

if __name__ == '__main__':
    main()