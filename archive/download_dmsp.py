#!/usr/bin/env python
"""
尝试使用多种方法下载DMSP数据
"""

import os
import requests
from datetime import datetime

# 事件日期
EVENT_DATE = '2015-04-10'
YEAR = 2015
DOY = 100  # 2015年4月10日是第100天

def download_ssusi_via_http():
    """尝试直接HTTP下载SSUSI数据"""
    print("=" * 60)
    print("尝试下载SSUSI数据...")
    print("=" * 60)

    # SSUSI数据URL模式
    # https://ssusi.jhuapl.edu/data/F17_AUR/2015/100/
    satellites = ['F16', 'F17', 'F18']

    for sat in satellites:
        # 尝试不同的URL模式
        urls_to_try = [
            f"https://ssusi.jhuapl.edu/data/{sat}_AUR/{YEAR}/{DOY:03d}",
            f"https://ssusi.jhuapl.edu/{sat}_AUR/{YEAR}/{DOY:03d}",
            f"https://ssusi.jhuapl.edu/data_products/{sat}_AUR/{YEAR}/{DOY:03d}",
        ]

        for url in urls_to_try:
            print(f"尝试: {url}")
            try:
                resp = requests.get(url, timeout=30)
                print(f"  状态码: {resp.status_code}")
                if resp.status_code == 200:
                    print(f"  成功! 返回长度: {len(resp.text)}")
                    # 检查是否是目录列表
                    if 'NetCDF' in resp.text or '.nc' in resp.text:
                        print(f"  找到NetCDF文件链接!")
                    break
            except Exception as e:
                print(f"  错误: {e}")

def download_ssj5_via_jhuapl():
    """从JHU/APL下载SSJ5数据"""
    print("\n" + "=" * 60)
    print("JHU/APL SSJ5数据下载")
    print("=" * 60)

    # JHU/APL在线数据检索
    print("""
    手动下载步骤:
    1. 访问: http://sd-www.jhuapl.edu/Aurora/data/data_step1.cgi
    2. 选择日期: 2015年4月10日
    3. 选择卫星: F16, F17, 或 F18
    4. 下载SSJ5数据文件

    数据格式: IDL保存文件 (.sav) 或 CDF格式
    """)

def try_pysatmadrigal():
    """尝试使用pysatMadrigal下载"""
    print("\n" + "=" * 60)
    print("尝试pysatMadrigal下载...")
    print("=" * 60)

    try:
        import pysat
        from pysatMadrigal.instruments import dmsp_ssj

        # 设置数据目录
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'dmsp')
        os.makedirs(data_dir, exist_ok=True)

        pysat.params['data_dirs'] = data_dir
        print(f"数据目录: {data_dir}")

        # 尝试加载SSJ5数据
        print("\n尝试加载F17 SSJ5数据...")

        # 注意: 可能需要Madrigal用户注册
        print("""
    需要Madrigal用户注册:
    1. 访问: https://cedar.openmadrigal.org/register
    2. 注册免费账户
    3. 设置用户信息:
       pysat.params['user_info'] = {{
           'user_name': 'your_name',
           'user_email': 'your_email',
           'user_affiliation': 'your_affiliation'
       }}
        """)

    except Exception as e:
        print(f"pysatMadrigal尝试失败: {e}")

def download_via_cdaWeb():
    """从NASA CDAWeb下载DMSP数据"""
    print("\n" + "=" * 60)
    print("从CDAWeb下载DMSP数据")
    print("=" * 60)

    # CDAWeb可能有的DMSP数据
    print("""
    CDAWeb DMSP数据:
    1. 访问: https://cdaweb.gsfc.nasa.gov/
    2. 搜索: DMSP
    3. 选择数据集:
       - DMSP-F17_SSJ5 (粒子数据)
       - DMSP-F18_SSJ5
    4. 选择时间: 2015-04-10
    5. 下载CDF文件
    """)

    # 尝试使用cdasws
    try:
        from cdasws import CdasWs
        cdas = CdasWs()

        # 搜索DMSP数据集
        print("\n搜索DMSP数据集...")
        datasets = cdas.get_datasets('DMSP')
        if datasets:
            print(f"找到 {len(datasets)} 个DMSP数据集:")
            for ds in datasets[:10]:
                print(f"  - {ds['Id']}: {ds.get('Name', '')}")
    except Exception as e:
        print(f"CDAWeb查询失败: {e}")

def main():
    print("=" * 60)
    print("DMSP数据下载尝试")
    print(f"事件日期: {EVENT_DATE} (DOY {DOY})")
    print("=" * 60)

    # 创建数据目录
    os.makedirs('data/dmsp', exist_ok=True)

    # 尝试各种下载方法
    download_ssusi_via_http()
    download_ssj5_via_jhuapl()
    try_pysatmadrigal()
    download_via_cdaWeb()

    print("\n" + "=" * 60)
    print("下载尝试完成")
    print("=" * 60)

if __name__ == '__main__':
    main()