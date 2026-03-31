#!/usr/bin/env python
"""
验证下载的数据并绘制IMF时间序列
论文: Wang et al. (2023) - Figure 1复现
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import datetime

def load_and_verify_omni():
    """加载并验证OMNI数据"""
    from pyspedas.projects.omni import load
    import pyspedas

    print("=" * 60)
    print("加载OMNI数据...")
    print("=" * 60)

    # 加载OMNI数据
    trange = ['2015-04-10 00:00:00', '2015-04-10 23:59:59']
    vars = load(
        trange=trange,
        datatype='1min',
        level='hro2',
        notplot=True  # 返回数据而不是tplot变量
    )

    # 打印可用变量
    if vars:
        print("可用变量:")
        for key in vars.keys():
            print(f"  - {key}")

    return vars

def load_and_verify_themis():
    """加载并验证THEMIS数据"""
    from pyspedas.projects.themis import fgm
    import pyspedas

    print("=" * 60)
    print("加载THEMIS FGM数据...")
    print("=" * 60)

    trange = ['2015-04-10 00:00:00', '2015-04-10 23:59:59']
    vars = fgm(
        trange=trange,
        probe='a',
        level='l2',
        notplot=True
    )

    if vars:
        print("可用变量:")
        for key in vars.keys():
            print(f"  - {key}")

    return vars

def plot_imf_timeseries(omni_vars):
    """绘制IMF时间序列图 (Figure 1复现)"""
    print("=" * 60)
    print("绘制IMF时间序列图...")
    print("=" * 60)

    # 获取IMF分量
    if omni_vars is None:
        print("错误: OMNI数据未加载")
        return

    # 查找IMF变量
    bx = None
    by = None
    bz = None
    time = None

    for key, data in omni_vars.items():
        if 'BX' in key.upper() and 'GSE' in key.upper():
            bx = data['y']
            time = data['x']
        elif 'BY' in key.upper() and 'GSE' in key.upper():
            by = data['y']
        elif 'BZ' in key.upper() and 'GSE' in key.upper():
            bz = data['y']
        elif 'Bx' in key and 'GSE' not in key.upper():
            # 尝试其他命名方式
            bx = data['y']
            time = data['x']

    # 如果没有找到，尝试直接访问
    if time is None:
        # 尝试查找时间变量
        for key, data in omni_vars.items():
            if 'Epoch' in key or 'Time' in key or 'time' in key:
                time = data
                break

    print(f"时间变量: {time is not None}")
    print(f"Bx变量: {bx is not None}")
    print(f"By变量: {by is not None}")
    print(f"Bz变量: {bz is not None}")

    # 如果找到数据，绘制
    if time is not None and bz is not None:
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

        # 转换时间格式
        if isinstance(time[0], (float, np.float64)):
            # pyspedas使用Unix时间戳
            times = [datetime.datetime.utcfromtimestamp(t) for t in time]
        else:
            times = time

        # Plot IMF components
        if bx is not None:
            axes[0].plot(times, bx, 'b-', linewidth=0.5)
            axes[0].set_ylabel('Bx (nT)')
            axes[0].set_ylim(-20, 20)
            axes[0].grid(True, alpha=0.3)

        if by is not None:
            axes[1].plot(times, by, 'g-', linewidth=0.5)
            axes[1].set_ylabel('By (nT)')
            axes[1].set_ylim(-30, 30)
            axes[1].grid(True, alpha=0.3)

        axes[2].plot(times, bz, 'r-', linewidth=0.5)
        axes[2].set_ylabel('Bz (nT)')
        axes[2].set_ylim(-30, 30)
        axes[2].grid(True, alpha=0.3)
        # 标记北向IMF区域
        axes[2].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        axes[2].fill_between(times, bz, 0, where=bz > 0, alpha=0.3, color='red', label='Northward IMF')

        # 绘制|B|
        if bx is not None and by is not None:
            b_mag = np.sqrt(bx**2 + by**2 + bz**2)
            axes[3].plot(times, b_mag, 'k-', linewidth=0.5)
            axes[3].set_ylabel('|B| (nT)')
            axes[3].grid(True, alpha=0.3)

        # 设置时间格式
        axes[-1].xaxis.set_major_formatter(DateFormatter('%H:%M'))
        axes[-1].set_xlabel('UTC Time (2015-04-10)')

        plt.suptitle('IMF Components - 2015 April 10 Event\n(Wang et al., 2023 - Figure 1 Reproduction)', fontsize=12)
        plt.tight_layout()

        # 保存图像
        output_file = 'output/imf_timeseries_20150410.png'
        plt.savefig(output_file, dpi=150)
        print(f"图像已保存: {output_file}")
        plt.close()
    else:
        print("无法绘制IMF图: 缺少必要数据变量")

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("数据验证和初步分析")
    print("=" * 60 + "\n")

    # 创建输出目录
    import os
    os.makedirs('output', exist_ok=True)

    # 加载OMNI数据
    omni_vars = load_and_verify_omni()

    # 加载THEMIS数据
    themis_vars = load_and_verify_themis()

    # 绘制IMF时间序列
    if omni_vars:
        plot_imf_timeseries(omni_vars)

    print("\n" + "=" * 60)
    print("数据验证完成!")
    print("=" * 60)

if __name__ == '__main__':
    main()