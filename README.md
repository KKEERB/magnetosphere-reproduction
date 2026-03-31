# Magnetosphere Reproduction: Wang et al. (2023)

复现 Wang et al. (2023) "Unusual shrinkage and reshaping of Earth's magnetosphere under a strong northward interplanetary magnetic field" (Communications Earth & Environment, DOI: 10.1038/s43247-023-00700-0)

## 事件概述

**2015年4月10日** — IMF Bz持续强北向（>15 nT达3小时），触发双叶重联（DLR），导致磁层顶大幅收缩、极盖区关闭、出现马蹄形极光（HCA）和反向对流。

## 项目结构

```
magnetosphere-reproduction/
├── scripts/
│   ├── read_ssj_binary.py      # [核心] SSJ5二进制解析器(128字节/记录)
│   ├── auroral_boundary.py     # [核心] 极光边界检测(标准+HCA双模式)
│   ├── dmsp_ssj5_analysis.py   # SSJ5光谱图生成(57张)
│   ├── dmsp_ssusi_analysis.py  # SSUSI极光成像处理
│   ├── superdarn_analysis.py   # SuperDARN对流分析(含自动下载)
│   ├── figure2_reproduction.py # 论文Figure 2/3复现
│   ├── download_data.py        # OMNI/THEMIS数据下载
│   └── verify_and_plot.py      # IMF时序图
├── data/
│   ├── dmsp/                   # JHU/APL SSJ5二进制文件(.f17, .f18)
│   ├── superdarn/              # SuperDARN对流图(30张JPG)
│   └── ssusi/                  # SSUSI极光数据
├── output/                     # 所有输出图表(134个PNG)
└── archive/                    # 废弃脚本和临时文件
```

## 快速使用

### 环境要求

- Python 3.10+ (本项目使用 C:/Python314/python.exe)
- 依赖: `pip install numpy matplotlib pillow pyspedas cdasws ssj_auroral_boundary`

### SSJ5二进制数据解析

```python
import sys; sys.path.insert(0, 'scripts')
from read_ssj_binary import read_ssj_file, find_polar_passes, CHANNEL_ENERGIES

data = read_ssj_file('data/dmsp/2015apr10.f17')
print(f"F{data['satellite']:02d}, {data['n_records']} records")
print(f"MLat: [{data['mlat'].min():.1f}, {data['mlat'].max():.1f}]")
print(f"MLT:  [{data['mlt'].min():.1f}, {data['mlt'].max():.1f}]")

passes = find_polar_passes(data['mlat'])
print(f"Polar passes: {len(passes)}")
```

### 极光边界检测

```python
from auroral_boundary import hardy_integrate, moving_average, detect_boundaries

s, e = passes[0]
eflux = data['eflux_19_rescaled'][:, s:e]
intflux = hardy_integrate(eflux, CHANNEL_ENERGIES)
intflux_smooth = moving_average(intflux, 15)
bnd = detect_boundaries(intflux_smooth, data['mlat'][s:e], data['sod'][s:e])
print(f"Mode: {bnd['mode']}, Poleward edge: {bnd['poleward_edge_mlat']}")
```

### 运行全部分析

```bash
python scripts/download_data.py        # 下载数据(首次)
python scripts/verify_and_plot.py      # IMF时序图
python scripts/dmsp_ssj5_analysis.py   # SSJ5光谱图
python scripts/auroral_boundary.py     # 极光边界检测
python scripts/superdarn_analysis.py   # SuperDARN分析
python scripts/figure2_reproduction.py # 论文图表复现
```

## JHU/APL SSJ5二进制格式

128字节/记录，小端序。完整解析见 `scripts/read_ssj_binary.py`。

| 偏移 | 大小 | 字段 |
|------|------|------|
| 0 | 1 | 卫星号 (uint8) |
| 1 | 1 | 年份后两位 |
| 2-3 | 2 | DOY (int16) |
| 4-5 | 2 | 地理纬度 x10 (int16) |
| 6-7 | 2 | 地理经度 x10 (int16) |
| 8-47 | 80 | 20通道电子计数 (20 x int16) |
| 48-87 | 80 | 20通道离子计数 (20 x int16) |
| 88-91 | 4 | 秒/天 (int32) |
| 92-93 | 2 | 版本号 x1000 |
| 94-97 | 4 | 标志位 |
| 98-121 | 24 | NORAD位置 + 磁坐标 + MLT (int32 x 6) |
| 122-127 | 6 | 填充 |

通道能量 (19物理通道, eV):
[30000, 20400, 13900, 9450, 6460, 4400, 3000, 2040, 1392, 949, 646, 440, 300, 204, 139, 95, 65, 44, 30]

## 关键科学结果

| 发现 | 详情 |
|------|------|
| 极光边界推进 | EQ1=80.6, PO1=83.1 (F17 Pass #22, 18:00 UT) |
| HCA模式 | 多个过境检测到极向边缘>80 MLat |
| 反向对流 | Phi_pc 从 62 kV 降至 29 kV |
| 极盖关闭 | SSJ5观测到极高纬度连续沉降(极盖基本消失) |

## 数据来源

| 数据 | 来源 | 状态 |
|------|------|------|
| OMNI (IMF/太阳风) | cdaweb.gsfc.nasa.gov | 已下载 |
| THEMIS FGM | themis.ssl.berkeley.edu | 已下载 |
| DMSP SSJ5 (F17/F18) | sd-www.jhuapl.edu | 已下载 |
| DMSP SSUSI (F17) | ssusi.jhuapl.edu | 已下载 |
| SuperDARN 对流图 | davit.ece.vt.edu | 30张已下载 |

## 参考文献

1. Wang et al. (2023) - Commun. Earth Environ., 主论文
2. Song & Russell (1992) - JGR, 双叶重联模型
3. Milan et al. (2020) - JGR, HCA与双叶重联
4. Hones et al. (1989) - GRL, HCA首次定义
5. Wing et al. (2014) - Space Sci. Rev., CDPS综述
