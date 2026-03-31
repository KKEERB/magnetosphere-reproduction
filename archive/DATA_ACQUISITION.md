# DMSP数据获取指南

## 问题说明

**CDAWeb数据限制**: CDAWeb的DMSP SSJ数据只到2014年底，没有2015年4月10日的数据。

## 替代数据源

### 1. JHU/APL在线数据检索 (推荐)

**SSJ5粒子数据**:
- 网址: http://sd-www.jhuapl.edu/Aurora/data/data_step1.cgi
- 步骤:
  1. 选择日期: 2015年4月10日
  2. 选择卫星: F16, F17, 或 F18
  3. 下载数据文件

**SSUSI极光成像**:
- 网址: https://ssusi.jhuapl.edu/data_products
- 选择: F17_AUR 或 F18_AUR
- 时间: 2015年 DOY 100

### 2. Madrigal数据库

- 网址: https://cedar.openmadrigal.org
- 需要注册免费账户
- 数据更完整

### 3. 直接联系论文作者

如果上述方法失败，可以联系论文作者获取数据。

## 代码示例 (获取数据后)

```python
# 读取SSUSI NetCDF文件
import netCDF4 as nc

dataset = nc.Dataset('ssusi_file.nc')
print(dataset.variables.keys())

# 读取SSJ5 CDF文件
import cdflib
data = cdflib.CDF('ssj5_file.cdf')
print(data.cdf_info())
```

## 已下载的数据

- ✅ OMNI太阳风数据 (CDAWeb)
- ✅ THEMIS FGM数据 (pyspedas)
- ⏳ DMSP SSUSI (需要手动下载)
- ⏳ DMSP SSJ5 (需要手动下载)