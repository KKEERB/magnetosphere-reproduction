# DMSP数据获取尝试总结

## 自动下载尝试结果

| 方法 | 状态 | 结果 |
|------|------|------|
| CDAWeb (cdasws) | ❌ | DMSP SSJ数据只到2014年底 |
| pyspedas | ❌ | 不支持DMSP |
| pysatMadrigal | ⚠️ | 只有ABI极光边界指数，无粒子数据 |
| JHU/APL HTTP | ❌ | 403/需要交互式表单 |
| Madrigal REST API | ❌ | 404 |
| Madrigal Web API | ⚠️ | 找到文件但无法直接下载 |

## 已确认可用但需手动操作

### 1. JHU/APL在线数据检索
- **网址**: http://sd-www.jhuapl.edu/Aurora/data/data_step1.cgi
- **可用卫星**: F13, F15 (2015-04-10)
- **步骤**:
  1. 选择日期: 2015年4月10日
  2. 选择卫星: F15
  3. 选择数据类型
  4. 下载

### 2. SSUSI极光成像
- **网址**: https://ssusi.jhuapl.edu/data_products
- **选择**: F16/F17/F18 AUR
- **时间**: 2015年 DOY 100

### 3. Madrigal注册后下载
- **注册**: https://cedar.openmadrigal.org/register
- **数据**: DMSP完整粒子数据

## 替代方案

### 使用其他卫星数据
如果DMSP数据确实难以获取，可考虑：
- **POES/MetOp**: 粒子沉降数据
- **THEMIS**: 已下载的磁场数据
- **SuperDARN**: 对流数据

### 联系论文作者
Wang et al. (2023)论文的补充材料可能包含数据链接。

## 代码已准备

以下脚本已创建，获取数据后可直接使用：
- `scripts/dmsp_ssusi_analysis.py`
- `scripts/dmsp_ssj5_analysis.py`
- `ssj_auroral_boundary` 库已安装

## 建议

由于2015年DMSP数据自动获取困难，建议：
1. **手动下载**: 访问上述网站手动获取
2. **继续Phase 3**: 先进行SuperDARN数据准备
3. **联系作者**: 获取论文补充数据