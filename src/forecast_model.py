"""
车企配件采购计划半自动生成系统
Phase 3: 预测模型核心算法

功能：
- 月均需求计算（加权平均法）
- 移动平均法 (SMA)
- 指数平滑法 (Exponential Smoothing)
- Holt-Winters 季节性模型
- Croston 法（间歇性需求）
- 模型自动选择
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import Tuple, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/forecast_model.log', encoding='utf-8', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DemandForecaster:
    """配件需求预测器"""

    def __init__(self, data: pd.DataFrame, monthly_cols: list):
        """
        初始化预测器

        Args:
            data: 配件数据 DataFrame
            monthly_cols: 月度数据列名列表（按时间排序）
        """
        self.data = data
        self.monthly_cols = monthly_cols
        self.forecast_results = None

        logger.info(f"初始化预测器，{len(monthly_cols)} 个月度列")

    def get_monthly_data(self, row_idx: int, lookback_months: int = 12) -> np.ndarray:
        """
        获取指定配件的月度需求数据

        Args:
            row_idx: 数据行索引
            lookback_months: 回溯月数

        Returns:
            月度需求数组
        """
        # 取最近 N 个月的数据
        recent_cols = self.monthly_cols[-lookback_months:] if len(self.monthly_cols) >= lookback_months else self.monthly_cols

        # 获取数据并转换为数值
        row_data = self.data.iloc[row_idx]
        demand = []
        for col in recent_cols:
            val = row_data.get(col, 0)
            try:
                demand.append(float(val) if pd.notna(val) else 0.0)
            except (ValueError, TypeError):
                demand.append(0.0)

        return np.array(demand)

    def calc_weighted_average(self, demand: np.ndarray) -> float:
        """
        计算加权平均需求

        权重公式：
        - 远期（前 3 个月）：20%
        - 中期（中间 2 月）：30%
        - 近期（最近 1 月）：50%

        Args:
            demand: 月度需求数组

        Returns:
            加权平均需求
        """
        if len(demand) < 1:
            return 0.0

        # 确保至少 6 个月数据
        n = min(6, len(demand))
        recent = demand[-n:]

        if n <= 3:
            # 数据不足 3 个月，简单平均
            return float(np.mean(recent))

        # 计算加权平均
        # J,K,L = 前 3 个月（远期，权重 20%）
        # M,N = 中间 2 月（中期，权重 30%）
        # O = 最近 1 月（近期，权重 50%）

        if n >= 6:
            JKL = np.mean(recent[:3])  # 前 3 个月平均
            MN = np.mean(recent[3:5])   # 中间 2 月平均
            O = recent[5]                # 最近 1 月

            weighted_avg = JKL * 0.2 + MN * 0.3 + O * 0.5
        elif n >= 5:
            JKL = np.mean(recent[:3])
            MN = np.mean(recent[3:5])
            O = recent[4]
            weighted_avg = JKL * 0.2 + MN * 0.3 + O * 0.5
        else:
            weighted_avg = np.mean(recent)

        return float(weighted_avg)

    def simple_moving_average(self, demand: np.ndarray, window: int = 12) -> float:
        """
        简单移动平均法 (SMA)

        Args:
            demand: 月度需求数组
            window: 窗口大小

        Returns:
            移动平均值
        """
        if len(demand) < window:
            window = len(demand)

        if window == 0:
            return 0.0

        return float(np.mean(demand[-window:]))

    def exponential_smoothing(self, demand: np.ndarray, alpha: float = 0.2) -> Tuple[float, float]:
        """
        指数平滑法

        公式：F(t+1) = α × A(t) + (1-α) × F(t)

        Args:
            demand: 月度需求数组
            alpha: 平滑系数 (0.1-0.3)

        Returns:
            (预测值，趋势值)
        """
        if len(demand) < 2:
            return (float(np.mean(demand)) if len(demand) > 0 else 0.0, 0.0)

        # 初始化
        forecast = demand[0]
        for actual in demand[1:]:
            forecast = alpha * actual + (1 - alpha) * forecast

        # 计算简单趋势（最后两点变化率）
        trend = demand[-1] - demand[-2] if len(demand) >= 2 else 0.0

        return (float(forecast), float(trend))

    def holt_winters(self, demand: np.ndarray, seasonal_periods: int = 12) -> float:
        """
        Holt-Winters 季节性模型（简化版）

        适用于有季节性波动的配件

        Args:
            demand: 月度需求数组
            seasonal_periods: 季节周期（默认 12 个月）

        Returns:
            下期预测值
        """
        if len(demand) < seasonal_periods * 2:
            # 数据不足，回退到指数平滑
            result, _ = self.exponential_smoothing(demand)
            return result

        # 计算水平分量 L
        L = np.mean(demand)

        # 计算季节因子
        seasonal_factors = []
        for i in range(seasonal_periods):
            season_vals = demand[i::seasonal_periods]
            if len(season_vals) > 0:
                season_avg = np.mean(season_vals)
                seasonal_factors.append(season_avg / L if L > 0 else 1.0)
            else:
                seasonal_factors.append(1.0)

        # 下期预测（假设下期是季节周期的第一个月）
        next_season_factor = seasonal_factors[0]

        # 简单趋势
        trend = (demand[-1] - demand[-seasonal_periods]) / seasonal_periods if len(demand) > seasonal_periods else 0

        forecast = (L + trend) * next_season_factor
        return max(0, float(forecast))

    def croston(self, demand: np.ndarray, alpha: float = 0.1) -> float:
        """
        Croston 法 - 用于间歇性需求

        分别预测：
        - 需求发生概率 p
        - 需求发生时的需求量 z

        最终预测：y = p × z

        Args:
            demand: 月度需求数组
            alpha: 平滑系数

        Returns:
            预测值
        """
        # 找到非零需求的位置和值
        non_zero_mask = demand > 0
        non_zero_indices = np.where(non_zero_mask)[0]
        non_zero_values = demand[non_zero_mask]

        if len(non_zero_values) == 0:
            return 0.0

        if len(non_zero_values) == 1:
            return float(non_zero_values[0])

        # 计算需求间隔
        intervals = np.diff(non_zero_indices)

        if len(intervals) == 0:
            return float(np.mean(non_zero_values))

        # 初始化 Croston 分量
        z = non_zero_values[0]  # 需求量
        p = 1.0 / intervals[0] if intervals[0] > 0 else 1.0  # 需求概率

        # 更新
        for i in range(1, len(non_zero_values)):
            z = alpha * non_zero_values[i] + (1 - alpha) * z
            if i < len(intervals):
                p = alpha * (1.0 / intervals[i]) + (1 - alpha) * p

        forecast = z * p
        return max(0, float(forecast))

    def detect_demand_pattern(self, demand: np.ndarray) -> str:
        """
        检测需求模式

        Args:
            demand: 月度需求数组

        Returns:
            需求模式：'stable', 'seasonal', 'intermittent', 'trend', 'new'
        """
        n = len(demand)

        if n < 3:
            return 'new'  # 新品，数据不足

        # 计算零需求比例
        zero_ratio = np.sum(demand == 0) / n

        # 间歇性需求：超过 50% 的月份需求为 0
        if zero_ratio > 0.5:
            return 'intermittent'

        # 计算变异系数 (CV)
        mean_demand = np.mean(demand)
        std_demand = np.std(demand)
        cv = std_demand / mean_demand if mean_demand > 0 else 0

        # 计算自相关系数（检测季节性）
        if n >= 24:
            autocorr = np.corrcoef(demand[:-12], demand[12:])[0, 1]
            if autocorr > 0.5:
                return 'seasonal'

        # 检测趋势
        if n >= 6:
            first_half = np.mean(demand[:n//2])
            second_half = np.mean(demand[n//2:])
            trend_change = (second_half - first_half) / first_half if first_half > 0 else 0
            if abs(trend_change) > 0.3:
                return 'trend'

        # 默认：稳定需求
        return 'stable'

    def select_model(self, demand: np.ndarray) -> str:
        """
        根据需求特征选择最佳预测模型

        Args:
            demand: 月度需求数组

        Returns:
            模型名称：'weighted_avg', 'sma', 'exp_smoothing', 'holt_winters', 'croston'
        """
        pattern = self.detect_demand_pattern(demand)

        logger.debug(f"需求模式：{pattern}, 零需求比：{np.sum(demand==0)/len(demand):.2f}")

        if pattern == 'new':
            return 'weighted_avg'  # 新品，使用加权平均
        elif pattern == 'intermittent':
            return 'croston'  # 间歇性需求
        elif pattern == 'seasonal':
            return 'holt_winters'  # 季节性需求
        elif pattern == 'trend':
            return 'exp_smoothing'  # 有趋势的需求
        else:
            return 'sma'  # 稳定需求，使用移动平均

    def forecast(self, row_idx: int, lookback_months: int = 12, auto_select: bool = True) -> dict:
        """
        执行预测

        Args:
            row_idx: 数据行索引
            lookback_months: 回溯月数
            auto_select: 是否自动选择模型

        Returns:
            预测结果字典
        """
        # 获取月度需求数据
        demand = self.get_monthly_data(row_idx, lookback_months)

        # 检测数据质量
        total_demand = np.sum(demand)
        zero_months = np.sum(demand == 0)
        non_zero_months = np.sum(demand > 0)

        # 如果完全没有需求
        if total_demand == 0:
            return {
                'model': 'none',
                'forecast': 0.0,
                'monthly_demand': 0.0,
                'confidence': 0.0,
                'demand_pattern': 'zero'
            }

        # 选择模型
        if auto_select:
            model = self.select_model(demand)
        else:
            model = 'weighted_avg'

        # 执行预测
        if model == 'croston':
            forecast_val = self.croston(demand)
        elif model == 'holt_winters':
            forecast_val = self.holt_winters(demand)
        elif model == 'exp_smoothing':
            forecast_val, _ = self.exponential_smoothing(demand)
        elif model == 'sma':
            forecast_val = self.simple_moving_average(demand)
        else:
            forecast_val = self.calc_weighted_average(demand)

        # 计算置信度（基于数据量）
        data_quality = non_zero_months / lookback_months
        confidence = min(1.0, data_quality + 0.2)  # 基础置信度

        # 检测需求模式
        pattern = self.detect_demand_pattern(demand)

        return {
            'model': model,
            'forecast': max(0, forecast_val),
            'monthly_demand': self.calc_weighted_average(demand),
            'confidence': round(confidence, 2),
            'demand_pattern': pattern,
            'total_demand_12m': float(total_demand),
            'zero_months': int(zero_months),
            'non_zero_months': int(non_zero_months)
        }

    def run_forecast_for_all(self, lookback_months: int = 12) -> pd.DataFrame:
        """
        对所有配件执行预测

        Args:
            lookback_months: 回溯月数

        Returns:
            预测结果 DataFrame
        """
        logger.info(f"开始对所有 {len(self.data)} 个配件执行预测...")

        results = []

        for idx in range(len(self.data)):
            forecast_result = self.forecast(idx, lookback_months)
            results.append(forecast_result)

            # 每 1000 条记录输出一次进度
            if (idx + 1) % 1000 == 0:
                logger.info(f"  进度：{idx + 1}/{len(self.data)}")

        # 转换为 DataFrame
        self.forecast_results = pd.DataFrame(results)

        # 统计模型使用情况
        model_counts = self.forecast_results['model'].value_counts()
        logger.info(f"预测模型使用情况:\n{model_counts}")

        # 统计置信度分布
        avg_confidence = self.forecast_results['confidence'].mean()
        logger.info(f"平均置信度：{avg_confidence:.2f}")

        return self.forecast_results

    def get_forecast_summary(self) -> dict:
        """
        获取预测摘要统计

        Returns:
            摘要统计字典
        """
        if self.forecast_results is None:
            raise ValueError("请先运行预测")

        summary = {
            'total_items': len(self.forecast_results),
            'model_distribution': self.forecast_results['model'].value_counts().to_dict(),
            'avg_confidence': self.forecast_results['confidence'].mean(),
            'avg_monthly_demand': self.forecast_results['monthly_demand'].mean(),
            'zero_demand_items': (self.forecast_results['forecast'] == 0).sum(),
        }

        # 按需求模式统计
        summary['pattern_distribution'] = self.forecast_results['demand_pattern'].value_counts().to_dict()

        return summary


def main():
    """主函数 - 测试预测模型模块"""

    # 配置路径
    input_dir = Path("C:/Users/Lenovo/配件采购系统/data/output")
    output_dir = Path("C:/Users/Lenovo/配件采购系统/data/output")

    logger.info("=" * 60)
    logger.info("车企配件采购系统 - 预测模型核心算法模块")
    logger.info("=" * 60)

    # 1. 加载数据
    print("\n[1/5] 加载数据...")

    # 加载 ABC 分类结果
    abc_files = sorted(input_dir.glob("abc_classification_*.xlsx"))
    if not abc_files:
        raise FileNotFoundError("未找到 ABC 分类结果，请先运行 Phase 2")

    abc_file = abc_files[-1]
    logger.info(f"加载 ABC 分类文件：{abc_file.name}")
    abc_data = pd.read_excel(abc_file, engine='openpyxl')
    logger.info(f"ABC 分类数据行数：{len(abc_data)}")

    # 加载清洗后的原始数据（包含月度列）
    cleaned_files = sorted(input_dir.glob("cleaned_data_*.xlsx"))
    if not cleaned_files:
        raise FileNotFoundError("未找到清洗后的数据文件，请先运行 Phase 1")

    cleaned_file = cleaned_files[-1]
    logger.info(f"加载清洗数据文件：{cleaned_file.name}")
    raw_data = pd.read_excel(cleaned_file, engine='openpyxl')
    logger.info(f"清洗数据行数：{len(raw_data)}")

    # 2. 从原始数据识别月度列
    print("\n[2/5] 识别月度列...")
    monthly_cols = []
    for col in raw_data.columns:
        col_str = str(col)
        if '年' in col_str and '月' in col_str:
            monthly_cols.append(col)

    # 排序月度列
    def parse_year_month(col):
        try:
            parts = str(col).split('年')
            if len(parts) >= 2:
                year = int(parts[0][-2:])
                month_part = parts[1].split('月')[0]
                month = int(month_part)
                return (year, month)
        except:
            pass
        return (99, 99)

    monthly_cols = sorted(monthly_cols, key=parse_year_month)
    logger.info(f"识别到 {len(monthly_cols)} 个月度列")

    # 合并 ABC 分类结果和原始数据（用于月度列）
    # 只取需要的列
    data = pd.concat([
        abc_data[['配件号', '名称', '单位', 'ABC 分类', '存储系数', 'H 系等级', 'H 系系数']].reset_index(drop=True),
        raw_data[[col for col in monthly_cols]].reset_index(drop=True)
    ], axis=1)
    logger.info(f"合并后数据行数：{len(data)}")

    # 3. 创建预测器并运行预测
    print("\n[3/5] 运行预测模型...")
    forecaster = DemandForecaster(data, monthly_cols)
    forecast_results = forecaster.run_forecast_for_all(lookback_months=12)

    # 合并结果到数据
    base_cols = ['配件号']
    if '名称' in data.columns:
        base_cols.append('名称')
    if 'ABC 分类' in data.columns:
        base_cols.extend(['ABC 分类', '存储系数', 'H 系等级', 'H 系系数'])

    result_df = pd.concat([
        data[base_cols].reset_index(drop=True),
        forecast_results
    ], axis=1)

    # 4. 显示摘要
    print("\n[4/5] 预测摘要:")
    summary = forecaster.get_forecast_summary()

    print(f"\n=== 模型使用分布 ===")
    for model, count in sorted(summary['model_distribution'].items()):
        pct = count / summary['total_items'] * 100
        print(f"  {model}: {count} 件 ({pct:.1f}%)")

    print(f"\n=== 需求模式分布 ===")
    for pattern, count in sorted(summary['pattern_distribution'].items()):
        pct = count / summary['total_items'] * 100
        print(f"  {pattern}: {count} 件 ({pct:.1f}%)")

    print(f"\n平均置信度：{summary['avg_confidence']:.2f}")
    print(f"零需求配件：{summary['zero_demand_items']} 件")

    # 5. 保存结果
    print("\n[5/5] 保存预测结果...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"forecast_result_{timestamp}.xlsx"
    result_df.to_excel(output_file, index=False, engine='openpyxl')
    logger.info(f"保存到：{output_file}")

    logger.info("=" * 60)
    logger.info("Phase 3 完成!")
    logger.info("=" * 60)

    return forecaster


if __name__ == "__main__":
    main()
