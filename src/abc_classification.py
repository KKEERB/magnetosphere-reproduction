"""
车企配件采购计划半自动生成系统
Phase 2: 需求特征分析和 ABC 分类模块

功能：
- 计算出库频次（T 系）
- 计算需求波动系数（H 系）
- 计算价格系数（E 系）
- 计算计划系数（P 系）
- ABC 分类判定
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/abc_classification.log', encoding='utf-8', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ABCClassifier:
    """配件 ABC 分类器"""

    # ABC 分类标准（基于 12 个月出库频次 T）
    # 已调整阈值以适应当前数据分布（原标准：A≥120, B 36-119, C 12-35, D 1-11, E=0）
    ABC_RULES = {
        'A': {'min_freq': 10, 'coef': 1.8},    # 高频需求：12 个月出库≥10 次（原 120 次）
        'B': {'min_freq': 6, 'max_freq': 9, 'coef': 1.5},  # 中高频需求：6-9 次（原 36-119 次）
        'C': {'min_freq': 4, 'max_freq': 5, 'coef': 1.3},  # 中频需求：4-5 次（原 12-35 次）
        'D': {'min_freq': 1, 'max_freq': 3, 'coef': 1.2},  # 低频需求：1-3 次（原 1-11 次）
        'E': {'max_freq': 0, 'coef': 0.0},     # 无需求
    }

    # 需求波动系数（H 系）- 基于 12 个月出库量
    H_RULES = {
        'H': {'min_qty': 6000, 'coef': 1.5},
        'M': {'min_qty': 120, 'max_qty': 5999, 'coef': 1.2},
        'L': {'min_qty': 1, 'max_qty': 119, 'coef': 1.0},
        'E': {'max_qty': 0, 'coef': 0.0},
    }

    # 价格系数（E 系）- 基于采购额
    E_RULES = {
        'E': {'max_amt': 0, 'coef': 0.0},
        'S': {'max_amt': 10, 'coef': 6.0},
        'U1': {'max_amt': 100, 'coef': 4.0},
        'U2': {'max_amt': 300, 'coef': 2.0},
        'U3': {'max_amt': 500, 'coef': 1.5},
        'V1': {'max_amt': 1000, 'coef': 1.2},
        'V2': {'max_amt': 3000, 'coef': 1.0},
        'V3': {'max_amt': 5000, 'coef': 1.0},
        'W1': {'max_amt': 10000, 'coef': 1.0},
        'W2': {'min_amt': 10000, 'coef': 1.0},
    }

    # 计划系数（P 系）- 基于采购频次
    P_RULES = {
        'P1': {'min_freq': 10, 'coef': 0.5},
        'P2': {'min_freq': 11, 'max_freq': 20, 'coef': 0.8},
        'P3': {'min_freq': 21, 'max_freq': 30, 'coef': 1.0},
        'P4': {'min_freq': 31, 'max_freq': 60, 'coef': 1.5},
        'P5': {'min_freq': 61, 'max_freq': 90, 'coef': 2.0},
        'P6': {'min_freq': 91, 'max_freq': 120, 'coef': 2.5},
        'P7': {'min_freq': 121, 'coef': 3.0},
        'Q': {'coef': 2.0},  # 无法确认
    }

    def __init__(self, data: pd.DataFrame):
        """
        初始化分类器

        Args:
            data: 清洗后的配件数据 DataFrame
        """
        self.data = data.copy()
        self.monthly_cols = []
        self.result = None

    def identify_monthly_columns(self) -> list:
        """
        识别月度数据列

        Returns:
            月度列名列表
        """
        # 匹配 "XX 年 X 月" 格式的列
        monthly_cols = []
        for col in self.data.columns:
            col_str = str(col)
            # 匹配 16 年 1 月 到 26 年 12 月 格式
            if '年' in col_str and '月' in col_str:
                monthly_cols.append(col)

        # 按年份和月份排序
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

        self.monthly_cols = sorted(monthly_cols, key=parse_year_month)
        logger.info(f"识别到 {len(self.monthly_cols)} 个月度列")
        return self.monthly_cols

    def calc_demand_frequency(self, lookback_months: int = 12) -> pd.Series:
        """
        计算出库频次（T 系）- 有需求的月份数

        Args:
            lookback_months: 回溯月数（默认 12 个月）

        Returns:
            出库频次 Series
        """
        if not self.monthly_cols:
            self.identify_monthly_columns()

        # 取最近 lookback_months 个月的数据
        recent_cols = self.monthly_cols[-lookback_months:] if len(self.monthly_cols) >= lookback_months else self.monthly_cols

        logger.info(f"计算出库频次，使用最近 {len(recent_cols)} 个月数据")

        # 将数据转换为数值类型（字符串转数字）
        monthly_data = self.data[recent_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

        # 计算有需求的月份数（值 > 0 的月份）
        freq = (monthly_data > 0).sum(axis=1)
        return freq

    def calc_demand_quantity(self, lookback_months: int = 12) -> pd.Series:
        """
        计算出库总量 - 最近 N 个月的出库总和

        Args:
            lookback_months: 回溯月数（默认 12 个月）

        Returns:
            出库总量 Series
        """
        if not self.monthly_cols:
            self.identify_monthly_columns()

        recent_cols = self.monthly_cols[-lookback_months:] if len(self.monthly_cols) >= lookback_months else self.monthly_cols

        # 转换为数值类型并计算总和（填充 NaN 为 0）
        monthly_data = self.data[recent_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        total = monthly_data.sum(axis=1)
        return total

    def classify_abc(self, freq: pd.Series) -> pd.DataFrame:
        """
        进行 ABC 分类

        Args:
            freq: 出库频次 Series

        Returns:
            包含分类结果的 DataFrame
        """
        logger.info("开始 ABC 分类...")

        result = pd.DataFrame()
        result['出库频次'] = freq

        # 初始化分类和系数列
        result['ABC 分类'] = 'Z'  # 默认观察件
        result['存储系数'] = 1.0

        # 应用 ABC 规则
        for grade, rule in self.ABC_RULES.items():
            if 'min_freq' in rule and 'max_freq' in rule:
                mask = (freq >= rule['min_freq']) & (freq <= rule['max_freq'])
            elif 'min_freq' in rule:
                mask = freq >= rule['min_freq']
            elif 'max_freq' in rule:
                mask = freq <= rule['max_freq']
            else:
                mask = pd.Series([False] * len(freq))

            result.loc[mask, 'ABC 分类'] = grade
            result.loc[mask, '存储系数'] = rule['coef']

        # 统计分类结果
        abc_counts = result['ABC 分类'].value_counts()
        logger.info(f"ABC 分类结果:\n{abc_counts}")

        return result

    def classify_h(self, qty: pd.Series) -> pd.DataFrame:
        """
        进行 H 系分类（需求波动系数）

        Args:
            qty: 出库总量 Series

        Returns:
            包含 H 系分类结果的 DataFrame
        """
        logger.info("开始 H 系分类（需求波动系数）...")

        result = pd.DataFrame()
        result['12 月出库总量'] = qty
        result['H 系等级'] = 'L'  # 默认
        result['H 系系数'] = 1.0

        # 应用 H 系规则（按优先级从高到低）
        for grade in ['H', 'M', 'L', 'E']:  # 按优先级排序
            rule = self.H_RULES[grade]
            if 'min_qty' in rule and 'max_qty' in rule:
                mask = (qty >= rule['min_qty']) & (qty <= rule['max_qty'])
            elif 'min_qty' in rule:
                mask = qty >= rule['min_qty']
            elif 'max_qty' in rule:
                mask = qty <= rule['max_qty']
            else:
                mask = pd.Series([False] * len(qty))

            result.loc[mask, 'H 系等级'] = grade
            result.loc[mask, 'H 系系数'] = rule['coef']

        h_counts = result['H 系等级'].value_counts()
        logger.info(f"H 系分类结果:\n{h_counts}")

        return result

    def classify_e(self, amount_col: str = None) -> pd.DataFrame:
        """
        进行 E 系分类（价格系数）

        Args:
            amount_col: 采购额列名

        Returns:
            包含 E 系分类结果的 DataFrame
        """
        logger.info("开始 E 系分类（价格系数）...")

        # 尝试自动查找采购额列
        if amount_col is None:
            for col in ['采购额', '采购金额', '金额']:
                if col in self.data.columns:
                    amount_col = col
                    break

        if amount_col is None or amount_col not in self.data.columns:
            logger.warning("未找到采购额列，跳过 E 系分类")
            return pd.DataFrame()

        amount = self.data[amount_col].fillna(0)

        result = pd.DataFrame()
        result['采购额'] = amount
        result['E 系等级'] = 'W2'  # 默认高额
        result['E 系系数'] = 1.0

        # 应用 E 系规则
        for grade in ['E', 'S', 'U1', 'U2', 'U3', 'V1', 'V2', 'V3', 'W1', 'W2']:
            rule = self.E_RULES[grade]
            if 'min_amt' in rule and 'max_amt' in rule:
                mask = (amount >= rule['min_amt']) & (amount <= rule['max_amt'])
            elif 'min_amt' in rule:
                mask = amount >= rule['min_amt']
            elif 'max_amt' in rule:
                mask = amount <= rule['max_amt']
            else:
                mask = pd.Series([False] * len(amount))

            result.loc[mask, 'E 系等级'] = grade
            result.loc[mask, 'E 系系数'] = rule['coef']

        e_counts = result['E 系等级'].value_counts()
        logger.info(f"E 系分类结果:\n{e_counts}")

        return result

    def classify_p(self, purchase_freq_col: str = None) -> pd.DataFrame:
        """
        进行 P 系分类（计划系数）

        Args:
            purchase_freq_col: 采购频次列名

        Returns:
            包含 P 系分类结果的 DataFrame
        """
        logger.info("开始 P 系分类（计划系数）...")

        # 尝试自动查找采购频次列
        if purchase_freq_col is None:
            # 使用已计算的出库频次作为替代
            if hasattr(self, '_last_freq'):
                freq = self._last_freq
            else:
                freq = self.calc_demand_frequency()
        else:
            freq = self.data[purchase_freq_col].fillna(0)

        result = pd.DataFrame()
        result['采购频次'] = freq
        result['P 系等级'] = 'Q'  # 默认无法确认
        result['P 系系数'] = 2.0

        # 应用 P 系规则
        for grade in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']:
            rule = self.P_RULES[grade]
            if 'min_freq' in rule and 'max_freq' in rule:
                mask = (freq >= rule['min_freq']) & (freq <= rule['max_freq'])
            elif 'min_freq' in rule:
                mask = freq >= rule['min_freq']
            else:
                mask = pd.Series([False] * len(freq))

            result.loc[mask, 'P 系等级'] = grade
            result.loc[mask, 'P 系系数'] = rule['coef']

        p_counts = result['P 系等级'].value_counts()
        logger.info(f"P 系分类结果:\n{p_counts}")

        return result

    def find_column(self, possible_names: list) -> str:
        """
        在数据中查找匹配列名

        Args:
            possible_names: 可能的列名列表

        Returns:
            匹配的列名，如果没有找到则返回 None
        """
        for col in self.data.columns:
            col_str = str(col)
            for name in possible_names:
                if name in col_str:
                    return col
        return None

    def run_full_classification(self) -> pd.DataFrame:
        """
        运行完整的分类流程

        Returns:
            包含所有分类结果的 DataFrame
        """
        logger.info("=" * 60)
        logger.info("开始完整 ABC 分类流程")
        logger.info("=" * 60)

        # 1. 识别月度列
        self.identify_monthly_columns()

        # 2. 计算出库频次
        freq = self.calc_demand_frequency(lookback_months=12)
        self._last_freq = freq

        # 3. 计算出库总量
        qty = self.calc_demand_quantity(lookback_months=12)

        # 4. 进行各类分类
        abc_result = self.classify_abc(freq)
        h_result = self.classify_h(qty)
        e_result = self.classify_e()
        p_result = self.classify_p()

        # 5. 查找关键字段（支持多种列名）
        part_no_col = self.find_column(['配件号', '配件', '图号', '编号'])
        name_col = self.find_column(['名称', '名字', '品名'])
        unit_col = self.find_column(['单位', '计量'])

        # 如果找不到，使用第一列作为配件号
        if part_no_col is None:
            part_no_col = self.data.columns[0]
            logger.warning(f"未找到配件号列，使用第一列：{part_no_col}")

        # 构建基础列 DataFrame
        base_cols = [part_no_col]
        if name_col and name_col in self.data.columns:
            base_cols.append(name_col)
        if unit_col and unit_col in self.data.columns:
            base_cols.append(unit_col)

        base_df = self.data[base_cols].reset_index(drop=True)
        # 重命名为标准名称
        base_df.columns = ['配件号', '名称', '单位'] if len(base_cols) >= 3 else ['配件号', '名称'][:len(base_cols)]

        # 6. 合并所有分类结果
        self.result = pd.concat([
            base_df,
            abc_result.reset_index(drop=True),
            h_result[['12 月出库总量', 'H 系等级', 'H 系系数']].reset_index(drop=True),
            e_result.reset_index(drop=True) if len(e_result) > 0 else pd.DataFrame(),
            p_result.reset_index(drop=True),
        ], axis=1)

        # 6. 计算综合系数
        if 'E 系系数' in self.result.columns and 'P 系系数' in self.result.columns:
            self.result['综合系数'] = (
                self.result['存储系数'] * 0.4 +
                self.result['H 系系数'] * 0.3 +
                self.result['E 系系数'] * 0.15 +
                self.result['P 系系数'] * 0.15
            )
        else:
            self.result['综合系数'] = self.result['存储系数']

        logger.info(f"分类完成，共 {len(self.result)} 条记录")
        logger.info(f"综合系数范围：{self.result['综合系数'].min():.2f} - {self.result['综合系数'].max():.2f}")

        return self.result

    def save_classification_result(self, output_path: str):
        """
        保存分类结果

        Args:
            output_path: 输出文件路径
        """
        if self.result is None:
            raise ValueError("请先运行分类")

        logger.info(f"保存分类结果到：{output_path}")
        self.result.to_excel(output_path, index=False, engine='openpyxl')
        logger.info("保存完成")

    def get_summary_stats(self) -> dict:
        """
        获取分类统计摘要

        Returns:
            统计摘要字典
        """
        if self.result is None:
            raise ValueError("请先运行分类")

        summary = {
            'total_items': len(self.result),
            'abc_distribution': self.result['ABC 分类'].value_counts().to_dict(),
            'h_distribution': self.result['H 系等级'].value_counts().to_dict() if 'H 系等级' in self.result.columns else {},
            'e_distribution': self.result['E 系等级'].value_counts().to_dict() if 'E 系等级' in self.result.columns else {},
            'p_distribution': self.result['P 系等级'].value_counts().to_dict() if 'P 系等级' in self.result.columns else {},
        }

        # 计算各类别占比
        summary['abc_percentage'] = {
            k: round(v / summary['total_items'] * 100, 2)
            for k, v in summary['abc_distribution'].items()
        }

        return summary


def main():
    """主函数 - 测试 ABC 分类模块"""

    # 配置路径
    input_dir = Path("C:/Users/Lenovo/配件采购系统/data/output")
    output_dir = Path("C:/Users/Lenovo/配件采购系统/data/output")

    logger.info("=" * 60)
    logger.info("车企配件采购系统 - 需求特征分析和 ABC 分类模块")
    logger.info("=" * 60)

    # 1. 加载清洗后的数据
    print("\n[1/4] 加载清洗后的数据...")
    # 找到最新的清洗数据文件
    cleaned_files = sorted(input_dir.glob("cleaned_data_*.xlsx"))
    if not cleaned_files:
        raise FileNotFoundError("未找到清洗后的数据文件，请先运行 Phase 1")

    latest_file = cleaned_files[-1]
    logger.info(f"加载文件：{latest_file.name}")
    data = pd.read_excel(latest_file, engine='openpyxl')
    logger.info(f"数据行数：{len(data)}")

    # 2. 创建分类器
    print("\n[2/4] 创建 ABC 分类器...")
    classifier = ABCClassifier(data)

    # 3. 运行完整分类
    print("\n[3/4] 运行分类流程...")
    result = classifier.run_full_classification()

    # 4. 显示统计摘要
    print("\n[4/4] 统计摘要:")
    summary = classifier.get_summary_stats()

    print(f"\n=== ABC 分类分布 ===")
    for grade, count in sorted(summary['abc_distribution'].items()):
        pct = summary['abc_percentage'].get(grade, 0)
        print(f"  {grade}类：{count} 件 ({pct}%)")

    print(f"\n=== H 系分布 ===")
    for grade, count in sorted(summary['h_distribution'].items()):
        print(f"  {grade}级：{count} 件")

    print(f"\n=== E 系分布 ===")
    for grade, count in sorted(summary['e_distribution'].items()):
        print(f"  {grade}级：{count} 件")

    print(f"\n=== P 系分布 ===")
    for grade, count in sorted(summary['p_distribution'].items()):
        print(f"  {grade}级：{count} 件")

    # 5. 保存结果
    print("\n[5/5] 保存分类结果...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"abc_classification_{timestamp}.xlsx"
    classifier.save_classification_result(str(output_file))

    logger.info("=" * 60)
    logger.info("Phase 2 完成!")
    logger.info("=" * 60)

    return classifier


if __name__ == "__main__":
    main()
