"""
车企配件采购计划半自动生成系统
Phase 4: 采购量计算模块

功能：
- 计算日均需求
- 计算最大/最小储备量
- 计算安全库存
- 计算建议采购量
- 应用业务规则（MOQ、包装规格等）
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
        logging.FileHandler('logs/procurement_calc.log', encoding='utf-8', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProcurementCalculator:
    """采购量计算器"""

    # 默认业务参数
    DEFAULT_PARAMS = {
        '备货周期': 15,  # 天
        '安全周期': 15,  # 天
        '服务水准': 0.95,  # 95% 服务水准
        'z_score': 1.65,  # 95% 服务水准对应的 Z 值
    }

    def __init__(self, forecast_data: pd.DataFrame, business_rules: dict = None):
        """
        初始化计算器

        Args:
            forecast_data: 包含预测结果的 DataFrame
            business_rules: 业务规则参数字典
        """
        self.data = forecast_data.copy()
        self.params = {**self.DEFAULT_PARAMS, **(business_rules or {})}
        self.result = None

        logger.info(f"初始化采购量计算器，{len(self.data)} 条记录")
        logger.info(f"业务参数：备货周期={self.params['备货周期']}天，安全周期={self.params['安全周期']}天")

    def calc_daily_demand(self, monthly_demand_col: str = 'monthly_demand') -> pd.Series:
        """
        计算日均需求

        公式：日均需求 = 月均需求 / 30

        Args:
            monthly_demand_col: 月均需求列名

        Returns:
            日均需求 Series
        """
        if monthly_demand_col not in self.data.columns:
            logger.warning(f"未找到列 {monthly_demand_col}，使用 0 代替")
            return pd.Series([0.0] * len(self.data))

        monthly_demand = self.data[monthly_demand_col].fillna(0)
        daily_demand = monthly_demand / 30.0
        return daily_demand

    def calc_safety_stock(self, demand_std: pd.Series = None) -> pd.Series:
        """
        计算安全库存

        公式：安全库存 = Z × σ_demand × √(提前期)

        Args:
            demand_std: 需求标准差 Series（可选）

        Returns:
            安全库存 Series
        """
        z_score = self.params.get('z_score', 1.65)
        lead_time = self.params.get('备货周期', 15)

        if demand_std is None:
            # 如果没有标准差数据，使用简化公式
            # 安全库存 = 日均需求 × 安全周期
            daily_demand = self.calc_daily_demand()
            safety_stock = daily_demand * self.params.get('安全周期', 15)
        else:
            # 完整公式
            safety_stock = z_score * demand_std * np.sqrt(lead_time)

        return safety_stock

    def calc_max_reserve(self, daily_demand: pd.Series, plan_coef: pd.Series = None) -> pd.Series:
        """
        计算最大储备量

        公式：最大储备量 = 日均需求 × 计划系数 + 备货周期 + 安全周期

        Args:
            daily_demand: 日均需求 Series
            plan_coef: 计划系数 Series（P 系系数）

        Returns:
            最大储备量 Series
        """
        if plan_coef is None:
            plan_coef = pd.Series([1.0] * len(daily_demand))

        # 从业务规则获取周期参数
        reserve_period = self.params.get('备货周期', 15)
        safety_period = self.params.get('安全周期', 15)

        # 最大储备量 = 日均需求 × 计划系数 × (备货周期 + 安全周期)
        # 简化公式：最大储备量 = 日均需求 × 计划系数 × 30 (假设总周期 30 天)
        max_reserve = daily_demand * plan_coef * (reserve_period + safety_period) / 30.0

        # 转换为整数（向上取整）
        max_reserve = np.ceil(max_reserve)

        return max_reserve

    def calc_min_reserve(self, daily_demand: pd.Series, plan_coef: pd.Series = None) -> pd.Series:
        """
        计算最小储备量

        公式：最小储备量 = 日均需求 × 计划系数 × 安全周期

        Args:
            daily_demand: 日均需求 Series
            plan_coef: 计划系数 Series（P 系系数）

        Returns:
            最小储备量 Series
        """
        if plan_coef is None:
            plan_coef = pd.Series([1.0] * len(daily_demand))

        safety_period = self.params.get('安全周期', 15)

        # 最小储备量 = 日均需求 × 计划系数 × 安全周期/30
        min_reserve = daily_demand * plan_coef * safety_period / 30.0
        min_reserve = np.ceil(min_reserve)

        return min_reserve

    def calc_procurement_qty(self, max_reserve: pd.Series,
                              current_stock: pd.Series = None,
                              in_transit: pd.Series = None) -> pd.Series:
        """
        计算建议采购量

        公式：建议采购量 = MAX(0, 最大储备量 - 当前库存 - 在途 + 未满足需求)

        Args:
            max_reserve: 最大储备量 Series
            current_stock: 当前库存 Series（可选）
            in_transit: 在途库存 Series（可选）

        Returns:
            建议采购量 Series
        """
        # 如果没有库存数据，假设库存为 0
        if current_stock is None:
            current_stock = pd.Series([0.0] * len(max_reserve))

        if in_transit is None:
            in_transit = pd.Series([0.0] * len(max_reserve))

        # 建议采购量 = MAX(0, 最大储备量 - 当前库存 - 在途库存)
        procurement_qty = max_reserve - current_stock - in_transit
        procurement_qty = procurement_qty.clip(lower=0)

        # 向上取整
        procurement_qty = np.ceil(procurement_qty)

        return procurement_qty

    def apply_moq(self, qty: pd.Series, moq_col: str = None) -> pd.Series:
        """
        应用最小起订量 (MOQ)

        Args:
            qty: 原始采购量 Series
            moq_col: MOQ 列名

        Returns:
            应用 MOQ 后的采购量
        """
        if moq_col is None or moq_col not in self.data.columns:
            return qty

        moq = self.data[moq_col].fillna(1)

        # 如果采购量大于 0 但小于 MOQ，则调整为 MOQ
        adjusted_qty = qty.copy()
        mask = (qty > 0) & (qty < moq)
        adjusted_qty[mask] = moq[mask]

        return adjusted_qty

    def apply_pack_size(self, qty: pd.Series, pack_size_col: str = None) -> pd.Series:
        """
        应用包装规格取整

        Args:
            qty: 原始采购量 Series
            pack_size_col: 包装规格列名

        Returns:
            取整后的采购量
        """
        if pack_size_col is None or pack_size_col not in self.data.columns:
            return qty

        pack_size = self.data[pack_size_col].fillna(1)

        # 向上取整到包装规格的倍数
        adjusted_qty = np.ceil(qty / pack_size) * pack_size

        return adjusted_qty

    def classify_procurement_priority(self, qty: pd.Series, abc_class: pd.Series) -> pd.Series:
        """
        判定采购优先级

        Args:
            qty: 采购量 Series
            abc_class: ABC 分类 Series

        Returns:
            优先级 Series（高/中/低）
        """
        priority = pd.Series(['低'] * len(qty))

        # A 类配件且采购量>0，标记为高优先级
        high_priority_mask = (abc_class == 'A') & (qty > 0)
        priority[high_priority_mask] = '高'

        # B 类或 C 类且采购量>0，标记为中优先级
        medium_priority_mask = (abc_class.isin(['B', 'C', 'D'])) & (qty > 0)
        priority[medium_priority_mask] = '中'

        # E 类（无需求）始终为低优先级
        priority[abc_class == 'E'] = '低'

        return priority

    def run_calculation(self) -> pd.DataFrame:
        """
        运行完整采购量计算流程

        Returns:
            包含计算结果的 DataFrame
        """
        logger.info("=" * 60)
        logger.info("开始采购量计算流程")
        logger.info("=" * 60)

        # 1. 计算日均需求
        logger.info("计算日均需求...")
        daily_demand = self.calc_daily_demand('monthly_demand')

        # 2. 获取计划系数（P 系系数）
        plan_coef_col = 'P 系系数'
        if plan_coef_col in self.data.columns:
            plan_coef = self.data[plan_coef_col].fillna(1.0)
        else:
            plan_coef = pd.Series([1.0] * len(self.data))
            logger.warning("未找到 P 系系数列，使用默认值 1.0")

        # 3. 计算最大储备量
        logger.info("计算最大储备量...")
        max_reserve = self.calc_max_reserve(daily_demand, plan_coef)

        # 4. 计算最小储备量
        logger.info("计算最小储备量...")
        min_reserve = self.calc_min_reserve(daily_demand, plan_coef)

        # 5. 计算安全库存
        logger.info("计算安全库存...")
        safety_stock = self.calc_safety_stock()

        # 6. 计算建议采购量
        logger.info("计算建议采购量...")
        procurement_qty = self.calc_procurement_qty(max_reserve)

        # 7. 应用 MOQ
        logger.info("应用最小起订量...")
        procurement_qty = self.apply_moq(procurement_qty)

        # 8. 应用包装规格
        logger.info("应用包装规格...")
        procurement_qty = self.apply_pack_size(procurement_qty)

        # 9. 判定采购优先级
        logger.info("判定采购优先级...")
        abc_class = self.data.get('ABC 分类', pd.Series(['Z'] * len(self.data)))
        priority = self.classify_procurement_priority(procurement_qty, abc_class)

        # 10. 组装结果
        self.result = pd.DataFrame({
            '配件号': self.data.get('配件号', pd.Series(range(len(self.data)))),
            '名称': self.data.get('名称', pd.Series([''] * len(self.data))),
            '单位': self.data.get('单位', pd.Series([''] * len(self.data))),
            'ABC 分类': abc_class,
            '月均需求': self.data.get('monthly_demand', daily_demand * 30).fillna(0),
            '日均需求': daily_demand,
            '计划系数': plan_coef,
            '最大储备量': max_reserve,
            '最小储备量': min_reserve,
            '安全库存': safety_stock,
            '建议采购量': procurement_qty,
            '采购优先级': priority,
            '预测模型': self.data.get('model', pd.Series(['unknown'] * len(self.data))),
            '置信度': self.data.get('confidence', pd.Series([0.0] * len(self.data))),
        })

        # 11. 统计摘要
        total_items = len(self.result)
        items_to_procure = (self.result['建议采购量'] > 0).sum()
        high_priority = (self.result['采购优先级'] == '高').sum()
        medium_priority = (self.result['采购优先级'] == '中').sum()

        logger.info(f"计算完成，共 {total_items} 条记录")
        logger.info(f"需要采购的配件：{items_to_procure} 件")
        logger.info(f"高优先级：{high_priority} 件")
        logger.info(f"中优先级：{medium_priority} 件")

        return self.result

    def get_summary_stats(self) -> dict:
        """
        获取采购计算摘要统计

        Returns:
            摘要统计字典
        """
        if self.result is None:
            raise ValueError("请先运行计算")

        summary = {
            'total_items': len(self.result),
            'items_to_procure': int((self.result['建议采购量'] > 0).sum()),
            'total_procurement_qty': int(self.result['建议采购量'].sum()),
            'priority_distribution': self.result['采购优先级'].value_counts().to_dict(),
            'abc_distribution': self.result['ABC 分类'].value_counts().to_dict(),
        }

        # 按 ABC 分类统计采购量
        abc_procurement = self.result.groupby('ABC 分类')['建议采购量'].sum().to_dict()
        summary['abc_procurement'] = abc_procurement

        return summary

    def save_result(self, output_path: str):
        """
        保存计算结果

        Args:
            output_path: 输出文件路径
        """
        if self.result is None:
            raise ValueError("请先运行计算")

        logger.info(f"保存计算结果到：{output_path}")
        self.result.to_excel(output_path, index=False, engine='openpyxl')
        logger.info("保存完成")


def main():
    """主函数 - 测试采购量计算模块"""

    # 配置路径
    input_dir = Path("C:/Users/Lenovo/配件采购系统/data/output")
    output_dir = Path("C:/Users/Lenovo/配件采购系统/data/output")

    logger.info("=" * 60)
    logger.info("车企配件采购系统 - 采购量计算模块")
    logger.info("=" * 60)

    # 1. 加载预测结果
    print("\n[1/5] 加载预测结果...")
    forecast_files = sorted(input_dir.glob("forecast_result_*.xlsx"))
    if not forecast_files:
        raise FileNotFoundError("未找到预测结果，请先运行 Phase 3")

    latest_file = forecast_files[-1]
    logger.info(f"加载文件：{latest_file.name}")
    data = pd.read_excel(latest_file, engine='openpyxl')
    logger.info(f"数据行数：{len(data)}")

    # 2. 创建计算器
    print("\n[2/5] 创建采购量计算器...")
    calculator = ProcurementCalculator(data)

    # 3. 运行计算
    print("\n[3/5] 运行采购量计算...")
    result = calculator.run_calculation()

    # 4. 显示摘要
    print("\n[4/5] 计算摘要:")
    summary = calculator.get_summary_stats()

    print(f"\n=== 采购统计 ===")
    print(f"  总配件数：{summary['total_items']} 件")
    print(f"  需要采购：{summary['items_to_procure']} 件")
    print(f"  总采购量：{summary['total_procurement_qty']} 件")

    print(f"\n=== 优先级分布 ===")
    for priority, count in summary['priority_distribution'].items():
        pct = count / summary['total_items'] * 100
        print(f"  {priority}优先级：{count} 件 ({pct:.1f}%)")

    print(f"\n=== ABC 分类采购量 ===")
    for abc, qty in sorted(summary['abc_procurement'].items()):
        print(f"  {abc}类：{qty} 件")

    # 5. 保存结果
    print("\n[5/5] 保存计算结果...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"procurement_calc_{timestamp}.xlsx"
    calculator.save_result(str(output_file))

    logger.info("=" * 60)
    logger.info("Phase 4 完成!")
    logger.info("=" * 60)

    return calculator


if __name__ == "__main__":
    main()
