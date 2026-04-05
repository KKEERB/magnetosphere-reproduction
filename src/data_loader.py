"""
车企配件采购计划半自动生成系统
Phase 1: 数据读取和清洗模块

功能：
- 读取多个 Excel 文件
- 数据清洗和标准化
- 缺失值处理
- 配件去重和标识匹配
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
        logging.FileHandler('logs/data_loader.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataLoader:
    """配件采购数据加载器"""

    def __init__(self, input_dir: str):
        """
        初始化数据加载器

        Args:
            input_dir: 输入 Excel 文件目录
        """
        self.input_dir = Path(input_dir)
        self.data = None
        self.business_rules = None

    def load_excel_files(self) -> pd.DataFrame:
        """
        加载目录下所有 Excel 文件并合并

        Returns:
            合并后的 DataFrame
        """
        logger.info(f"开始扫描目录：{self.input_dir}")

        # 找到所有 Excel 文件
        excel_files = list(self.input_dir.glob("*.xlsx")) + list(self.input_dir.glob("*.xls"))

        if not excel_files:
            raise FileNotFoundError(f"在 {self.input_dir} 目录下未找到 Excel 文件")

        logger.info(f"找到 {len(excel_files)} 个 Excel 文件")

        # 读取并合并所有文件
        dataframes = []
        for file in excel_files:
            try:
                logger.info(f"读取文件：{file.name}")
                df = pd.read_excel(file, engine='openpyxl')
                df['_source_file'] = file.name  # 添加来源标记
                dataframes.append(df)
                logger.info(f"  - 行数：{len(df)}, 列数：{len(df.columns)}")
            except Exception as e:
                logger.error(f"读取文件 {file.name} 失败：{e}")
                continue

        if not dataframes:
            raise ValueError("未能成功读取任何 Excel 文件")

        # 合并所有数据
        self.data = pd.concat(dataframes, ignore_index=True)
        logger.info(f"合并后总行数：{len(self.data)}, 总列数：{len(self.data.columns)}")

        return self.data

    def load_business_rules(self, rules_file: str) -> dict:
        """
        加载业务规则文件

        Args:
            rules_file: 业务规则 Excel 文件路径

        Returns:
            业务规则字典
        """
        logger.info(f"加载业务规则文件：{rules_file}")

        try:
            xl = pd.ExcelFile(rules_file, engine='openpyxl')
            self.business_rules = {}

            # 读取关键 Sheet
            sheets_to_load = ['系数标准', '数据逻辑', '计算模型']

            for sheet in xl.sheet_names:
                try:
                    df = pd.read_excel(xl, sheet_name=sheet)
                    # 清理 Sheet 名称（移除空白字符）
                    clean_name = sheet.strip()
                    self.business_rules[clean_name] = df
                    logger.info(f"  - 加载 Sheet '{clean_name}': {len(df)} 行")
                except Exception as e:
                    logger.warning(f"加载 Sheet '{sheet}' 失败：{e}")

            return self.business_rules

        except Exception as e:
            logger.error(f"加载业务规则文件失败：{e}")
            return {}

    # PMS 数据清洗规则（来自"当量制作.docx"）
    PMS_FILTER_RULES = {
        # 直供供应商标识（需要删除）
        'direct_suppliers': ['1218', '7237', '0155', '4862', '7479', '2076', '6393', 'F5693'],

        # 需要删除的配件类型（名称关键词）
        'exclude_part_types': [
            '轮胎', '桥总成', '车架总成', '柴油机总成', '供气系统总成',
            '滤芯', '油品', '磁铁', '封胶', '冷却液', '发动机液',
            '单瓶', '变速箱', '变速器', '缓速取力器',
        ],

        # 需要删除的关键词（备注/订单类型）
        'exclude_keywords': [
            'KB', '快报', '质量快报', '快报单号',
            '整改', '改制整改', '整改报单',
            '维修', '维修报单', '维修指导', '维修厂',
            '专项订单', '作业指导', '保外转保内',
            '招标出库', '新产品方案', '重点产品',
            '星辉专用件', '星辉新能源',
        ],

        # 需要删除的订单状态
        'exclude_order_status': ['撤单', '新增', '作废', '终止'],

        # 需要删除的仓库/虚拟库
        'exclude_warehouses': ['汇川', '绿控', '亿纬', '虚拟库', '空入空出'],
    }

    def clean_data(self, apply_pms_rules: bool = True) -> pd.DataFrame:
        """
        清洗数据

        处理内容：
        - 移除全空行和列
        - 标准化列名
        - 处理缺失值
        - 去除重复记录
        - 应用 PMS 业务规则（当量制作.docx 中的规则）

        Args:
            apply_pms_rules: 是否应用 PMS 数据清洗规则

        Returns:
            清洗后的 DataFrame
        """
        if self.data is None:
            raise ValueError("请先加载数据")

        logger.info("开始数据清洗...")
        initial_rows = len(self.data)
        initial_cols = len(self.data.columns)

        # 1. 移除全空行
        self.data = self.data.dropna(how='all')
        logger.info(f"移除全空行后行数：{len(self.data)} (移除 {initial_rows - len(self.data)} 行)")

        # 2. 移除全空列
        self.data = self.data.dropna(axis=1, how='all')
        logger.info(f"移除全空列后列数：{len(self.data.columns)} (移除 {initial_cols - len(self.data.columns)} 列)")

        # 3. 标准化列名（去除空白字符）
        self.data.columns = [str(col).strip() for col in self.data.columns]

        # 4. 处理配件号列（确保为字符串类型）
        key_columns = ['配件号', '短号', '图号', '名称']
        for col in key_columns:
            if col in self.data.columns:
                self.data[col] = self.data[col].astype(str).replace('nan', None)

        # 5. 去除重复记录（基于配件号）
        if '配件号' in self.data.columns:
            dup_count = self.data.duplicated(subset=['配件号'], keep='first').sum()
            self.data = self.data.drop_duplicates(subset=['配件号'], keep='first')
            logger.info(f"去除重复记录后行数：{len(self.data)} (移除 {dup_count} 条重复)")

        # 6. 填充数值型缺失值为 0
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_cols] = self.data[numeric_cols].fillna(0)

        # 7. 应用 PMS 业务规则清洗
        if apply_pms_rules:
            logger.info("应用 PMS 数据清洗规则...")
            self._apply_pms_filter_rules()

        logger.info("数据清洗完成")
        return self.data

    def _apply_pms_filter_rules(self):
        """
        应用 PMS 数据清洗规则（来自当量制作.docx）

        规则包括：
        - 删除直供供应商标识
        - 删除特定配件类型（轮胎、总成类、胶液类等）
        - 删除关键词订单（KB 开头、快报、整改、维修等）
        - 删除无效状态订单（撤单、新增、作废、终止）
        """
        initial_count = len(self.data)
        filter_mask = pd.Series([True] * len(self.data), index=self.data.index)
        filter_reasons = []

        # 1. 删除直供供应商标识（配件号后 5 位匹配）
        if '配件号' in self.data.columns:
            supplier_mask = pd.Series([False] * len(self.data), index=self.data.index)
            for supplier in self.PMS_FILTER_RULES['direct_suppliers']:
                # 配件号后 5 位匹配直供供应商标识
                mask = self.data['配件号'].str.endswith(supplier, na=False)
                supplier_mask |= mask
            if supplier_mask.any():
                filter_reasons.append(f"直供供应商标识：{supplier_mask.sum()} 条")
            filter_mask &= ~supplier_mask

        # 2. 删除特定配件类型（名称关键词）
        if '名称' in self.data.columns:
            part_type_mask = pd.Series([False] * len(self.data), index=self.data.index)
            for keyword in self.PMS_FILTER_RULES['exclude_part_types']:
                mask = self.data['名称'].str.contains(keyword, regex=False, na=False)
                part_type_mask |= mask
            if part_type_mask.any():
                filter_reasons.append(f"配件类型过滤：{part_type_mask.sum()} 条")
            filter_mask &= ~part_type_mask

        # 3. 删除关键词订单（备注/订单类型）
        # 检查可能的备注列
        remark_cols = ['备注', '订单类型', '订单备注', '说明']
        existing_remark_cols = [c for c in remark_cols if c in self.data.columns]

        if existing_remark_cols:
            keyword_mask = pd.Series([False] * len(self.data), index=self.data.index)
            for col in existing_remark_cols:
                for keyword in self.PMS_FILTER_RULES['exclude_keywords']:
                    mask = self.data[col].str.contains(keyword, regex=False, na=False)
                    keyword_mask |= mask
            if keyword_mask.any():
                filter_reasons.append(f"关键词过滤：{keyword_mask.sum()} 条")
            filter_mask &= ~keyword_mask

        # 4. 删除无效状态订单
        status_cols = ['状态', '订单状态', '单据状态']
        existing_status_cols = [c for c in status_cols if c in self.data.columns]

        if existing_status_cols:
            status_mask = pd.Series([False] * len(self.data), index=self.data.index)
            for col in existing_status_cols:
                for status in self.PMS_FILTER_RULES['exclude_order_status']:
                    mask = self.data[col].str.contains(status, regex=False, na=False)
                    status_mask |= mask
            if status_mask.any():
                filter_reasons.append(f"无效状态过滤：{status_mask.sum()} 条")
            filter_mask &= ~status_mask

        # 5. 应用过滤
        self.data = self.data[filter_mask].reset_index(drop=True)

        filtered_count = initial_count - len(self.data)
        logger.info(f"PMS 规则过滤后行数：{len(self.data)} (移除 {filtered_count} 条)")
        for reason in filter_reasons:
            logger.info(f"  - {reason}")

    def validate_data(self) -> dict:
        """
        验证数据质量

        Returns:
            数据质量报告字典
        """
        if self.data is None:
            raise ValueError("请先加载数据")

        logger.info("开始数据质量验证...")

        report = {
            'total_rows': len(self.data),
            'total_cols': len(self.data.columns),
            'missing_values': {},
            'zero_demand_count': 0,
            'negative_values': [],
            'data_quality_score': 100.0
        }

        # 1. 检查关键字段缺失情况
        key_fields = ['配件号', '名称', '单位']
        for field in key_fields:
            if field in self.data.columns:
                missing = self.data[field].isna().sum()
                missing_pct = missing / len(self.data) * 100
                report['missing_values'][field] = {
                    'count': missing,
                    'percentage': round(missing_pct, 2)
                }
                if missing_pct > 50:
                    report['data_quality_score'] -= 20
                    logger.warning(f"字段 '{field}' 缺失率过高：{missing_pct:.2f}%")

        # 2. 检查零需求记录
        year_cols = [col for col in self.data.columns if '年' in str(col) and col not in ['年度']]
        if year_cols:
            zero_demand_mask = (self.data[year_cols] == 0).all(axis=1)
            report['zero_demand_count'] = zero_demand_mask.sum()
            logger.info(f"零需求配件数量：{report['zero_demand_count']}")

        # 3. 检查负值
        for col in self.data.select_dtypes(include=[np.number]).columns:
            negative_count = (self.data[col] < 0).sum()
            if negative_count > 0:
                report['negative_values'].append({
                    'column': col,
                    'count': negative_count
                })

        logger.info(f"数据质量评分：{report['data_quality_score']}")
        return report

    def save_cleaned_data(self, output_path: str):
        """
        保存清洗后的数据

        Args:
            output_path: 输出文件路径
        """
        if self.data is None:
            raise ValueError("没有可保存的数据")

        logger.info(f"保存清洗后的数据到：{output_path}")
        self.data.to_excel(output_path, index=False, engine='openpyxl')
        logger.info("保存完成")

    def get_column_info(self) -> pd.DataFrame:
        """
        获取列信息摘要

        Returns:
            列信息 DataFrame
        """
        if self.data is None:
            raise ValueError("请先加载数据")

        info = []
        for col in self.data.columns:
            non_null = self.data[col].notna().sum()
            unique = self.data[col].nunique()
            dtype = self.data[col].dtype

            info.append({
                '列名': col,
                '类型': dtype,
                '非空值数': non_null,
                '唯一值数': unique,
                '非空率': f"{non_null/len(self.data)*100:.1f}%"
            })

        return pd.DataFrame(info)


def main():
    """主函数 - 测试数据加载和清洗"""

    # 配置路径
    input_dir = Path("C:/Users/Lenovo/.cc-connect/attachments")
    output_dir = Path("C:/Users/Lenovo/配件采购系统/data/output")
    rules_file = input_dir / "保内总库储备原则及标准 (1).xlsx"

    logger.info("=" * 60)
    logger.info("车企配件采购系统 - 数据读取和清洗模块")
    logger.info("=" * 60)

    # 创建数据加载器
    loader = DataLoader(input_dir)

    # 1. 加载 Excel 文件
    print("\n[1/5] 加载 Excel 文件...")
    loader.load_excel_files()

    # 2. 加载业务规则
    print("\n[2/5] 加载业务规则...")
    loader.load_business_rules(str(rules_file))

    # 3. 清洗数据
    print("\n[3/5] 清洗数据...")
    loader.clean_data()

    # 4. 验证数据质量
    print("\n[4/5] 验证数据质量...")
    quality_report = loader.validate_data()
    print(f"\n数据质量报告:")
    print(f"  - 总行数：{quality_report['total_rows']}")
    print(f"  - 总列数：{quality_report['total_cols']}")
    print(f"  - 零需求配件：{quality_report['zero_demand_count']}")
    print(f"  - 质量评分：{quality_report['data_quality_score']}")

    # 5. 保存清洗后的数据
    print("\n[5/5] 保存清洗后的数据...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"cleaned_data_{timestamp}.xlsx"
    loader.save_cleaned_data(str(output_file))

    # 显示列信息
    print("\n列信息摘要:")
    col_info = loader.get_column_info()
    print(col_info.to_string())

    logger.info("=" * 60)
    logger.info("Phase 1 完成!")
    logger.info("=" * 60)

    return loader


if __name__ == "__main__":
    main()
