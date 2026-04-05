"""
车企配件采购计划半自动生成系统
主入口脚本

使用方法：
    python main.py

功能：
    一键运行所有模块，从数据读取到最终报告生成
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import DataLoader
from abc_classification import ABCClassifier
from forecast_model import DemandForecaster
from procurement_calc import ProcurementCalculator
from export_module import ExcelExporter, WordExporter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/main.log', encoding='utf-8', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_pipeline(input_dir: str = None, output_dir: str = None):
    """
    运行完整的数据处理管道

    Args:
        input_dir: 输入数据目录（Excel 文件所在目录）
        output_dir: 输出结果目录
    """
    # 默认路径
    if input_dir is None:
        input_dir = Path.home() / '.cc-connect' / 'attachments'
    else:
        input_dir = Path(input_dir)

    if output_dir is None:
        output_dir = Path(__file__).parent / 'data' / 'output'
    else:
        output_dir = Path(output_dir)

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("车企配件采购计划半自动生成系统")
    logger.info("=" * 80)
    logger.info(f"输入目录：{input_dir}")
    logger.info(f"输出目录：{output_dir}")
    logger.info("=" * 80)

    # ========== Phase 1: 数据读取和清洗 ==========
    logger.info("\n" + "=" * 60)
    logger.info("Phase 1: 数据读取和清洗")
    logger.info("=" * 60)

    loader = DataLoader(input_dir)
    loader.load_excel_files()
    loader.clean_data()
    quality_report = loader.validate_data()

    print(f"\n[OK] Phase 1 完成")
    print(f"  - 数据行数：{quality_report['total_rows']:,} 行")
    print(f"  - 数据列数：{quality_report['total_cols']} 列")
    print(f"  - 质量评分：{quality_report['data_quality_score']}")

    # ========== Phase 2: ABC 分类 ==========
    logger.info("\n" + "=" * 60)
    logger.info("Phase 2: 需求特征分析和 ABC 分类")
    logger.info("=" * 60)

    classifier = ABCClassifier(loader.data)
    abc_result = classifier.run_full_classification()
    abc_summary = classifier.get_summary_stats()

    print(f"\n[OK] Phase 2 完成")
    print(f"  ABC 分类分布:")
    for grade, count in sorted(abc_summary['abc_distribution'].items()):
        pct = abc_summary['abc_percentage'].get(grade, 0)
        print(f"    {grade}类：{count:,} 件 ({pct}%)")

    # ========== Phase 3: 需求预测 ==========
    logger.info("\n" + "=" * 60)
    logger.info("Phase 3: 需求预测模型")
    logger.info("=" * 60)

    # 重新识别月度列（从原始数据）
    monthly_cols = []
    for col in loader.data.columns:
        col_str = str(col)
        if '年' in col_str and '月' in col_str:
            monthly_cols.append(col)

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

    # 合并 ABC 结果和原始数据
    forecast_data = pd.concat([
        abc_result.reset_index(drop=True),
        loader.data[monthly_cols].reset_index(drop=True)
    ], axis=1)

    forecaster = DemandForecaster(forecast_data, monthly_cols)
    forecast_results = forecaster.run_forecast_for_all(lookback_months=12)
    forecast_summary = forecaster.get_forecast_summary()

    print(f"\n[OK] Phase 3 完成")
    print(f"  预测模型分布:")
    for model, count in sorted(forecast_summary['model_distribution'].items()):
        pct = count / forecast_summary['total_items'] * 100
        print(f"    {model}: {count:,} 件 ({pct:.1f}%)")

    # ========== Phase 4: 采购量计算 ==========
    logger.info("\n" + "=" * 60)
    logger.info("Phase 4: 采购量计算")
    logger.info("=" * 60)

    # 合并预测结果
    calc_data = pd.concat([
        abc_result.reset_index(drop=True),
        forecast_results.reset_index(drop=True)
    ], axis=1)

    calculator = ProcurementCalculator(calc_data)
    calc_result = calculator.run_calculation()
    calc_summary = calculator.get_summary_stats()

    print(f"\n[OK] Phase 4 完成")
    print(f"  采购统计:")
    print(f"    总配件数：{calc_summary['total_items']:,} 件")
    print(f"    需要采购：{calc_summary['items_to_procure']:,} 件")
    print(f"    总采购量：{calc_summary['total_procurement_qty']:,} 件")

    # ========== Phase 5 & 6: 导出报告 ==========
    logger.info("\n" + "=" * 60)
    logger.info("Phase 5 & 6: 导出报告")
    logger.info("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 导出 Excel
    excel_exporter = ExcelExporter(calc_result)
    excel_file = output_dir / f"采购计划_{timestamp}.xlsx"
    excel_exporter.export(str(excel_file))

    # 导出 Word
    word_exporter = WordExporter(calc_result, calc_summary)
    word_file = output_dir / f"采购计划说明_{timestamp}.docx"
    word_file_path = word_exporter.export(str(word_file))

    print(f"\n[OK] Phase 5 & 6 完成")
    print(f"  Excel 报告：{excel_file.name}")
    print(f"  Word 说明：{Path(word_file_path).name}")

    # ========== 完成 ==========
    logger.info("\n" + "=" * 80)
    logger.info("所有阶段完成!")
    logger.info("=" * 80)

    print("\n" + "=" * 80)
    print("[OK] 车企配件采购计划半自动生成系统 - 执行完成")
    print("=" * 80)
    print(f"\n输出文件位于：{output_dir}")
    print(f"  - 采购计划：{excel_file.name}")
    print(f"  - 说明文档：{Path(word_file_path).name}")
    print("=" * 80)

    return {
        'excel_file': str(excel_file),
        'word_file': str(word_file_path),
        'summary': calc_summary
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='车企配件采购计划半自动生成系统')
    parser.add_argument('--input', '-i', type=str, help='输入数据目录（Excel 文件所在目录）')
    parser.add_argument('--output', '-o', type=str, help='输出结果目录')

    args = parser.parse_args()

    try:
        result = run_pipeline(args.input, args.output)
        print("\n执行成功!")
    except Exception as e:
        logger.exception(f"执行失败：{e}")
        print(f"\n执行失败：{e}")
        sys.exit(1)
