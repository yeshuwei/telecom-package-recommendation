import argparse
import logging
import pandas as pd

from configs.config import PRICE_SHEET_PATH
from scripts.tariff_repository import TariffRepository

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="导入在售资费Excel到MySQL")
    parser.add_argument("--path", default=PRICE_SHEET_PATH, help="Excel文件路径")
    parser.add_argument("--version", default=None, help="导入批次版本标签")
    args = parser.parse_args()

    logger.info(f"读取Excel: {args.path}")
    book = pd.read_excel(args.path, sheet_name=None)
    repo = TariffRepository()
    total_series = 0
    total_rows = 0
    try:
        for sheet_name, df in book.items():
            series = str(sheet_name).strip()
            logger.info(f"导入系列: {series}")
            cnt = repo.import_series_dataframe(series, df, version=args.version)
            total_series += 1
            total_rows += cnt
        logger.info(f"导入完成：系列 {total_series} 个，记录 {total_rows} 条")
    finally:
        repo.close()


if __name__ == "__main__":
    main()