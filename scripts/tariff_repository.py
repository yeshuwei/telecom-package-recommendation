import re
import logging
from typing import Any, Dict, List, Optional, Tuple

import pymysql

from mcp_tools.mysql_tools import mysql_tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


NAME_CANDIDATES = ["套餐名称", "名称", "资费名称", "产品名称"]
PRICE_CANDIDATES = ["价格", "月费", "套餐价", "资费价格", "售价"]


def extract_numeric(value: Any) -> Optional[float]:
    if value is None:
        return None
    s = str(value)
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


# def normalize_value_num(key: str, value: Any) -> Optional[float]:
#     if value is None:
#         return None
#     s = str(value)
#     if re.search(r"不限|无限", s):
#         if re.search(r"流量|GB|G", key):
#             return 100.0
#         if re.search(r"通话|语音|分钟", key):
#             return 1000.0
#     num = extract_numeric(s)
#     return num


def normalize_feature_key(col: str) -> str:
    c = str(col).strip()
    if re.search(r"国内流量", c):
        return "国内流量"
    if re.search(r"国内通话", c):
        return "国内通话"
    return c


def create_tables_if_not_exists(conn: pymysql.Connection) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tariff_plan (
                id BIGINT PRIMARY KEY AUTO_INCREMENT,
                series VARCHAR(128) NOT NULL,
                plan_name VARCHAR(256) NOT NULL,
                price DECIMAL(10,2) NOT NULL,
                version VARCHAR(32) NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_series(series),
                INDEX idx_series_price(series, price),
                INDEX idx_plan_name(plan_name)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tariff_feature (
                id BIGINT PRIMARY KEY AUTO_INCREMENT,
                plan_id BIGINT NOT NULL,
                feature_key VARCHAR(128) NOT NULL,
                feature_value VARCHAR(512) NULL,
                INDEX idx_plan_fk(plan_id),
                INDEX idx_feature_kv(feature_key, feature_value),
                CONSTRAINT fk_plan FOREIGN KEY (plan_id) REFERENCES tariff_plan(id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
    conn.commit()
    logger.info("MySQL表校验创建完成")


def upsert_plan(conn: pymysql.Connection, series: str, plan_name: str, price: float, version: Optional[str]) -> int:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT id FROM tariff_plan WHERE series=%s AND plan_name=%s AND IFNULL(version,'')=IFNULL(%s,'')
            """,
            (series, plan_name, version)
        )
        row = cursor.fetchone()
        if row and row.get("id"):
            plan_id = int(row["id"]) if isinstance(row, dict) else int(row[0])
            cursor.execute(
                """
                UPDATE tariff_plan SET price=%s WHERE id=%s
                """,
                (price, plan_id)
            )
            conn.commit()
            return plan_id
        cursor.execute(
            """
            INSERT INTO tariff_plan(series, plan_name, price, version) VALUES(%s,%s,%s,%s)
            """,
            (series, plan_name, price, version)
        )
        conn.commit()
        return int(cursor.lastrowid)


def insert_features(conn: pymysql.Connection, plan_id: int, features: List[Tuple[str, Any]]) -> None:
    rows = []
    for key, value in features:
        fkey = normalize_feature_key(key)
        fval = str(value) if value is not None else None
        rows.append((plan_id, fkey, fval))
    if not rows:
        return
    with conn.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO tariff_feature(plan_id, feature_key, feature_value) VALUES(%s,%s,%s)
            """,
            rows
        )
    conn.commit()


class TariffRepository:
    def __init__(self):
        self.conn: Optional[pymysql.Connection] = None

    def connect(self) -> pymysql.Connection:
        if not self.conn:
            self.conn = mysql_tools.connect_to_mysql()
            create_tables_if_not_exists(self.conn)
        return self.conn

    def close(self) -> None:
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

    def import_series_dataframe(self, series: str, df, version: Optional[str] = None) -> int:
        conn = self.connect()
        total = 0
        # 寻找通用列
        name_col = next((c for c in NAME_CANDIDATES if c in df.columns), None)
        price_col = next((c for c in PRICE_CANDIDATES if c in df.columns), None)
        if not name_col or not price_col:
            logger.warning(f"系列{series}缺少通用列，跳过")
            return 0
        for _, row in df.iterrows():
            plan_name = str(row[name_col]).strip()
            price_val = extract_numeric(row[price_col])
            if not plan_name or price_val is None:
                continue
            plan_id = upsert_plan(conn, series, plan_name, float(price_val), version)
            # 其余列作为特征
            feature_cols = [c for c in df.columns if c not in [name_col, price_col]]
            features = [(c, row[c]) for c in feature_cols]
            insert_features(conn, plan_id, features)
            total += 1
        logger.info(f"系列{series}导入完成，共{total}条")
        return total