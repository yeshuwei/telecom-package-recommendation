"""
MCP 工具 数据库操作相关函数
"""
import logging
from typing import Dict, Any, Optional
import pymysql
import pymysql.cursors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MYSQLTools:
    """数据库操作相关工具"""
    def __init__(self, mysql_config: Dict[str, Any] = None):
        self.mysql_config = mysql_config or {
            "host": "localhost",
            "port": 3306,
            "user": "root",
            "password": "123",
            "database": "data_demo",
            "charset": "utf8mb4"
        }

        logger.info("MYSQL数据库配置完毕")

    # ===================== 通用与用户画像 =====================

    def connect_to_mysql(self) -> pymysql.Connection:
        """连接到mysql数据库 返回连接对象"""
        try:
            connection = pymysql.connect(
                host=self.mysql_config.get("host", "localhost"),
                port=self.mysql_config.get("port", 3306),
                user=self.mysql_config.get("user", "root"),
                password=self.mysql_config.get("password", ""),
                database=self.mysql_config.get("database", "data_demo"),
                charset=self.mysql_config.get("charset", "utf8mb4"),
                cursorclass=pymysql.cursors.DictCursor
            )

            # logger.info("数据库连接成功")
            return connection
        except Exception as e:
            logger.error(f"连接数据库失败{e}")
            raise

    def query_cursor_by_phone_number(self, phone_number: int) -> Optional[Dict[str, Any]]:
        try:
            connection = self.connect_to_mysql()
            cursor = connection.cursor(pymysql.cursors.DictCursor)
            sql = "SELECT * FROM user_data WHERE 业务号码 = %s"
            cursor.execute(sql, (phone_number,))

            user_info: Dict[str, Any] = cursor.fetchone()

            cursor.close()
            connection.close()

            new_user_info = {}
            for key, value in user_info.items():
                if "偏好" in key:
                    try:
                        num_value = float(value)
                        if num_value >= 2:
                            new_user_info[key] = num_value
                    except (ValueError, TypeError):
                        continue
                else:
                    new_user_info[key] = value

            if new_user_info:
                logger.info(f"已查询到业务号码为{phone_number}的用户信息")
                return new_user_info
            else:
                logger.info(f"未找到到业务号码为{phone_number}的用户信息")
                return None
        except Exception as e:
            logger.error(f"查询用户信息失败{e}")
            return None

    def query_package_by_id(self, package_id: int) -> Optional[Dict[str, Any]]:
        """
        根据套餐ID查询套餐简介信息
        
        Args:
            package_id: 套餐ID
            
        Returns:
            套餐信息字典，如果未找到返回None
        """
        try:
            connection = self.connect_to_mysql()
            cursor = connection.cursor(pymysql.cursors.DictCursor)

            sql = "SELECT * FROM package_intro WHERE id = %s"
            cursor.execute(sql, (package_id,))

            package_info = cursor.fetchone()

            cursor.close()
            connection.close()

            if package_info:
                logger.info(f"已查询到ID为{package_id}的套餐信息: {package_info['package_name']}")
                return package_info
            else:
                logger.info(f"未找到ID为{package_id}的套餐信息")
                return None
        except Exception as e:
            logger.error(f"查询套餐信息失败: {e}")
            return None

    def query_package_by_name(self, package_name: str) -> list:
        """
        根据套餐名称模糊查询套餐简介信息
        
        Args:
            package_name: 套餐名称（支持模糊匹配）
            
        Returns:
            套餐信息列表
        """
        try:
            connection = self.connect_to_mysql()
            cursor = connection.cursor(pymysql.cursors.DictCursor)

            sql = "SELECT * FROM package_intro WHERE package_name LIKE %s"
            cursor.execute(sql, (f"%{package_name}%",))

            package_list = cursor.fetchall()

            cursor.close()
            connection.close()

            if package_list:
                logger.info(f"查询到{len(package_list)}个包含'{package_name}'的套餐")
                return package_list
            else:
                logger.info(f"未找到包含'{package_name}'的套餐")
                return []
        except Exception as e:
            logger.error(f"查询套餐信息失败: {e}")
            return []

    def query_package_intro_by_name(self, package_name: str) -> Optional[Dict[str, Any]]:
        """
        根据套餐名称模糊查询套餐简介（返回第一个匹配结果）
        
        Args:
            package_name: 套餐名称（支持模糊匹配）
            
        Returns:
            套餐信息字典（包含description字段），如果未找到返回None
        """
        try:
            connection = self.connect_to_mysql()
            cursor = connection.cursor(pymysql.cursors.DictCursor)

            sql = "SELECT package_name, description FROM package_intro WHERE package_name LIKE %s LIMIT 1"
            cursor.execute(sql, (f"%{package_name}%",))

            package_info = cursor.fetchone()

            cursor.close()
            connection.close()

            if package_info:
                logger.info(f"查询到套餐简介: {package_name}")
                return package_info
            else:
                logger.warning(f"未找到套餐简介: {package_name}")
                return None
        except Exception as e:
            logger.error(f"查询套餐简介失败: {e}")
            return None

    def query_all_packages(self) -> list:
        """
        查询所有套餐简介信息
        
        Returns:
            所有套餐信息列表
        """
        try:
            connection = self.connect_to_mysql()
            cursor = connection.cursor(pymysql.cursors.DictCursor)

            sql = "SELECT * FROM package_intro ORDER BY id"
            cursor.execute(sql)

            package_list = cursor.fetchall()

            cursor.close()
            connection.close()

            logger.info(f"查询到{len(package_list)}个套餐")
            return package_list
        except Exception as e:
            logger.error(f"查询所有套餐信息失败: {e}")
            return []

    # ===================== 套餐系列/价位/属性 =====================
    def query_all_series_names(self) -> list:
        """返回所有套餐系列名称列表"""
        try:
            connection = self.connect_to_mysql()
            cursor = connection.cursor(pymysql.cursors.DictCursor)
            cursor.execute("SELECT package_name FROM package_intro ORDER BY id")
            rows = cursor.fetchall()
            cursor.close()
            connection.close()
            return [r["package_name"] for r in rows] if rows else []
        except Exception as e:
            logger.error(f"查询系列名称失败: {e}")
            return []

    def query_package_intro_by_series(self, series_name: str) -> Optional[Dict[str, Any]]:
        """根据系列名精确查询套餐系列简介（含description）"""
        try:
            connection = self.connect_to_mysql()
            cursor = connection.cursor(pymysql.cursors.DictCursor)
            sql = "SELECT id, package_name, description FROM package_intro WHERE package_name = %s LIMIT 1"
            cursor.execute(sql, (series_name,))
            row = cursor.fetchone()
            cursor.close()
            connection.close()
            return row
        except Exception as e:
            logger.error(f"查询系列简介失败: {e}")
            return None

    def query_plans_by_series(self, series_name: str) -> list:
        """查询某系列下所有价位套餐（要求tariff_plan表包含id列作为plan_id）"""
        try:
            connection = self.connect_to_mysql()
            cursor = connection.cursor(pymysql.cursors.DictCursor)
            # 假设tariff_plan表存在id主键 将id查询出来之后重命名为plan_id
            sql = "SELECT id AS plan_id, plan_name, price FROM tariff_plan WHERE series = %s ORDER BY price"
            cursor.execute(sql, (series_name,))
            rows = cursor.fetchall()
            cursor.close()
            connection.close()
            return rows or []
        except Exception as e:
            logger.error(f"查询系列价位失败: {e}")
            return []

    def query_features_by_plan_id(self, plan_id: int) -> Dict[str, Any]:
        """查询某价位套餐的所有特性，返回字典{feature_key: feature_value}"""
        try:
            connection = self.connect_to_mysql()
            cursor = connection.cursor(pymysql.cursors.DictCursor)
            sql = "SELECT feature_key, feature_value FROM tariff_feature WHERE plan_id = %s"
            cursor.execute(sql, (plan_id,))
            rows = cursor.fetchall()
            cursor.close()
            connection.close()
            features: Dict[str, Any] = {}
            if rows:
                for r in rows:
                    features[r["feature_key"]] = r["feature_value"]
            return features
        except Exception as e:
            logger.error(f"查询套餐特性失败: {e}")
            return {}


# 创建全局单例
mysql_tools = MYSQLTools()
