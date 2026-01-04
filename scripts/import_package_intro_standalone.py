"""
导入套餐简介到MySQL数据库（独立版本）
从markdown文件中解析套餐信息并存储到数据库
不依赖项目的其他模块，可独立运行
"""
import re
import pymysql
import pymysql.cursors
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MySQL配置
MYSQL_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "123",
    "database": "data_demo",
    "charset": "utf8mb4"
}


def connect_to_mysql():
    """连接到MySQL数据库"""
    try:
        connection = pymysql.connect(
            host=MYSQL_CONFIG["host"],
            port=MYSQL_CONFIG["port"],
            user=MYSQL_CONFIG["user"],
            password=MYSQL_CONFIG["password"],
            database=MYSQL_CONFIG["database"],
            charset=MYSQL_CONFIG["charset"],
            cursorclass=pymysql.cursors.DictCursor
        )
        logger.info("数据库连接成功")
        return connection
    except Exception as e:
        logger.error(f"连接数据库失败: {e}")
        raise


def parse_package_intro(md_file_path: str) -> list:
    """
    解析套餐简介markdown文件
    
    Args:
        md_file_path: markdown文件路径
        
    Returns:
        套餐信息列表，每个元素是一个字典
    """
    packages = []
    
    try:
        with open(md_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 按照 ### 标题分割内容
        sections = re.split(r'\n### ', content)
        
        for section in sections:
            if not section.strip():
                continue
            
            # 提取标题和内容
            lines = section.strip().split('\n', 1)
            if len(lines) < 2:
                continue
            
            title_line = lines[0].strip()
            description = lines[1].strip() if len(lines) > 1 else ""
            
            # 解析标题（格式：序号. 套餐名称）
            match = re.match(r'(\d+)\.\s*(.+)', title_line)
            if match:
                package_id = int(match.group(1))
                package_name = match.group(2).strip()
                
                packages.append({
                    'package_id': package_id,
                    'package_name': package_name,
                    'description': description
                })
                
                logger.info(f"解析套餐: {package_id}. {package_name}")
        
        logger.info(f"共解析到 {len(packages)} 个套餐")
        return packages
        
    except Exception as e:
        logger.error(f"解析markdown文件失败: {e}")
        raise


def create_package_table():
    """创建套餐简介表"""
    try:
        connection = connect_to_mysql()
        cursor = connection.cursor()
        
        # 创建表（如果不存在）
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS package_intro (
            id INT PRIMARY KEY,
            package_name VARCHAR(255) NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """
        
        cursor.execute(create_table_sql)
        connection.commit()
        
        logger.info("套餐简介表创建成功或已存在")
        
        cursor.close()
        connection.close()
        
    except Exception as e:
        logger.error(f"创建表失败: {e}")
        raise


def insert_packages(packages: list):
    """
    插入套餐信息到数据库
    
    Args:
        packages: 套餐信息列表
    """
    try:
        connection = connect_to_mysql()
        cursor = connection.cursor()
        
        # 清空现有数据
        cursor.execute("DELETE FROM package_intro")
        logger.info("已清空现有套餐数据")
        
        # 插入新数据
        insert_sql = """
        INSERT INTO package_intro (id, package_name, description)
        VALUES (%s, %s, %s)
        """
        
        for package in packages:
            cursor.execute(insert_sql, (
                package['package_id'],
                package['package_name'],
                package['description']
            ))
        
        connection.commit()
        logger.info(f"成功插入 {len(packages)} 条套餐记录")
        
        # 验证插入
        cursor.execute("SELECT COUNT(*) as count FROM package_intro")
        result = cursor.fetchone()
        logger.info(f"数据库中共有 {result['count']} 条套餐记录")
        
        cursor.close()
        connection.close()
        
    except Exception as e:
        logger.error(f"插入数据失败: {e}")
        raise


def main():
    """主函数"""
    # 获取markdown文件路径
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    md_file_path = project_root / "sources" / "套餐简介_smDkXtsMAi7TacipsvazkS.md"
    
    if not md_file_path.exists():
        logger.error(f"文件不存在: {md_file_path}")
        return
    
    logger.info(f"开始处理文件: {md_file_path}")
    
    # 创建表
    create_package_table()
    
    # 解析markdown文件
    packages = parse_package_intro(str(md_file_path))
    
    # 插入数据库
    insert_packages(packages)
    
    logger.info("套餐简介导入完成！")


if __name__ == "__main__":
    main()

