"""
文档处理和分块模块
"""
import re
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self):
        self.principles = []
    
    def parse_document(self, file_path: str) -> List[Dict[str, Any]]:
        """解析电信产品推荐原则文档"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 按推荐原则分块
            principles = self._extract_principles(content)
            
            logger.info(f"成功解析文档，提取到 {len(principles)} 个推荐原则")
            return principles
            
        except Exception as e:
            logger.error(f"解析文档失败: {e}")
            return []
    
    def _extract_principles(self, content: str) -> List[Dict[str, Any]]:
        """提取推荐原则"""
        principles = []
        
        # 按数字编号分割原则
        pattern = r'\d+\.\s*[\s\S]*?(?=\n\d+\.|$)'
        matches = re.findall(pattern, content)
        
        for i, match in enumerate(matches, 1):
            principle = self._parse_single_principle(match, i)
            if principle:
                principles.append(principle)
        
        return principles
    
    def _parse_single_principle(self, text: str, principle_id: int) -> Dict[str, Any]:
        """解析单个推荐原则"""
        try:
            # 提取标题
            title_match = re.search(r'\d+\.\s*(.*)', text)
            title = title_match.group(1) if title_match else f"推荐原则{principle_id}"
            title = re.sub(r'[*_`]', '', title)  # 去掉 Markdown 修饰符

            # 提取触发条件
            trigger_match = re.search(r'触发条件\*\*[：:]\s*(.+?)(?:\n|$)', text)
            trigger_conditions = trigger_match.group(1).strip() if trigger_match else ""
            trigger_conditions = re.sub(r'[*_`]', '', trigger_conditions)

            # 提取推荐产品
            product_match = re.search(r'推荐产品\*\*[：:]\s*(.+?)(?:\n|$)', text)
            recommended_products = product_match.group(1).strip() if product_match else ""
            recommended_products = re.sub(r'[*_`]', '', recommended_products)

            # 提取套餐简介
            brief_match = re.search(r'套餐简介\*\*[：:]\s*(.+?)(?:\n|$)', text)
            package_brief = brief_match.group(1).strip() if brief_match else ""
            package_brief = re.sub(r'[*_`]', '', package_brief)

            # 确定分类
            category = self._determine_category(title)
            
            # 构建完整内容
            content = f"{title}\n触发条件：{trigger_conditions}\n推荐产品：{recommended_products}\n套餐简介：{package_brief}"
            
            return {
                'principle_id': f"P{principle_id:02d}",
                'title': title,
                'content': content,
                'trigger_conditions': trigger_conditions,
                'recommended_products': recommended_products,
                'category': category,
                'package_brief': package_brief
            }
            
        except Exception as e:
            logger.error(f"解析原则失败: {e}")
            return None
    
    def _determine_category(self, title: str) -> str:
        """根据标题确定分类"""
        if "老年" in title or "孝心" in title:
            return "老年用户"
        elif "残疾" in title or "爱心" in title:
            return "特殊群体"
        elif "军人" in title or "拥军" in title:
            return "军人群体"
        elif "5G" in title or "终端" in title:
            return "网络终端"
        elif "宽带" in title or "千兆" in title:
            return "宽带升级"
        elif "流量" in title:
            return "流量使用"
        elif "视频" in title or "影视" in title:
            return "视频娱乐"
        elif "少儿" in title or "教育" in title:
            return "教育场景"
        elif "家庭" in title or "WiFi" in title or "安防" in title:
            return "智慧家庭"
        elif "高端" in title or "徽金" in title:
            return "高端客户"
        elif "扶贫" in title or "乡村" in title:
            return "扶贫用户"
        elif "语音" in title:
            return "语音通话"
        elif "会员" in title or "畅玩" in title:
            return "会员权益"
        else:
            return "其他"
    
    def get_search_text(self, principle: Dict[str, Any]) -> str:
        """生成用于搜索的文本"""
        # 组合标题、触发条件、推荐产品和套餐简介作为搜索文本
        search_text = (f"{principle['title']} {principle['trigger_conditions']} {principle['recommended_products']} "
                       f"{principle.get('package_brief', '')}")
        return search_text.strip()
