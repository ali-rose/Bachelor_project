import pandas as pd
import re
import spacy
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional,Any
import jieba
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from itertools import groupby
import os
import jieba.analyse
import json



class PolicyDataConverter:
    """
    将政策数据转换为三种不同格式:
    1. 结构化数据 (CSV/表格)
    2. 半结构化数据 (JSON)
    3. 非结构化数据 (纯文本)
    """

    def __init__(self, input_file: str):
        """
        初始化数据转换器

        参数:
            input_file: Excel文件路径
        """
        self.df = pd.read_excel(input_file)
        # 确保所有必要的列都存在
        required_columns = ['标题', '发布时间', '网址', '发文机关', '文件种类', '主题类别', '正文']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"输入文件缺少必要的列: {col}")

                # 初始化NLP组件
        self._initialize_nlp()

                # 加载类别-目标对象映射字典
        self._load_category_target_map()

                # 构建目标对象词典和上下文词典
        self._build_target_dictionaries()

        print(f"成功加载数据，共 {len(self.df)} 条记录")

    def _initialize_nlp(self):
        """初始化NLP相关组件"""
        # 加载中文spaCy模型 (需要提前安装: pip install spacy && python -m spacy download zh_core_web_sm)
        try:
            self.nlp = spacy.load("zh_core_web_sm")
        except:
            # 如果spaCy模型加载失败，使用备用方法
            self.nlp = None
            print("警告: spaCy中文模型加载失败，将使用备用分析方法")

        # 加载自定义词典到jieba
        self._add_custom_words_to_jieba()

    def _add_custom_words_to_jieba(self):
        """向jieba添加自定义词典"""
        # 政策相关词汇
        policy_terms = [
            "适用对象", "适用范围", "政策对象", "扶持对象", "补贴对象", "奖励对象",
            "受益人群", "服务对象", "惠及群体", "主要面向", "专项资金", "申请条件"
        ]

        # 组织实体词汇
        org_terms = [
            "中小企业", "小微企业", "初创企业", "高新技术企业", "科技型企业",
            "民营企业", "外资企业", "合资企业", "独资企业", "上市公司",
            "研究机构", "事业单位", "社会组织", "非营利组织", "行业协会"
        ]

        # 人群类别词汇
        people_terms = [
            "大学生", "毕业生", "创业者", "创业团队", "失业人员", "下岗职工",
            "农民工", "就业困难人员", "残疾人", "老年人", "退役军人", "烈士家属",
            "低收入群体", "贫困人口", "脱贫人口", "外来务工人员"
        ]

        # 添加到jieba词典
        for term in policy_terms + org_terms + people_terms:
            jieba.add_word(term)

    def _load_category_target_map(self):
        """加载类别-目标对象映射字典"""
        self.category_target_map = {
            "中央有关文件": "政府机关、各级单位",
            "人口与计划生育、妇女儿童工作\\人口与计划生育": "计划生育服务对象、家庭、个人",
            "人口与计划生育、妇女儿童工作\\妇女儿童": "妇女、儿童、家庭",
            "公安、安全、司法\\公安": "公安机关、治安管理人员、公安工作者",
            "公安、安全、司法\\其他": "其他公安相关人员",
            "公安、安全、司法\\司法": "司法工作者、法院人员、律师",
            "其他": "一般社会群体、无特定对象",
            "农业、林业、水利\\其他": "农业从业者、农民、林业工人、水利工作者",
            "农业、林业、水利\\农业、畜牧业、渔业": "农民、畜牧业工作者、渔业从业者",
            "农业、林业、水利\\林业": "林业工作者、生态环保人员",
            "农业、林业、水利\\水利": "水利工作人员、农民、农业从业者",
            "劳动、人事、监察\\其他": "劳动者、就业人员",
            "劳动、人事、监察\\劳动就业": "劳动者、求职者、失业人员、就业者",
            "劳动、人事、监察\\监察": "监察人员、监察员、工作人员",
            "劳动、人事、监察\\社会保障": "参保人员、退休人员、社会保障对象",
            "劳动、人事、监察\\纠正行业不正之风": "企业员工、监察对象",
            "卫生、体育\\体育": "运动员、体育工作者、体育人员",
            "卫生、体育\\其他": "卫生人员、一般民众",
            "卫生、体育\\医药管理": "医疗行业从业者、医务工作者",
            "卫生、体育\\卫生": "医疗机构工作人员、患者、居民",
            "商贸、海关、旅游\\其他": "商贸工作人员、海关工作人员、旅游从业者",
            "商贸、海关、旅游\\国内贸易（含供销）": "贸易从业人员、商贸人员",
            "商贸、海关、旅游\\对外经贸合作": "国际贸易人员、进出口公司员工",
            "商贸、海关、旅游\\旅游": "旅游从业者、游客、旅游相关单位",
            "商贸、海关、旅游\\海关": "海关工作人员、外贸人员",
            "国务院组织机构\\其他": "政府工作人员、机关工作人员",
            "国务院组织机构\\国务院": "政府工作人员、国务院人员",
            "国务院组织机构\\国务院直属事业单位": "事业单位人员",
            "国务院组织机构\\国务院直属机构": "直属机构工作人员",
            "国务院组织机构\\国务院组成部门": "政府部门工作人员",
            "国务院组织机构\\国务院部委管理的国家局": "政府局工作人员",
            "国土资源、能源\\其他": "土地管理人员、能源管理人员",
            "国土资源、能源\\土地": "土地使用者、农民、土地所有者",
            "国土资源、能源\\海洋": "海洋管理者、渔业从业者",
            "国土资源、能源\\煤炭": "煤炭开采工人、煤矿工作人员",
            "国土资源、能源\\电力": "电力公司工作人员、电力行业人员",
            "国民经济管理、国有资产监管\\其他": "经济管理者、国有企业员工",
            "国民经济管理、国有资产监管\\国有资产监管": "国有企业员工、资产监管人员",
            "国民经济管理、国有资产监管\\宏观经济": "经济学者、宏观经济分析人员",
            "国民经济管理、国有资产监管\\物价": "物价监管人员、消费者",
            "国民经济管理、国有资产监管\\经济体制改革": "改革实施人员、经济学者",
            "国民经济管理、国有资产监管\\统计": "统计人员、研究人员",
            "国防\\国防动员": "军人、退役军人",
            "国防\\国防建设": "国防建设人员、军人、军事人员",
            "城乡建设、环境保护\\其他": "城市建设人员、环保人员",
            "城乡建设、环境保护\\城乡建设（含住房）": "城市规划人员、建筑从业者",
            "城乡建设、环境保护\\城市规划": "城市规划人员、建筑设计师",
            "城乡建设、环境保护\\气象、水文、测绘、地震": "气象学者、水文人员、测绘师",
            "城乡建设、环境保护\\环境监测、保护与治理": "环保从业者、环保工作人员",
            "城乡建设、环境保护\\节能与资源综合利用": "节能环保人员、能源使用者",
            "对外事务\\国际条约、国际组织": "外交人员、国际事务人员",
            "工业、交通\\信息产业（含电信）": "信息产业工作人员、电信行业人员",
            "工业、交通\\公路": "公路运输人员、司机、运输人员",
            "工业、交通\\其他": "工业从业者、交通行业人员",
            "工业、交通\\机械制造与重工业": "机械制造工人、重工业工人",
            "工业、交通\\民航": "民航工作人员、航空人员",
            "工业、交通\\邮政": "邮政工作人员、快递人员",
            "工业、交通\\铁路": "铁路工作人员、铁路职员",
            "市场监管、安全生产监管\\其他": "市场监管人员、安全生产监管人员",
            "市场监管、安全生产监管\\安全生产监管": "安全生产人员、生产监督人员",
            "市场监管、安全生产监管\\工商": "工商管理人员、企业管理人员",
            "市场监管、安全生产监管\\质量监督": "质量监督人员、检测人员",
            "市场监管、安全生产监管\\食品药品监管": "食品药品监管人员、监管人员",
            "文化、广电、新闻出版\\文化": "文化工作者、文艺工作者",
            "文化、广电、新闻出版\\文物": "文物管理人员、考古人员",
            "文化、广电、新闻出版\\新闻出版": "新闻人员、出版人员",
            "民政、扶贫、救灾\\优抚安置": "退役军人、优抚人员",
            "民政、扶贫、救灾\\其他": "贫困人群、受灾人员",
            "民政、扶贫、救灾\\扶贫": "贫困家庭、贫困人群、农民",
            "民政、扶贫、救灾\\社会福利": "社会福利对象、福利人员",
            "民政、扶贫、救灾\\社团管理": "社团成员、组织人员",
            "民政、扶贫、救灾\\行政区划与地名": "地方行政人员、社区人员",
            "民族、宗教\\宗教事务": "宗教团体、宗教人员",
            "港澳台侨工作\\侨务工作": "侨务人员、港澳台侨民",
            "港澳台侨工作\\港澳工作": "港澳居民、港澳工作人员",
            "科技、教育\\教育": "学生、教师、教育工作者",
            "科技、教育\\知识产权": "知识产权从业者、企业家",
            "科技、教育\\科技": "科技工作者、研究人员",
            "综合政务\\其他": "政府人员、各级工作人员",
            "综合政务\\参事、文史": "文史专家、历史学者",
            "综合政务\\应急管理": "应急管理人员、灾难救援人员",
            "综合政务\\政务公开": "政府工作人员、政务公开人员",
            "综合政务\\政务督查": "政务督查人员、督查人员",
            "综合政务\\文秘工作": "文秘工作人员",
            "综合政务\\电子政务": "电子政务人员、技术人员",
            "财政、金融、审计\\保险": "保险从业人员、保险消费者",
            "财政、金融、审计\\其他": "金融行业从业人员",
            "财政、金融、审计\\社会信用体系建设": "信用体系建设人员、监管人员",
            "财政、金融、审计\\税务": "税务人员、纳税人",
            "财政、金融、审计\\证券": "证券从业人员、投资者",
            "财政、金融、审计\\财政": "财政人员、财务人员",
            "财政、金融、审计\\银行": "银行从业人员、储户"
        }

    def _build_target_dictionaries(self):
        """构建目标对象词典和上下文词典"""
        # 定义常见的适用对象类型及其相关词汇
        self.entity_types = {
            "企业类": [
                "企业", "公司", "单位", "法人", "经营者", "创业者", "创业团队",
                "中小企业", "小微企业", "初创企业", "高新技术企业", "科技型企业",
                "民营企业", "外资企业", "合资企业", "独资企业", "上市公司"
            ],
            "个人类": [
                "个人", "居民", "公民", "自然人", "家庭", "户籍人口", "常住人口",
                "纳税人", "参保人", "缴费人", "申请人", "购房者", "消费者"
            ],
            "特定群体": [
                "学生", "毕业生", "大学生", "研究生", "博士", "教师", "医生", "教职工",
                "农民", "农户", "村民", "职工", "劳动者", "工人", "员工", "职员",
                "失业人员", "下岗职工", "农民工", "就业困难人员", "残疾人",
                "老年人", "儿童", "妇女", "退役军人", "烈士家属", "低收入群体",
                "贫困人口", "脱贫人口", "外来务工人员"
            ],
            "机构类": [
                "学校", "医院", "研究所", "高校", "科研机构", "事业单位", "社会组织",
                "非营利组织", "协会", "基金会", "行业协会", "商会", "政府机构"
            ],
            "地域类": [
                "中央", "各省", "各市", "各县", "乡镇", "街道", "社区", "城市", "农村",
                "边远地区", "老少边穷地区", "特殊区域"
            ]
        }

        # 扁平化词典，便于查找
        self.all_entity_words = {}
        for entity_type, words in self.entity_types.items():
            for word in words:
                self.all_entity_words[word] = entity_type

        # 构建适用对象上下文词典
        self.target_context_words = [
            "适用于", "适用对象", "适用范围", "政策对象", "覆盖范围", "服务对象",
            "惠及群体", "本政策针对", "面向", "惠及", "帮助", "支持", "扶持对象",
            "补贴对象", "资助对象", "奖励对象", "受益人群", "主要服务", "主要面对",
            "主要针对", "主要覆盖", "可申请", "可享受", "可获得"
        ]
    def convert_to_structured(self, output_file: str = "structured_data.xlsx") -> None:
        """
        将数据转换为结构化格式并保存为Excel

        结构化数据特点:
        - 严格的表格形式
        - 固定的字段和格式
        - 标准化的值
        - 规范的条目提取
        """
        print("开始转换为结构化数据...")

        # 创建新的结构化DataFrame
        structured_df = pd.DataFrame()

        # 复制基本字段
        structured_df['政策ID'] = range(1, len(self.df) + 1)
        structured_df['政策标题'] = self.df['标题']
        # 修改1: 去除发布日期中的时分秒，只保留年月日
        structured_df['发布日期'] = pd.to_datetime(self.df['发布时间'], errors='coerce').dt.date
        structured_df['发文机关'] = self.df['发文机关']
        structured_df['文件分类'] = self.df['文件种类']
        structured_df['政策领域'] = self.df['主题类别']
        structured_df['政策链接'] = self.df['网址']

        # 提取结构化信息
        structured_df['生效状态'] = self.df.apply(lambda row: self._extract_status(row['正文']), axis=1)
        # 修改2: 去除发文字号、实施日期、关键条款数量
        structured_df['主要措施'] = self.df.apply(lambda row: self._extract_measures(row['正文']), axis=1)
        # 使用改进后的适用对象提取方法
        structured_df['适用对象'] = self.df.apply(
            lambda row: self._extract_target_groups_enhanced(
                row['正文'] if pd.notna(row['正文']) else "",
                row['标题'] if pd.notna(row['标题']) else "",
                row['主题类别'] if pd.notna(row['主题类别']) else ""
            ),
            axis=1
        )

        structured_df['关键词'] = self.df.apply(lambda row: ','.join(self._extract_keywords(
            f"{row['标题']} {row['正文'][:1000]}" if pd.notna(row['正文']) else row['标题']
        )), axis=1)

        # 保存为Excel
        structured_df.to_excel(output_file, index=False)
        print(f"结构化数据已保存至: {output_file}")

        return structured_df

    def convert_to_semi_structured(self, output_file: str = "semi_structured_data.json") -> None:
        """
        将数据转换为半结构化格式并保存为JSON

        半结构化数据特点:
        - 具有层次结构
        - 字段可嵌套
        - 保留文本的语义结构
        - 灵活的数据组织
        """
        print("开始转换为半结构化数据...")

        semi_structured_data = []

        for idx, row in self.df.iterrows():
            # 构建半结构化文档
            document = {
                "document_id": idx + 1,
                "metadata": {
                    "title": row['标题'],
                    "publish_date": row['发布时间'],
                    "url": row['网址'],
                    "issuing_authority": row['发文机关'],
                    "document_type": row['文件种类'],
                    "policy_category": row['主题类别']
                },
                "content": self._structure_content(row['正文']) if pd.notna(row['正文']) else {}
            }
            semi_structured_data.append(document)

        # 保存为JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(semi_structured_data, f, ensure_ascii=False, indent=2)

        print(f"半结构化数据已保存至: {output_file}")

        return semi_structured_data

    def convert_to_unstructured(self, output_file: str = "unstructured_data.txt") -> None:
        """
        将数据转换为非结构化格式并保存为单个文本文件

        非结构化数据特点:
        - 连续的自然语言文本
        - 无明确分隔的字段
        - 保留原文格式和流动性

        修改3: 所有政策保存到同一个文件，而不是拆分为多个
        """
        print("开始转换为非结构化数据...")

        # 创建输出目录
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 将所有政策写入同一个文件
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, row in self.df.iterrows():
                # 构建非结构化文本
                text_content = f"""[文档ID: {idx + 1}]
标题: {row['标题']}

发布时间: {row['发布时间']}
发文机关: {row['发文机关']}
文件种类: {row['文件种类']}
主题类别: {row['主题类别']}
网址链接: {row['网址']}

正文内容:
{row['正文'] if pd.notna(row['正文']) else '无正文内容'}

{'-' * 80}

"""
                f.write(text_content)

        print(f"非结构化数据已保存至文件: {output_file}")

        return output_file

    def convert_all_formats(self, output_dir: str = "converted_data") -> Dict[str, Any]:
        """
        转换为所有三种格式并返回结果
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 转换并保存各种格式
        structured_df = self.convert_to_structured(os.path.join(output_dir, "structured_data.xlsx"))
        semi_structured_data = self.convert_to_semi_structured(os.path.join(output_dir, "semi_structured_data.json"))
        unstructured_path = self.convert_to_unstructured(os.path.join(output_dir, "unstructured_data.txt"))

        # 创建一个README文件说明数据格式
        readme_content = """# 政策数据格式转换结果

## 文件说明
1. `structured_data.xlsx`: 结构化数据，表格形式，具有固定字段和格式
2. `semi_structured_data.json`: 半结构化数据，JSON格式，保留文档层次结构
3. `unstructured_data.txt`: 非结构化数据，单个文本文件，保留原始流动性

## 数据格式特点
- 结构化数据: 严格的表格形式，标准化字段，适合关系型数据库
- 半结构化数据: 有层次结构但灵活，保留文档语义结构，适合NoSQL数据库
- 非结构化数据: 连续文本，无明确字段分隔，保留原文格式和流动性，整合在单一文件中便于RAG模型处理

## 使用建议
根据实验需求，可以选择相应格式进行RAG模型的训练和测试。
"""
        with open(os.path.join(output_dir, "README.md"), 'w', encoding='utf-8') as f:
            f.write(readme_content)

        results = {
            "structured": structured_df,
            "semi_structured": semi_structured_data,
            "unstructured": unstructured_path
        }

        print(f"所有格式转换完成，数据已保存至: {output_dir}")
        return results

    # 辅助方法
    def _extract_status(self, text: str) -> str:
        """提取政策生效状态"""
        if not isinstance(text, str):
            return "未知"

        if "废止" in text or "已废止" in text:
            return "已废止"
        elif "暂行" in text or "试行" in text:
            return "试行"
        elif "修订" in text or "修正" in text:
            return "已修订"
        else:
            return "有效"

    def _extract_document_number(self, text: str) -> str:
        """提取发文字号"""
        if not isinstance(text, str):
            return ""

        # 常见发文字号模式
        patterns = [
            r'[\[（\(][国地省市][a-zA-Z0-9发委政字]{1,20}[\]）\)]第?[0-9０-９〇一二三四五六七八九十百千]+号',
            r'[国地省市][a-zA-Z0-9发委政字]{1,20}[\[（\(]?[0-9０-９]{4}[\]）\)]?第?[0-9０-９]+号'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[0]

        return ""

    def _extract_implementation_date(self, text: str) -> str:
        """提取实施日期"""
        if not isinstance(text, str):
            return ""

        # 查找实施日期的模式
        patterns = [
            r'[自从]([0-9０-９]{4}年[0-9０-９]{1,2}月[0-9０-９]{1,2}日)起[实施执行]',
            r'[实施执行]日期[是为][：:]?([0-9０-９]{4}年[0-9０-９]{1,2}月[0-9０-９]{1,2}日)',
            r'[于将]([0-9０-９]{4}年[0-9０-９]{1,2}月[0-9０-９]{1,2}日)[起开始][实施执行]'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[0]

        return ""


    def _extract_measures(self, text: str) -> str:
        """提取政策主要措施"""
        if not isinstance(text, str):
            return ""

        # 查找关键措施部分
        measures = []

        # 1. 先尝试找到明确标记的措施部分
        measure_section_patterns = [
            r'主要措施[如下为是：:]+[\s\S]{0,100}?([一二三四五六七八九十(（][^。]{5,}[\s\S]*?)(?=$|[^一二三四五六七八九十(（])',
            r'政策[内容要点][如下为是：:]+[\s\S]{0,100}?([一二三四五六七八九十(（][^。]{5,}[\s\S]*?)(?=$|[^一二三四五六七八九十(（])',
            r'[具体主要重点关键][措施办法][包括有是为：:]+[\s\S]{0,100}?([一二三四五六七八九十(（][^。]{5,}[\s\S]*?)(?=$|[^一二三四五六七八九十(（])'
        ]

        for pattern in measure_section_patterns:
            matches = re.findall(pattern, text)
            if matches:
                section_text = matches[0]
                # 提取编号措施
                numbered_measures = self._extract_numbered_items_continuous(section_text)
                if numbered_measures:
                    return "; ".join(
                        [item[:1000] + ("..." if len(item) > 1000 else "") for item in numbered_measures[:5]])

        # 2. 直接在全文中寻找数字标记的项目
        numbered_measures = self._extract_numbered_items_continuous(text)
        if numbered_measures:
            # 过滤掉过短的措施项
            valid_measures = [item for item in numbered_measures if len(item) > 10]
            if valid_measures:
                return "; ".join([item[:1000] + ("..." if len(item) > 1000 else "") for item in valid_measures[:5]])

        # 3. 如果还是没找到，使用关键词定位策略
        sentences = re.split(r'[。！？]', text)
        for sentence in sentences:
            if len(sentence) > 30 and len(sentence) < 3000 and (
                    "措施" in sentence or "办法" in sentence or "补贴" in sentence or "支持" in sentence):
                measures.append(sentence[:1000] + ("..." if len(sentence) > 1000 else ""))
                if len(measures) >= 2:
                    break

        if not measures:
            return text

        return "; ".join(measures)

    def _is_numbered_item(self, text: str) -> bool:
        """判断文本是否以数字编号开头"""
        # 中文数字编号模式: 一、二、三、 或 (一)(二)(三) 或 （一）（二）（三） 等
        patterns = [
            r'^[一二三四五六七八九十][、.．，,。：:]',  # 一、二、三、
            r'^[（\(][一二三四五六七八九十][）\)]',  # (一)(二)(三) 或 （一）（二）（三）
            r'^[0-9]+[、.．，,。：:]',  # 1、2、3、
            r'^[（\(]?[0-9]+[）\)]',  # (1)(2)(3) 或 （1）（2）（3）
            r'^第[一二三四五六七八九十0-9]+条',  # 第一条、第二条
        ]

        for pattern in patterns:
            if re.match(pattern, text.strip()):
                return True
        return False

    def _extract_numbered_items_continuous(self, text: str) -> list:
        """从连续文本中提取编号项目"""
        # 匹配各种编号格式
        number_pattern = r'([一二三四五六七八九十]{1,2}[、.．]|[0-9]{1,2}[、.．]|[(（][一二三四五六七八九十0-9]{1,2}[)）]|第[一二三四五六七八九十0-9]+条)'

        # 找到所有编号的位置
        matches = list(re.finditer(number_pattern, text))

        if not matches:
            return []

        # 提取每个编号项的内容
        items = []
        for i in range(len(matches)):
            start_pos = matches[i].start()
            # 如果不是最后一个编号，则取到下一个编号的位置
            if i < len(matches) - 1:
                # 检查两个编号之间的距离，如果太近可能是同一个项目中的子项
                if matches[i + 1].start() - start_pos < 10:
                    continue
                end_pos = matches[i + 1].start()
            else:
                # 如果是最后一个编号，则取到合理的结束位置
                # 尝试找到句号、分号等作为结束
                end_text = text[start_pos:]
                sentence_end = re.search(r'[。；;!！?？]', end_text)
                if sentence_end and sentence_end.start() < 500:  # 限制在500字符内查找
                    end_pos = start_pos + sentence_end.start() + 1
                else:
                    # 如果找不到合适的结束标点，则取一个合理的长度
                    end_pos = min(start_pos + 300, len(text))

            # 提取编号项内容
            item_text = text[start_pos:end_pos].strip()

            # 确保编号项不会过短
            if len(item_text) > 10:
                items.append(item_text)

        return items

    def _extract_target_groups_enhanced(self, text: str, title: str, category: str) -> str:
        """
        增强版的适用对象提取方法，结合多种NLP技术

        参数:
            text: 政策正文
            title: 政策标题
            category: 政策类别

        返回:
            提取的适用对象文本
        """
        if not isinstance(text, str) or not text.strip():
            if title and category in self.category_target_map:
                return f"从政策类别推断: {self.category_target_map[category]}"
            return "未明确指定适用对象"

        # 1. 基于规则的显式提取
        explicit_targets = self._extract_explicit_targets(text)
        if explicit_targets:
            return explicit_targets

        # 2. 基于关键段落的语义提取
        semantic_targets = self._extract_semantic_targets(text)
        if semantic_targets:
            return semantic_targets

        # 3. 基于实体识别的提取
        ner_targets = self._extract_targets_by_ner(text, title)
        if ner_targets:
            return ner_targets

        # 4. 从标题中提取指示性信息
        title_targets = self._extract_from_title(title)
        if title_targets:
            return f"从标题推断: {title_targets}"

        # 5. 根据政策类别推断
        if category in self.category_target_map:
            return f"从政策类别推断: {self.category_target_map[category]}"

        return "未明确指定适用对象"

    def _extract_explicit_targets(self, text: str) -> str:
        """从文本中提取明确指定的适用对象"""
        # 减少文本量，只分析前3000个字符
        text_start = text[:3000] if len(text) > 3000 else text

        # 更精确的模式匹配
        target_patterns = [
            # 直接指明适用对象的强匹配模式
            r'(适用对象|适用范围|政策对象)[为是：:]([\s\S]{5,100}?)(?=。|\n|；|$)',
            r'(本[文件通知办法规定方案计划])[的]?(适用对象|适用范围)[为是：:]([\s\S]{5,100}?)(?=。|\n|；|$)',
            r'(本[文件通知办法规定方案计划])适用于([\s\S]{5,100}?)(?=。|\n|；|$)',
            r'(扶持对象|补贴对象|资助对象|奖励对象|受益人群)[为是：:]([\s\S]{5,100}?)(?=。|\n|；|$)',
            r'(主要面向|重点支持)[的：:]([\s\S]{5,100}?)(?=。|\n|；|$)',
        ]

        # 附近行模式（适用对象可能在冒号后单独成行）
        newline_patterns = [
            r'(适用对象|适用范围|政策对象)[：:]\s*\n+\s*([\s\S]{5,200}?)(?=\n\n|\n#|\n第)',
            r'(申请条件|申请对象)[：:]\s*\n+\s*([\s\S]{5,200}?)(?=\n\n|\n#|\n第)',
        ]

        # 尝试不同的模式
        for pattern in target_patterns:
            matches = re.findall(pattern, text_start)
            if matches:
                # 处理匹配结果
                if len(matches[0]) >= 3:  # 有些模式有3个捕获组
                    target_text = matches[0][2].strip()
                else:
                    target_text = matches[0][1].strip()

                # 清理提取的文本
                return self._clean_and_format_target_text(target_text)

        # 检查换行模式
        for pattern in newline_patterns:
            matches = re.findall(pattern, text_start)
            if matches:
                return self._clean_and_format_target_text(matches[0][1].strip())

        return ""

    def _extract_semantic_targets(self, text: str) -> str:
        """基于语义的目标提取，寻找与目标对象相关的关键段落"""
        # 仅分析前5000个字符
        text_to_analyze = text[:5000] if len(text) > 5000 else text

        # 按段落拆分文本
        paragraphs = re.split(r'\n+', text_to_analyze)
        relevant_paragraphs = []

        # 找出与适用对象相关的段落
        for para in paragraphs:
            # 跳过太短的段落
            if len(para) < 15:
                continue

            # 计算相关性分数
            score = 0
            for keyword in self.target_context_words:
                if keyword in para:
                    score += 3  # 关键词加权

            # 检查是否包含实体词汇
            for entity in self.all_entity_words.keys():
                if entity in para:
                    score += 1  # 实体词汇加权

            # 收集高分段落
            if score >= 3:  # 设定一个阈值
                relevant_paragraphs.append((para, score))

        # 如果找到相关段落，处理分数最高的段落
        if relevant_paragraphs:
            # 按相关性分数排序
            relevant_paragraphs.sort(key=lambda x: x[1], reverse=True)
            top_paragraph = relevant_paragraphs[0][0]

            # 尝试从段落中提取关键句子
            key_sentences = re.split(r'[。！？]', top_paragraph)
            target_sentences = []

            for sentence in key_sentences:
                for keyword in self.target_context_words:
                    if keyword in sentence and len(sentence) > 10:
                        target_sentences.append(sentence)
                        break

            if target_sentences:
                # 连接找到的关键句子
                result = "。".join(target_sentences)
                # 提取实体并构建结构化结果
                return self._extract_and_structure_entities(result)

        return ""

    def _extract_targets_by_ner(self, text: str, title: str) -> str:
        """使用命名实体识别提取目标对象"""
        # 分析文本前3000字符和标题
        text_to_analyze = text[:3000] if len(text) > 3000 else text
        combined_text = title + "。" + text_to_analyze

        # 使用实体提取方法
        entities = self._identify_target_entities_enhanced(combined_text)

        if entities:
            return entities

        return ""

    def _identify_target_entities_enhanced(self, text: str) -> str:
        """增强版实体识别方法"""
        found_entities = defaultdict(set)

        # 1. 使用jieba进行词性标注
        words = pseg.cut(text)
        for word, flag in words:
            # 检查是否是我们定义的目标实体
            if word in self.all_entity_words:
                entity_type = self.all_entity_words[word]
                # 尝试获取实体的完整形式
                extended_entity = self._get_extended_entity(text, word)
                found_entities[entity_type].add(extended_entity)

            # 针对特定词性的词，也考虑为潜在实体
            if flag in ['n', 'ni', 'ns', 'nt', 'nz'] and len(word) >= 2:
                for entity_type, entities in self.entity_types.items():
                    # 检查是否是实体类型的子字符串
                    for entity in entities:
                        if word in entity or entity in word:
                            extended_entity = self._get_extended_entity(text, word)
                            found_entities[entity_type].add(extended_entity)
                            break

        # 2. 使用spaCy进行命名实体识别（如果可用）
        if self.nlp is not None:
            doc = self.nlp(text[:10000])  # 限制处理文本长度
            for ent in doc.ents:
                # 只考虑ORG、PERSON、GPE类型的实体
                if ent.label_ in ["ORG", "PERSON", "GPE"]:
                    # 映射到我们的实体类型
                    entity_type = self._map_spacy_entity_to_type(ent.label_)
                    if entity_type:
                        found_entities[entity_type].add(ent.text)

        # 构建结果字符串
        if found_entities:
            result_parts = []
            for entity_type, entities in found_entities.items():
                # 过滤掉重复和子字符串
                filtered_entities = self._filter_overlapping_entities(entities)
                # 限制每类最多5个实体
                entity_list = list(filtered_entities)[:5]
                if entity_list:
                    result_parts.append(f"{entity_type}: {', '.join(entity_list)}")

            if result_parts:
                return "; ".join(result_parts)

        return ""

    def _map_spacy_entity_to_type(self, spacy_label: str) -> str:
        """将spaCy实体类型映射到我们的实体类型"""
        mapping = {
            "ORG": "机构类",
            "PERSON": "个人类",
            "GPE": "地域类",
        }
        return mapping.get(spacy_label, "")

    def _filter_overlapping_entities(self, entities: Set[str]) -> Set[str]:
        """过滤重复和重叠的实体"""
        result = set()
        entity_list = sorted(list(entities), key=len, reverse=True)

        for i, entity in enumerate(entity_list):
            should_add = True
            for j, other_entity in enumerate(entity_list):
                if i != j and entity in other_entity and len(entity) < len(other_entity):
                    should_add = False
                    break
            if should_add:
                result.add(entity)

        return result

    def _get_extended_entity(self, text: str, entity: str) -> str:
        """获取实体的扩展形式（前后修饰词）"""
        idx = text.find(entity)
        if idx == -1:
            return entity

        # 向前查找可能的修饰词
        start = max(0, idx - 10)
        end = min(len(text), idx + len(entity) + 2)
        context = text[start:end]

        # 尝试提取更完整的实体表达
        pattern = r'([^。，；、！？\s]{0,10}' + re.escape(entity) + r'[^。，；、！？\s]{0,5})'
        match = re.search(pattern, context)

        if match:
            return match.group(1)

        return entity

    def _extract_from_title(self, title: str) -> str:
        """从标题中提取适用对象指示信息"""
        if not title:
            return ""

        # 标题模式
        title_patterns = [
            r"关于(.*?)(支持|促进|加强|鼓励|扶持|帮助)(.*?)(发展|建设|工作|保障)",
            r"(.*?)面向(.*?)的(.*?)(政策|措施|办法|规定|方案)",
            r"(支持|促进|加强|鼓励|扶持|帮助)(.*?)(发展|建设|工作|保障)的(.*?)(政策|措施|办法|规定|方案)"
        ]

        for pattern in title_patterns:
            match = re.search(pattern, title)
            if match:
                # 从匹配组中提取可能的目标对象
                for group in match.groups():
                    if group and len(group) < 15:
                        # 检查是否包含实体词
                        for entity in self.all_entity_words:
                            if entity in group:
                                return group

        # 使用jieba分词提取可能的名词短语
        words = pseg.cut(title)
        noun_phrases = []

        for word, flag in words:
            if flag.startswith('n') and len(word) >= 2:
                # 检查是否是我们定义的目标实体
                for entity_type, entities in self.entity_types.items():
                    if word in entities or any(entity in word for entity in entities):
                        return word

                # 收集名词，可能是特殊领域名词
                noun_phrases.append(word)

        if noun_phrases:
            # 返回最长的名词短语
            return max(noun_phrases, key=len)

        return ""

    def _clean_and_format_target_text(self, text: str) -> str:
        """清理和格式化提取的目标文本"""
        if not text:
            return ""

        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text.strip())

        # 移除常见的无用前缀
        prefixes = ["包括但不限于", "主要包括", "具体包括", "主要是", "具体为", "即"]
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()

        # 移除括号中的解释性内容
        text = re.sub(r'\([^)]*\)', '', text)
        text = re.sub(r'（[^）]*）', '', text)

        # 移除冗余标点
        text = re.sub(r'[,，;；、]{2,}', '、', text)

        # 限制长度
        if len(text) > 150:
            # 提取实体并结构化，而不是简单截断
            return self._extract_and_structure_entities(text)

        return text.strip()

    def _extract_and_structure_entities(self, text: str) -> str:
        """从长文本中提取实体并进行结构化"""
        # 使用jieba分词和词性标注
        words = pseg.cut(text)

        # 收集可能的实体
        potential_entities = defaultdict(list)
        current_entity = ""
        current_type = ""

        # 遍历分词结果
        for word, flag in words:
            # 如果是我们定义的实体
            if word in self.all_entity_words:
                if current_entity:
                    # 保存之前收集的实体
                    if current_type:
                        potential_entities[current_type].append(current_entity)

                # 开始新的实体
                current_entity = word
                current_type = self.all_entity_words[word]

            # 如果是名词，可能是实体的一部分
            elif flag.startswith('n') and len(word) >= 2:
                # 检查是否属于某个实体类型
                entity_type = ""
                for type_name, entities in self.entity_types.items():
                    if any(entity in word for entity in entities):
                        entity_type = type_name
                        break

                if entity_type:
                    if current_entity and current_type != entity_type:
                        # 保存之前收集的实体
                        potential_entities[current_type].append(current_entity)
                        current_entity = word
                        current_type = entity_type
                    else:
                        current_entity = word
                        current_type = entity_type
                elif current_entity and len(current_entity) + len(word) <= 10:
                    # 扩展当前实体
                    current_entity += word
                else:
                    # 保存当前实体，开始新的
                    if current_entity and current_type:
                        potential_entities[current_type].append(current_entity)
                    current_entity = ""
                    current_type = ""

            # 如果是标点或连词，可能是实体的分隔符
            elif word in ['，', '、', '和', '与', '及']:
                if current_entity and current_type:
                    potential_entities[current_type].append(current_entity)
                    current_entity = ""
            else:
                # 其他词，结束当前实体
                if current_entity and current_type:
                    potential_entities[current_type].append(current_entity)
                current_entity = ""
                current_type = ""

        # 保存最后一个实体
        if current_entity and current_type:
            potential_entities[current_type].append(current_entity)

        # 过滤和整合实体
        result_parts = []
        for entity_type, entities in potential_entities.items():
            # 过滤重复和短实体
            filtered_entities = set()
            for entity in entities:
                if len(entity) >= 2:
                    filtered_entities.add(entity)

            # 进一步过滤重叠实体
            final_entities = self._filter_overlapping_entities(filtered_entities)

            # 构建类型结果
            if final_entities:
                result_parts.append(f"{entity_type}: {', '.join(sorted(final_entities)[:5])}")

        if result_parts:
            return "; ".join(result_parts)

        # 如果没有提取到结构化实体，返回原文的一部分
        return text[:150] + "..." if len(text) > 150 else text


    def _extract_keywords(self, text: str, topK: int = 10) -> List[str]:
        """提取文本关键词"""
        if not isinstance(text, str) or not text:
            return []

        # 使用jieba提取关键词
        keywords = jieba.analyse.extract_tags(text, topK=topK)
        return keywords

    def _structure_content(self, text: str) -> Dict[str, Any]:
        """结构化文档内容"""
        if not isinstance(text, str):
            return {"raw_text": ""}

        # 分割段落
        paragraphs = [p for p in text.split('\n') if p.strip()]

        # 识别章节结构
        chapters = []
        current_chapter = None
        current_articles = []

        # 章节模式
        chapter_pattern = r'^第[一二三四五六七八九十百千0-9０-９]+[章节编]'
        article_pattern = r'^第[一二三四五六七八九十百千0-9０-９]+条'

        for para in paragraphs:
            if re.match(chapter_pattern, para):
                # 保存前一章节
                if current_chapter:
                    chapters.append({
                        "title": current_chapter,
                        "articles": current_articles
                    })

                current_chapter = para
                current_articles = []
            elif re.match(article_pattern, para):
                current_articles.append(para)
            elif current_articles:
                # 附加到最后一个条款
                if current_articles[-1].endswith('。'):
                    current_articles[-1] += " " + para
                else:
                    current_articles[-1] += para
            elif current_chapter:
                # 章节描述
                current_chapter += " " + para

        # 添加最后一章
        if current_chapter:
            chapters.append({
                "title": current_chapter,
                "articles": current_articles
            })

        # 提取关键信息
        structure = {
            "document_number": self._extract_document_number(text),
            "implementation_date": self._extract_implementation_date(text),
            "status": self._extract_status(text),
            "chapters": chapters if chapters else [],
            "key_points": self._extract_key_points(text),
            "raw_text": text
        }

        return structure

    def _extract_key_points(self, text: str, max_points: int = 3) -> List[str]:
        """提取文本关键要点"""
        if not isinstance(text, str):
            return []

        # 查找关键要点标记
        key_points = []

        # 要点可能的标记
        point_markers = [
            r'要点[如下包括为：:]([\s\S]{20,200}?)\n',
            r'重点[内容如下为：:]([\s\S]{20,200}?)\n',
            r'(主要内容[如下为：:][\s\S]{20,200}?)\n'
        ]

        for marker in point_markers:
            matches = re.findall(marker, text)
            if matches:
                for match in matches[:max_points]:
                    key_points.append(match.strip())
                return key_points

        # 如果没有找到明确的要点标记，尝试提取重要段落
        intro_pattern = r'^([^第\n]{20,200}?。)'
        intro_matches = re.findall(intro_pattern, text)
        if intro_matches:
            key_points.append(intro_matches[0].strip())

        # 提取包含关键词的段落
        important_keywords = ["目的", "意义", "总体要求", "基本原则", "主要任务"]
        paragraphs = text.split('\n')

        for keyword in important_keywords:
            if len(key_points) >= max_points:
                break

            for para in paragraphs:
                if keyword in para and 20 <= len(para) <= 200:
                    key_points.append(para.strip())
                    break

        return key_points[:max_points]

    def _extract_context(self, text: str, keyword: str, context_size: int) -> str:
        """提取关键词上下文"""
        if not isinstance(text, str) or keyword not in text:
            return ""

        # 找到关键词位置
        pos = text.find(keyword)
        start = max(0, pos - context_size)
        end = min(len(text), pos + len(keyword) + context_size)

        return text[start:end]


# 使用示例
if __name__ == "__main__":
    # 初始化转换器
    converter = PolicyDataConverter("merged_deduplicated.xlsx")

    # 转换所有格式
    results = converter.convert_all_formats("政策数据_三种格式")

    print("转换完成!")