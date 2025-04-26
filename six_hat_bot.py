# -*- coding: utf-8 -*-
"""
基于六顶思考帽的多Agent分析系统

该系统利用六顶思考帽方法论从多角度分析系统需求，
使用多个Agent协同工作，提供全面且深入的分析结果。

作者：AI助手
日期：2023-04-25
"""

import os
import sys
from tabnanny import verbose
import time
import json
import re
import requests
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
from abc import ABC, abstractmethod
import threading
import logging
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# 导入dotenv处理环境变量
from dotenv import load_dotenv
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("SixHatsSystem")

# 尝试导入搜索库
try:
    from googlesearch import search as google_search
except ImportError:
    logger.warning("googlesearch-python 库未安装，将使用备用搜索方法")
    def google_search(query, num_results=10, lang='en'):
        """备用的搜索函数"""
        logger.warning(f"正在使用备用搜索方法搜索: {query}")
        return []

try:
    import duckduckgo_search as ddg
except ImportError:
    logger.warning("duckduckgo_search 库未安装，将使用备用搜索方法")
    ddg = None

# 工具函数
def google_search_tool(query: str, num_results: int = 5) -> List[str]:
    """
    使用Google搜索工具
    
    参数:
        query: 搜索查询
        num_results: 结果数量
        
    返回:
        搜索结果URL列表
    """
    try:
        results = list(google_search(query, num_results=num_results))
        return results
    except Exception as e:
        logger.error(f"Google搜索失败(line {e.__traceback__.tb_lineno}): {str(e)}")
        return []

def search_duckduckgo_tool(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    使用DuckDuckGo搜索工具
    
    参数:
        query: 搜索查询
        num_results: 结果数量
        
    返回:
        搜索结果列表
    """
    try:
        if ddg is None:
            logger.warning("DuckDuckGo搜索不可用，请安装duckduckgo_search库")
            return []
            
        results = ddg.DDGS().text(query, max_results=num_results)
        return list(results)
    except Exception as e:
        logger.error(f"DuckDuckGo搜索失败(line {e.__traceback__.tb_lineno}): {str(e)}")
        return []

def fetch_webpage_tool(url: str, max_length: int = 5000) -> str:
    """
    获取网页内容工具
    
    参数:
        url: 网页URL
        max_length: 最大内容长度
        
    返回:
        提取的网页文本内容
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # 使用BeautifulSoup解析内容
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 提取正文内容
        for script in soup(["script", "style", "meta", "noscript"]):
            script.extract()
            
        text = soup.get_text(separator='\n', strip=True)
        
        # 删除多余空行并限制长度
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)[:max_length]
        
        if len(text) >= max_length:
            text += "\n[内容已被截断，仅显示前部分]\n"
            
        return text
    except Exception as e:
        logger.error(f"获取网页内容失败(line {e.__traceback__.tb_lineno}): {str(e)}")
        return f"无法获取网页内容: {str(e)}"

# 异常类
class ModelAPIError(Exception):
    """模型API错误"""
    pass

class AgentError(Exception):
    """Agent错误"""
    pass

class ToolError(Exception):
    """工具错误"""
    pass

# 模型API接口
class ModelAPI(ABC):
    """
    模型API抽象基类，定义与大语言模型交互的接口。
    可以是Azure OpenAI API或OpenRouter API的具体实现。
    """
    
    @abstractmethod
    def generate_response(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """
        生成模型响应
        
        参数:
            messages: 消息历史列表，包含角色和内容
            **kwargs: 额外的模型参数
            
        返回:
            模型生成的响应文本
        """
        pass
    
    @abstractmethod
    def generate_stream(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """
        流式生成模型响应
        
        参数:
            messages: 消息历史列表，包含角色和内容
            **kwargs: 额外的模型参数
            
        返回:
            模型生成的响应文本
        """
        pass

# Azure OpenAI API实现
class AzureOpenAIAPI(ModelAPI):
    """Azure OpenAI API的具体实现"""
    
    def __init__(self, api_key: str, endpoint: str, deployment_name: str):
        """
        初始化Azure OpenAI API客户端
        
        参数:
            api_key: Azure OpenAI API密钥
            endpoint: Azure OpenAI API端点URL
            deployment_name: 部署名称
        """
        self.api_key = api_key
        self.endpoint = endpoint 
        self.deployment_name = deployment_name
        self.api_version = "2023-05-15"  # 可以根据需要更新
        logger.info(f"初始化AzureOpenAIAPI: endpoint={endpoint}, deployment={deployment_name}")
        
        try:
            import openai
            self.client = openai.AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.endpoint
            )
        except ImportError:
            logger.error("openai 库未安装，请使用pip install openai进行安装")
            raise
    
    def generate_response(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """
        生成模型响应
        
        参数:
            messages: 消息历史列表
            **kwargs: 额外的模型参数
            
        返回:
            模型生成的响应文本
        """
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Azure OpenAI API调用失败(line {e.__traceback__.tb_lineno}): {str(e)}")
            raise ModelAPIError(f"Azure OpenAI API调用失败: {str(e)}")
    
    def generate_stream(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """
        流式生成模型响应
        
        参数:
            messages: 消息历史列表
            **kwargs: 额外的模型参数
            
        返回:
            模型生成的响应文本
        """
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                stream=True,
                **kwargs
            )
            
            full_content = ""
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_content += content
                    # 可以在这里实现实时输出
                    # print(content, end="", flush=True)
                    
            return full_content
        except Exception as e:
            logger.error(f"Azure OpenAI API流式调用失败(line {e.__traceback__.tb_lineno}): {str(e)}")
            raise ModelAPIError(f"Azure OpenAI API流式调用失败: {str(e)}")

# OpenRouter API实现
class OpenRouterAPI(ModelAPI):
    """OpenRouter API的具体实现"""
    
    def __init__(self, api_key: str, model: str = "anthropic/claude-3-opus:beta"):
        """
        初始化OpenRouter API客户端
        
        参数:
            api_key: OpenRouter API密钥
            model: 要使用的模型名称
        """
        self.api_key = api_key
        self.model = model
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        logger.info(f"初始化OpenRouterAPI: model={model}")
        
        try:
            import openai
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1"
            )
        except ImportError:
            logger.error("openai 库未安装，请使用pip install openai进行安装")
            raise
    
    def generate_response(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """
        生成模型响应
        
        参数:
            messages: 消息历史列表
            **kwargs: 额外的模型参数
            
        返回:
            模型生成的响应文本
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenRouter API调用失败(line {e.__traceback__.tb_lineno}): {str(e)}")
            raise ModelAPIError(f"OpenRouter API调用失败: {str(e)}")
    
    def generate_stream(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """
        流式生成模型响应
        
        参数:
            messages: 消息历史列表
            **kwargs: 额外的模型参数
            
        返回:
            模型生成的响应文本
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                **kwargs
            )
            
            full_content = ""
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_content += content
                    # 可以在这里实现实时输出
                    # print(content, end="", flush=True)
                    
            return full_content
        except Exception as e:
            logger.error(f"OpenRouter API流式调用失败(line {e.__traceback__.tb_lineno}): {str(e)}")
            raise ModelAPIError(f"OpenRouter API流式调用失败: {str(e)}")

# 工具管理器
class ToolManager:
    """工具管理器，用于管理和调用各种工具"""
    
    def __init__(self):
        """初始化工具管理器"""
        self.tools: Dict[str, Callable] = {}
        logger.info("初始化工具管理器")
    
    def register_tool(self, name: str, tool_func: Callable):
        """
        注册工具
        
        参数:
            name: 工具名称
            tool_func: 工具函数
        """
        self.tools[name] = tool_func
        logger.info(f"注册工具: {name}")
    
    def get_tool(self, name: str) -> Optional[Callable]:
        """
        获取工具
        
        参数:
            name: 工具名称
            
        返回:
            工具函数，如果不存在则返回None
        """
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """
        列出所有可用工具
        
        返回:
            工具名称列表
        """
        return list(self.tools.keys())
    
    def call_tool(self, name: str, *args, **kwargs) -> Any:
        """
        调用工具
        
        参数:
            name: 工具名称
            *args: 位置参数
            **kwargs: 关键字参数
            
        返回:
            工具执行结果
        """
        tool = self.get_tool(name)
        if not tool:
            raise ToolError(f"工具不存在: {name}")
        
        try:
            return tool(*args, **kwargs)
        except Exception as e:
            error_msg = f"工具执行失败: {str(e)}"
            logger.error(error_msg)
            raise ToolError(error_msg)

# 共享内存组件
class SharedMemory:
    """共享内存，用于Agent之间共享信息"""
    
    def __init__(self):
        """初始化共享内存"""
        self.memory: Dict[str, Any] = {}
        self.lock = threading.Lock()
        logger.info("初始化共享内存")
    
    def set(self, key: str, value: Any):
        """
        设置内存值
        
        参数:
            key: 键名
            value: 值
        """
        with self.lock:
            self.memory[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取内存值
        
        参数:
            key: 键名
            default: 默认值，当键不存在时返回
            
        返回:
            存储的值或默认值
        """
        with self.lock:
            return self.memory.get(key, default)
    
    def delete(self, key: str):
        """
        删除内存值
        
        参数:
            key: 键名
        """
        with self.lock:
            if key in self.memory:
                del self.memory[key]
    
    def list_keys(self) -> List[str]:
        """
        列出所有键
        
        返回:
            键名列表
        """
        with self.lock:
            return list(self.memory.keys())
    
    def clear(self):
        """清空内存"""
        with self.lock:
            self.memory.clear()

# Agent基类
class Agent(ABC):
    """
    Agent抽象基类，定义了Agent的基本行为和属性。
    """
    
    def __init__(self, name: str, role: str, model_api: ModelAPI, shared_memory: SharedMemory, tool_manager: ToolManager, verbose: bool = False):
        """
        初始化Agent
        
        参数:
            name: Agent名称
            role: Agent角色
            model_api: 模型API实例
            shared_memory: 共享内存实例
            tool_manager: 工具管理器实例
            verbose: 是否启用详细日志模式
        """
        self.name = name
        self.role = role
        self.model_api = model_api
        self.shared_memory = shared_memory
        self.tool_manager = tool_manager
        self.verbose = verbose
        self.message_history: List[Dict[str, Any]] = []
        self.verbose = verbose
        if self.verbose:
            logger.info(f"Agent {self.name} ({self.role}) initialized with verbose mode.")

    def add_message(self, role: str, content: Any):
        """
        添加消息到历史记录
        
        参数:
            role: 消息角色 (system, user, assistant)
            content: 消息内容
        """
        message = {"role": role, "content": content}
        self.message_history.append(message)
        if self.verbose:
            logger.info(f"[{self.name}] Added message: Role={role}, Content Snippet='{str(content)[:100]}...'" if len(str(content)) > 100 else f"[{self.name}] Added message: Role={role}, Content='{content}'")

    
    def get_messages(self) -> List[Dict[str, str]]:
        """
        获取完整消息历史
        
        返回:
            消息历史列表
        """
        return self.message_history
    
    def clear_messages(self):
        """清空消息历史"""
        self.message_history = []
    
    def share_info(self, key: str, value: Any):
        """
        分享信息到共享内存
        
        参数:
            key: 信息键名
            value: 信息内容
        """
        self.shared_memory.set(f"{self.name}_{key}", value)
    
    def get_shared_info(self, agent_name: str, key: str, default: Any = None) -> Any:
        """
        从共享内存获取指定Agent分享的信息
        
        参数:
            agent_name: Agent名称
            key: 信息键名
            default: 默认值
            
        返回:
            共享信息或默认值
        """
        return self.shared_memory.get(f"{agent_name}_{key}", default)
    
    @abstractmethod
    async def process(self, message: str) -> str:
        """
        处理消息并生成响应
        
        参数:
            message: 输入消息
            
        返回:
            响应消息
        """
        pass

# 思考帽Agent类
class HatAgent(Agent):
    """
    思考帽Agent，继承自Agent基类，实现六顶思考帽的特定思考方式
    """
    
    def __init__(self, name: str, role: str, hat_color: str, model_api: ModelAPI, shared_memory: SharedMemory, tool_manager: ToolManager, verbose: bool = False):
        """
        初始化思考帽Agent
        
        参数:
            name: Agent名称
            role: Agent角色
            hat_color: 思考帽颜色
            model_api: 模型API实例
            shared_memory: 共享内存实例
            tool_manager: 工具管理器实例
            verbose: 是否启用详细日志模式
        """
        logger.info(f"初始化思考帽Agent: {name}, 颜色: {hat_color}")
        super().__init__(name, role, model_api, shared_memory, tool_manager, verbose)
        self.hat_color = hat_color
        self.system_prompt = self._get_hat_prompt(hat_color)
        # 设置系统提示
        self.add_message("system", self.system_prompt)
        
    
    def _get_hat_prompt(self, hat_color: str) -> str:
        """
        根据思考帽颜色获取相应的系统提示
        
        参数:
            hat_color: 思考帽颜色
            
        返回:
            系统提示文本
        """
        prompts = {
            "blue": """你是蓝帽思考者，负责定义问题、组织思考流程、监督讨论进度和总结结论。
蓝帽代表元认知，关注思考的过程而非内容。

你的职责包括:
1. 在分析开始时，明确定义问题和目标
2. 设计思考流程和各思考帽的发言顺序
3. 确保讨论不偏离主题
4. 定期总结已达成的共识
5. 在讨论结束时做出最终总结

在发言时，请明确使用"蓝帽思考："开头，并专注于过程控制而非具体内容分析。
典型的蓝帽问题包括："我们的目标是什么？"、"接下来我们应该关注什么？"、"我们已经取得了哪些进展？""",
            
            "white": """你是白帽思考者，负责收集、整理和分析与需求相关的客观事实和数据。
白帽代表中立和客观，只关注数据和事实，不做价值判断。

你的职责包括:
1. 提供与需求相关的客观事实和数据
2. 指出当前信息中的缺失和不确定性
3. 区分已知事实和假设
4. 在需要时，提出获取更多信息的建议
5. 使用统计数据和参考资料支持论点

在发言时，请明确使用"白帽思考："开头，并严格基于事实进行分析，避免主观判断。
典型的白帽问题包括："我们已知哪些信息？"、"还需要哪些数据？"、"这些数字表明了什么？""",
            
            "red": """你是红帽思考者，负责表达对需求的感受、直觉和情感反应。
红帽代表情感和直觉，允许表达主观感受，不需要理性解释。

你的职责包括:
1. 表达对需求的直接感受和情感反应
2. 分享直觉判断，而无需逻辑证明
3. 预测用户和利益相关者的情感反应
4. 指出需求中可能引起共鸣或抵触的方面
5. 评估需求的情感吸引力和用户体验

在发言时，请明确使用"红帽思考："开头，并放心表达主观感受，不需要逻辑证明。
典型的红帽表达包括："我对这个需求的感觉是..."、"直觉告诉我..."、"用户可能会感到...""",
            
            "yellow": """你是黄帽思考者，负责寻找需求的价值、优势和可行性。
黄帽代表乐观和建设性思考，专注于机会和好处。

你的职责包括:
1. 识别需求的价值和优势
2. 分析实现需求的可行性
3. 寻找克服障碍的方法
4. 强调成功的可能性
5. 提出使方案更可行的改进建议

在发言时，请明确使用"黄帽思考："开头，并保持建设性和乐观，寻找积极的可能性。
典型的黄帽问题包括："实现这个需求有什么好处？"、"如何使这个想法可行？"、"这个方案的价值在哪里？""",
            
            "black": """你是黑帽思考者，负责识别需求中的风险、缺陷和潜在问题。
黑帽代表谨慎和逻辑判断，关注风险和可能的失败点。

你的职责包括:
1. 识别需求中的逻辑缺陷和矛盾
2. 评估潜在风险和挑战
3. 指出方案中的弱点和不足
4. 预测可能的失败场景
5. 分析资源限制和成本问题

在发言时，请明确使用"黑帽思考："开头，并保持谨慎和批判性，但避免过度悲观。
典型的黑帽问题包括："可能出现什么问题？"、"有什么风险需要考虑？"、"我们忽略了什么？""",
            
            "green": """你是绿帽思考者，负责提出创新的解决方案和替代方案。
绿帽代表创造性思维，寻求突破性想法和新视角。

你的职责包括:
1. 提出创新的解决方案和方法
2. 挑战传统思维和假设
3. 寻找替代视角和方法
4. 进行头脑风暴，产生多种可能性
5. 整合不同想法，创造新的组合

在发言时，请明确使用"绿帽思考："开头，并保持创造性和开放性，不受限于传统思维。
典型的绿帽问题包括："还有其他解决方案吗？"、"如何突破当前限制？"、"有没有完全不同的方法？"""
        }
        
        return prompts.get(hat_color.lower(), "你是一个思考帽Agent，请根据你的角色特性进行思考。")
    
    async def process(self, message: str) -> str:
        """
        处理消息并根据思考帽特性生成响应
        
        参数:
            message: 输入消息
            
        返回:
            响应消息
        """
        # 准备提示
        self.add_message("user", message)
        
        # 获取完整消息历史
        messages = self.get_messages()
        
        # 调用模型API生成响应
        try:
            response = self.model_api.generate_response(messages, temperature=0.7, max_tokens=1000)
            
            # 将响应添加到历史
            self.add_message("assistant", response)
            
            # 共享思考结果
            self.share_info(f"思考结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}", response)
            
            return response
        except Exception as e:
            error_msg = f"思考过程出错: {str(e)}"
            logger.error(error_msg)
            return error_msg

# 特定思考帽Agent实现
class BlueHatAgent(HatAgent):
    """蓝帽思考者，负责过程控制和总结"""
    
    def __init__(self, name: str, role: str, model_api: ModelAPI, shared_memory: SharedMemory, tool_manager: ToolManager, verbose: bool = False):
        logger.info(f"初始化蓝帽思考者: {name}")
        super().__init__(name, role, "blue", model_api, shared_memory, tool_manager, verbose)


class WhiteHatAgent(HatAgent):
    """白帽思考者，负责客观事实和数据"""
    
    def __init__(self, name: str, role: str, model_api: ModelAPI, shared_memory: SharedMemory, tool_manager: ToolManager, verbose: bool = False):
        logger.info(f"初始化白帽思考者: {name}")
        super().__init__(name, role, "white", model_api, shared_memory, tool_manager, verbose)


class RedHatAgent(HatAgent):
    """红帽思考者，负责情感和直觉"""
    def __init__(self, name: str, role: str, model_api: ModelAPI, shared_memory: SharedMemory, tool_manager: ToolManager, verbose: bool = False):
        logger.info(f"初始化红帽思考者: {name}")
        super().__init__(name, role, "red", model_api, shared_memory, tool_manager, verbose)


class YellowHatAgent(HatAgent):
    """黄帽思考者，负责积极和可行性"""
    def __init__(self, name: str, role: str, model_api: ModelAPI, shared_memory: SharedMemory, tool_manager: ToolManager, verbose: bool = False):
        logger.info(f"初始化黄帽思考者: {name}")
        super().__init__(name, role, "yellow", model_api, shared_memory, tool_manager, verbose)


class BlackHatAgent(HatAgent):
    """黑帽思考者，负责风险评估和批判"""
    
    def __init__(self, name: str, role: str, model_api: ModelAPI, shared_memory: SharedMemory, tool_manager: ToolManager, verbose: bool = False):
        logger.info(f"初始化黑帽思考者: {name}")
        super().__init__(name, role, "black", model_api, shared_memory, tool_manager, verbose)


class GreenHatAgent(HatAgent):
    """绿帽思考者，负责创新思维和替代方案"""
    
    def __init__(self, name: str, role: str, model_api: ModelAPI, shared_memory: SharedMemory, tool_manager: ToolManager, verbose: bool = False):
        logger.info(f"初始化绿帽思考者: {name}")
        super().__init__(name, role, "green", model_api, shared_memory, tool_manager, verbose)

    
    async def generate_ideas(self, topic: str, idea_count: int = 3) -> str:
        """
        针对特定主题生成创新想法
        
        参数:
            topic: 主题
            idea_count: 想法数量
            
        返回:
            创新想法列表
        """
        prompt = f"请对'{topic}'提出{idea_count}个创新性的解决方案或替代方案，突破常规思维。"
        return await self.process(prompt)
    
    async def lateral_thinking(self, problem: str) -> str:
        """
        应用横向思维解决问题
        
        参数:
            problem: 问题描述
            
        返回:
            横向思维分析结果
        """
        prompt = f"请使用横向思维方法，从完全不同的角度分析这个问题：{problem}"
        return await self.process(prompt)

# 信息搜集Agent
# 信息搜集Agent
class InfoAgent(Agent):
    """信息搜集Agent，负责搜索和获取外部信息，并能提取网页内容。"""
    
    def __init__(self, name: str, role: str, model_api: ModelAPI, shared_memory: SharedMemory, tool_manager: ToolManager, verbose: bool = False):
        """
        初始化信息搜集Agent
        
        参数:
            name: Agent名称
            role: Agent角色
            model_api: 模型API实例
            shared_memory: 共享内存实例
            tool_manager: 工具管理器实例
            verbose: 是否启用详细日志模式
        """
        super().__init__(name, role, model_api, shared_memory, tool_manager, verbose)
        # 设置系统提示
        system_prompt = """你是信息收集者，负责搜索和收集与需求相关的外部信息。
你能够使用搜索工具获取相关资料，并能提取网页核心内容，并以Markdown格式返回。

你的职责包括:
1. 根据讨论需要，搜索相关信息
2. 整理搜索结果，提取有价值的内容
3. 获取网页内容后，提取核心信息并以Markdown格式总结
4. 确保信息来源可靠，并提供引用
5. 不提供个人观点，只负责信息的收集和整理

请在需要时主动使用搜索和网页内容获取工具，并明确使用「信息收集：」开头汇报结果。
"""
        self.add_message("system", system_prompt)
        logger.info(f"初始化信息搜集Agent: {name}")
    
    async def search_info(self, query: str, num_results: int = 10, fetch_content: bool = True) -> str:
        """
        搜索信息，并可选择性地获取网页内容
        
        参数:
            query: 搜索查询
            num_results: 结果数量
            fetch_content: 是否获取搜索结果的网页内容
            
        返回:
            搜索结果和（可选的）网页内容摘要
        """
        if self.verbose:
            logger.info(f"[{self.name}] Starting search for query: '{query}' with num_results={num_results}, fetch_content={fetch_content}")
            
        search_results_urls = []
        search_summary = ""
        
        try:
            # 尝试使用DuckDuckGo搜索
            ddg_tool = self.tool_manager.get_tool("duckduckgo_search")
            if ddg_tool:
                duckduckgo_results = self.tool_manager.call_tool("duckduckgo_search", query=query, num_results=num_results)
                if duckduckgo_results:
                    search_summary += f"DuckDuckGo搜索结果：\n"
                    for i, result in enumerate(duckduckgo_results):
                        search_summary += f"{i+1}. {result.get('title', 'N/A')} - {result.get('href', 'N/A')}\n   {result.get('body', '')[:150]}...\n"
                        if result.get('href'):
                            search_results_urls.append(result['href'])
                    search_summary += "\n"
                    if self.verbose:
                        logger.info(f"[{self.name}] DuckDuckGo search successful, found {len(duckduckgo_results)} results.")
                else:
                    if self.verbose:
                        logger.info(f"[{self.name}] DuckDuckGo search returned no results.")
            else:
                 if self.verbose:
                    logger.warning(f"[{self.name}] DuckDuckGo search tool not available.")

            # 备选：尝试使用Google搜索 (如果DDG失败或未找到结果)
            if not search_results_urls:
                google_tool = self.tool_manager.get_tool("google_search")
                if google_tool:
                    google_results = self.tool_manager.call_tool("google_search", query=query, num_results=num_results)
                    if google_results:
                        search_summary += f"Google搜索结果：\n"
                        for i, url in enumerate(google_results):
                            search_summary += f"{i+1}. {url}\n"
                            search_results_urls.append(url)
                        search_summary += "\n"
                        if self.verbose:
                            logger.info(f"[{self.name}] Google search successful, found {len(google_results)} results.")
                    else:
                        if self.verbose:
                            logger.info(f"[{self.name}] Google search returned no results.")
                else:
                    if self.verbose:
                        logger.warning(f"[{self.name}] Google search tool not available.")

            if not search_results_urls:
                return "所有搜索工具均不可用或未找到结果，无法获取信息"

            # 获取网页内容
            fetched_content_summary = ""
            if fetch_content and search_results_urls:
                if self.verbose:
                    logger.info(f"[{self.name}] Fetching content for {len(search_results_urls)} URLs.")
                tasks = [self.fetch_webpage(url) for url in search_results_urls]
                import asyncio
                fetched_contents = await asyncio.gather(*tasks)
                
                # 过滤掉获取失败的内容
                successful_fetches = [content for content in fetched_contents if not content.startswith("无法获取网页内容")]
                
                if successful_fetches:
                    fetched_content_summary = "\n\n网页内容摘要 (Markdown格式):\n" + "\n\n---\n\n".join(successful_fetches)
                    if self.verbose:
                        logger.info(f"[{self.name}] Successfully fetched content for {len(successful_fetches)} URLs.")
                else:
                     if self.verbose:
                        logger.info(f"[{self.name}] Failed to fetch content for all URLs.")
            
            # 组合最终结果
            final_result = f"信息收集：关于「{query}」的搜索结果\n{search_summary}{fetched_content_summary}"
            
            # 将结果存入共享内存
            search_key = f"搜索结果_{query}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.share_info(search_key, final_result)
            
            return final_result
        except Exception as e:
            error_msg = f"搜索信息出错: {str(e)}"
            logger.error(f"[{self.name}] Error during search_info: {error_msg}")
            return error_msg
    
    async def fetch_webpage(self, url: str, max_length: int = 3000) -> str:
        """
        获取网页内容并尝试提取核心信息，以Markdown格式返回
        
        参数:
            url: 网页URL
            max_length: 提取内容的最大长度
            
        返回:
            提取的网页核心内容 (Markdown格式) 或错误信息
        """
        if self.verbose:
            logger.info(f"[{self.name}] Attempting to fetch content from URL: {url}")
            
        try:
            fetch_tool = self.tool_manager.get_tool("fetch_webpage")
            if not fetch_tool:
                logger.warning(f"[{self.name}] Fetch webpage tool not available.")
                return "网页获取工具不可用"
                
            # 调用工具获取原始文本
            raw_content = self.tool_manager.call_tool("fetch_webpage", url=url, max_length=max_length * 2) # 获取稍多内容供模型处理
            
            if raw_content.startswith("无法获取网页内容"):
                if self.verbose:
                    logger.warning(f"[{self.name}] Failed to fetch raw content from {url}: {raw_content}")
                return raw_content # 直接返回工具的错误信息

            if self.verbose:
                logger.info(f"[{self.name}] Successfully fetched raw content from {url}. Length: {len(raw_content)}")

            # 使用LLM提取核心内容并格式化为Markdown
            prompt = f"请从以下网页文本内容中提取核心信息，并以简洁的Markdown格式进行总结。请专注于主要观点和关键信息，忽略导航、广告和页脚等无关内容。内容来源URL: {url}\n\n原始文本内容：\n{raw_content[:max_length]}"
            
            # 使用当前Agent的模型进行处理
            messages = [
                {"role": "system", "content": "你是一个网页内容总结助手，请将提供的文本内容提取核心信息并以Markdown格式输出。"},
                {"role": "user", "content": prompt}
            ]
            
            summary = self.model_api.generate_response(messages, temperature=0.3, max_tokens=1000)
            
            if self.verbose:
                logger.info(f"[{self.name}] Generated Markdown summary for {url}. Length: {len(summary)}")
                
            # 返回Markdown格式的总结
            return f"**来源: {url}**\n\n{summary}"
            
        except Exception as e:
            error_msg = f"获取并处理网页内容出错 ({url}): {str(e)}"
            logger.error(f"[{self.name}] {error_msg}")
            return f"无法获取网页内容: {str(e)}"
    
    async def process(self, message: str) -> str:
        """
        处理消息，根据内容决定是搜索信息还是直接回答
        
        参数:
            message: 输入消息
            
        返回:
            响应消息
        """
        if self.verbose:
             logger.info(f"[{self.name}] Processing message: '{message[:100]}...'" if len(message) > 100 else f"[{self.name}] Processing message: '{message}'")
             
        # 添加用户消息
        self.add_message("user", message)
        
        # 判断是否需要搜索信息 (更智能的判断)
        keywords = ["搜索", "查找", "信息", "资料", "查询", "了解", "告诉我关于"]
        if any(keyword in message for keyword in keywords):
            # 提取搜索关键词 (尝试去除指令词)
            search_query = message
            for keyword in keywords:
                search_query = search_query.replace(keyword, "")
            search_query = search_query.strip()
            
            if not search_query:
                # 如果去除关键词后为空，可能需要模型判断意图
                # 暂时使用原始消息作为查询
                search_query = message 
                if self.verbose:
                    logger.info(f"[{self.name}] No specific query extracted, using full message for search: '{search_query}'")
            else:
                if self.verbose:
                    logger.info(f"[{self.name}] Extracted search query: '{search_query}'")
                
            # 执行搜索并获取内容
            search_result = await self.search_info(search_query, fetch_content=True)
            
            # 添加结果到历史并返回
            self.add_message("assistant", search_result)
            return search_result
        else:
            # 普通消息处理 (可能需要调用LLM)
            if self.verbose:
                logger.info(f"[{self.name}] Message does not seem like a search request. Processing with LLM.")
            messages = self.get_messages()
            try:
                response = self.model_api.generate_response(messages, temperature=0.7, max_tokens=1000)
                self.add_message("assistant", response)
                return response
            except Exception as e:
                error_msg = f"处理消息出错: {str(e)}"
                logger.error(f"[{self.name}] Error processing message with LLM: {error_msg}")
    """信息搜集Agent，负责搜索和获取外部信息"""
    


# 报告生成Agent
class ReportAgent(Agent):
    """报告生成Agent，负责总结思考帽讨论结果并生成分析报告"""
    
    def __init__(self, name: str, role: str, model_api: ModelAPI, shared_memory: SharedMemory, tool_manager: ToolManager, verbose: bool = False):
        """
        初始化报告生成Agent
        
        参数:
            name: Agent名称
            role: Agent角色
            model_api: 模型API实例
            shared_memory: 共享内存实例
            tool_manager: 工具管理器实例
            verbose: 是否启用详细日志模式
        """
        super().__init__(name, role, model_api, shared_memory, tool_manager)
        # 设置系统提示
        system_prompt = """你是报告生成者，负责整理和总结六顶思考帽的分析结果，形成最终分析报告。

你的职责包括:
1. 收集所有思考帽的讨论内容
2. 整理各个思考角度的关键点
3. 按照客观性原则组织内容
4. 确保报告结构清晰，内容全面
5. 确保报告客观反映所有思考帽的视角

报告应包括以下部分:
1. 需求描述: 简要描述原始需求
2. 事实基础(白帽): 客观数据和事实分析
3. 情感反应(红帽): 直觉和情感评估
4. 价值与优势(黄帽): 积极方面和可行性
5. 风险与挑战(黑帽): 潜在问题和限制
6. 创新可能(绿帽): 创新思路和替代方案
7. 总结与建议(蓝帽): 过程总结和后续建议

在生成报告时，请以「六顶思考帽分析报告」作为标题，并明确每个部分的边界。
"""
        self.add_message("system", system_prompt)
        logger.info(f"初始化报告生成Agent: {name}")
    
    def collect_hat_thoughts(self) -> Dict[str, str]:
        """
        收集所有思考帽的思考结果
        
        返回:
            思考结果字典
        """
        results = {}
        hat_colors = ["blue", "white", "red", "yellow", "black", "green"]
        hat_names = {
            "blue": "蓝帽", 
            "white": "白帽", 
            "red": "红帽", 
            "yellow": "黄帽", 
            "black": "黑帽", 
            "green": "绿帽"
        }
        
        # 从共享内存中获取各思考帽的最新结果
        for color in hat_colors:
            # 获取该颜色帽子所有内容
            keys = self.shared_memory.list_keys()
            hat_keys = [k for k in keys if k.startswith(f"{hat_names[color]}思考者_思考结果_")]
            
            if hat_keys:
                # 排序获取最新结果
                sorted_keys = sorted(hat_keys, reverse=True)
                latest_key = sorted_keys[0] if sorted_keys else None
                if latest_key:
                    results[color] = self.shared_memory.get(latest_key, "未提供分析内容")
            else:
                results[color] = "未提供分析内容"
        
        # 获取搜索结果信息
        search_keys = [k for k in self.shared_memory.list_keys() if k.startswith("搜索结果_")]
        if search_keys:
            # 按时间排序，获取最新的几条搜索结果
            sorted_search_keys = sorted(search_keys, reverse=True)[:3]  # 取最新的3条
            search_results = []
            for key in sorted_search_keys:
                search_results.append(self.shared_memory.get(key, ""))
            
            # 将搜索结果添加到收集的内容中
            results["search"] = "\n\n".join(filter(None, search_results))
        else:
            results["search"] = ""
        
        return results
    
    async def process(self, message: str) -> str:
        """
        处理消息，生成分析报告
        
        参数:
            message: 输入消息
            
        返回:
            生成的报告
        """
        # 添加用户消息
        self.add_message("user", message)
        
        # 如果是请求生成报告
        if "生成报告" in message or "总结" in message or "报告" in message:
            # 收集思考帽的思考结果
            thoughts = self.collect_hat_thoughts()
            
            # 准备提供给模型的提示
            report_prompt = f"请根据以下六顶思考帽的分析结果，生成一份完整的分析报告：\n\n"
            
            # 添加原始需求信息
            original_requirement = self.shared_memory.get("原始需求", "未提供原始需求")
            report_prompt += f"原始需求：\n{original_requirement}\n\n"
            
            # 添加各思考帽的分析结果
            hat_colors = ["blue", "white", "red", "yellow", "black", "green"]
            hat_display_names = {
                "blue": "蓝帽(过程控制)", 
                "white": "白帽(事实数据)", 
                "red": "红帽(情感直觉)", 
                "yellow": "黄帽(价值优势)", 
                "black": "黑帽(风险评估)", 
                "green": "绿帽(创新思维)",
                "search": "搜集的外部信息"
            }
            
            # 首先添加蓝帽的分析结果（过程控制）
            if "blue" in thoughts and thoughts["blue"]:
                report_prompt += f"{hat_display_names['blue']}分析结果：\n{thoughts['blue']}\n\n"
            
            # 添加白帽的分析结果（客观事实）和搜索信息
            if "white" in thoughts and thoughts["white"]:
                report_prompt += f"{hat_display_names['white']}分析结果：\n{thoughts['white']}\n\n"
            
            # 添加搜索结果（如果有）
            if "search" in thoughts and thoughts["search"]:
                report_prompt += f"{hat_display_names['search']}：\n{thoughts['search']}\n\n"
            
            # 添加其他思考帽的分析结果
            for color in ["red", "yellow", "black", "green"]:
                if color in thoughts and thoughts[color]:
                    report_prompt += f"{hat_display_names[color]}分析结果：\n{thoughts[color]}\n\n"
            
            # 添加报告结构指导
            report_prompt += "请根据以上信息生成一份结构完整、内容全面的分析报告，报告应包括以下部分：\n"
            report_prompt += "1. 需求概述：简要说明原始需求\n"
            report_prompt += "2. 客观事实分析：基于白帽思考和收集的信息\n"
            report_prompt += "3. 情感与直觉反应：基于红帽思考\n"
            report_prompt += "4. 价值与优势分析：基于黄帽思考\n"
            report_prompt += "5. 风险与挑战分析：基于黑帽思考\n"
            report_prompt += "6. 创新方案建议：基于绿帽思考\n"
            report_prompt += "7. 结论与建议：综合所有分析的最终建议\n\n"
            report_prompt += "请使用markdown格式，确保报告结构清晰，内容全面且有深度。"
            
            # 调用模型生成报告
            self.add_message("user", report_prompt)
            messages = self.get_messages()
            
            try:
                report = self.model_api.generate_response(messages, temperature=0.5, max_tokens=3000)
                # 添加到历史
                self.add_message("assistant", report)
                # 保存报告到共享内存
                self.share_info(f"分析报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}", report)
                return report
            except Exception as e:
                error_msg = f"生成报告出错: {str(e)}"
                logger.error(error_msg)
                return error_msg
        else:
            # 普通消息处理
            messages = self.get_messages()
            try:
                response = self.model_api.generate_response(messages, temperature=0.7, max_tokens=1000)
                self.add_message("assistant", response)
                return response
            except Exception as e:
                error_msg = f"处理消息出错: {str(e)}"
                logger.error(error_msg)
                return error_msg


# Agent工厂类
class AgentFactory:
    """Agent工厂类，负责创建不同类型的Agent"""
    
    def __init__(self, model_api: ModelAPI, shared_memory: SharedMemory, tool_manager: ToolManager, verbose: bool = False) -> None:
        """
        初始化Agent工厂
        
        参数:
            model_api: 模型API实例
            shared_memory: 共享内存实例
            tool_manager: 工具管理器实例
            verbose: 是否启用详细日志模式
        """
        self.model_api = model_api
        self.shared_memory = shared_memory
        self.tool_manager = tool_manager
        self.verbose = verbose
        logger.info(f"AgentFactory initialized with verbose mode: {self.verbose}")

    def create_hat_agent(self, color: str) -> HatAgent:
        """
        创建思考帽Agent
        
        参数:
            color: 思考帽颜色
            
        返回:
            HatAgent实例
        """
        hat_classes = {
            "blue": BlueHatAgent,
            "white": WhiteHatAgent,
            "red": RedHatAgent,
            "yellow": YellowHatAgent,
            "black": BlackHatAgent,
            "green": GreenHatAgent
        }
        hat_names = {
            "blue": "蓝帽思考者",
            "white": "白帽思考者",
            "red": "红帽思考者",
            "yellow": "黄帽思考者",
            "black": "黑帽思考者",
            "green": "绿帽思考者"
        }
        hat_roles = {
            "blue": "负责思考过程的控制和协调",
            "white": "负责客观事实和数据的分析",
            "red": "负责情感、直觉和预感的表达",
            "yellow": "负责价值、优势和可行性的发掘",
            "black": "负责风险、问题和挑战的评估",
            "green": "负责创新思维和替代方案的提出"
        }
        
        agent_class = hat_classes.get(color.lower())
        if not agent_class:
            raise ValueError(f"未知的思考帽颜色: {color}")
            
        return agent_class(
            name=hat_names[color.lower()],
            role=hat_roles[color.lower()],
            model_api=self.model_api,
            shared_memory=self.shared_memory,
            tool_manager=self.tool_manager,
            verbose=self.verbose
        )

    def create_info_agent(self) -> InfoAgent:
        """
        创建信息搜集Agent
        
        返回:
            InfoAgent实例
        """
        return InfoAgent(
            name="信息搜集者",
            role="负责搜索和获取外部信息",
            model_api=self.model_api,
            shared_memory=self.shared_memory,
            tool_manager=self.tool_manager,
            verbose=self.verbose
        )

    def create_report_agent(self) -> ReportAgent:
        """
        创建报告生成Agent
        
        返回:
            ReportAgent实例
        """
        return ReportAgent(
            name="报告生成者",
            role="负责总结讨论结果并生成报告",
            model_api=self.model_api,
            shared_memory=self.shared_memory,
            tool_manager=self.tool_manager,
            verbose=self.verbose
        )



# 六顶思考帽系统控制器
class SixHatsSystem:
    """六顶思考帽系统控制器，负责协调所有Agent的工作"""
    
    def __init__(self, api_type: str = "azure", verbose: bool = False):
        """
        初始化六顶思考帽系统
        
        参数:
            api_type: 使用的模型API类型 ('azure' 或 'openrouter')
            verbose: 是否启用详细日志模式
        """
        self.verbose = verbose
        self.shared_memory = SharedMemory()
        self.tool_manager = ToolManager()
        self._register_tools()
        self.model_api = self._init_model_api(api_type)
        self.agent_factory = AgentFactory(
            model_api=self.model_api,
            shared_memory=self.shared_memory,
            tool_manager=self.tool_manager,
            verbose=self.verbose
        )
        self.agents: Dict[str, Agent] = {}
        self._init_agents()
        
        logger.info(f"初始化六顶思考帽系统，API类型: {api_type}, Verbose: {self.verbose}")
    
    def _register_tools(self):
        """
        注册工具到工具管理器
        """
        self.tool_manager.register_tool("google_search", google_search_tool)
        self.tool_manager.register_tool("duckduckgo_search", search_duckduckgo_tool)
        self.tool_manager.register_tool("fetch_webpage", fetch_webpage_tool)
        logger.info("工具注册完成")
    
    def _init_model_api(self, api_type: str) -> ModelAPI:
        """
        初始化模型API
        
        参数:
            api_type: API类型
            
        返回:
            ModelAPI实例
        """
        if api_type.lower() == "azure":
            # 从环境变量获取Azure OpenAI配置
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
            
            if not all([api_key, endpoint, deployment_name]):
                error_msg = "未设置Azure OpenAI的环境变量，请检查.env文件"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            return AzureOpenAIAPI(api_key, endpoint, deployment_name)
        elif api_type.lower() == "openrouter":
            # 从环境变量获取OpenRouter配置
            api_key = os.getenv("OPENROUTER_API_KEY")
            model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3-opus:beta")
            
            if not api_key:
                error_msg = "未设置OpenRouter的API密钥，请检查.env文件"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            return OpenRouterAPI(api_key, model)
        else:
            error_msg = f"不支持的API类型: {api_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _init_agents(self):
        """
        初始化所有Agent
        """
        # 创建六顶思考帽Agent
        hat_colors = ["blue", "white", "red", "yellow", "black", "green"]
        for color in hat_colors:
            self.agents[color] = self.agent_factory.create_hat_agent(color )
        
        # 创建信息搜集Agent
        self.agents["info"] = self.agent_factory.create_info_agent()
        
        # 创建报告生成Agent
        self.agents["report"] = self.agent_factory.create_report_agent()
        
        logger.info("初始化Agent完成")
    
    def set_requirement(self, requirement: str):
        """
        设置原始需求
        
        参数:
            requirement: 需求描述
        """
        self.shared_memory.set("原始需求", requirement)
        logger.info("设置原始需求完成")
    
    async def process_with_hat(self, hat_color: str, message: str) -> str:
        """
        使用指定思考帽处理消息
        
        参数:
            hat_color: 思考帽颜色
            message: 输入消息
            
        返回:
            处理结果
        """
        agent = self.agents.get(hat_color.lower())
        if not agent:
            error_msg = f"未找到{hat_color}帽Agent"
            logger.error(error_msg)
            return error_msg
            
        return await agent.process(message)
    
    async def search_info(self, query: str) -> str:
        """
        搜索信息
        
        参数:
            query: 搜索查询
            
        返回:
            搜索结果
        """
        info_agent = self.agents.get("info")
        if not info_agent:
            error_msg = "未找到信息搜集Agent"
            logger.error(error_msg)
            return error_msg
            
        return await info_agent.search_info(query)
    
    async def generate_report(self, message: str = "生成分析报告") -> str:
        """
        生成分析报告
        
        参数:
            message: 消息提示
            
        返回:
            生成的报告
        """
        report_agent = self.agents.get("report")
        if not report_agent:
            error_msg = "未找到报告生成Agent"
            logger.error(error_msg)
            return error_msg
            
        return await report_agent.process(message)
    
    async def analyze_requirement(self, requirement: str) -> str:
        """
        分析需求，使用所有思考帽进行分析
        
        参数:
            requirement: 需求描述
            
        返回:
            分析报告
        """
        # 设置原始需求
        self.set_requirement(requirement)
        
        # 使用蓝帽先进行过程设计
        blue_result = await self.process_with_hat("blue", f"请对以下需求进行思考流程设计：\n{requirement}")
        logger.info("蓝帽思考完成")
        
        # 收集相关信息
        info_agent = self.agents.get("info")
        if info_agent:
            # 自动提取搜索关键词并进行搜索
            search_keywords = requirement.split()[:3]  # 简单地取前三个词作为搜索关键词
            search_query = " ".join(search_keywords)
            await info_agent.search_info(search_query)
            logger.info(f"信息搜集完成: {search_query}")
        
        # 并行处理其他思考帽
        tasks = [
            self.process_with_hat("white", f"请对以下需求进行客观事实分析：\n{requirement}"),
            self.process_with_hat("red", f"请对以下需求进行情感和直觉反应：\n{requirement}"),
            self.process_with_hat("yellow", f"请对以下需求进行积极和可行性分析：\n{requirement}"),
            self.process_with_hat("black", f"请对以下需求进行风险和问题分析：\n{requirement}")
        ]
        
        # 等待白帽、红帽、黄帽和黑帽完成分析
        import asyncio
        results = await asyncio.gather(*tasks)
        logger.info("基础思考帽分析完成")
        
        # 使用绿帽进行创新思维分析，使用增强的创新方法
        green_hat = self.agents.get("green")
        if isinstance(green_hat, GreenHatAgent):
            # 使用横向思维方法
            await green_hat.lateral_thinking(f"请对这个需求使用横向思维：\n{requirement}")
            
            # 生成创新想法
            await green_hat.generate_ideas(requirement, 5)
            logger.info("绿帽创新思维分析完成")
        else:
            # 常规绿帽思考
            await self.process_with_hat("green", f"请对以下需求进行创新思维和新方案分析：\n{requirement}")
            logger.info("绿帽常规分析完成")
        
        # 生成分析报告
        report = await self.generate_report()
        logger.info("分析报告生成完成")
        
        return report


# 主程序入口
async def main(verbose_mode: bool = False):
    """
    主程序入口
    
    参数:
        verbose_mode: 是否启用详细日志模式
    """
    # 初始化六顶思考帽系统
    try:
        # 默认使用Azure OpenAI，可以通过环境变量设置API类型
        api_type = os.getenv("API_TYPE", "azure")
        system = SixHatsSystem(api_type, verbose=verbose_mode)
        
        # 获取要分析的需求
        print("===== 六顶思考帽分析系统 =====\n")
        print("请输入需要分析的系统需求描述：")
        requirement = input("> ").strip()
        
        if not requirement:
            print("需求描述不能为空！")
            return
        
        print("\n正在进行多维度分析，请稍候...")
        if verbose_mode:
            logger.info("Verbose mode enabled. Starting analysis...")
        
        # 开始分析
        import asyncio
        report = await system.analyze_requirement(requirement)
        
        print("\n===== 分析报告 =====\n")
        print(report)
        
    except Exception as e:
        logger.error(f"程序执行出错(line {e.__traceback__.tb_lineno}): {str(e)}")
        print(f"\n程序执行出错(line {e.__traceback__.tb_lineno}): {str(e)}")


# 程序入口
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="基于六顶思考帽的多Agent分析系统")
    parser.add_argument("-v", "--verbose", action="store_true", help="启用详细日志模式")
    args = parser.parse_args()
    
    import asyncio
    
    # 检查是否安装了必要的库
    missing_libs = []
    try:
        import openai
    except ImportError:
        missing_libs.append("openai")
    
    try:
        from googlesearch import search
    except ImportError:
        # 允许 googlesearch-python 可选
        pass 
    
    try:
        import duckduckgo_search
    except ImportError:
         # 允许 duckduckgo-search 可选
        pass
        
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        if "requests" not in missing_libs:
            missing_libs.append("requests")
        missing_libs.append("beautifulsoup4") # 注意库名是 beautifulsoup4
    
    try:
        from dotenv import load_dotenv
    except ImportError:
        missing_libs.append("python-dotenv")
    
    if missing_libs:
        print("错误：缺少必要的Python库。")
        print("请尝试使用以下命令安装缺失的库:")
        print(f"  pip install {' '.join(missing_libs)}")
        print("安装完成后，请重新运行程序。")
        sys.exit(1)
    
    # 运行主程序，并传递verbose标志
    asyncio.run(main(verbose_mode=args.verbose))
