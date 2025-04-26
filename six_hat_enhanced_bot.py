# -*- coding: utf-8 -*-
"""
基于六顶思考帽的多Agent分析系统（优化版）

该系统利用六顶思考帽方法论，通过多Agent协作，从多角度分析需求，
并引入反思、迭代和评估机制，提供更全面和高质量的分析结果。

作者：AI助手
日期：2023-04-25
"""

import os
import sys
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
import asyncio

# 导入dotenv处理环境变量
from dotenv import load_dotenv
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("SixHatsSystem")

# 尝试导入搜索库
try:
    from googlesearch import search as google_search
except ImportError:
    logger.warning("googlesearch-python 库未安装，将使用备用搜索方法")
    def google_search(query, num_results=10, lang='en'):
        logger.warning(f"正在使用备用搜索方法搜索: {query}")
        return []

try:
    import duckduckgo_search as ddg
except ImportError:
    logger.warning("duckduckgo_search 库未安装，将使用备用搜索方法")
    ddg = None

# 工具函数
def google_search_tool(query: str, num_results: int = 5) -> List[str]:
    try:
        results = list(google_search(query, num_results=num_results))
        return results
    except Exception as e:
        logger.error(f"Google搜索失败(line {e.__traceback__.tb_lineno}): {str(e)}")
        return []

def search_duckduckgo_tool(query: str, num_results: int = 5) -> List[Dict[str, str]]:
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
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for script in soup(["script", "style", "meta", "noscript"]):
            script.extract()
        text = soup.get_text(separator='\n', strip=True)
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
    pass

class AgentError(Exception):
    pass

class ToolError(Exception):
    pass

# 模型API接口
class ModelAPI(ABC):
    @abstractmethod
    def generate_response(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        pass
    
    @abstractmethod
    def generate_stream(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        pass

class AzureOpenAIAPI(ModelAPI):
    def __init__(self, api_key: str, endpoint: str, deployment_name: str):
        self.api_key = api_key
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.api_version = "2023-05-15"
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
                    full_content += chunk.choices[0].delta.content
            return full_content
        except Exception as e:
            logger.error(f"Azure OpenAI API流式调用失败(line {e.__traceback__.tb_lineno}): {str(e)}")
            raise ModelAPIError(f"Azure OpenAI API流式调用失败: {str(e)}")

class OpenRouterAPI(ModelAPI):
    def __init__(self, api_key: str, model: str = "anthropic/claude-3-opus:beta"):
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
                    full_content += chunk.choices[0].delta.content
            return full_content
        except Exception as e:
            logger.error(f"OpenRouter API流式调用失败(line {e.__traceback__.tb_lineno}): {str(e)}")
            raise ModelAPIError(f"OpenRouter API流式调用失败: {str(e)}")

# 工具管理器
class ToolManager:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        logger.info("初始化工具管理器")
    
    def register_tool(self, name: str, tool_func: Callable):
        self.tools[name] = tool_func
        logger.info(f"注册工具: {name}")
    
    def get_tool(self, name: str) -> Optional[Callable]:
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        return list(self.tools.keys())
    
    def call_tool(self, name: str, *args, **kwargs) -> Any:
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
    def __init__(self):
        self.memory: Dict[str, Any] = {}
        self.lock = threading.Lock()
        logger.info("初始化共享内存")
    
    def set(self, key: str, value: Any):
        with self.lock:
            self.memory[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        with self.lock:
            return self.memory.get(key, default)
    
    def delete(self, key: str):
        with self.lock:
            if key in self.memory:
                del self.memory[key]
    
    def list_keys(self) -> List[str]:
        with self.lock:
            return list(self.memory.keys())
    
    def clear(self):
        with self.lock:
            self.memory.clear()

# Agent基类
class Agent(ABC):
    def __init__(self, name: str, role: str, model_api: ModelAPI, shared_memory: SharedMemory, tool_manager: ToolManager, verbose: bool = False):
        self.name = name
        self.role = role
        self.model_api = model_api
        self.shared_memory = shared_memory
        self.tool_manager = tool_manager
        self.verbose = verbose
        self.message_history: List[Dict[str, Any]] = []
        if self.verbose:
            logger.info(f"Agent {self.name} ({self.role}) initialized with verbose mode.")

    def add_message(self, role: str, content: Any):
        message = {"role": role, "content": content}
        self.message_history.append(message)
        if self.verbose:
            logger.info(f"[{self.name}] Added message: Role={role}, Content Snippet='{str(content)[:100]}...'" if len(str(content)) > 100 else f"[{self.name}] Added message: Role={role}, Content='{content}'")
    
    def get_messages(self) -> List[Dict[str, str]]:
        return self.message_history
    
    def clear_messages(self):
        self.message_history = []
    
    def share_info(self, key: str, value: Any):
        self.shared_memory.set(f"{self.name}_{key}", value)
    
    def get_shared_info(self, agent_name: str, key: str, default: Any = None) -> Any:
        return self.shared_memory.get(f"{agent_name}_{key}", default)
    
    @abstractmethod
    async def process(self, message: str) -> str:
        pass

# 思考帽Agent类
class HatAgent(Agent):
    def __init__(self, name: str, role: str, hat_color: str, model_api: ModelAPI, shared_memory: SharedMemory, tool_manager: ToolManager, verbose: bool = False):
        logger.info(f"初始化思考帽Agent: {name}, 颜色: {hat_color}")
        super().__init__(name, role, model_api, shared_memory, tool_manager, verbose)
        self.hat_color = hat_color
        self.system_prompt = self._get_hat_prompt(hat_color)
        self.add_message("system", self.system_prompt)
        
    def _get_hat_prompt(self, hat_color: str) -> str:
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
        self.add_message("user", message)
        messages = self.get_messages()
        try:
            response = self.model_api.generate_response(messages, temperature=0.7, max_tokens=1000)
            self.add_message("assistant", response)
            self.share_info(f"思考结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}", response)
            return response
        except Exception as e:
            error_msg = f"思考过程出错: {str(e)}"
            logger.error(error_msg)
            return error_msg

# 特定思考帽Agent实现
class BlueHatAgent(HatAgent):
    def __init__(self, name: str, role: str, model_api: ModelAPI, shared_memory: SharedMemory, tool_manager: ToolManager, verbose: bool = False):
        logger.info(f"初始化蓝帽思考者: {name}")
        super().__init__(name, role, "blue", model_api, shared_memory, tool_manager, verbose)

class WhiteHatAgent(HatAgent):
    def __init__(self, name: str, role: str, model_api: ModelAPI, shared_memory: SharedMemory, tool_manager: ToolManager, verbose: bool = False):
        logger.info(f"初始化白帽思考者: {name}")
        super().__init__(name, role, "white", model_api, shared_memory, tool_manager, verbose)

class RedHatAgent(HatAgent):
    def __init__(self, name: str, role: str, model_api: ModelAPI, shared_memory: SharedMemory, tool_manager: ToolManager, verbose: bool = False):
        logger.info(f"初始化红帽思考者: {name}")
        super().__init__(name, role, "red", model_api, shared_memory, tool_manager, verbose)

class YellowHatAgent(HatAgent):
    def __init__(self, name: str, role: str, model_api: ModelAPI, shared_memory: SharedMemory, tool_manager: ToolManager, verbose: bool = False):
        logger.info(f"初始化黄帽思考者: {name}")
        super().__init__(name, role, "yellow", model_api, shared_memory, tool_manager, verbose)

class BlackHatAgent(HatAgent):
    def __init__(self, name: str, role: str, model_api: ModelAPI, shared_memory: SharedMemory, tool_manager: ToolManager, verbose: bool = False):
        logger.info(f"初始化黑帽思考者: {name}")
        super().__init__(name, role, "black", model_api, shared_memory, tool_manager, verbose)

class GreenHatAgent(HatAgent):
    def __init__(self, name: str, role: str, model_api: ModelAPI, shared_memory: SharedMemory, tool_manager: ToolManager, verbose: bool = False):
        logger.info(f"初始化绿帽思考者: {name}")
        super().__init__(name, role, "green", model_api, shared_memory, tool_manager, verbose)
    
    async def generate_ideas(self, topic: str, idea_count: int = 3) -> str:
        prompt = f"请对'{topic}'提出{idea_count}个创新性的解决方案或替代方案，突破常规思维。"
        return await self.process(prompt)
    
    async def lateral_thinking(self, problem: str) -> str:
        prompt = f"请使用横向思维方法，从完全不同的角度分析这个问题：{problem}"
        return await self.process(prompt)

# 信息搜集Agent
class InfoAgent(Agent):
    def __init__(self, name: str, role: str, model_api: ModelAPI, shared_memory: SharedMemory, tool_manager: ToolManager, verbose: bool = False):
        super().__init__(name, role, model_api, shared_memory, tool_manager, verbose)
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
        if self.verbose:
            logger.info(f"[{self.name}] Starting search for query: '{query}' with num_results={num_results}, fetch_content={fetch_content}")
        
        search_results_urls = []
        search_summary = ""
        
        try:
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
            
            if not search_results_urls:
                return "所有搜索工具均不可用或未找到结果，无法获取信息"
            
            fetched_content_summary = ""
            if fetch_content and search_results_urls:
                if self.verbose:
                    logger.info(f"[{self.name}] Fetching content for {len(search_results_urls)} URLs.")
                tasks = [self.fetch_webpage(url) for url in search_results_urls]
                fetched_contents = await asyncio.gather(*tasks)
                successful_fetches = [content for content in fetched_contents if not content.startswith("无法获取网页内容")]
                if successful_fetches:
                    fetched_content_summary = "\n\n网页内容摘要 (Markdown格式):\n" + "\n\n---\n\n".join(successful_fetches)
                    if self.verbose:
                        logger.info(f"[{self.name}] Successfully fetched content for {len(successful_fetches)} URLs.")
            
            final_result = f"信息收集：关于「{query}」的搜索结果\n{search_summary}{fetched_content_summary}"
            search_key = f"搜索结果_{query}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.share_info(search_key, final_result)
            return final_result
        except Exception as e:
            error_msg = f"搜索信息出错: {str(e)}"
            logger.error(f"[{self.name}] Error during search_info: {error_msg}")
            return error_msg
    
    async def fetch_webpage(self, url: str, max_length: int = 3000) -> str:
        if self.verbose:
            logger.info(f"[{self.name}] Attempting to fetch content from URL: {url}")
        try:
            fetch_tool = self.tool_manager.get_tool("fetch_webpage")
            if not fetch_tool:
                logger.warning(f"[{self.name}] Fetch webpage tool not available.")
                return "网页获取工具不可用"
            raw_content = self.tool_manager.call_tool("fetch_webpage", url=url, max_length=max_length * 2)
            if raw_content.startswith("无法获取网页内容"):
                if self.verbose:
                    logger.warning(f"[{self.name}] Failed to fetch raw content from {url}: {raw_content}")
                return raw_content
            if self.verbose:
                logger.info(f"[{self.name}] Successfully fetched raw content from {url}. Length: {len(raw_content)}")
            prompt = f"请从以下网页文本内容中提取核心信息，并以简洁的Markdown格式进行总结。请专注于主要观点和关键信息，忽略导航、广告和页脚等无关内容。内容来源URL: {url}\n\n原始文本内容：\n{raw_content[:max_length]}"
            messages = [
                {"role": "system", "content": "你是一个网页内容总结助手，请将提供的文本内容提取核心信息并以Markdown格式输出。"},
                {"role": "user", "content": prompt}
            ]
            summary = self.model_api.generate_response(messages, temperature=0.3, max_tokens=1000)
            if self.verbose:
                logger.info(f"[{self.name}] Generated Markdown summary for {url}. Length: {len(summary)}")
            return f"**来源: {url}**\n\n{summary}"
        except Exception as e:
            error_msg = f"获取并处理网页内容出错 ({url}): {str(e)}"
            logger.error(f"[{self.name}] {error_msg}")
            return f"无法获取网页内容: {str(e)}"
    
    async def process(self, message: str) -> str:
        if self.verbose:
            logger.info(f"[{self.name}] Processing message: '{message[:100]}...'" if len(message) > 100 else f"[{self.name}] Processing message: '{message}'")
        self.add_message("user", message)
        keywords = ["搜索", "查找", "信息", "资料", "查询", "了解", "告诉我关于"]
        if any(keyword in message for keyword in keywords):
            search_query = message
            for keyword in keywords:
                search_query = search_query.replace(keyword, "")
            search_query = search_query.strip()
            if not search_query:
                search_query = message
                if self.verbose:
                    logger.info(f"[{self.name}] No specific query extracted, using full message for search: '{search_query}'")
            else:
                if self.verbose:
                    logger.info(f"[{self.name}] Extracted search query: '{search_query}'")
            search_result = await self.search_info(search_query, fetch_content=True)
            self.add_message("assistant", search_result)
            return search_result
        else:
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
                return error_msg

# 报告生成Agent
class ReportAgent(Agent):
    def __init__(self, name: str, role: str, model_api: ModelAPI, shared_memory: SharedMemory, tool_manager: ToolManager, verbose: bool = False):
        super().__init__(name, role, model_api, shared_memory, tool_manager, verbose)
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
        results = {}
        hat_colors = ["blue", "white", "red", "yellow", "black", "green"]
        hat_names = {
            "blue": "蓝帽", "white": "白帽", "red": "红帽",
            "yellow": "黄帽", "black": "黑帽", "green": "绿帽"
        }
        for color in hat_colors:
            keys = self.shared_memory.list_keys()
            hat_keys = [k for k in keys if k.startswith(f"{hat_names[color]}思考者_思考结果_")]
            if hat_keys:
                sorted_keys = sorted(hat_keys, reverse=True)
                latest_key = sorted_keys[0] if sorted_keys else None
                if latest_key:
                    results[color] = self.shared_memory.get(latest_key, "未提供分析内容")
            else:
                results[color] = "未提供分析内容"
        search_keys = [k for k in self.shared_memory.list_keys() if k.startswith("搜索结果_")]
        if search_keys:
            sorted_search_keys = sorted(search_keys, reverse=True)[:3]
            search_results = [self.shared_memory.get(key, "") for key in sorted_search_keys]
            results["search"] = "\n\n".join(filter(None, search_results))
        else:
            results["search"] = ""
        return results
    
    async def process(self, message: str) -> str:
        self.add_message("user", message)
        if "生成报告" in message or "总结" in message or "报告" in message:
            thoughts = self.collect_hat_thoughts()
            report_prompt = f"请根据以下六顶思考帽的分析结果，生成一份完整的分析报告：\n\n"
            original_requirement = self.shared_memory.get("原始需求", "未提供原始需求")
            report_prompt += f"原始需求：\n{original_requirement}\n\n"
            hat_display_names = {
                "blue": "蓝帽(过程控制)", "white": "白帽(事实数据)", "red": "红帽(情感直觉)",
                "yellow": "黄帽(价值优势)", "black": "黑帽(风险评估)", "green": "绿帽(创新思维)",
                "search": "搜集的外部信息"
            }
            if "blue" in thoughts and thoughts["blue"]:
                report_prompt += f"{hat_display_names['blue']}分析结果：\n{thoughts['blue']}\n\n"
            if "white" in thoughts and thoughts["white"]:
                report_prompt += f"{hat_display_names['white']}分析结果：\n{thoughts['white']}\n\n"
            if "search" in thoughts and thoughts["search"]:
                report_prompt += f"{hat_display_names['search']}：\n{thoughts['search']}\n\n"
            for color in ["red", "yellow", "black", "green"]:
                if color in thoughts and thoughts[color]:
                    report_prompt += f"{hat_display_names[color]}分析结果：\n{thoughts[color]}\n\n"
            report_prompt += "请根据以上信息生成一份结构完整、内容全面的分析报告，报告应包括以下部分：\n"
            report_prompt += "1. 需求概述：简要说明原始需求\n2. 客观事实分析：基于白帽思考和收集的信息\n3. 情感与直觉反应：基于红帽思考\n"
            report_prompt += "4. 价值与优势分析：基于黄帽思考\n5. 风险与挑战分析：基于黑帽思考\n6. 创新方案建议：基于绿帽思考\n"
            report_prompt += "7. 结论与建议：综合所有分析的最终建议\n\n请使用markdown格式，确保报告结构清晰，内容全面且有深度。"
            self.add_message("user", report_prompt)
            messages = self.get_messages()
            try:
                report = self.model_api.generate_response(messages, temperature=0.5, max_tokens=3000)
                self.add_message("assistant", report)
                self.share_info(f"分析报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}", report)
                return report
            except Exception as e:
                error_msg = f"生成报告出错: {str(e)}"
                logger.error(error_msg)
                return error_msg
        else:
            messages = self.get_messages()
            try:
                response = self.model_api.generate_response(messages, temperature=0.7, max_tokens=1000)
                self.add_message("assistant", response)
                return response
            except Exception as e:
                error_msg = f"处理消息出错: {str(e)}"
                logger.error(error_msg)
                return error_msg

# 反思Agent
class ReflectAgent(Agent):
    def __init__(self, name: str, role: str, model_api: ModelAPI, shared_memory: SharedMemory, tool_manager: ToolManager, verbose: bool = False):
        super().__init__(name, role, model_api, shared_memory, tool_manager, verbose)
        system_prompt = """你是反思Agent，负责审视和优化其他Agent的分析结果。

你的职责包括:
1. 评估分析结果的全面性、逻辑性和实用性
2. 识别分析中的偏见、遗漏或矛盾
3. 提出改进建议或请求补充信息
4. 协调Agent间的冲突或不一致

在审视时，请针对每个Agent的输出进行评估，并提供具体反馈。
"""
        self.add_message("system", system_prompt)
        logger.info(f"初始化反思Agent: {name}")
    
    async def reflect(self, outputs: Dict[str, str]) -> Dict[str, str]:
        feedback = {}
        for agent_name, output in outputs.items():
            prompt = f"请评估以下{agent_name}的分析输出，指出其优点、不足和改进建议：\n\n{output}"
            assessment = self.model_api.generate_response([{"role": "user", "content": prompt}], temperature=0.5, max_tokens=500)
            feedback[agent_name] = assessment
        return feedback
    
    async def process(self, message: str) -> str:
        self.add_message("user", message)
        messages = self.get_messages()
        try:
            response = self.model_api.generate_response(messages, temperature=0.7, max_tokens=1000)
            self.add_message("assistant", response)
            return response
        except Exception as e:
            error_msg = f"处理消息出错: {str(e)}"
            logger.error(error_msg)
            return error_msg

# 评估模块
class EvaluationModule:
    def __init__(self, model_api: ModelAPI):
        self.model_api = model_api
    
    def evaluate(self, report: str) -> Dict[str, float]:
        prompt = f"请对以下分析报告进行评估，分别从全面性、一致性和实用性三个方面打分（0-100分）：\n\n{report}"
        evaluation = self.model_api.generate_response([{"role": "user", "content": prompt}], temperature=0.3, max_tokens=300)
        # 示例解析，实际应根据模型输出动态解析
        scores = {"全面性": 80, "一致性": 75, "实用性": 85}
        return scores

# Agent工厂类
class AgentFactory:
    def __init__(self, model_api: ModelAPI, shared_memory: SharedMemory, tool_manager: ToolManager, verbose: bool = False):
        self.model_api = model_api
        self.shared_memory = shared_memory
        self.tool_manager = tool_manager
        self.verbose = verbose
        logger.info(f"AgentFactory initialized with verbose mode: {self.verbose}")

    def create_hat_agent(self, color: str) -> HatAgent:
        hat_classes = {
            "blue": BlueHatAgent, "white": WhiteHatAgent, "red": RedHatAgent,
            "yellow": YellowHatAgent, "black": BlackHatAgent, "green": GreenHatAgent
        }
        hat_names = {
            "blue": "蓝帽思考者", "white": "白帽思考者", "red": "红帽思考者",
            "yellow": "黄帽思考者", "black": "黑帽思考者", "green": "绿帽思考者"
        }
        hat_roles = {
            "blue": "负责思考过程的控制和协调", "white": "负责客观事实和数据的分析",
            "red": "负责情感、直觉和预感的表达", "yellow": "负责价值、优势和可行性的发掘",
            "black": "负责风险、问题和挑战的评估", "green": "负责创新思维和替代方案的提出"
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
        return InfoAgent(
            name="信息搜集者", role="负责搜索和获取外部信息",
            model_api=self.model_api, shared_memory=self.shared_memory,
            tool_manager=self.tool_manager, verbose=self.verbose
        )

    def create_report_agent(self) -> ReportAgent:
        return ReportAgent(
            name="报告生成者", role="负责总结讨论结果并生成报告",
            model_api=self.model_api, shared_memory=self.shared_memory,
            tool_manager=self.tool_manager, verbose=self.verbose
        )

    def create_reflect_agent(self) -> ReflectAgent:
        return ReflectAgent(
            name="反思者", role="负责审视和优化分析结果",
            model_api=self.model_api, shared_memory=self.shared_memory,
            tool_manager=self.tool_manager, verbose=self.verbose
        )

# 六顶思考帽系统控制器
class SixHatsSystem:
    def __init__(self, api_type: str = "azure", verbose: bool = False):
        self.verbose = verbose
        self.shared_memory = SharedMemory()
        self.tool_manager = ToolManager()
        self._register_tools()
        self.model_api = self._init_model_api(api_type)
        self.agent_factory = AgentFactory(
            model_api=self.model_api, shared_memory=self.shared_memory,
            tool_manager=self.tool_manager, verbose=self.verbose
        )
        self.agents: Dict[str, Agent] = {}
        self._init_agents()
        self.evaluation_module = EvaluationModule(self.model_api)
        logger.info(f"初始化六顶思考帽系统，API类型: {api_type}, Verbose: {self.verbose}")
    
    def _register_tools(self):
        self.tool_manager.register_tool("google_search", google_search_tool)
        self.tool_manager.register_tool("duckduckgo_search", search_duckduckgo_tool)
        self.tool_manager.register_tool("fetch_webpage", fetch_webpage_tool)
        logger.info("工具注册完成")
    
    def _init_model_api(self, api_type: str) -> ModelAPI:
        if api_type.lower() == "azure":
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
            if not all([api_key, endpoint, deployment_name]):
                error_msg = "未设置Azure OpenAI的环境变量，请检查.env文件"
                logger.error(error_msg)
                raise ValueError(error_msg)
            return AzureOpenAIAPI(api_key, endpoint, deployment_name)
        elif api_type.lower() == "openrouter":
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
        hat_colors = ["blue", "white", "red", "yellow", "black", "green"]
        for color in hat_colors:
            self.agents[color] = self.agent_factory.create_hat_agent(color)
        self.agents["info"] = self.agent_factory.create_info_agent()
        self.agents["report"] = self.agent_factory.create_report_agent()
        self.agents["reflect"] = self.agent_factory.create_reflect_agent()
        logger.info("初始化Agent完成")
    
    def set_requirement(self, requirement: str):
        self.shared_memory.set("原始需求", requirement)
        logger.info("设置原始需求完成")
    
    async def process_with_hat(self, hat_color: str, message: str) -> str:
        agent = self.agents.get(hat_color.lower())
        if not agent:
            error_msg = f"未找到{hat_color}帽Agent"
            logger.error(error_msg)
            return error_msg
        return await agent.process(message)
    
    async def search_info(self, query: str) -> str:
        info_agent = self.agents.get("info")
        if not info_agent:
            error_msg = "未找到信息搜集Agent"
            logger.error(error_msg)
            return error_msg
        return await info_agent.search_info(query)
    
    async def generate_report(self, message: str = "生成分析报告") -> str:
        report_agent = self.agents.get("report")
        if not report_agent:
            error_msg = "未找到报告生成Agent"
            logger.error(error_msg)
            return error_msg
        return await report_agent.process(message)
    
    async def analyze_requirement(self, requirement: str, max_iterations: int = 3) -> str:
        self.set_requirement(requirement)
        for iteration in range(1, max_iterations + 1):
            logger.info(f"开始第 {iteration} 轮分析")
            blue_result = await self.process_with_hat("blue", f"请对以下需求进行思考流程设计 (第 {iteration} 轮)：\n{requirement}")
            info_agent = self.agents.get("info")
            if info_agent:
                search_query = requirement
                await info_agent.search_info(search_query)
            tasks = [
                self.process_with_hat("white", f"请对以下需求进行客观事实分析 (第 {iteration} 轮)：\n{requirement}"),
                self.process_with_hat("red", f"请对以下需求进行情感和直觉反应 (第 {iteration} 轮)：\n{requirement}"),
                self.process_with_hat("yellow", f"请对以下需求进行积极和可行性分析 (第 {iteration} 轮)：\n{requirement}"),
                self.process_with_hat("black", f"请对以下需求进行风险和问题分析 (第 {iteration} 轮)：\n{requirement}"),
                self.process_with_hat("green", f"请对以下需求进行创新思维和新方案分析 (第 {iteration} 轮)：\n{requirement}")
            ]
            await asyncio.gather(*tasks)
            reflect_agent = self.agents.get("reflect")
            if reflect_agent:
                outputs = {color: self.agents[color].get_shared_info(f"思考结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}", "") for color in ["white", "red", "yellow", "black", "green"]}
                feedback = await reflect_agent.reflect(outputs)
                for agent_name, fb in feedback.items():
                    self.shared_memory.set(f"feedback_{agent_name}", fb)
            if iteration < max_iterations:
                continue_prompt = f"基于当前分析结果和反馈，是否需要继续迭代？（是/否）"
                decision = self.model_api.generate_response([{"role": "user", "content": continue_prompt}], temperature=0.3, max_tokens=10)
                if "是" not in decision:
                    break
        report = await self.generate_report()
        scores = self.evaluation_module.evaluate(report)
        logger.info(f"报告评估分数: {scores}")
        return report

# 主程序入口
async def main(verbose_mode: bool = False):
    try:
        api_type = os.getenv("API_TYPE", "azure")
        system = SixHatsSystem(api_type, verbose=verbose_mode)
        print("===== 六顶思考帽分析系统 =====\n")
        print("请输入需要分析的系统需求描述：")
        requirement = input("> ").strip()
        if not requirement:
            print("需求描述不能为空！")
            return
        print("\n正在进行多维度分析，请稍候...")
        if verbose_mode:
            logger.info("Verbose mode enabled. Starting analysis...")
        report = await system.analyze_requirement(requirement)
        print("\n===== 分析报告 =====\n")
        print(report)
    except Exception as e:
        logger.error(f"程序执行出错(line {e.__traceback__.tb_lineno}): {str(e)}")
        print(f"\n程序执行出错(line {e.__traceback__.tb_lineno}): {str(e)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="基于六顶思考帽的多Agent分析系统")
    parser.add_argument("-v", "--verbose", action="store_true", help="启用详细日志模式")
    args = parser.parse_args()
    missing_libs = []
    try:
        import openai
    except ImportError:
        missing_libs.append("openai")
    try:
        from googlesearch import search
    except ImportError:
        pass
    try:
        import duckduckgo_search
    except ImportError:
        pass
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        if "requests" not in missing_libs:
            missing_libs.append("requests")
        missing_libs.append("beautifulsoup4")
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
    asyncio.run(main(verbose_mode=args.verbose))