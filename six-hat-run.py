#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
六顶思考帽分析系统启动程序

该文件提供了一个简单的命令行界面来运行六顶思考帽分析系统，
用户可以直接输入需求描述，系统将自动进行多角度分析并生成报告。

使用方法：python 6bot-run.py

作者：AI助手
日期：2023-04-25
"""

import os
import sys
import asyncio
import logging
import argparse
from dotenv import load_dotenv

# 确保当前目录在模块搜索路径中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 加载环境变量
load_dotenv()

# 设置日志级别
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# 导入六顶思考帽系统
from six_hat_bot import SixHatsSystem


class SixHatsAnalyzer:
    """六顶思考帽分析系统界面类"""
    
    def __init__(self):
        """初始化分析器"""
        # 获取API类型，默认使用OpenRouter
        self.api_type = os.getenv("API_TYPE", "openrouter")
        self.system = None
        
    async def initialize(self, verbose: bool = False):
        """初始化系统"""
        try:
            print("正在初始化六顶思考帽分析系统...")
            self.system = SixHatsSystem(self.api_type, verbose=verbose)
            print("初始化完成！")
            return True
        except Exception as e:
            print(f"初始化系统出错: {str(e)}")
            return False
    
    async def run_analysis(self):
        """运行分析流程"""
        print("\n" + "=" * 50)
        print("欢迎使用六顶思考帽分析系统")
        print("=" * 50)
        print("\n本系统将从多个角度分析您的需求，包括：")
        print("  · 蓝帽：过程控制和整体规划")
        print("  · 白帽：客观事实和数据收集")
        print("  · 红帽：情感反应和直觉感受")
        print("  · 黄帽：积极价值和可行性")
        print("  · 黑帽：风险评估和问题识别")
        print("  · 绿帽：创新思维和替代方案")
        print("\n系统将自动整合各方面的分析，生成全面的分析报告。")
        print("=" * 50)
        
        # 获取需求描述
        print("\n请描述您需要分析的需求或问题：")
        requirement = input("> ").strip()
        
        if not requirement:
            print("需求描述不能为空！")
            return False
        
        # 开始分析
        print("\n正在进行多维度分析，请稍候...\n")
        print("* 启动蓝帽思考流程...")
        
        try:
            # 执行分析
            report = await self.system.analyze_requirement(requirement)
            
            # 输出报告
            print("\n" + "=" * 50)
            print("六顶思考帽分析报告")
            print("=" * 50 + "\n")
            print(report)
            print("\n" + "=" * 50)
            print("分析完成，感谢使用！")
            print("=" * 50)
            
            return True
        except Exception as e:
            print(f"\n分析过程出错: {str(e)}")
            return False


async def main(verbose: bool = False):
    """主函数"""
    # 检查环境变量
    if not os.getenv("OPENROUTER_API_KEY") and not (os. getenv("AZURE_OPENAI_API_KEY") and 
                                                os.getenv("AZURE_OPENAI_ENDPOINT") and 
                                                os.getenv("AZURE_OPENAI_DEPLOYMENT")):
        print("错误：未设置API密钥。请确保已正确配置.env文件。")
        print("  - 对于OpenRouter，需要设置OPENROUTER_API_KEY")
        print("  - 对于Azure OpenAI，需要设置AZURE_OPENAI_API_KEY、AZURE_OPENAI_ENDPOINT和AZURE_OPENAI_DEPLOYMENT")
        return
    
    # 初始化分析器
    analyzer = SixHatsAnalyzer()
    if not await analyzer.initialize(verbose=verbose):
        return
    
    # 运行分析
    await analyzer.run_analysis()


if __name__ == "__main__":
    # 检查必要的库
    missing_libs = []
    
    # 检查openai库
    try:
        import openai
    except ImportError:
        missing_libs.append("openai")
    
    # 检查requests和BeautifulSoup
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        if "requests" not in missing_libs:
            missing_libs.append("requests")
        missing_libs.append("bs4")
    
    # 检查python-dotenv
    try:
        from dotenv import load_dotenv
    except ImportError:
        missing_libs.append("python-dotenv")
    
    # 如果缺少库，提示安装
    if missing_libs:
        print("缺少必要的库，请使用以下命令安装：")
        print(f"pip install {' '.join(missing_libs)}")
        print("然后重新运行程序。")
        sys.exit(1)
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="运行六顶思考帽分析系统")
    parser.add_argument("-v", "--verbose", action="store_true", help="启用详细日志模式")
    args = parser.parse_args()

    # 运行主程序
    asyncio.run(main(verbose=args.verbose))