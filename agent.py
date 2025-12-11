# Messages
from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import trim_messages
# Models
from langchain_core.language_models import BaseChatModel
from langchain_openai.chat_models import ChatOpenAI
from langchain_anthropic.chat_models import ChatAnthropic
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
# Tools
from tools import (
    get_google_search_tool,
    get_duckduckgo_search_tools,
    get_code_interpreter_tool,
    get_web_loader_tool,
    get_docx_loader_tool,
    get_text_file_loader_tool,
    get_text_attachment_writer_tool
)
# Agent
from langchain_core.tools import Tool
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent

import dotenv
import json
import os
from typing import Optional, List, Dict, Any, Tuple
from error_logger import log_agent_error
from pydantic import BaseModel, Field

class AgentResponse(BaseModel):
    """エージェントの応答形式モデル。"""
    response: str = Field(description="Agent's response.")
    attachments: List[str] = Field(
        description="Filenames to attach to the response. "+
                    "Treated as DOWNLOAD_DIR/<filename>."
    )

class Agent:
    """Langchainを使用したAIエージェントクラス。"""
    @classmethod
    async def create(
        cls,
        model_name: str,
        provider: str = "openai",
        system_prompt: Optional[str] = None,
        tools: Optional[List[str]] = None,
        reasoning_effort: Optional[str] = None,
        input_dir_path: Optional[str] = os.path.join(os.path.dirname(__file__), 'tmp', 'user_attach'),
        output_dir_path: Optional[str] = os.path.join(os.path.dirname(__file__), 'tmp', 'agent_attach'),
        debug: bool = False
    ):
        """
        エージェントを非同期に初期化します。

        Args:
            model_name (str): 使用するモデル名。
            provider (str): LLMプロバイダー ("openai", "anthropic", "google")。
            system_prompt (Optional[str]): システムプロンプト。
            tools (Optional[List[str]]): 使用するツールのリスト (例: ["ggl_search"])。
            reasoning_effort (Optional[str]): (OpenAIのみ) 推論の取り組み度合い。
            input_dir_path (Optional[str]): 入力ディレクトリパス。
            output_dir_path (Optional[str]): 出力ディレクトリパス。
            debug (bool): デバッグモード。
        """
        instance = cls(model_name, provider, system_prompt, tools, reasoning_effort, input_dir_path, output_dir_path, debug, _create_mode=True)
        # mcpツールを非同期に初期化
        await instance._async_init_components(tools_arg_for_async=tools)
        return instance

    def __init__(
        self,
        model_name: str,
        provider: str = "openai",
        system_prompt: Optional[str] = None,
        tools: Optional[List[str]] = None,
        reasoning_effort: Optional[str] = None,
        input_dir_path: Optional[str] = None,
        output_dir_path: Optional[str] = None,
        debug: bool = False,
        _create_mode: bool = False
    ):
        """
        エージェントを初期化します。

        Args:
            model_name (str): 使用するモデル名。
            provider (str): LLMプロバイダー ("openai", "anthropic", "google")。
            system_prompt (Optional[str]): システムプロンプト。
            tools (Optional[List[str]]): 使用するツールのリスト (例: ["ggl_search"])。
            reasoning_effort (Optional[str]): (OpenAIのみ) 推論の取り組み度合い。
            input_dir_path (Optional[str]): 入力ディレクトリパス。
            output_dir_path (Optional[str]): 出力ディレクトリパス。
            debug (bool): デバッグモード。
            _create_mode (bool): インスタンス作成モード。
        """
        
        if not _create_mode:
            raise TypeError("Agentインスタンスは Agent.create() 非同期クラスメソッドを使用して作成する必要があります。")
        
        dotenv.load_dotenv()

        self.debug = debug
        self.model_name = model_name
        self.provider = provider.lower()
        self.system_prompt = system_prompt or "あなたはAIアシスタントです。"
        self.input_dir_path = input_dir_path
        self.output_dir_path = output_dir_path
        self.tools = self._initialize_tools(tools)
        if self.provider == "openai" or self.provider == "openai-responses":
            self.reasoning_effort = reasoning_effort
        self.llm: BaseChatModel = self._initialize_llm()
        self.agent_executor: Optional[CompiledGraph] = None


    async def _async_init_components(self, tools_arg_for_async: Optional[List[str]] = None):
        """
        非同期で初期化が必要なコンポーネント（MCPツールなど）をセットアップし、
        エージェントエクゼキュータを初期化します。
        
        Args:
            tools_arg_for_async (Optional[List[str]]): 非同期で初期化が必要なツールのリスト。
        """
        if tools_arg_for_async and 'mcp' in tools_arg_for_async:
            from langchain_mcp_adapters.client import MultiServerMCPClient
            config_file = os.getenv("MCP_CONFIG_FILE") or "mcp.json"
            try:
                with open(config_file, 'r') as f:
                    raw = f.read()
                raw = raw.replace("${USER_ATTACHMENTS_DIR}", self.input_dir_path)
                raw = raw.replace("${AGENT_ATTACHMENTS_DIR}", self.output_dir_path)
                config = json.loads(raw)
                
                if 'mcpServers' in config:
                    mcp_client = MultiServerMCPClient(config['mcpServers'])
                    mcp_tools_list = await mcp_client.get_tools()
                    self.tools.extend(mcp_tools_list)
                else:
                    print(f"警告: MCP設定ファイル '{config_file}' に 'mcpServers' キーが見つかりません。")
            except FileNotFoundError:
                print(f"警告: MCP設定ファイル '{config_file}' が見つかりません。")
            except json.JSONDecodeError:
                print(f"警告: MCP設定ファイル '{config_file}' のJSONデコードに失敗しました。")
            except Exception as e:
                print(f"警告: MCPツールの初期化中にエラーが発生しました: {e}")

        self.agent_executor = self._initialize_agent_executor()


    def _initialize_llm(self) -> BaseChatModel:
        """LLMを初期化します。"""
        if self.provider == "openai":
            return ChatOpenAI(
                model=self.model_name,
                reasoning_effort=self.reasoning_effort,
                store=False)
        elif self.provider == "anthropic":
            return ChatAnthropic(model=self.model_name)
        elif self.provider == "google" or self.provider == "gemini":
            return ChatGoogleGenerativeAI(model=self.model_name)
        elif self.provider == "openai-responses":
            return ChatOpenAI(
                model=self.model_name,
                reasoning_effort=self.reasoning_effort,
                use_responses_api=True)
        else:
            raise ValueError(f"E: 不明なプロバイダーです: {self.provider}")


    def _initialize_agent_executor(self) -> CompiledGraph:
        """Agent Executorを初期化します。"""
        if not self.llm:
            print("E: LLMが初期化されていません。Agent Executorを作成できません。")
            return None

        return create_react_agent(
            model=self.llm,
            tools=self.tools
        )
    
    
    def _initialize_tools(self, tools: Optional[List[str]] = None) -> List[Tool]:
        """
        ツールを初期化します。

        Args:
            tools (Optional[List[str]]): 使用するツールのリスト (例: ["ggl_search"])。

        Returns:
            List[Tool]: 初期化されたツールのリスト。
        """
        download_dir = self.input_dir_path
        upload_dir = self.output_dir_path
        tools_list = [
            get_docx_loader_tool(download_dir),
            get_text_file_loader_tool(download_dir)
        ]
        if upload_dir:
            tools_list.append(get_text_attachment_writer_tool(upload_dir))
        if 'ggl_search' in tools:
            tools_list.append(get_google_search_tool())
        if 'ddg_search' in tools:
            tools_list.extend(get_duckduckgo_search_tools())
        if 'code_interpreter' in tools:
            tools_list.append(get_code_interpreter_tool())
        if 'web_loader' in tools:
            tools_list.append(get_web_loader_tool())
        return tools_list


    async def invoke(
        self,
        messages: List[BaseMessage]
    ) -> Tuple[str, List[str]]:
        """
        エージェントを実行し、応答を返します。

        Args:
            messages (List[BaseMessage]): ユーザー入力とチャット履歴のリスト。

        Returns:
            Tuple[str, List[str]]: エージェントからの応答と添付ファイルのパスのリスト。
        """
        if not self.agent_executor:
            print("E: Agent Executorが初期化されていません。")
            return "エラー: Agent Executorが初期化されていません。", []

        input_messages: List[BaseMessage] = trim_messages(
            messages,
            token_counter=len, # トークンカウンターとしてlen関数を使用
            max_tokens=20, # 会話ターン数
            include_system=True, # システムメッセージを含める
        )
        
        if self.debug: print(f"Input messages: {input_messages}")

        try:
            result = await self.agent_executor.ainvoke({"messages": input_messages})
            if self.debug:
                print(f"Raw agent response: {result}")

            # Extract response text: handle structured_response, messages list, or raw string
            if isinstance(result, dict):
                if "structured_response" in result and hasattr(result["structured_response"], "response"):
                    response_text = result["structured_response"].response
                elif "messages" in result:
                    if self.provider == "openai-responses":
                        response_text = result["messages"][-1].content[0]["text"]
                    else:
                        response_text = str(result["messages"][-1].content)
                else:
                    response_text = str(result)
            elif isinstance(result, str):
                response_text = result
            else:
                response_text = str(result)

            attachments = []
            attachments_dir = self.output_dir_path
            if attachments_dir and os.path.isdir(attachments_dir):
                for root, _, files in os.walk(attachments_dir):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if os.path.isfile(file_path):
                            attachments.append(file_path)
            if self.debug:
                print(f"Collected attachments: {attachments}")
            return response_text, attachments

        except Exception as e:
            if self.debug:
                print(f"エージェント実行中にエラーが発生しました: {e}")
            try:
                log_agent_error(input_messages, self.input_dir_path, e)
            except Exception:
                pass
            return f"処理中にエラーが発生しました。 ({type(e).__name__})", []


if __name__ == "__main__":
    # テスト用のコード
    import asyncio
    from langchain_core.messages import HumanMessage
    from prompts import SytemPrompts

    async def main():
        agent = await Agent.create(
            model_name="o4-mini", 
            provider="openai", 
            tools=["ggl_web_search", "web_loader", "mcp"], 
            system_prompt=SytemPrompts.prompts["assistant"],
            debug=True
        )
        messages_input = [HumanMessage(content="pythonで y = x^2 のグラフを描いて")]
        # invokeメソッドはコルーチンなのでawaitで待機
        response, attachments = await agent.invoke(messages_input)
        print(response)
        print(attachments)

    # async関数を実行
    asyncio.run(main())
