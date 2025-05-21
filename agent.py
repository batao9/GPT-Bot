# Messages
from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import trim_messages
# Models
from langchain_core.language_models import BaseChatModel
from langchain_openai.chat_models import ChatOpenAI
from langchain_anthropic.chat_models import ChatAnthropic
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
# Tools
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.tools.riza.command import ExecPython
from langchain_community.document_loaders import WebBaseLoader
# Agent
from langchain_core.tools import Tool
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent

import dotenv
import json
import os
from typing import Optional, List, Dict, Any, Tuple
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
            debug (bool): デバッグモード。
        """
        instance = cls(model_name, provider, system_prompt, tools, reasoning_effort, debug, _create_mode=True)
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
        self.tools = []
        if tools:
            if 'ddg_search' in tools:
                self.tools.append(DuckDuckGoSearchRun())
                self.tools.append(DuckDuckGoSearchResults())
            if 'ggl_search' in tools:
                search = GoogleSearchAPIWrapper()
                self.tools.append(
                    Tool(
                        name="Google_Search",
                        description='Googleで情報を検索します。引数は検索クエリです。',
                        func=search.run
                    )
                )
            if 'code_interpreter' in tools:
                self.tools.append(
                    Tool(
                        name="Python_Interpreter",
                        description='Run Python Code: '+
                            'Use this tool when you need to run Python code to get an answer. '+
                            'Input must be a valid Python expression or statement. '+
                            'You can use standard Python libraries. '+
                            'Results are displayed in the console, so use functions like print() to print the results. '+
                            'It is not possible to output graph images, so please use it for simple processing that can be completed in just a few lines.',
                        func=ExecPython().invoke
                    )
                )
            if 'web_loader' in tools:
                def web_loader_func(url: str) -> List[Any]:
                    loader = WebBaseLoader(url)
                    return loader.load()
                self.tools.append(
                    Tool(
                        name="Web_Loader",
                        description='Web Loader: '+
                            'Use this tool to load web pages. '+
                            'Input must be a valid URL.',
                        func=web_loader_func
                    )
                )
        if self.provider == "openai":
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
                    config = json.load(f)
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
        else:
            raise ValueError(f"E: 不明なプロバイダーです: {self.provider}")


    def _initialize_agent_executor(self) -> CompiledGraph:
        """Agent Executorを初期化します。"""
        if not self.llm:
            print("E: LLMが初期化されていません。Agent Executorを作成できません。")
            return None

        return create_react_agent(
            model=self.llm,
            tools=self.tools,
            response_format=AgentResponse
        )


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
            return "エラー: Agent Executorが初期化されていません。"

        input_messages: List[BaseMessage] = trim_messages(
            messages,
            token_counter=len, # トークンカウンターとしてlen関数を使用
            max_tokens=20, # 会話ターン数
            include_system=True, # システムメッセージを含める
        )
        
        if self.debug: print(f"Input messages: {input_messages}")

        try:
            response = await self.agent_executor.ainvoke(
                {"messages": input_messages}
            )
            if self.debug: print(f"Response: {response['messages'][-1].text()}")

            agent_response_obj = response["structured_response"]
            response_text = agent_response_obj.response
            requested_attachments = agent_response_obj.attachments

            if not requested_attachments:
                return response_text, []

            attachments_base_dir = os.getenv("AGENT_ATTACHMENTS_DIR")
            if not attachments_base_dir:
                if self.debug:
                    print(f"W: 環境変数 AGENT_ATTACHMENTS_DIR が設定されていません。添付ファイルのパスを検証できません。")
                return response_text, []

            # 添付ファイルが存在するか検証
            verified_attachments = []
            for filename in requested_attachments:
                if not isinstance(filename, str) or not filename.strip():
                    if self.debug:
                        print(f"W: 添付ファイルリストに無効なファイル名が含まれています: {filename}")
                    continue
                file_path = os.path.join(attachments_base_dir, filename)
                if os.path.exists(file_path) and os.path.isfile(file_path):
                    verified_attachments.append(file_path)
                elif self.debug:
                    print(f"W: 添付ファイル '{filename}' が指定されたパス '{file_path}' に見つからないか、通常のファイルではありません。")
            return response_text, verified_attachments
        
        except Exception as e:
            if self.debug: print(f"エージェント実行中にエラーが発生しました: {e}")
            return f"処理中にエラーが発生しました。 ({type(e).__name__})"


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