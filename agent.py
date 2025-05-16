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
from typing import Optional, List, Dict

class Agent:
    """Langchainを使用したAIエージェントクラス。"""

    def __init__(
        self,
        model_name: str,
        provider: str = "openai",
        system_prompt: Optional[str] = None,
        tools: Optional[List[str]] = None,
        reasoning_effort: Optional[str] = None,
    ):
        """
        エージェントを初期化します。

        Args:
            model_name (str): 使用するモデル名。
            provider (str): LLMプロバイダー ("openai", "anthropic", "google")。
            system_prompt (Optional[str]): システムプロンプト。
            tools (Optional[List[str]]): 使用するツールのリスト (例: ["ggl_search"])。
            reasoning_effort (Optional[str]): (OpenAIのみ) 推論の取り組み度合い。
        """
        dotenv.load_dotenv()

        self.model_name = model_name
        self.provider = provider.lower()
        self.system_prompt = system_prompt or "あなたはAIアシスタントです。Python_Interpreterを使った場合はコードと結果を表示してください。"
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
                            'Results are displayed in the console, so use functions like print() to print the results.',
                        func=ExecPython().invoke
                    )
                )
            if 'web_loader' in tools:
                def web_loader_func(url: str) -> str:
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
        self.agent_executor: CompiledGraph = self._initialize_agent_executor()


    def _initialize_llm(self) -> BaseChatModel:
        """LLMを初期化します。"""
        if self.provider == "openai":
            return ChatOpenAI(
                model=self.model_name,
                reasoning_effort=self.reasoning_effort,
                use_responses_api=True)
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
            tools=self.tools
        )


    def invoke(
        self,
        messages: List[BaseMessage]
    ) -> str:
        """
        エージェントを実行し、応答を返します。

        Args:
            messages (List[BaseMessage]): ユーザー入力とチャット履歴のリスト。

        Returns:
            str: エージェントからの応答。
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
        
        print(f"Input messages: {input_messages}")

        try:
            response = self.agent_executor.invoke(
                {"messages": input_messages},
                stream_mode="values"
            )
            print(f"Response: {response['messages'][-1].text()}")
            return response["messages"][-1].text()
        except Exception as e:
            print(f"エージェント実行中にエラーが発生しました: {e}")
            return f"処理中にエラーが発生しました。 ({type(e).__name__})"


if __name__ == "__main__":
    # テスト用のコード
    agent = Agent(model_name="gpt-4o", provider="openai", tools=["ggl_web_search", "code_interpreter", "web_loader"])
    response = agent.invoke([{"role": "user", "content": "0から1の間の乱数を生成して"}])
    print(response)