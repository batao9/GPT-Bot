# discord bot
import discord
# agent
from agent import Agent
# gen thread name
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
# messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, merge_message_runs
# load env
import os
import dotenv
# process attachments
import base64
import aiohttp
# utils
import tempfile
import re
import sympy
import charset_normalizer
import asyncio
from typing import Optional, List, Dict, Tuple
from LLM_Models import Models
from prompts import SytemPrompts


class Utils:
    @staticmethod
    def latex_to_image(latex_code: str, save_path: str = None) -> str:
        """LaTeXコードを画像に変換する。

        Args:
            latex_code (str): 変換するLaTeXコード。
            save_path (str, optional): 画像を保存するパス。指定しない場合は一時ファイルに保存。Defaults to None.

        Returns:
            str: 保存された画像ファイルのパス。
        """
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            sympy.preview(latex_code, viewer='file', filename=tmpfile.name, euler=False,
                        dvioptions=["-T", "tight", "-z", "0", "--truecolor", "-D 600"],
                        dpi=60)
            if save_path:
                os.rename(tmpfile.name, save_path)
                return save_path
            else:
                return tmpfile.name

    @staticmethod
    def is_probably_text(raw: bytes, threshold: float = 0.75) -> bool:
        """バイト列がテキストである可能性が高いか判定する。

        1. NULL バイトがあれば False を返す。
        2. charset_normalizer でベストマッチを取得する。
        3. language_confidence（言語推定の信頼度）が閾値以上なら True を返す。

        Args:
            raw (bytes): 判定するバイト列。
            threshold (float, optional): テキストと判定するための信頼度の閾値。Defaults to 0.75.

        Returns:
            bool: テキストである可能性が高い場合は True、そうでない場合は False。
        """
        if b'\x00' in raw:
            return False
        match = charset_normalizer.detect(raw)
        if not match:
            return False
        return match['confidence'] >= threshold

    @staticmethod
    def detect_encoding(raw: bytes) -> str:
        """バイト列のエンコーディングを推定する。

        charset-normalizer でエンコーディングを推定する。
        見つからなければ UTF‑8 を返す。

        Args:
            raw (bytes): エンコーディングを推定するバイト列。

        Returns:
            str: 推定されたエンコーディング名。
        """
        best = charset_normalizer.from_bytes(raw).best()
        if best and best.encoding:
            enc = best.encoding.replace('_', '-').lower()
            return enc
        return 'utf-8'


class MyClient(discord.Client):
    async def on_ready(self) -> None:
        """クライアントが準備完了したときに呼び出される。
        """
        print(f'Logged on as {self.user}!')


    async def on_message(self, message: discord.Message) -> None:
        """メッセージを受信したときに呼び出される。

        Args:
            message (discord.Message): 受信したメッセージオブジェクト。
        """
        if (
            message.author == self.user or
            message.author.bot or
            message.is_system() or
            not Models.is_channel_configured(message.channel)
        ):
            return

        llm_agent = Agent(
            model_name=Models.get_field(message.channel, "model"),
            provider=Models.get_field(message.channel, "provider"),
            tools=Models.get_field(message.channel, "tools"),
            reasoning_effort=Models.get_field(message.channel, "reasoning_effort"),
        )

        print(f"user: {message.content}")
        thread, messages = await self.get_thread_and_messages(message)
        thread_name_future = self.generate_thread_name(messages)
        converted_messages = await self.convert_message(
            messages,
            provider=Models.get_field(message.channel, "provider"),
            system_prompt=SytemPrompts.prompts['assistant']
        )
        response_future = llm_agent.invoke(messages = converted_messages)

        thread_name, response = await asyncio.gather(
            thread_name_future,
            response_future
        )

        if thread and thread_name:
            await thread.edit(name=thread_name)
        await self.send_response(thread, response)


    async def get_thread_and_messages(
        self,
        message: discord.Message
    ) -> Tuple[Optional[discord.Thread], List[discord.Message]]:
        """メッセージを送るスレッドと、スレッドのメッセージ履歴を取得する。

        Args:
            message (discord.Message): 元となるメッセージオブジェクト。

        Returns:
            Tuple[Optional[discord.Thread], List[discord.Message]]: スレッドオブジェクトとメッセージ履歴のリストのタプル。
        """
        if isinstance(message.channel, discord.Thread):
            thread = message.channel
            messages = [msg async for msg in thread.history(limit=100)]
            messages.append(await thread.parent.fetch_message(thread.id))
        elif (
            isinstance(message.channel, discord.TextChannel) and
            Models.is_channel_configured(message.channel)
        ):
            thread = await message.create_thread(name="スレッド名生成中...")
            messages = [message]
        else:
            thread = None
            messages = []
        return thread, messages


    async def generate_thread_name(
        self,
        messages: List[discord.Message]
    ) -> Optional[str]:
        """スレッドの名前を生成する。

        Args:
            messages (List[discord.Message]): スレッド名の生成に使用するメッセージのリスト。

        Returns:
            Optional[str]: 生成されたスレッド名、または生成できなかった場合は None。
        """
        class ThreadName(BaseModel):
            thread_name: str = Field(description="thread name")

        if len(messages) != 1:
            return None

        llm = init_chat_model("gpt-4.1-nano", model_provider="openai", use_responses_api=True, store=False)
        structured_llm = llm.with_structured_output(ThreadName)

        converted_messages = await self.convert_message(
            messages,
            provider="openai",
            system_prompt=SytemPrompts.prompts['thread']
        )
        response = structured_llm.invoke(converted_messages)

        thread_name = response.thread_name
        return thread_name[:99] if len(thread_name) > 99 else thread_name


    async def convert_message(
        self,
        messages: List[discord.Message],
        provider: str = "openai",
        system_prompt: Optional[str] = None
    ) -> List[BaseMessage]:
        """discord.Messageオブジェクトのリストをモデルへの入力形式に変換する。

        Args:
            messages (List[discord.Message]): 変換するdiscord.Messageオブジェクトのリスト。
            provider (str, optional): 使用するモデルプロバイダー ('openai', 'anthropic', 'gemini'など)。Defaults to "openai".
            system_prompt (Optional[str], optional): 追加するシステムプロンプト。Defaults to None.

        Returns:
            List[BaseMessage]: 変換されたメッセージのリスト。
        """
        converted_messages = []
        for msg in reversed(messages):
            if msg.is_system():
                continue

            if msg.author.bot:
                converted_messages.append(AIMessage(content=msg.content))
                continue

            contents = []
            # 添付ファイルの処理
            msg_content = msg.content
            if msg.attachments:
                for attachment in msg.attachments:
                    filename = attachment.filename
                    async with aiohttp.ClientSession() as session:
                        async with session.get(attachment.url) as resp:
                            if resp.status != 200:
                                continue
                            raw = await resp.read()
                            # pdf file
                            if filename.endswith('.pdf'):
                                pdf_base64 = base64.b64encode(raw).decode('utf-8')
                                if provider == "openai":
                                    contents.append({
                                        "type": "input_file",
                                        "filename": filename,
                                        "file_data": f"data:application/pdf;base64,{pdf_base64}",
                                    })
                                elif provider == "anthropic" or "gemini":
                                    contents.append({
                                        "type": "file",
                                        "source_type": "base64",
                                        "mime_type": "application/pdf",
                                        "data": pdf_base64,
                                    })
                            # image file
                            elif filename.endswith(('.png', '.jpg', '.gif', 'webp', 'jpeg')):
                                img_base64 = base64.b64encode(raw).decode('utf-8')
                                extension = filename.split('.')[-1]
                                contents.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/{extension};base64,{img_base64}"},
                                })
                            # text file
                            elif Utils.is_probably_text(raw):
                                encoding = Utils.detect_encoding(raw)
                                try:
                                    file_text = raw.decode(encoding)
                                    msg_content = (
                                        f"{msg_content}"
                                        f"\n__file_start__{filename=}\n"
                                        f"{file_text}\n__file_end__"
                                    )
                                except UnicodeError as e:
                                    msg_content = (
                                        f"{msg_content}"
                                        f"\n__file_start__{filename=}\n"
                                        f"Error decoding file: {e}\n__file_end__"
                                    )
                            # other file
                            else:
                                msg_content = (
                                    f"{msg_content}"
                                    f"\n__file_start__{filename=}\n"
                                    "Cannot open file. Only image files and files that can be opened in plane text are supported.\n"
                                    "__file_end__"
                                )
            # メッセージの変換
            contents.append({"type": "text", "text": msg_content})
            converted_messages.append(HumanMessage(contents))

        # システムプロンプトの追加
        if system_prompt:
            converted_messages.insert(0, SystemMessage(content=system_prompt))

        return merge_message_runs(converted_messages)


    async def send_response(
        self,
        thread: discord.TextChannel,
        response: str
    ) -> None:
        """レスポンスを指定されたスレッドに送信する。

        レスポンスをDiscordの文字数制限に合わせて分割し、LaTeX数式は画像に変換して送信する。

        Args:
            thread (discord.TextChannel): レスポンスを送信するスレッドまたはテキストチャンネル。
            response (str): 送信するレスポンス文字列。
        """
        print(f'bot:{response}')
        CODE_BLOCK_DELIMITER = "```"
        LATEX_DELIMITER = "$$"
        MAX_LENGTH = 2000

        # responseをパーツに分割
        parts = self.split_response(response, CODE_BLOCK_DELIMITER, LATEX_DELIMITER, MAX_LENGTH)

        # パーツを送信
        buff = ''
        for part in parts:
            # LaTeX数式を検出して画像に変換
            if part.startswith("$$") and part.endswith("$$"):
                latex_code = part
                try:
                    image_path = Utils.latex_to_image(latex_code)
                    if len(buff) + len(latex_code) > MAX_LENGTH:
                        await thread.send(buff)
                        buff = ''
                    await thread.send(buff + latex_code, file=discord.File(image_path))
                    os.remove(image_path)
                    buff = ''
                except Exception as e:
                    print(f'latex_to_image error: {e}')
                    if len(buff) + len(latex_code) > MAX_LENGTH:
                        await thread.send(buff)
                        buff = ''
                    buff += part
            else:
                if len(buff) + len(part) > MAX_LENGTH:
                    await thread.send(buff)
                    buff = ''
                buff += part

        # 残りを送信
        if buff.strip():
            await thread.send(buff)


    def split_response(
        self,
        response: str,
        code_block_delimiter: str = "```",
        latex_delimiter: str = "$$",
        max_length: int = 2000,
    ) -> List[str]:
        """Discordの文字数制限を守りつつレスポンスを分割する。

        ・```...``` で囲まれたコードブロック
        ・$$...$$   で囲まれた LaTeX ブロック
        └ いずれも max_length を超える場合は分割する
            ・コードブロックは改行単位で分割し、各チャンクを
            <code_block_delimiter><lang>\n ... \n<code_block_delimiter>
            で再ラップして壊れないようにする
            ・LaTeX ブロックは $$ ... $$ で再ラップして送る
        ・通常テキストは「空白/改行/句読点」で賢く分割 (_smart_split)

        Args:
            response (str): 分割するレスポンス文字列。
            code_block_delimiter (str, optional): コードブロックの区切り文字。Defaults to "```".
            latex_delimiter (str, optional): LaTeXブロックの区切り文字。Defaults to "$$".
            max_length (int, optional): 分割する最大文字数。Defaults to 2000.

        Returns:
            List[str]: 分割された文字列のリスト。
        """
        # ------------------------------------------------------------------
        # 共通ユーティリティ
        # ------------------------------------------------------------------
        def _smart_split(text: str, limit: int) -> List[str]:
            """通常テキストを limit 以下で自然な区切りで分割する。

            Args:
                text (str): 分割するテキスト。
                limit (int): 分割の最大文字数。

            Returns:
                List[str]: 分割された文字列のリスト。
            """
            delimiters = [
                " ", "\n", "。", "、", ",", ".", "，", "．", ";",
                "!", "?", "！", "？"
            ]
            chunks: List[str] = []
            start = 0
            length = len(text)

            while start < length:
                if length - start <= limit:
                    chunks.append(text[start:])
                    break

                end = start + limit
                segment = text[start:end]

                # segment 内の最後の区切り文字を探す
                split_pos = -1
                for d in delimiters:
                    pos = segment.rfind(d)
                    if pos > split_pos:
                        split_pos = pos

                if split_pos <= 0:
                    split_pos = limit
                else:
                    split_pos += 1

                chunks.append(text[start:start + split_pos])
                start += split_pos
            return chunks

        def _split_code_block(block: str, limit: int) -> List[str]:
            """コードブロックを改行単位で分割し、再ラップして返す。

            Args:
                block (str): 分割するコードブロック文字列。
                limit (int): 分割の最大文字数。

            Returns:
                List[str]: 分割され、再ラップされた文字列のリスト。
            """
            first_nl = block.find("\n")
            if first_nl == -1:
                return [block]  # 1 行しか無い異常系

            header_line = block[:first_nl]                      # ```lang
            language     = header_line[len(code_block_delimiter):]  # "lang" or ""

            footer  = code_block_delimiter
            header  = f"{code_block_delimiter}{language}"
            overhead = len(header) + len(footer) + 2            # ヘッダ + フッタ + 改行2つ

            # 末尾の ``` を除き、中身だけ取り出す
            if block.rstrip().endswith(code_block_delimiter):
                inner = block[first_nl + 1 : block.rstrip().rfind(code_block_delimiter)]
            else:
                inner = block[first_nl + 1 :]

            inner = inner.rstrip("\n")

            # そもそも全体で収まるならそのまま返す
            if len(inner) + overhead <= limit:
                return [block]

            lines = inner.split("\n")
            chunks: List[str] = []
            buf = ""
            max_body = max(limit - overhead, 1)  # コンテンツ部で許容される最大長

            for line in lines:
                candidate = f"{buf}\n{line}" if buf else line
                if len(candidate) > max_body:
                    if buf:
                        chunks.append(buf)
                        buf = ""
                    # 単一行が max_body を超える場合は強制分割
                    while len(line) > max_body:
                        chunks.append(line[:max_body])
                        line = line[max_body:]
                    buf = line
                else:
                    buf = candidate
            if buf:
                chunks.append(buf)

            return [f"{header}\n{chunk}\n{footer}" for chunk in chunks]

        def _split_latex_block(block: str, limit: int) -> List[str]:
            """LaTeX ブロックを分割し、再ラップして返す。

            Args:
                block (str): 分割するLaTeXブロック文字列。
                limit (int): 分割の最大文字数。

            Returns:
                List[str]: 分割され、再ラップされた文字列のリスト。
            """
            overhead = len(latex_delimiter) * 2
            inner_limit = max(limit - overhead, 1)

            inner = block[len(latex_delimiter) : -len(latex_delimiter)]
            if len(inner) <= inner_limit:
                return [block]

            # オーバーヘッドを差し引いた長さでスマート分割
            segments = _smart_split(inner, inner_limit)
            return [f"{latex_delimiter}{seg}{latex_delimiter}" for seg in segments]

        # ------------------------------------------------------------------
        # 正規表現で特殊ブロック（コード/LaTeX）を検出
        # ------------------------------------------------------------------
        pattern = re.compile(
            rf"({re.escape(code_block_delimiter)}.*?{re.escape(code_block_delimiter)}"
            rf"|{re.escape(latex_delimiter)}.*?{re.escape(latex_delimiter)})",
            re.DOTALL,
        )

        blocks: List[str] = []
        last_idx = 0
        for m in pattern.finditer(response):
            if m.start() > last_idx:               # 直前の通常テキスト
                blocks.append(response[last_idx : m.start()])
            blocks.append(m.group(0))              # 特殊ブロック
            last_idx = m.end()
        if last_idx < len(response):               # 残りの通常テキスト
            blocks.append(response[last_idx:])

        # ------------------------------------------------------------------
        # ブロックごとに分割処理
        # ------------------------------------------------------------------
        parts: List[str] = []
        for block in blocks:
            if block.startswith(code_block_delimiter):
                # コードブロック
                if len(block) <= max_length:
                    parts.append(block)
                else:
                    parts.extend(_split_code_block(block, max_length))
            elif block.startswith(latex_delimiter):
                # LaTeX ブロック
                if len(block) <= max_length:
                    parts.append(block)
                else:
                    parts.extend(_split_latex_block(block, max_length))
            else:
                # 通常テキスト
                if len(block) <= max_length:
                    parts.append(block)
                else:
                    parts.extend(_smart_split(block, max_length))

        # 空文字列および空白・改行のみから成る文字列は除外して返す
        return [p for p in parts if p.strip()]

if __name__ == "__main__":
    dotenv.load_dotenv()
    TOKEN = os.getenv('TOKEN')
    sympy.init_printing()
    intents = discord.Intents.default()
    intents.message_content = True
    discord_client = MyClient(intents=intents)
    discord_client.run(TOKEN)