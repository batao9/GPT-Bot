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
import re
import sympy
import asyncio
from typing import Optional, List, Dict, Tuple
from LLM_Models import Models
from prompts import SytemPrompts
from utils import Utils


class MyClient(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dir_path = os.getenv("USER_ATTACHMENTS_DIR") or os.path.join(os.path.dirname(__file__), 'tmp', 'user_attach')
        self.output_dir_path = os.getenv("AGENT_ATTACHMENTS_DIR") or os.path.join(os.path.dirname(__file__), 'tmp', 'agent_attach')
        os.makedirs(self.input_dir_path, exist_ok=True)
        os.makedirs(self.output_dir_path, exist_ok=True)

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

        print(f"user: {message.content}")
        thread, messages = await self.get_thread_and_messages(message)
        
        session_id = message.id
        session_input_dir = os.path.join(self.input_dir_path, str(session_id))
        session_output_dir = os.path.join(self.output_dir_path, str(session_id))
        os.makedirs(session_input_dir, exist_ok=True)
        os.makedirs(session_output_dir, exist_ok=True)
        
        llm_agent: Agent = await Agent.create(
            model_name=Models.get_field(message.channel, "model"),
            provider=Models.get_field(message.channel, "provider"),
            tools=Models.get_field(message.channel, "tools"),
            reasoning_effort=Models.get_field(message.channel, "reasoning_effort"),
            input_dir_path=session_input_dir,
            output_dir_path=session_output_dir
        )

        thread_name_coro = self.generate_thread_name(messages)
        converted_messages_for_agent = await self.convert_message(
            messages,
            provider=Models.get_field(message.channel, "provider"),
            system_prompt=SytemPrompts.prompts['assistant'],
            save_dir_path = session_input_dir
        )
        response_coro = llm_agent.invoke(messages=converted_messages_for_agent)

        task_generate_thread_name = asyncio.create_task(thread_name_coro)
        task_get_response = asyncio.create_task(response_coro)

        try:
            thread_name = await task_generate_thread_name
            if thread and thread_name:
                try:
                    await thread.edit(name=thread_name)
                except discord.errors.NotFound:
                    print(f"スレッド {thread.id} が見つかりませんでした。名前編集前に削除された可能性があります。")
                except discord.errors.Forbidden:
                    print(f"スレッド {thread.id} の名前を編集する権限がありません。")
                except Exception as e:
                    print(f"スレッド名編集中にエラーが発生しました ({thread.id}): {e}")
        except Exception as e:
            print(f"スレッド名生成中にエラーが発生しました: {e}")

        async with thread.typing():
            try:
                response, attachments = await task_get_response
                await self.send_response(
                    thread=thread, 
                    response=response, 
                    attachments=attachments,
                    input_dir_path = session_input_dir,
                    output_dir_path = session_output_dir)
            except Exception as e:
                print(f"エージェントのレスポンス処理または送信中にエラーが発生しました: {e}")
                if thread:
                    try:
                        await self.send_response(
                            thread=thread, 
                            response="エラーが発生しました。処理を完了できませんでした。", 
                            attachments=[], 
                            input_dir_path = session_input_dir,
                            output_dir_path = session_output_dir)
                    except Exception as send_e:
                        print(f"エラーメッセージの送信中にさらにエラーが発生しました: {send_e}")

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
        system_prompt: Optional[str] = None,
        save_dir_path: str = None,
    ) -> List[BaseMessage]:
        """discord.Messageオブジェクトのリストをモデルへの入力形式に変換する。

        Args:
            messages (List[discord.Message]): 変換するdiscord.Messageオブジェクトのリスト。
            provider (str, optional): 使用するモデルプロバイダー ('openai', 'anthropic', 'gemini'など)。Defaults to "openai".
            system_prompt (Optional[str], optional): 追加するシステムプロンプト。Defaults to None.
            save_dir_path: ファイルを保存するディレクトリのパス。Defaults to None.

        Returns:
            List[BaseMessage]: 変換されたメッセージのリスト。
        """
        converted_messages = []
        
        for msg_idx, msg in enumerate(reversed(messages)):
            if msg.is_system():
                continue
            
            current_msg_content_text = msg.content if msg.content is not None else ""
            if msg.author.bot and not msg.attachments and not current_msg_content_text.strip():
                print(f"Skipping empty bot message (no attachments, no content): {msg.id}")
                continue
            
            if msg.author.bot and not msg.attachments:
                converted_messages.append(AIMessage(content=current_msg_content_text))
                continue

            contents = []
            if current_msg_content_text.strip():
                contents.append({"type": "text", "text": current_msg_content_text})
            
            # 添付ファイルの処理
            attachments_for_llm = []
            user_uploaded_files_info = []
            if msg.attachments:
                for attachment_idx, attachment in enumerate(msg.attachments):
                    filename = attachment.filename
                    # download file info
                    if save_dir_path:
                        _, ext = os.path.splitext(filename)
                        downloaded_filename = f"attach{msg_idx:02d}_{attachment_idx:02d}{ext}"
                        download_path = os.path.join(save_dir_path, downloaded_filename)
                    else:
                        download_path = None
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.get(attachment.url) as resp:
                            if resp.status != 200:
                                continue
                            raw = await resp.read()
                            
                            # download file
                            try:
                                if download_path:
                                    with open(download_path, "wb") as f:
                                        f.write(raw)
                                    user_uploaded_files_info.append(
                                        {"origin": filename, "downloaded_path": downloaded_filename}
                                    )
                                    print(f"File downloaded: {filename} -> {downloaded_filename}")
                            except Exception as e:
                                print(f"Error downloading file {filename}: {e}")
                                continue
                            
                            # pdf file to base64
                            if filename.endswith('.pdf') and not msg.author.bot:
                                pdf_base64 = base64.b64encode(raw).decode('utf-8')
                                if provider == "openai":
                                    imgs_base64 = Utils.pdf_url_to_base64_images(attachment.url)
                                    for img in imgs_base64:
                                        attachments_for_llm.append({
                                            "type": "image_url",
                                            "image_url": {"url": f"data:image/png;base64,{img}"},
                                        })
                                elif provider == "anthropic" or "gemini":
                                    attachments_for_llm.append({
                                        "type": "file",
                                        "source_type": "base64",
                                        "mime_type": "application/pdf",
                                        "data": pdf_base64,
                                    })
                            # image file to base64
                            elif filename.endswith(('.png', '.jpg', '.gif', 'webp', 'jpeg')):
                                if filename.startswith('latex_formula_'):
                                    continue
                                img_base64 = base64.b64encode(raw).decode('utf-8')
                                extension = filename.split('.')[-1]
                                attachments_for_llm.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/{extension};base64,{img_base64}"},
                                })
                            # text file 
                            elif Utils.is_probably_text(raw):
                                # text fileはAgentがツールで読む想定
                                pass
                            # other file
                            else:
                                pass
            
            # 添付ファイル情報をメッセージに追加
            if user_uploaded_files_info:
                file_info_text = "\n\navailable files:\n"
                for info in user_uploaded_files_info:
                    file_info_text += f"- path: {info['downloaded_path']} (original filename: {info['origin']})\n"
                
                # contentsの最初のtext要素にファイル情報を追記
                if contents and contents[0]["type"] == "text":
                    contents[0]["text"] += file_info_text
                else: # text要素がない場合は先頭に追加
                    contents.insert(0, {"type": "text", "text": file_info_text.strip()})

            # メッセージの変換
            if not msg.author.bot:
                converted_messages.append(HumanMessage(contents + attachments_for_llm))
            if msg.author.bot:
                if provider == "gemini":
                    # geminiはmodelが画像を送信することも想定している
                    converted_messages.append(AIMessage(contents + attachments_for_llm))
                else:
                    converted_messages.append(AIMessage(contents))
                    # ほかのproviderでもbotが画像を送信した場合の処理を追加予定
                    # code interpreterの結果を画像で送信する場合など
                    

        # システムプロンプトの追加
        if system_prompt:
            converted_messages.insert(0, SystemMessage(content=system_prompt))

        return merge_message_runs(converted_messages)


    async def send_response(
        self,
        thread: discord.TextChannel,
        response: str,
        attachments: List[str] = [],
        input_dir_path: str = None,
        output_dir_path: str = None
    ) -> None:
        """レスポンスを指定されたスレッドに送信する。

        レスポンスをDiscordの文字数制限に合わせて分割し、LaTeX数式は画像に変換して送信する。

        Args:
            thread (discord.TextChannel): レスポンスを送信するスレッドまたはテキストチャンネル。
            response (str): 送信するレスポンス文字列。
            attachments (List[str], optional): 添付ファイルのパスのリスト。Defaults to [].
            input_dir_path: userの添付ファイルを保存するディレクトリのパス。Defaults to None.
            output_dir_path: agentの添付ファイルを保存するディレクトリのパス。Defaults to None.
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
                        if buff.strip():
                            await thread.send(buff)
                        buff = ''
                    buff += part
            else:
                if len(buff) + len(part) > MAX_LENGTH:
                    await thread.send(buff)
                    buff = ''
                buff += part

        # 残りのバッファ内容と、引数で渡された添付ファイルを送信
        if buff.strip() or attachments:
            files_to_send = []
            if attachments:
                for file_path in attachments:
                    files_to_send.append(discord.File(file_path))
            
            content_to_send = buff if buff.strip() else None
            await thread.send(content=content_to_send, files=files_to_send)
        
        # self.input_dir_path と self.output_dir_path の中身を削除
        out_dir = output_dir_path or self.output_dir_path
        in_dir = input_dir_path or self.input_dir_path
        dirs_to_cleanup = [d for d in (out_dir, in_dir) if d and os.path.isdir(d)]
        for dir_path in dirs_to_cleanup:
            print(f"Cleaning up directory: {dir_path}")
            for item_name in os.listdir(dir_path):
                item_path = os.path.join(dir_path, item_name)
                try:
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        import shutil
                        shutil.rmtree(item_path)
                except Exception as e:
                    print(f'Failed to delete {item_path}. Reason: {e}')
            
            try:
                os.rmdir(dir_path)
            except Exception as e:
                print(f'Failed to remove directory {dir_path}. Reason: {e}')


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