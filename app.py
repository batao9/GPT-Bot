import discord
import openai
import dotenv
import os
import re
import aiohttp
import pymupdf4llm
import sympy
import tempfile
import asyncio
import json
import base64
from pydantic import BaseModel
from typing import Optional, Tuple, List, Dict
import charset_normalizer


class GPT_Models:
    @staticmethod
    def load_mapping(file_path: str = 'models.json') -> Dict:
        """モデルのマッピングをファイルから読み込む"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: {file_path} が見つかりません。")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error: JSONの読み込みに失敗しました: {e}")
            return {}

    # マッピングを初期化
    mappling: Dict = load_mapping.__func__()

    @staticmethod
    def get_field(channel: discord.TextChannel, key: str) -> Optional[str]:
        """チャンネルに対応するモデルを取得する"""
        channel_name = getattr(channel, 'name', None)
        parent_name = getattr(channel.parent, 'name', None) if hasattr(channel, 'parent') else None
        try:
            mapping = GPT_Models.mappling.get(channel_name) or GPT_Models.mappling.get(parent_name)
            if mapping:
                return mapping[key]
        except:
            return None
    
    @staticmethod
    def is_channel_configured(channel_name: str) -> bool:
        """指定されたチャンネル名が設定されているか確認する"""
        return (channel_name in GPT_Models.mappling)
    

class SytemPrompts:
    prompts = { 'assistant': """
あなたはAIアシスタントです。userのメッセージに対して返答を行ってください。
応答の際は以下のルールに従ってください。
- userから特に指示がない場合は同じ言語で返答を行う
- メールの作成の依頼では、拝啓・敬具などは含めない
- Latex math symbolsを含む数式は$$で囲む
    例：
    ニュートン流体（一般的な液体や気体を想定）で、圧縮性を考慮すると、次のような形になります。
    $$
    \rho \left( \frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u} \right)
    = -\nabla p + \mu \nabla^2 \mathbf{u} + \mathbf{f}
    $$  # メインの数式は$$で囲む
    - $\rho$ : 流体の密度   # 記号の説明などの場合は$で囲む
    - $\mathbf{u}$ : 流体の速度ベクトル # 記号の説明などの場合は$で囲む
    ...
- プログラムの修正を行う場合は基本的に修正箇所のみを示す
- プログラムコードを含む場合は関数ごとなどで細かくコードブロックを分け、長いコードブロックは可能な限り避ける
    例：
    修正ポイントは2カ所です。
    ```python   # diff pythonは使わない
    # 修正箇所
    def func():
    -   psss
    +   func2()
    ```
    ```python
    # 追加関数
    def func2():
        print("Hello World")
    ```
""",
        'thread': """
あなたはスレッド管理システムです。
以下のルールを守ってスレッドのタイトルを生成してください。
- userのメッセージを元にスレッドのタイトルを生成する
- メッセージに対して直接返答を行わない
- タイトルは20文字程度以内
"""
    }


def latex_to_image(latex_code: str, save_path: str = None) -> str:
    """LaTeXコードを画像に変換する"""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        sympy.preview(latex_code, viewer='file', filename=tmpfile.name, euler=False,
                      dvioptions=["-T", "tight", "-z", "0", "--truecolor", "-D 600"], 
                      dpi=60)
        if save_path:
            os.rename(tmpfile.name, save_path)
            return save_path
        else:
            return tmpfile.name


def is_probably_text(raw: bytes, threshold: float = 0.75) -> bool:
    """
    1. NULL バイトがあれば False
    2. charset_normalizer でベストマッチを取得
    3. language_confidence（言語推定の信頼度）が閾値以上なら True
    """
    if b'\x00' in raw:
        return False
    match = charset_normalizer.detect(raw)
    if not match:
        return False
    return match['confidence'] >= threshold


def detect_encoding(raw: bytes) -> str:
    """
    charset-normalizer でエンコーディング推定．
    見つからなければ UTF‑8 を返す。
    """
    best = charset_normalizer.from_bytes(raw).best()
    if best and best.encoding:
        enc = best.encoding.replace('_', '-').lower()
        return enc
    return 'utf-8'


async def process_message_content(
    msg: discord.Message,
    is_img_input: bool = False
) -> Tuple[str, List[str], List[Dict[str, str]]]:
    """メッセージからコンテンツと画像URLを取得する"""
    content = msg.content
    pdf_content: List[Dict[str, str]] = []
    img_urls: List[str] = []
    if msg.attachments:
        for attachment in msg.attachments:
            filename = attachment.filename
            url = attachment.url
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        continue
                    raw = await resp.read()
                    # pdf file
                    if filename.endswith('.pdf'):
                        if is_img_input:
                            base64_str = base64.b64encode(raw).decode('utf-8')
                            pdf_content.append({
                                "filename": filename,
                                "base64_str": base64_str
                            })
                        else:
                            with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as tmp_pdf:
                                tmp_pdf.write(raw)
                                tmp_pdf.flush()
                                file_text = pymupdf4llm.to_markdown(tmp_pdf.name)
                                pdf_str = (
                                    f"{pdf_str}"
                                    f"\n__file_start__{filename=}\n"
                                    f"{file_text}\n__file_end__"
                                )
                    # img file
                    elif is_img_input and filename.endswith(('.png', '.jpg', '.gif', 'webp')):
                        img_urls.append(attachment.url)
                    # text file
                    elif is_probably_text(raw):
                        encoding = detect_encoding(raw)
                        try:
                            file_text = raw.decode(encoding)
                            content = (
                                f"{content}"
                                f"\n__file_start__{filename=}\n"
                                f"{file_text}\n__file_end__"
                            )
                        except UnicodeError as e:
                            content = (
                                f"{content}"
                                f"\n__file_start__{filename=}\n"
                                f"{e}\n__file_end__"
                            )
                    # other file
                    else:
                        content = (
                            f"{content}"
                            f"\n__file_start__{filename=}\n"
                            "Cannot open file. Only image files and files that can be opened in text format are supported.\n"
                            "__file_end__"
                        )
    return content, img_urls, pdf_content


async def get_gpt_response(
    messages: List[discord.Message],
    channel: discord.TextChannel
) -> str:
    """ChatGPTのレスポンスを取得する"""
    model = GPT_Models.get_field(channel, 'model')
    reasoning_effort = GPT_Models.get_field(channel, 'reasoning_effort') or None
    is_use_web_search = GPT_Models.get_field(channel, 'web_search') or False
    is_use_code_interpreter = GPT_Models.get_field(channel, 'code_interpreter') or False
    is_img_input = GPT_Models.get_field(channel, 'img_input') or False
    
    prompt = []
    for msg in messages:
        if msg.is_system():
            continue
        if msg.author.bot:
            role = 'assistant'
            typ = 'output_text'
        else:
            role = 'user'
            typ = 'input_text'
        
        try:
            content, img_urls, pdf_content = await process_message_content(msg, is_img_input)
        except Exception as e:
            print(f"Error processing message content: {e}")
            continue
        
        prompt.insert(0, {
            "role": role,
            "content": [{"type": typ, "text": content}]
        })

        if is_img_input and role == 'user':
            for url in img_urls:
                prompt[0]["content"].append({
                    "type": "input_image",
                    "image_url": url,
                })
            for pdf in pdf_content:
                pdf_base64 = pdf["base64_str"]
                prompt[0]["content"].append({
                    "type": "input_file",
                    "filename": pdf["filename"],
                    "file_data": f"data:application/pdf;base64,{pdf_base64}",
                })

    tools = []
    if is_use_web_search:
        tools.append({"type": "web_search_preview"})
    if is_use_code_interpreter:
        tools.append({"type": "code_interpreter"})
    
    response = openai_client.responses.create(
        model=model,
        instructions=SytemPrompts.prompts['assistant'],
        reasoning={"effort": reasoning_effort} if reasoning_effort else None,
        input=prompt,
        tools=tools,
        tool_choice="auto",
        store=False
    )
    return response.output_text


async def get_thread_name(
    messages: List[discord.Message]
) -> str:
    """スレッド名を生成する"""
    class ThreadName(BaseModel):
        thread_name: str
    
    model = 'gpt-4.1-nano'
    is_img_input = GPT_Models.get_field(messages[0].channel, 'img_input') or False
    msg = messages[0]
    
    try:
        content, img_urls, pdf_content = await process_message_content(msg, is_img_input)
    except Exception as e:
        print(f"Error processing message content: {e}")
        return 'title gen error'
        
    prompt = [{
        "role": "user",
        "content": [{"type": "input_text", "text": content}]
    }]
    if is_img_input:
        for url in img_urls:
            prompt[0]["content"].append({
                "type": "input_image",
                "image_url": url,
            })
        for pdf in pdf_content:
            pdf_base64 = pdf["base64_str"]
            prompt[0]["content"].append({
                "type": "input_file",
                "filename": pdf["filename"],
                "file_data": f"data:application/pdf;base64,{pdf_base64}",
            })
        
    response = openai_client.responses.parse(
        model=model,
        instructions=SytemPrompts.prompts['thread'],
        input=prompt,
        text_format=ThreadName,
        store=False
    )
    
    return response.output_parsed.thread_name


class MyClient(discord.Client):
    async def on_ready(self) -> None:
        """クライアントが準備完了したときに呼び出される"""
        print(f'Logged on as {self.user}!')

    async def on_message(self, message: discord.Message) -> None:
        """メッセージを受信したときに呼び出される"""
        if (
            message.author == self.user or
            message.author.bot or
            message.is_system() or
            GPT_Models.get_field(message.channel, 'model') is None
        ):
            return

        print('user:' + message.content)
        thread, messages = await self.get_thread_and_messages(message)
        
        thread_name_future = self.generate_thread_name(messages)
        gpt_response_future = get_gpt_response(messages, message.channel)
        thread_name, gpt_response = await asyncio.gather(
            thread_name_future,
            gpt_response_future
        )

        if thread and thread_name:
            await thread.edit(name=thread_name)
        await self.send_response(thread, gpt_response)
    
    async def get_thread_and_messages(
        self,
        message: discord.Message
    ) -> Tuple[Optional[discord.Thread], List[discord.Message]]:
        """メッセージを送るスレッドと，スレッドのメッセージ履歴を返す"""
        if isinstance(message.channel, discord.Thread):
            thread = message.channel
            messages = [msg async for msg in thread.history(limit=100)]
            messages.append(await thread.parent.fetch_message(thread.id))
        elif (
            isinstance(message.channel, discord.TextChannel) and
            GPT_Models.is_channel_configured(message.channel.name)
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
        """スレッドの名前を付ける"""
        if len(messages) != 1:
            return None
        
        thread_name = await get_thread_name(messages)
        return thread_name[:99] if len(thread_name) > 99 else thread_name
            
    
    def split_response(
        self,
        response: str,
        code_block_delimiter: str = "```",
        latex_delimiter: str = "$$",
        max_length: int = 2000,
    ) -> List[str]:
        """
        Discord の 2000 文字制限を守りつつレスポンスを分割する。

        ・```...``` で囲まれたコードブロック  
        ・$$...$$   で囲まれた LaTeX ブロック  
        └ いずれも max_length を超える場合は分割する  
            ・コードブロックは改行単位で分割し、各チャンクを  
            <code_block_delimiter><lang>\n ... \n<code_block_delimiter>  
            で再ラップして壊れないようにする  
            ・LaTeX ブロックは $$ ... $$ で再ラップして送る  
        ・通常テキストは「空白/改行/句読点」で賢く分割 (_smart_split)
        """
        # ------------------------------------------------------------------
        # 共通ユーティリティ
        # ------------------------------------------------------------------
        def _smart_split(text: str, limit: int) -> List[str]:
            """通常テキストを limit 以下で自然な区切りで分割"""
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
            """
            コードブロックを改行単位で分割し、再ラップして返す
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
            """
            LaTeX ブロックを分割し、再ラップして返す
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
    

    async def send_response(
        self,
        thread: discord.TextChannel,
        response: str
    ) -> None:
        """レスポンスを送信する"""
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
                    image_path = latex_to_image(latex_code)
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


if __name__ == '__main__':
    dotenv.load_dotenv()
    TOKEN = os.getenv('TOKEN')
    sympy.init_printing()

    openai_client = openai.OpenAI()
    intents = discord.Intents.default()
    intents.message_content = True
    client = MyClient(intents=intents)
    client.run(TOKEN)
