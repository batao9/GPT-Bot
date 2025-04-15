import discord
import openai
import dotenv
import os
import aiohttp
import io
from pdfminer.high_level import extract_text
import sympy
import tempfile
import asyncio
import json
from pydantic import BaseModel


class GPT_Models:
    @staticmethod
    def load_mapping(file_path='models.json'):
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
    mappling = load_mapping.__func__()

    @staticmethod
    def get_field(channel: discord.TextChannel, key: str):
        """チャンネルに対応するモデルを取得する"""
        channel_name = getattr(channel, 'name', None)
        parent_name = getattr(channel.parent, 'name', None) if hasattr(channel, 'parent') else None
        mapping = GPT_Models.mappling.get(channel_name) or GPT_Models.mappling.get(parent_name)
        try:
            if mapping:
                return mapping[key]
        except KeyError:
            return None
    
    @staticmethod
    def is_channel_configured(channel_name: str):
        """指定されたチャンネル名が設定されているか確認する"""
        return (channel_name in GPT_Models.mappling)
    
class SytemPrompts:
    prompts = {
        'assistant': 
                    """
                    あなたはAIアシスタントです。userのメッセージに対して返答を行ってください。
                    応答の際は以下のルールに従ってください。
                    - userから特に指示がない場合は日本語で返答を行う
                    - 必要な場合はweb上のリソースを参照する
                    - Latex math symbolsを含む数式は$$で囲む
                    - プログラムの修正を行う場合は全体をメッセージに含めず，修正箇所のみを示す
                    - プログラムコードを含む場合は関数ごとなどで細かくコードブロックを分け、2000字を超える長いコードブロックは避ける
                    """,
        'thread':
                """
                以下のルールを守ってスレッドのタイトルを生成してください。
                - userのメッセージを元にスレッドのタイトルを生成する
                - メッセージに対して直接返答を行わない
                - タイトルは20文字程度以内
                - タイトルは日本語で記載
                - タイトルのみを返答する
                """
    }


def latex_to_image(latex_code: str, save_path=None):
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

async def process_message_content(msg: discord.Message):
    """メッセージからコンテンツと画像URLを取得する"""
    content = msg.content
    img_urls = []
    if msg.attachments:
        for attachment in msg.attachments:
            filename = attachment.filename
            if filename.endswith(('.txt', '.py', '.md', '.csv', '.c', '.cpp', '.java', 'pdf')):
                url = attachment.url
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            data = io.BytesIO(await resp.read())
                            file_text = extract_text(data) if filename.endswith('pdf') else data.read().decode('utf-8')
                            content = f'{content}\n{filename}\n{file_text}'
            elif filename.endswith(('.png', '.jpg', '.gif', 'webp')):
                img_urls.append(attachment.url)
    return content, img_urls

async def get_gpt_response(messages: list[discord.Message], channel: discord.TextChannel):
    """ChatGPTのレスポンスを取得する"""
    model = GPT_Models.get_field(channel, 'model')
    web_search = GPT_Models.get_field(channel, 'web_search') or False
    img_input = GPT_Models.get_field(channel, 'img_input') or False
    
    prompt = []
    for msg in messages:
        if msg.is_system(): 
            # systemメッセージは無視
            continue
        if msg.author.bot:
            role = 'assistant'
            type = 'output_text'
        else:
            role = 'user'
            type = 'input_text'
        
        # content, img_urlsを取得
        try:
            content, img_urls = await process_message_content(msg)
        except Exception as e:
            print(f"Error processing message content: {e}")
            continue
        
        prompt.insert(0, {
            "role": role,
            "content": [
                { "type": type, "text": content}
            ]})

        # 画像のURLを追加
        if img_input and role == 'user':
            for url in img_urls:
                prompt[0]["content"].append({
                    "type": "input_image",
                    "image_url": url,
                })

    response = openai_clinet.responses.create(
        model=model,
        instructions=SytemPrompts.prompts['assistant'],
        input=prompt,
        tools=[{"type": "web_search_preview"}] if web_search else None,
        tool_choice="auto",
        store=False
    )
    return response.output_text

async def get_thread_name(messages: list[discord.Message]):
    """スレッド名を生成する"""
    class ThreadName(BaseModel):
        thread_name: str
    
    model = 'gpt-4o-mini'
    
    msg = messages[0]
    try:
        content, img_urls = await process_message_content(msg)
    except Exception as e:
        print(f"Error processing message content: {e}")
        
    prompt = [{"role": "user",
            "content": [
                { "type": "text", "text": content}
                ]}]
    for url in img_urls:
        prompt[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": url},
        })
    sys_prompts = {"role": "developer",
                    "content": [{
                        "type": "text",
                        "text": SytemPrompts.prompts['thread']
                        }]
                    }
    prompt.insert(0, sys_prompts)
        
    # レスポンスを生成
    response = openai.beta.chat.completions.parse(
        model=model,
        messages=prompt,
        response_format=ThreadName
    )
    
    return response.choices[0].message.parsed.thread_name    

class MyClient(discord.Client):
    async def on_ready(self):
        """クライアントが準備完了したときに呼び出される"""
        print(f'Logged on as {self.user}!')

    async def on_message(self, message: discord.Message):
        """メッセージを受信したときに呼び出される"""
        if message.author == self.user or message.author.bot or message.is_system():
            return

        # スレッドが存在する場合は過去のメッセージを取得し、存在しない場合は新規スレッドを作成
        print ('user:' + message.content)
        thread, messages = await self.get_thread_and_messages(message)
        
        # スレッド名を生成, GPTのレスポンスを取得
        thread_name_future = self.generate_thread_name(messages)
        gpt_response_future = get_gpt_response(messages, message.channel)
        thread_name, gpt_response = await asyncio.gather(thread_name_future, gpt_response_future)

        # スレッド名を更新, GPTのレスポンスを送信
        if thread and thread_name:
            await thread.edit(name=thread_name)
        await self.send_response(thread, gpt_response)
    
    async def get_thread_and_messages(self, message: discord.Message):
        """メッセージを送るスレッドと，スレッドのメッセージ履歴を返す"""
        if isinstance(message.channel, discord.Thread):
            thread = message.channel
            messages = [msg async for msg in thread.history(limit=50)]
            messages.append(await thread.parent.fetch_message(thread.id))
        # チャンネルでメッセージが送られた場合
        elif isinstance(message.channel, discord.TextChannel) and GPT_Models.is_channel_configured(message.channel.name):
            thread = await message.create_thread(name="スレッド名生成中...")
            messages = [message]
        else:
            thread = None
            messages = []
        return thread, messages
    
    async def generate_thread_name(self, messages: list[discord.Message]):
        """スレッドの名前を付ける"""
        if len(messages) != 1:
            return None
        
        thred_name = await get_thread_name(messages)
        
        if len(thred_name) > 99:
            thred_name = thred_name[:99]
        
        return thred_name
            
    
    def split_response(self, response: str, code_block_delimiter="```", latex_delimiter="$$", max_length=2000):
        """レスポンスを区切る"""
        parts = []
        current_index = 0
        while current_index < len(response):
            start_index_code = response.find(code_block_delimiter, current_index)
            start_index_latex = response.find(latex_delimiter, current_index)
            
            if start_index_code == -1:
                start_index = start_index_latex
            elif start_index_latex == -1:
                start_index = start_index_code
            else:
                start_index = min(start_index_code, start_index_latex)
            
            if start_index == -1:
                parts.append(response[current_index:])
                break

            if start_index == start_index_code:
                end_index = response.find(code_block_delimiter, start_index + len(code_block_delimiter))
                delimiter_length = len(code_block_delimiter)
            else:
                end_index = response.find(latex_delimiter, start_index + len(latex_delimiter))
                delimiter_length = len(latex_delimiter)
            
            if end_index == -1:
                parts.append(response[current_index:])
                break

            end_index += delimiter_length

            if start_index > current_index:
                parts.extend([response[current_index:start_index][i:i+max_length]
                            for i in range(0, len(response[current_index:start_index]), max_length)])
            parts.append(response[start_index:end_index])

            current_index = end_index

        return parts
    

    async def send_response(self, thread: discord.TextChannel, response: str):
        """レスポンスを送信する"""
        print (f'bot:{response}')
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
                if len(buff) > 0: 
                    await thread.send(buff)
                    buff = ''
                latex_code = part
                try:
                    image_path = latex_to_image(latex_code)
                    await thread.send(file=discord.File(image_path))
                    os.remove(image_path)
                except Exception as e:
                    print(f'latex_to_image error: {e}')
                    await thread.send(latex_code)

            else:
                # パートが最大長を超えている場合はさらに分割
                if len(part) > MAX_LENGTH:
                    # 一旦送信
                    if len(buff) > 0:
                        await thread.send(buff)
                        buff = ''
                    for i in range(0, len(part), MAX_LENGTH):
                        # 2000字ごとに送信
                        buff = ''
                        buff += part[i:i+MAX_LENGTH]
                        if len(buff) >= MAX_LENGTH:
                            await thread.send(part[i:i+MAX_LENGTH])
                    
                else:
                    if len(buff) + len(part) > MAX_LENGTH:
                        # 2000字を超える場合は一旦送信
                        await thread.send(buff)
                        buff = ''
                        buff += part
                    else:
                        # 2000字を超えない場合はバッファに貯める
                        buff += part
                        
        # 残りを送信
        if len(buff) > 0:
            await thread.send(buff)


if __name__ == '__main__':
    dotenv.load_dotenv()
    TOKEN = os.getenv('TOKEN')
    sympy.init_printing()

    openai_clinet = openai.OpenAI()
    intents = discord.Intents.default()
    intents.message_content = True
    client = MyClient(intents=intents)
    client.run(TOKEN)
