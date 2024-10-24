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


class GPT_Models:
    # channel name
    gpt4o_channel = 'gpt-4o'
    gpt4omini_channel = 'gpt-4o-mini'
    gpt4_channel = 'gpt-4'
    gpt35_channel = 'gpt-3'
    o1_channel = 'o1-preview'
    o1mini_channel = 'o1-mini'
    
    # model name
    gpt4o = 'gpt-4o'
    gpt4omini = 'gpt-4o-mini'
    gpt4 = 'gpt-4-turbo'
    gpt35 = 'gpt-3.5-turbo'
    o1_preview = 'o1-preview'
    o1mini = 'o1-mini'
    
    # mapping
    mappling = {
                gpt4_channel: gpt4,
                gpt4o_channel: gpt4o,
                gpt4omini_channel: gpt4omini,
                gpt35_channel: gpt35,
                o1_channel: o1_preview,
                o1mini: o1mini_channel
            }
    
class SytemPrompts:
    prompts = {
        'assistant': 
            {
                "role": "system",
                "content": [
                    {
                    "type": "text",
                    "text": """
                        あなたはAIアシスタントです。userのメッセージに対して返答を行ってください。
                        応答の際は以下のルールに従ってください。
                        - userから特に指示がない場合は日本語で返答を行う
                        - Markdown形式での応答を行う
                        - Latex math symbolsを含む数式は$$で囲む
                        - プログラムの修正を行う場合は全体をメッセージに含めず，修正箇所のみを示す
                        - プログラムコードを含む場合は関数ごとなどで細かくコードブロックを分け、2000字を超える長いコードブロックは避ける
                        """
                    }
                ]
            },
        'thread':
            {
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": """
                以下のルールを守ってスレッドのタイトルを生成してください。
                - userのメッセージを元にスレッドのタイトルを生成する
                - メッセージに対して直接返答を行わない
                - タイトルは20文字程度以内
                - タイトルは日本語で記載
                - タイトルのみを返答する
                """
                }
            ]
        }
    }


# latex_to_image関数の定義
def latex_to_image(latex_code, save_path=None):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        sympy.preview(latex_code, viewer='file', filename=tmpfile.name, euler=False,
                      dvioptions=["-T", "tight", "-z", "0", "--truecolor", "-D 600"], 
                      dpi=60)
        if save_path:
            os.rename(tmpfile.name, save_path)
            return save_path
        else:
            return tmpfile.name

# メッセージからcontent, img_urlsを取得
async def process_message_content(msg):
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

# ChatGPTのレスポンスを取得する
async def get_gpt_response(messages, model, system_message=None):
    prompt = []
    for msg in messages:
        if msg.is_system(): 
            # systemメッセージは無視
            continue
        if msg.author.bot:
            role = 'assistant'
        else:
            role = 'user'
        
        # content, img_urlsを取得
        try:
            content, img_urls = await process_message_content(msg)
        except Exception as e:
            print(f"Error processing message content: {e}")
            continue
        
        # roleをuserに
        prompt.insert(0, {
            "role": role,
            "content": [
                { "type": "text", "text": content}
            ]})

        # 画像のURLを追加
        if role == 'user':
            for url in img_urls:
                prompt[0]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": url},
                })

    # system messageを追加
    if system_message is not None:
        prompt.insert(0, system_message)

    # レスポンスを生成
    response = openai.chat.completions.create(
        model=model,  
        temperature=0.7,
        top_p=0.9,
        messages=prompt
    )
    return response.choices[0].message.content


class MyClient(discord.Client):
    async def on_ready(self):
        print(f'Logged on as {self.user}!')

    async def on_message(self, message):
        # メッセージ送信者がボット自身か、他のボット、またはシステムメッセージの場合は無視
        if message.author == self.user or message.author.bot or message.is_system():
            return

        # チャンネル名に基づいてモデルを選択
        model = self.get_model_based_on_channel(message.channel)
        if not model:
            return

        # スレッドが存在する場合は過去のメッセージを取得し、存在しない場合は新規スレッドを作成
        print ('user:' + message.content)
        thread, messages = await self.get_thread_and_messages(message)
        
        # スレッド名を生成, GPTのレスポンスを取得
        thread_name_future = self.generate_thread_name(messages)
        gpt_response_future = get_gpt_response(messages, model, SytemPrompts.prompts['assistant'])
        thread_name, gpt_response = await asyncio.gather(thread_name_future, gpt_response_future)

        # スレッド名を更新, GPTのレスポンスを送信
        if thread and thread_name:
            await thread.edit(name=thread_name)
        await self.send_response(thread, gpt_response)

    # チャンネル名に基づいてモデルを返す
    def get_model_based_on_channel(self, channel):
        channel_name = getattr(channel, 'name', None)
        parent_name = getattr(channel.parent, 'name', None) if hasattr(channel, 'parent') else None

        model_mapping = GPT_Models.mappling
        return model_mapping.get(channel_name) or model_mapping.get(parent_name)
    
    # メッセージを送るスレッドと，スレッドのメッセージ履歴を返す
    async def get_thread_and_messages(self, message):
        # スレッドでメッセージが送られた場合
        if isinstance(message.channel, discord.Thread):
            thread = message.channel
            messages = [msg async for msg in thread.history(limit=50)]
            messages.append(await thread.parent.fetch_message(thread.id))
        # チャンネルでメッセージが送られた場合
        elif isinstance(message.channel, discord.TextChannel):
            thread = await message.create_thread(name="スレッド名生成中...")  # 一時的なスレッド名で作成
            messages = [message]
        else:
            thread = None
            messages = []
        return thread, messages
    
    # スレッドの名前を付ける
    async def generate_thread_name(self, messages):
        if len(messages) != 1:
            return None
        
        thred_name = await get_gpt_response(messages, GPT_Models.gpt4omini, SytemPrompts.prompts['thread'])
        
        if len(thred_name) > 99: # タイトルが99文字を超える場合は切り捨て
            thred_name = thred_name[:99]
        
        return thred_name
            
    
    # responseを区切る
    def split_response(self, response, code_block_delimiter="```", latex_delimiter="$$", max_length=2000):
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
    

    # responseを送信
    async def send_response(self, thread, response):
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
    openai.api_key = os.getenv('OPENAI_API_KEY')
    sympy.init_printing()

    intents = discord.Intents.default()
    intents.message_content = True
    client = MyClient(intents=intents)
    client.run(TOKEN)
