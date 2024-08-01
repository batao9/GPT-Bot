import discord
import openai
import dotenv
import os
import aiohttp
import io
from pdfminer.high_level import extract_text

CHANNEL_NAME_GPT4o = 'gpt-4o'
CHANNEL_NAME_GPT4o_MINI = 'gpt-4o-mini'
CHANNEL_NAME_GPT4 = 'gpt-4'
CHANNEL_NAME_GPT35 = 'gpt3'
MODEL_GPT4o = 'gpt-4o'
MODEL_GPT4o_MINI = 'gpt-4o-mini'
MODEL_GPT4 = 'gpt-4-turbo'
MODEL_GPT35 = 'gpt-3.5-turbo'

dotenv.load_dotenv()
TOKEN = os.getenv('TOKEN')
openai.api_key = os.getenv('OPENAI_API_KEY')


# ChatGPTのレスポンスを取得する
async def get_gpt_response(messages, model, system_message=None):
    prompt = []
    for msg in messages:
        # systemメッセージは無視
        if msg.is_system():
            continue
        # メッセージの中身を取り出して，APIに投げる形に変換
        if msg.author.bot:
            # botからのメッセージはroleをassistantに
            prompt.insert(0, {
                "role": "assistant",
                "content": [
                    {
                        'type': 'text',
                        'text': msg.content
                    }
                ]
            })
        else:
            content = msg.content
            img_urls = []
            # 添付ファイルがある場合はcontentに追加
            if msg.attachments:
                for attachment in msg.attachments:
                    # 添付ファイルのファイル名から拡張子を取得
                    filename = attachment.filename
                    if filename.endswith(('.txt', '.py', '.md', '.csv', '.c', '.cpp', '.java', 'pdf')):  # txt形式のファイル
                        # 添付ファイルのURLを取得
                        url = attachment.url
                        # 添付ファイルの内容を非同期でダウsンロード
                        async with aiohttp.ClientSession() as session:
                            async with session.get(url) as resp:
                                if resp.status == 200:
                                    # ダウンロードした内容をメモリ上に保持
                                    data = io.BytesIO(await resp.read())
                                    # テキストとして読み込み（エンコーディングに注意）
                                    file_text = ''
                                    if filename.endswith(('pdf')):
                                        file_text = extract_text(data)
                                    else:
                                        file_text = data.read().decode('utf-8')
                                    # ファイルの内容を結合
                                    content = f'{content}\n{filename}\n{file_text}'
                    
                    elif filename.endswith(('.png', '.jpg', '.jpeg')):  # 画像の場合
                        img_urls.append(attachment.url)
            
            # roleをuserに
            prompt.insert(0, {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": content
                    }
                ]
            })

            # 画像のURLを追加
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
        
        # GPTのレスポンスを取得し、送信
        system_message = {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": """
                        あなたはAIアシスタントです。userのメッセージに対して返答を行ってください。
                        応答の際は以下のルールに従ってください。
                        - userから特に指示がない場合は日本語で返答を行う
                        - Markdown形式での応答を行う
                        - 数式を含む場合は$$で数式を囲む数式記法を使用する
                        - プログラムの修正を行う場合は全体をメッセージに含めず，修正箇所のみを示す
                        - プログラムコードを含む場合は関数ごとなどで細かくコードブロックを分け、2000字を超える長いコードブロックは避ける
                        """
                }
            ]
        }
        
        gpt_response_future = get_gpt_response(messages, model, system_message)  # GPTのレスポンスを非同期で取得

        # GPTのレスポンスを取得し、送信
        gpt_response = await gpt_response_future
        await self.send_response_in_parts(thread, gpt_response)


    # チャンネル名に基づいてモデルを返す
    def get_model_based_on_channel(self, channel):
        if hasattr(channel, 'name'):
            if channel.name == CHANNEL_NAME_GPT4:
                return MODEL_GPT4
            elif channel.name == CHANNEL_NAME_GPT4o:
                return MODEL_GPT4o
            elif channel.name == CHANNEL_NAME_GPT4o_MINI:
                return MODEL_GPT4o_MINI
            elif channel.name == CHANNEL_NAME_GPT35:
                return MODEL_GPT35
            
        if hasattr(channel, 'parent'):
            if channel.parent.name == CHANNEL_NAME_GPT4:
                return MODEL_GPT4
            elif channel.parent.name == CHANNEL_NAME_GPT4o:
                return MODEL_GPT4o
            elif channel.parent.name == CHANNEL_NAME_GPT4o_MINI:
                return MODEL_GPT4o_MINI
            elif channel.parent.name == CHANNEL_NAME_GPT35:
                return MODEL_GPT35
        return None
    
    
    # メッセージを送るスレッドと，スレッドのメッセージ履歴を返す
    async def get_thread_and_messages(self, message):
        # スレッドでメッセージが送られた場合
        if isinstance(message.channel, discord.Thread):
            thread = message.channel
            messages = [msg async for msg in thread.history(limit=30)]
            # オリジナルメッセージを追加
            messages.append(await thread.parent.fetch_message(thread.id))
        # チャンネルでメッセージが送られた場合
        elif isinstance(message.channel, discord.TextChannel):
            thread_name_future = self.generate_thread_name([message])  # スレッド名の生成を非同期で開始
            thread = await message.create_thread(name="スレッド名生成中...")  # 一時的なスレッド名で作成
            messages = [message]
            thread_name = await thread_name_future  # スレッド名の生成を待機
            await thread.edit(name=thread_name)  # スレッド名を更新
        else:
            thread = None
            messages = []
        return thread, messages
    
    # スレッドの名前を付ける
    async def generate_thread_name(self, messages):
        # GPT-4oでスレッド名を生成
        system_message = {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": """
                        以下のルールを守ってスレッドのタイトルを生成してください。
                        - userのメッセージを元にスレッドのタイトルを生成する
                        - メッセージに対して直接返答を行わない
                        - タイトルは10文字程度
                        - タイトルは日本語で記載
                        - タイトルのみを返答する
                        """
                }
            ]
        }
        thred_name = await get_gpt_response(messages, MODEL_GPT4o_MINI, system_message)
        
        if len(thred_name) > 15: # タイトルが15文字を超える場合は15文字に切り捨て
            thred_name = thred_name[:15]
        
        return thred_name
            
    
    
    # responseをコードブロックで区切って送信，2000文字超える場合はさらに区切って送信（Discordの文字数上限は2000文字）
    async def send_response_in_parts(self, thread, response):
        print (f'bot:{response}')
        CODE_BLOCK_DELIMITER = "```"
        MAX_LENGTH = 2000

        # コードブロックを検出してリストに格納
        parts = []
        current_index = 0
        while current_index < len(response):
            # 次のコードブロックの開始位置を探す
            start_index = response.find(CODE_BLOCK_DELIMITER, current_index)
            # コードブロックが見つからない場合は、残りの文字列を追加して終了
            if start_index == -1:
                parts.append(response[current_index:])
                break

            # コードブロックの終了位置を探す
            end_index = response.find(CODE_BLOCK_DELIMITER, start_index + len(CODE_BLOCK_DELIMITER))
            # コードブロックが閉じられていない場合は、全体を一つのパートとして扱う
            if end_index == -1:
                parts.append(response[current_index:])
                break

            # コードブロックの終わりを含む位置
            end_index += len(CODE_BLOCK_DELIMITER)

            # コードブロック前のテキストを追加
            if start_index > current_index:
                parts.extend([response[current_index:start_index][i:i+MAX_LENGTH]
                            for i in range(0, len(response[current_index:start_index]), MAX_LENGTH)])
            # コードブロックを追加
            parts.append(response[start_index:end_index])

            # 現在のインデックスを更新
            current_index = end_index

        # パーツを送信
        buff = ''
        for part in parts:
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

    
intents = discord.Intents.default()
intents.message_content = True
client = MyClient(intents=intents)
client.run(TOKEN)
