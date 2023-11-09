import discord
import openai
import dotenv
import os

CHANNEL_NAME_GPT4 = 'chat-with-gpt4'
CHANNEL_NAME_GPT4_VISION = 'chat-with-gpt4-vision'
CHANNEL_NAME_GPT35 = 'chat-with-gpt3'
MODEL_GPT4 = 'gpt-4-1106-preview'
MODEL_GPT4_VISION = 'gpt-4-vision-preview'
MODEL_GPT35 = 'gpt-3.5-turbo-1106'

dotenv.load_dotenv()
TOKEN = os.getenv('TOKEN')
openai.api_key = os.getenv('OPENAI_API_KEY')


# ChatGPTのレスポンスを取得する
def get_gpt_response(messages, model):
    if model == MODEL_GPT4_VISION:
        return get_gpt_response_vision(messages, model)
    
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
                "content": msg.content
            })
        else:
            # roleをuserに
            prompt.insert(0, {
                "role": "user",
                "content": msg.content
            })
                
    # レスポンスを生成
    response = openai.ChatCompletion.create(
        model=model,  
        temperature=0.7,
        top_p=0.9,
        messages=prompt
    )
    
    return response['choices'][0]['message']['content']

def get_gpt_response_vision(messages, model):
    prompt = []
    for msg in messages:
        # systemメッセージは無視
        if msg.is_system():
            continue
        # メッセージの中身を取り出して，APIに投げる形に変換
        if msg.author.bot:
            # botからのメッセージはroleをassistantに
            # 画像が添付されているならpromptに画像urlを含める
            if msg.attachments and any(msg.attachments[0].filename.lower().endswith(ext) for ext in ['png', 'jpg', 'jpeg']):
                prompt.insert(0, {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": msg.content},
                        {
                            "type": "image_url",
                            "image_url": msg.attachments[0].url
                        },
                    ]
                })
            else:
                prompt.insert(0, {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": msg.content},
                    ]
                })
        else:
            # roleをuserに
            # 画像が添付されているならpromptに画像urlを含める
            if msg.attachments and any(msg.attachments[0].filename.lower().endswith(ext) for ext in ['png', 'jpg', 'jpeg']):
                prompt.insert(0, {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": msg.content},
                        {
                            "type": "image_url",
                            "image_url": msg.attachments[0].url
                        },
                    ]
                })
            else:
                prompt.insert(0, {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": msg.content},
                    ]
                })
    print(prompt)
    # レスポンスを生成
    response = openai.ChatCompletion.create(
        model=model,  
        temperature=0.7,
        top_p=0.9,
        messages=prompt,
        max_tokens=1000
    )
    
    return response['choices'][0]['message']['content']

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
        gpt_response = get_gpt_response(messages, model)
        await self.send_response_in_parts(thread, gpt_response)


    # チャンネル名に基づいてモデルを返す
    def get_model_based_on_channel(self, channel):
        if hasattr(channel, 'name'):
            if channel.name == CHANNEL_NAME_GPT4:
                return MODEL_GPT4
            elif channel.name == CHANNEL_NAME_GPT4_VISION:
                return MODEL_GPT4_VISION
            elif channel.name == CHANNEL_NAME_GPT35:
                return MODEL_GPT35
            
        if hasattr(channel, 'parent'):
            if channel.parent.name == CHANNEL_NAME_GPT4:
                return MODEL_GPT4
            elif channel.parent.name == CHANNEL_NAME_GPT4_VISION:
                return MODEL_GPT4_VISION
            elif channel.parent.name == CHANNEL_NAME_GPT35:
                return MODEL_GPT35
        return None
    
    
    # メッセージを送るスレッドと，スレッドのメッセージ履歴を返す
    async def get_thread_and_messages(self, message):
        # スレッドでメッセージが送られた場合
        if isinstance(message.channel, discord.Thread):
            thread = message.channel
            messages = [msg async for msg in thread.history(limit=10)]
            # オリジナルメッセージを追加
            messages.append(await thread.parent.fetch_message(thread.id))
        # チャンネルでメッセージが送られた場合
        elif isinstance(message.channel, discord.TextChannel):
            thread = await message.create_thread(name=message.content[:10])
            messages = [message]
        else:
            thread = None
            messages = []
        return thread, messages
    
    
    # responseを2000文字ごとに区切って送信（Discordの文字数上限は2000文字）
    async def send_response_in_parts(self, thread, response):
        for i in range(0, len(response), 2000):
            print('bot:' + response[i:i+2000])
            await thread.send(response[i:i+2000])

    
intents = discord.Intents.default()
intents.message_content = True
client = MyClient(intents=intents)
client.run(TOKEN)