GPT bot
=======
GPT bot は ChatGPT を Discord 上で利用するための Discord Bot です．

## 実行前の準備
1. [OpenAI APIキーの発行](https://openai.com/blog/openai-api)


2. [Discord Botの作成とTOKENの取得](https://discordpy.readthedocs.io/ja/latest/discord.html) ※PermissionsはAdministratorでOKです

3. 必要なライブラリをインストール  
    ```
    pip install discord.py
    pip install openai
    pip install python-dotenv
    ```

4. 取得したTOKENとAPIキーを `.env` に書き込む
    ```
    TOKEN = your discord token
    OPENAI_API_KEY = your open ai api key
    ```

5. Botを招待したサーバに `chat-with-gpt4` と `chat-with-gpt3` という名前のチャンネルを作る  
Botとのやり取りはこのチャンネルで行います．

## 使用方法
1. スクリプトを実行します.  

    ```
    python app.py
    ```
    ここで
    ```
    Logged on as ボットの名前!
    ```
    と出れば起動には成功しています．

2. `chat-with-gpt4` または `chat-with-gpt3` チャンネルにプロンプトを書き込む  
`chat-with-gpt4` チャンネルに書き込むとGPT-4との会話をスタートさせることができます．会話が始まると自動でスレッドが作られ，以降はそのスレッドで会話を続けることができます．  
別の話題について話したい場合は，新たにチャンネルに書き込むと別のスレッドが作成され，新たに会話を行うことができます．`chat-with-gpt3` チャネルでは GPT-3.5 turbo と会話できます．
