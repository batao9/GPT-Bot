GPT bot
=======
GPT bot は ChatGPT を Discord 上で利用するための Discord Bot です．

## 実行前の準備
1. OpenAI APIキーの発行  
[調べて頑張ってください．](https://openai.com/blog/openai-api)

2. Discord Botの作成とTOKENの取得  
[調べて頑張ってください．](https://discordpy.readthedocs.io/ja/latest/discord.html) ※PermissionsはAdministratorでOKです

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

## 注意点
* このプログラムではモデルは `gpt-4-0613` を利用しています．GPT‐4のAPIが一般開放されていない場合は，[GPT-4 API waitlist](https://openai.com/waitlist/gpt-4-api)に登録してください．
* スレッドに書き込まれた場合，スレッド内の過去10件のメッセージを読み込んで，GPTに渡しています．そのため，会話が長くなると記憶が消えるので注意してください．
* 読み込む過去のメッセージ数を増やすこともできますが，その場合のAPIの利用料金には注意してください．
* ~~現在のプログラムでは，`chat-with-gpt4` チャンネル以外のスレッドのメッセージにもBotが反応するようになっています．~~ 修正済み
* 自分1人だけのサーバ用に作ったので，それ以外の用途で使う場合は必要に応じて調整してください．