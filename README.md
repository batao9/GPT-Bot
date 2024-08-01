GPT bot
=======
GPT bot は ChatGPT を Discord 上で利用するための Discord Bot です．

## 実行前の準備
1. .envファイルを作成しAPIキーを追記
    ```
    TOKEN = XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX # discord API key
    OPENAI_API_KEY = sk-XXXXXXXXXXXXXXXXXXX # open ai API key
    ```
2. 必要なライブラリをインストール  
    ```
    pip install -r requirements.txt
    ```
## 使用方法
`gpt-4o`, `gpt-4o-mini`, `gpt-4` または `gpt-3` チャンネルを作成し書き込む．
画像は`.png`, `.jpg`, `.gif`, `.webp` に対応
文章は`.txt`, `.py`, `.md`, `.csv`, `.c`, `.cpp`, `.java`, `pdf`に対応
