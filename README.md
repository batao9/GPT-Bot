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
3. 数式を画像に変換する機能を使用する場合はlatexをインストール
    ```
    sudo apt update
    sudo apt install texlive texlive-latex-extra dvipng
    ```
## 使用方法
`models.json` で指定したチャンネルを作成し書き込む．

## 注意点
画像は`.png`, `.jpg`, `.gif`, `.webp` に対応
文章は`.txt`, `.py`, `.md`, `.csv`, `.c`, `.cpp`, `.java`, `pdf`に対応
