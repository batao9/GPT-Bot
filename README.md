GPT bot
=======
GPT bot は ChatGPT を Discord 上で利用するための Discord Bot です．

## 実行前の準備
1. .envファイルを作成しAPIキーを追記
    ```
    # discord API key
    TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # llm API keys
    OPENAI_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxx
    GOOGLE_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxx
    ANTHROPIC_API_KEY=xxxxxxxxxxxxxxxxxxxxxxx
    # google search CSE ID
    GOOGLE_CSE_ID=xxxxxxxxxxxxxxxxxxxxxxxxxxx # google search tool を使う場合

    # agent attachment files
    USER_ATTACHMENTS_DIR='/path/to/user_attach'     # userが送信したファイルのパス
    AGENT_ATTACHMENTS_DIR='/path/to/agent_attach'   # agentが送信するファイルのパス
    ```
2. 数式を画像に変換する機能を使用する場合はlatexをインストール
    ```
    sudo apt update
    sudo apt install texlive texlive-latex-extra dvipng
    ```
3. 実行
    ```
    uv run python app.py
    ```
## 使用方法
### models.jsonの設定
`models.json` で指定したチャンネルを作成し書き込む．
```json
"discord channel name": { 
    "model": "model name / required.",
    "provider": "openai, anthropic or gemini, xai / required.",
    "tools": [
        "ggl_search / google search",
        "ddg_search / duckduckgo search",
        "web_loader / web loader from url",
        "mcp / mcp server"
    ],
    "reasoning_effort": "low, medium or high / default is medium. OpenAI's o-series models only."
}
```
エージェントは上記設定に関係なく、長いコードやテキストを添付ファイルとして送信するための`Text_Attachment_Writer`ツールを常に利用できます。`content`（必須）、`filename`、`encoding`の各引数を個別に指定する形式で呼び出すと、`AGENT_ATTACHMENTS_DIR`配下にファイルが作成され、応答時に添付されます。`filename`を省略すると自動で`.txt`ファイル名が生成されます。
### MCP Serverの利用
`mcp.json` を設定することでMCP Serverへの接続が可能
```json
{
    "mcpServers": {
        "tool name": {
            "command": "uv",
            "args": ["--directory", "/path/to/python", "run", "python", "server.py"],
            "transport": "stdio"
        }
    }
}
```
## 注意点
画像は`.png`, `.jpg`, `.gif`, `.webp` に対応
文章は一般的な文字コードでエンコード可能なテキスト形式、および`pdf`に対応
