from langchain_core.tools import Tool, StructuredTool
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.tools.riza.command import ExecPython
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import Docx2txtLoader
from typing import List, Any, Optional
import uuid
from utils import Utils
import os
from pydantic import BaseModel, Field

def get_google_search_tool() -> Tool:
    """Google検索ツールを返します。"""
    search = GoogleSearchAPIWrapper()
    return Tool(
        name="Google_Search",
        description='Search the web using Google. Input must be a valid search query.',
        func=search.run
    )

def get_duckduckgo_search_tools() -> List[Tool]:
    """DuckDuckGo検索ツール（通常検索と結果検索）のリストを返します。"""
    return [
        DuckDuckGoSearchRun(),
        DuckDuckGoSearchResults()
    ]

def get_code_interpreter_tool() -> Tool:
    """Pythonコードインタプリタツールを返します。"""
    return Tool(
        name="Python_Interpreter",
        description='Run Python Code: ' +
                    'Use this tool when you need to run Python code to get an answer. ' +
                    'Input must be a valid Python expression or statement. ' +
                    'You can use standard Python libraries. ' +
                    'Results are displayed in the console, so use functions like print() to print the results. ' +
                    'It is not possible to output graph images, so please use it for simple processing that can be completed in just a few lines.',
        func=ExecPython().invoke
    )

def get_web_loader_tool() -> Tool:
    """Webローダーツールを返します。"""
    def web_loader_func(url: str) -> List[Any]:
        loader = WebBaseLoader(url)
        return loader.load()
    return Tool(
        name="Web_Loader",
        description='Web Loader: ' +
                    'Use this tool to load web pages. ' +
                    'Input must be a valid URL.',
        func=web_loader_func
    )

def get_docx_loader_tool(download_dir: str = None) -> Tool:
    """docxローダーツールを返します。"""
    def docx_loader_func(path: str) -> List[Any]:
        loader = Docx2txtLoader(os.path.join(download_dir, path))
        return loader.load()
    return Tool(
        name="Docx_Loader",
        description='Docx Loader: ' +
                    'Use this tool to load docx files. ' +
                    'Input must be available filename of a docx file.',
        func=docx_loader_func
    )

def get_text_file_loader_tool(download_dir: str = None) -> Tool:
    """textファイルローダーツールを返します。"""
    def text_file_loader_func(path: str) -> str:
        with open(os.path.join(download_dir, path), 'rb') as f:
            raw = f.read()
        if Utils.is_probably_text(raw):
            return raw.decode(Utils.detect_encoding(raw))
        else:
            return "Cannot read this file. Please check if it is a text file."
        
    return Tool(
        name="Text_File_Loader",
        description='Text File Loader: ' +
                    'Use this tool to read non-binary files such as source code or plain text. ' +
                    'Input must be available filename of a file.',
        func=text_file_loader_func
    )


def get_text_attachment_writer_tool(upload_dir: str = None) -> Tool:
    """テキスト添付ファイル作成ツールを返します。"""

    class TextAttachmentWriterArgs(BaseModel):
        content: str = Field(description="Text to write into the attachment file.")
        filename: Optional[str] = Field(default=None, description="Optional filename for the attachment.")
        encoding: str = Field(default="utf-8", description="Encoding used when writing the file.")

    def text_attachment_writer_func(
        content: str,
        filename: Optional[str] = None,
        encoding: str = "utf-8"
    ) -> str:
        if not upload_dir:
            raise ValueError("Attachment directory is not configured.")

        if content is None:
            raise ValueError("'content' is required to create an attachment.")

        os.makedirs(upload_dir, exist_ok=True)

        if filename:
            filename = os.path.basename(filename)
        else:
            filename = f"attachment_{uuid.uuid4().hex}.txt"

        if not filename:
            raise ValueError("A valid filename could not be determined.")

        if any(sep in filename for sep in ('/', '\\')):
            raise ValueError("Filename must not contain path separators.")

        file_path = os.path.join(upload_dir, filename)

        try:
            with open(file_path, 'w', encoding=encoding) as f:
                f.write(content)
        except LookupError as e:
            raise ValueError(f"Unsupported encoding specified: {encoding}") from e

        return f"Attachment saved as {filename}. The file will be delivered with the final response."

    return StructuredTool.from_function(
        func=text_attachment_writer_func,
        name="Text_Attachment_Writer",
        description='Text Attachment Writer: Create a text-based attachment. Provide content, optional filename, and encoding.',
        args_schema=TextAttachmentWriterArgs
    )
