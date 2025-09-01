import asyncio
import base64
import os
import types
import pytest
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app import MyClient
from prompts import SytemPrompts

class Author:
    def __init__(self, bot=False):
        self.bot = bot

class Attachment:
    def __init__(self, filename, url):
        self.filename = filename
        self.url = url

class Message:
    def __init__(self, content, author, attachments=None):
        self.content = content
        self.author = author
        self.attachments = attachments or []
    def is_system(self):
        return False

class FakeResp:
    def __init__(self, status=200, raw=b""):
        self.status = status
        self._raw = raw
    async def read(self):
        return self._raw

class FakeRespCM:
    def __init__(self, resp):
        self._resp = resp
    async def __aenter__(self):
        return self._resp
    async def __aexit__(self, exc_type, exc, tb):
        return False

class FakeClientSession:
    def __init__(self, mapping):
        self._mapping = mapping  # url -> (status, raw)
    async def __aenter__(self):
        return self
    async def __aexit__(self, exc_type, exc, tb):
        return False
    def get(self, url):
        status, raw = self._mapping.get(url, (404, b""))
        return FakeRespCM(FakeResp(status, raw))

@pytest.mark.asyncio
async def test_convert_message_no_attachments(tmp_path, monkeypatch):
    client = MyClient(intents=None)
    msg = Message("Hello", Author(bot=False))
    messages = [msg]
    out = await client.convert_message(
        messages,
        provider="openai",
        system_prompt=SytemPrompts.prompts['assistant'],
        save_dir_path=str(tmp_path),
    )
    # System + Human
    assert len(out) >= 2
    assert out[0].type == 'system'
    assert 'AIアシスタント' in out[0].content
    assert out[-1].type == 'human'
    assert isinstance(out[-1].content, list)
    assert any(part.get('type') == 'text' and 'Hello' in part.get('text','') for part in out[-1].content)


@pytest.mark.asyncio
async def test_convert_message_image_attachment(tmp_path, monkeypatch):
    client = MyClient(intents=None)
    url = "https://example.com/img.png"
    raw = b"PNG_BYTES"
    monkeypatch.setattr(
        "aiohttp.ClientSession",
        lambda: FakeClientSession({url: (200, raw)})
    )
    msg = Message("See image", Author(bot=False), attachments=[Attachment("img.png", url)])
    out = await client.convert_message([msg], provider="openai", save_dir_path=str(tmp_path))
    human = [m for m in out if getattr(m, 'type', '') == 'human'][-1]
    parts = human.content
    # text part exists
    assert any(p.get('type') == 'text' for p in parts)
    # image_url part created
    imgs = [p for p in parts if p.get('type') == 'image_url']
    assert len(imgs) == 1
    assert imgs[0]['image_url']['url'].startswith('data:image/png;base64,')


@pytest.mark.asyncio
async def test_convert_message_pdf_openai(tmp_path, monkeypatch):
    client = MyClient(intents=None)
    url = "https://example.com/file.pdf"
    raw = b"%PDF-1.4 MOCK"
    monkeypatch.setattr(
        "aiohttp.ClientSession",
        lambda: FakeClientSession({url: (200, raw)})
    )
    # mock Utils.pdf_url_to_base64_images
    from utils import Utils
    monkeypatch.setattr(Utils, 'pdf_url_to_base64_images', staticmethod(lambda u: ['AAA', 'BBB']))

    msg = Message("PDF here", Author(bot=False), attachments=[Attachment("file.pdf", url)])
    out = await client.convert_message([msg], provider="openai", save_dir_path=str(tmp_path))
    human = [m for m in out if getattr(m, 'type', '') == 'human'][-1]
    parts = human.content
    imgs = [p for p in parts if p.get('type') == 'image_url']
    assert len(imgs) == 2
    # Also file info injected
    texts = [p for p in parts if p.get('type') == 'text']
    assert any('available files:' in p['text'] for p in texts)


@pytest.mark.asyncio
async def test_convert_message_pdf_anthropic(tmp_path, monkeypatch):
    client = MyClient(intents=None)
    url = "https://example.com/anthropic.pdf"
    raw = b"%PDF-1.4 MOCK"
    monkeypatch.setattr(
        "aiohttp.ClientSession",
        lambda: FakeClientSession({url: (200, raw)})
    )
    msg = Message("PDF A", Author(bot=False), attachments=[Attachment("paper.pdf", url)])
    out = await client.convert_message([msg], provider="anthropic", save_dir_path=str(tmp_path))
    human = [m for m in out if getattr(m, 'type', '') == 'human'][-1]
    parts = human.content
    files = [p for p in parts if p.get('type') == 'file']
    assert len(files) == 1
    assert files[0]['mime_type'] == 'application/pdf'
    # data is base64 of raw
    import base64 as b64
    assert files[0]['data'] == base64.b64encode(raw).decode('utf-8')


@pytest.mark.asyncio
async def test_convert_message_text_attachment_injects_file_info(tmp_path, monkeypatch):
    client = MyClient(intents=None)
    url = "https://example.com/data.txt"
    raw = b"hello text file"
    monkeypatch.setattr(
        "aiohttp.ClientSession",
        lambda: FakeClientSession({url: (200, raw)})
    )
    # Force classify as text
    from utils import Utils
    monkeypatch.setattr(Utils, 'is_probably_text', staticmethod(lambda b: True))
    msg = Message("read this", Author(bot=False), attachments=[Attachment("data.txt", url)])
    out = await client.convert_message([msg], provider="openai", save_dir_path=str(tmp_path))
    human = [m for m in out if getattr(m, 'type', '') == 'human'][-1]
    parts = human.content
    # no image_url or file entries
    assert not any(p.get('type') == 'image_url' for p in parts)
    assert not any(p.get('type') == 'file' for p in parts)
    # but file info text is injected
    assert any('available files:' in p.get('text', '') for p in parts if p.get('type') == 'text')
