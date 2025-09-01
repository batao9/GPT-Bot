import os
import sys
import types
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app import MyClient
from LLM_Models import Models

class FakeAuthor:
    def __init__(self, bot=False):
        self.bot = bot

class FakeMessage:
    def __init__(self, content, author, channel, mid=123):
        self.content = content
        self.author = author
        self.channel = channel
        self.id = mid
    def is_system(self):
        return False

class FakeThread:
    def __init__(self):
        self.edited = []
        self.sent = []
    async def edit(self, name=None):
        self.edited.append(name)
    async def send(self, content=None, file=None, files=None):
        self.sent.append({
            'content': content,
            'file': file,
            'files': files,
        })
    class _TypingCtx:
        async def __aenter__(self):
            return None
        async def __aexit__(self, exc_type, exc, tb):
            return False
    def typing(self):
        return FakeThread._TypingCtx()

@pytest.mark.asyncio
async def test_on_message_happy_path(monkeypatch, tmp_path):
    client = MyClient(intents=None)

    # Always configured
    monkeypatch.setattr(Models, 'is_channel_configured', staticmethod(lambda ch: True))

    # Fake thread and messages from helper
    thread = FakeThread()
    user_msg = FakeMessage("Hi", FakeAuthor(bot=False), channel=object())
    async def fake_get_thread_and_messages(self, message):
        return thread, [user_msg]
    monkeypatch.setattr(MyClient, 'get_thread_and_messages', fake_get_thread_and_messages)

    # Deterministic title
    async def fake_generate_thread_name(self, messages):
        return "Test Title"
    monkeypatch.setattr(MyClient, 'generate_thread_name', fake_generate_thread_name)

    # Avoid heavy conversion
    async def fake_convert_message(self, messages, provider, system_prompt=None, save_dir_path=None):
        class HM:
            type = 'human'
            content = 'hi'
        class SM:
            type = 'system'
            content = 'sys'
        return [SM(), HM()]
    monkeypatch.setattr(MyClient, 'convert_message', fake_convert_message)

    # Fake Agent
    class FakeAgent:
        async def invoke(self, messages):
            return "OK", []
    async def fake_create(**kwargs):
        return FakeAgent()
    # Patch at import site
    import agent as agent_module
    monkeypatch.setattr(agent_module.Agent, 'create', staticmethod(fake_create))

    # Spy send_response to capture args
    calls = {}
    async def fake_send_response(self, thread, response, attachments, input_dir_path=None, output_dir_path=None):
        calls['response'] = response
        calls['attachments'] = attachments
        calls['in_dir'] = input_dir_path
        calls['out_dir'] = output_dir_path
    monkeypatch.setattr(MyClient, 'send_response', fake_send_response)

    # Run
    msg = FakeMessage("trigger", FakeAuthor(bot=False), channel=object(), mid=999)
    await client.on_message(msg)

    # Title edited
    assert thread.edited and thread.edited[-1] == "Test Title"
    # send_response got called with OK
    assert calls.get('response') == 'OK'
    assert calls.get('attachments') == []
    # session dirs passed
    assert calls.get('in_dir') and calls.get('out_dir')


@pytest.mark.asyncio
async def test_on_message_agent_error_path(monkeypatch):
    client = MyClient(intents=None)

    # Always configured
    monkeypatch.setattr(Models, 'is_channel_configured', staticmethod(lambda ch: True))

    # Fake thread and messages
    thread = FakeThread()
    user_msg = FakeMessage("Hi", FakeAuthor(bot=False), channel=object())
    async def fake_get_thread_and_messages(self, message):
        return thread, [user_msg]
    monkeypatch.setattr(MyClient, 'get_thread_and_messages', fake_get_thread_and_messages)

    async def fake_generate_thread_name(self, messages):
        return "Err Title"
    monkeypatch.setattr(MyClient, 'generate_thread_name', fake_generate_thread_name)

    async def fake_convert_message(self, messages, provider, system_prompt=None, save_dir_path=None):
        class HM:
            type = 'human'
            content = 'hi'
        class SM:
            type = 'system'
            content = 'sys'
        return [SM(), HM()]
    monkeypatch.setattr(MyClient, 'convert_message', fake_convert_message)

    # Agent that raises on invoke await
    class FakeAgent:
        async def invoke(self, messages):
            raise RuntimeError("boom")
    async def fake_create(**kwargs):
        return FakeAgent()
    import agent as agent_module
    monkeypatch.setattr(agent_module.Agent, 'create', staticmethod(fake_create))

    sent = []
    async def fake_send_response(self, thread, response, attachments, input_dir_path=None, output_dir_path=None):
        sent.append((response, attachments))
    monkeypatch.setattr(MyClient, 'send_response', fake_send_response)

    msg = FakeMessage("trigger", FakeAuthor(bot=False), channel=object(), mid=1000)
    await client.on_message(msg)

    # Fallback error message should be sent once in except block
    assert sent, "send_response should be called"
    assert any("エラー" in r for r, _ in sent)
