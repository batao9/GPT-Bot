import os
import json
import asyncio
import pytest

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from agent import Agent
from langchain_core.messages import HumanMessage, SystemMessage


class RaisingExecutor:
    async def ainvoke(self, inputs):
        raise ValueError("boom for test")


@pytest.mark.asyncio
async def test_agent_error_logging_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("ERROR_LOG_ENABLED", "true")
    monkeypatch.setenv("ERROR_LOG_DIR", str(tmp_path))

    agent = await Agent.create(
        model_name="gpt-4o-mini",
        provider="openai",
        tools=[],
        debug=False,
    )
    # Inject raising executor to avoid network
    agent.agent_executor = RaisingExecutor()

    msgs = [
        SystemMessage(content="sys"),
        HumanMessage(content="hello world"),
    ]

    resp, atts = await agent.invoke(msgs)
    assert "エラー" in resp
    # a log file should be created
    files = list(tmp_path.iterdir())
    assert files, "No error log file was created"
    p = files[0]
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data.get("scope") == "agent.invoke"
    # input messages contain our human text
    texts = json.dumps(data.get("inputs", {}), ensure_ascii=False)
    assert "hello world" in texts
    # error info present
    err = data.get("error", {})
    assert err.get("type") == "ValueError"
    assert "boom for test" in err.get("message", "")


@pytest.mark.asyncio
async def test_agent_error_logging_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("ERROR_LOG_ENABLED", "false")
    monkeypatch.setenv("ERROR_LOG_DIR", str(tmp_path))

    agent = await Agent.create(
        model_name="gpt-4o-mini",
        provider="openai",
        tools=[],
        debug=False,
    )
    agent.agent_executor = RaisingExecutor()

    resp, atts = await agent.invoke([HumanMessage(content="x")])
    assert "エラー" in resp
    assert not any(tmp_path.iterdir()), "Logs should not be written when disabled"
