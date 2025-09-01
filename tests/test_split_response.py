import pytest
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from app import MyClient
import discord


def make_client():
    intents = discord.Intents.none()
    return MyClient(intents=intents)


def test_split_response_plain_short():
    client = make_client()
    s = "hello world"
    parts = client.split_response(s, max_length=50)
    assert parts == [s]


def test_split_response_plain_long():
    client = make_client()
    s = ("abcde " * 30) * 3  # > 50 chars
    parts = client.split_response(s, max_length=50)
    assert all(len(p) <= 50 for p in parts)
    assert "".join(parts) == s


def test_split_response_codeblock_preserve_lang_and_wrap():
    client = make_client()
    body = "\n".join(["x = 1" for _ in range(80)])  # make it long
    code = f"```python\n{body}\n```"
    parts = client.split_response(code, max_length=120)
    # 1) 全てコードブロックで再ラップされている
    assert all(p.startswith("```python") and p.rstrip().endswith("```") for p in parts)
    # 2) 長さ制限内
    assert all(len(p) <= 120 for p in parts)
    # 3) 再結合しても中身の行が失われていない（ラップを剥がして比較）
    inner = "".join(
        p.split("\n", 1)[1].rsplit("\n" + "```", 1)[0] if p.rstrip().endswith("```") else p
        for p in parts
    )
    assert inner.strip().startswith("x = 1")
    assert inner.count("x = 1") == body.count("x = 1")


def test_split_response_latex_wrap_and_split():
    client = make_client()
    inner = " ".join(["x^2 + y^2 = 1" for _ in range(40)])
    latex = f"$${inner}$$"
    parts = client.split_response(latex, max_length=80)
    assert all(p.startswith("$$") and p.endswith("$$") for p in parts)
    assert all(len(p) <= 80 for p in parts)
