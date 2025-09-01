import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import Utils


def test_is_probably_text_ascii():
    raw = b"Hello, world!\nThis is ASCII."
    assert Utils.is_probably_text(raw) is True


def test_is_probably_text_binary_nullbyte():
    raw = b"\x00\x01\x02\xff\x00"
    assert Utils.is_probably_text(raw) is False


def test_detect_encoding_roundtrip_utf8():
    s = "こんにちは世界"
    raw = s.encode("utf-8")
    enc = Utils.detect_encoding(raw)
    # 返る値は 'utf-8' or 'ascii' など環境差がありうるので、復元できることを確認
    assert raw.decode(enc) == s
