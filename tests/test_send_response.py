import os
import asyncio
import pytest
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app import MyClient
from utils import Utils

class FakeThread:
    def __init__(self):
        self.sent = []
    async def send(self, content=None, file=None, files=None):
        self.sent.append({
            'content': content,
            'file': file,
            'files': files,
        })

@pytest.mark.asyncio
async def test_send_response_latex_and_text(tmp_path, monkeypatch):
    client = MyClient(intents=None)
    thread = FakeThread()

    # stub latex_to_image to create a real temp file
    img_path = tmp_path / 'latex.png'
    def fake_latex_to_image(latex_code: str):
        img_path.write_bytes(b'PNG')
        return str(img_path)
    monkeypatch.setattr(Utils, 'latex_to_image', staticmethod(fake_latex_to_image))

    resp = "before $$x+y$$ after"
    await client.send_response(thread=thread, response=resp)

    # Expect two sends: one for latex with file, one for trailing text
    assert len(thread.sent) >= 1
    first = thread.sent[0]
    assert first['file'] is not None  # latex image attached
    assert 'before $$x+y$$' in (first['content'] or '')
    # After processing, the image temp should be deleted
    assert not img_path.exists()

@pytest.mark.asyncio
async def test_send_response_attachments_and_cleanup(tmp_path):
    client = MyClient(intents=None)
    thread = FakeThread()

    # create attachments
    f1 = tmp_path / 'a.txt'; f1.write_text('A')
    f2 = tmp_path / 'b.txt'; f2.write_text('B')

    # create dirs to be cleaned
    in_dir = tmp_path / 'in'; in_dir.mkdir(); (in_dir / 'x').write_text('x')
    out_dir = tmp_path / 'out'; out_dir.mkdir(); (out_dir / 'y').write_text('y')

    await client.send_response(
        thread=thread,
        response="\n\n",  # blank
        attachments=[str(f1), str(f2)],
        input_dir_path=str(in_dir),
        output_dir_path=str(out_dir),
    )

    # One final send with files
    assert any(call['files'] and len(call['files']) == 2 for call in thread.sent)
    # Dirs cleaned up
    assert not in_dir.exists()
    assert not out_dir.exists()
