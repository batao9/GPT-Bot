import os
import tempfile
import base64
import httpx
import fitz
import sympy
import charset_normalizer

class Utils:
    @staticmethod
    def latex_to_image(latex_code: str, prefix: str = 'latex_formula_', save_path: str = None) -> str:
        """LaTeXコードを画像に変換する。

        Args:
            latex_code (str): 変換するLaTeXコード。
            prefix (str, optional): 一時ファイル名のプレフィックス。Defaults to 'latex_formula_'.
            save_path (str, optional): 画像を保存するパス。指定しない場合は一時ファイルに保存。Defaults to None.

        Returns:
            str: 保存された画像ファイルのパス。
        """
        with tempfile.NamedTemporaryFile(prefix=prefix, suffix=".png", delete=False) as tmpfile:
            sympy.preview(latex_code, viewer='file', filename=tmpfile.name, euler=False,
                        dvioptions=["-T", "tight", "-z", "0", "--truecolor", "-D 600"],
                        dpi=60)
            if save_path:
                os.rename(tmpfile.name, save_path)
                return save_path
            else:
                return tmpfile.name

    @staticmethod
    def is_probably_text(raw: bytes, threshold: float = 0.75) -> bool:
        """バイト列がテキストである可能性が高いか判定する。

        1. NULL バイトがあれば False を返す。
        2. charset_normalizer でベストマッチを取得する。
        3. language_confidence（言語推定の信頼度）が閾値以上なら True を返す。

        Args:
            raw (bytes): 判定するバイト列。
            threshold (float, optional): テキストと判定するための信頼度の閾値。Defaults to 0.75.

        Returns:
            bool: テキストである可能性が高い場合は True、そうでない場合は False。
        """
        if b'\x00' in raw:
            return False
        match = charset_normalizer.detect(raw)
        if not match:
            return False
        return match['confidence'] >= threshold

    @staticmethod
    def detect_encoding(raw: bytes) -> str:
        """バイト列のエンコーディングを推定する。

        charset-normalizer でエンコーディングを推定する。
        見つからなければ UTF‑8 を返す。

        Args:
            raw (bytes): エンコーディングを推定するバイト列。

        Returns:
            str: 推定されたエンコーディング名。
        """
        best = charset_normalizer.from_bytes(raw).best()
        if best and best.encoding:
            enc = best.encoding.replace('_', '-').lower()
            return enc
        return 'utf-8'

    @staticmethod
    def pdf_url_to_base64_images(pdf_url: str) -> list[str]:
        """URLからPDFを取得し、1ページずつ画像に変換する。
        
        URLからPDFを取得し、1ページずつ画像に変換し、base64エンコードし、
        最終的にbase64エンコードされたPDFの画像の文字列のリストを返す。

        Args:
            pdf_url (str): PDFのURL。

        Returns:
            list[str]: 各ページをbase64エンコードした画像データの文字列のリスト。
                    エラーが発生した場合は空のリストを返します。
        """
        base64_images = []
        try:
            response = httpx.get(pdf_url)
            response.raise_for_status()  # HTTPエラーの場合は例外を発生させる
            pdf_bytes = response.content

            doc = fitz.open(stream=pdf_bytes, filetype="pdf")

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap()  # デフォルトDPI (96) で画像を取得
                img_bytes = pix.tobytes("png")  # PNG形式で画像のバイト列を取得

                base64_encoded_image = base64.b64encode(img_bytes).decode('utf-8')
                base64_images.append(base64_encoded_image)

            doc.close()
        except httpx.HTTPStatusError as e:
            print(f"URLからのPDF取得中にHTTPエラーが発生しました ({pdf_url}): {e}")
            return []
        except Exception as e:
            print(f"PDF処理中にエラーが発生しました ({pdf_url}): {e}")
            return []
        
        return base64_images