from discord import TextChannel
import json
from typing import Dict, Optional

class Models:
    @staticmethod
    def load_mapping(file_path: str = 'models.json') -> Dict:
        """モデルのマッピングをファイルから読み込む"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: {file_path} が見つかりません。")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error: JSONの読み込みに失敗しました: {e}")
            return {}

    @staticmethod
    def get_field(channel: TextChannel, key: str) -> Optional[str]:
        """チャンネルに対応するモデルを取得する"""
        channel_name = getattr(channel, 'name', None)
        parent_name = getattr(channel.parent, 'name', None) if hasattr(channel, 'parent') else None
        try:
            mapping = Models.mappling.get(channel_name) or Models.mappling.get(parent_name)
            if mapping:
                return mapping[key]
        except:
            return None
    
    @staticmethod
    def is_channel_configured(channel: TextChannel) -> bool:
        """指定されたチャンネルが設定されているか確認する"""
        channel_name = getattr(channel, 'name', None)
        parent_name = getattr(channel.parent, 'name', None) if hasattr(channel, 'parent') else None
        return channel_name in Models.mappling or parent_name in Models.mappling
    
    mappling: Dict = load_mapping.__func__()