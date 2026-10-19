"""One bundled showcase catalog for the file page and realtime test page."""
import json
from pathlib import Path


def load_demo_samples(directory):
    directory = Path(directory)
    manifest = directory / "realtime_samples.json"
    if not manifest.is_file():
        return []
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError("展示样本清单必须是 JSON 列表。")
    seen = set()
    for item in entries:
        filename = item["file"]
        if (not isinstance(filename, str) or not filename or Path(filename).name != filename
                or not filename.endswith(".mp3") or filename in seen):
            raise ValueError("展示样本必须是 test_samples 下不重复的 MP3 文件名。")
        if (directory / filename).resolve().parent != directory.resolve():
            raise ValueError("展示样本不能指向 test_samples 目录外。")
        if not all(isinstance(item.get(key), str) and item[key] for key in ["label", "source_lang", "target_lang"]):
            raise ValueError("展示样本缺少有效标签或语言代码。")
        seen.add(filename)
    return entries
