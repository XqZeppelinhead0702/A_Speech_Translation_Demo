# pip install -U datasets

from pathlib import Path
import csv
from datasets import load_dataset, Audio

out = Path("test_samples")
out.mkdir(exist_ok=True)
records = []

for pair in ["zh-CN_en", "en_zh-CN", "fr_en"]:
    ds = load_dataset(
        "fixie-ai/covost2", pair, split="test", streaming=True
    ).cast_column("audio", Audio(decode=False))

    for i, row in enumerate(ds.take(5)):
        audio = row["audio"]
        suffix = Path(audio["path"] or "").suffix or ".mp3"
        filename = f"{pair}_{i:02d}{suffix}"
        (out / filename).write_bytes(audio["bytes"])

        records.append({
            "file": filename,
            "pair": pair,
            "source_text": row["sentence"],
            "reference_translation": row["translation"],
        })

with (out / "references.csv").open(
    "w", newline="", encoding="utf-8-sig"
) as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)