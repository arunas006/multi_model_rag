import json
from pathlib import Path

CHUNKS_DIR = Path(r"C:\dev\multi_model_rag\data\chunks")
META_FILE = Path(r"C:\dev\multi_model_rag\data\source_files.json")

def build_source_metadata():
    sources = set()

    for file in CHUNKS_DIR.glob("*.json"):
        try:
            data = json.loads(file.read_text())

            for item in data:
                if item.get("source_file"):
                    sources.add(item["source_file"])
                    break  # one per file is enough

        except Exception as e:
            print(f"Skipping {file}: {e}")

    META_FILE.parent.mkdir(parents=True, exist_ok=True)
    META_FILE.write_text(json.dumps(sorted(sources), indent=2))

    print(f"✅ Saved {len(sources)} source files to {META_FILE}")

if __name__ == "__main__":
    build_source_metadata()