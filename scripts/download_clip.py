"""Download the paper's image encoder at a pinned public revision."""
import argparse
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    print(snapshot_download("openai/clip-vit-large-patch14",revision="32bd64288804d66eefd0ccbe215aa642df71cc41",
        allow_patterns=["config.json","preprocessor_config.json","model.safetensors"],local_dir=args.output))
