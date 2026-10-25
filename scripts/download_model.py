"""Download a public base model separately from the source repository."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', choices=['3b', '7b'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    snapshot_download(f'Qwen/Qwen2.5-{args.size.upper()}-Instruct', local_dir=args.output)


if __name__ == '__main__': main()
