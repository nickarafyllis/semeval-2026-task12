#!/usr/bin/env python3
"""
Build BM25 index for contextualized chunks
Input: chunk_meta.json
Output: BM25 index (pickle)
"""
import json
import pickle
import argparse
from pathlib import Path
from rank_bm25 import BM25Okapi

def main():
    parser = argparse.ArgumentParser(description="Build BM25 index")
    parser.add_argument("--input", required=True, help="Chunk metadata JSON")
    parser.add_argument("--output", required=True, help="Output BM25 pickle")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    chunks = []
    for topic in data:
        chunks.extend(topic['chunks'])
        
    corpus = [chunk['content'].lower().split() for chunk in chunks]
    print(f"BM25 index corpus: {len(corpus)} chunks")

    bm25 = BM25Okapi(corpus)
    
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.output, "wb") as f:
        pickle.dump(bm25, f)
    print(f"BM25 index saved to {args.output}")


if __name__ == "__main__":
    main()
