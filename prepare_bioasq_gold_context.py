#!/usr/bin/env python3
"""
Build LoCoMo-style samples from local BioASQ JSON files (e.g., 10B*_golden.json merged).

- Uses ONLY BioASQ question types: factoid, list
- factoid -> category 4
- list    -> category 6
- answer_str: first synonym of each answer-item joined by " ; "
"""

import argparse
import json
import os
import random
import re
import glob
from typing import Any, Dict, List, Optional
from tqdm import tqdm

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def split_into_sentences(text: str) -> List[str]:
    if not text:
        return []
    t = re.sub(r"\s+", " ", str(text)).strip() 
    if not t:
        return []
    return [s.strip() for s in _SENT_SPLIT.split(t) if s and s.strip()]

def normalize_text_for_dedup(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip().lower()
    return s

def dedup_snippets_by_text(snippets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for snip in snippets:
        if not isinstance(snip, dict):
            continue
        txt = snip.get("text")
        if not isinstance(txt, str) or not txt.strip():
            continue
        key = normalize_text_for_dedup(txt)
        if key in seen:
            continue
        seen.add(key)
        out.append(snip)
    return out

def flatten_bioasq_answer(exact_answer: Optional[List[List[str]]]) -> str:
    """
    BioASQ exact_answer example:
      - factoid: [[syn1, syn2, ...]] -> "syn1"
      - list:    [[item1_syns...], [item2_syns...], ...] -> "item1 ; item2 ; ..."
    We take the first synonym of each answer item and join with ' ; '.
    """
    if not exact_answer or not isinstance(exact_answer, list):
        return ""

    flat_answers: List[str] = []
    for item_list in exact_answer:
        if item_list and isinstance(item_list, list) and len(item_list) > 0:
            first_synonym = item_list[0]
            if isinstance(first_synonym, str) and first_synonym.strip():
                flat_answers.append(first_synonym.strip())

    if not flat_answers:
        return ""
    return " ; ".join(flat_answers)

def load_and_merge_local_files(input_pattern: str) -> List[Dict[str, Any]]:
    file_list = sorted(glob.glob(input_pattern))
    if not file_list:
        raise FileNotFoundError(f"No files found matching pattern: {input_pattern}")

    print(f"[Load] Found {len(file_list)} files matching '{input_pattern}'")
    merged_questions: List[Dict[str, Any]] = []
    for file_path in tqdm(file_list, desc="Merging files"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "questions" in data and isinstance(data["questions"], list):
                merged_questions.extend(data["questions"])
            else:
                print(f"Warning: Skipping {file_path} - no root 'questions' list found.")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {file_path}. Skipping.")
        except Exception as e:
            print(f"Error reading {file_path}: {e}. Skipping.")

    print(f"[Load] Total merged questions: {len(merged_questions)}")
    return merged_questions

def normalize_text_for_dedup(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

def transform_bioasq_to_grouped_records(raw_questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped_records: List[Dict[str, Any]] = []
    print("[Transform] Keeping only types: ['factoid', 'list'] with categories {factoid:4, list:6} ...")

    for q_obj in tqdm(raw_questions, desc="Processing questions"):
        q_type = q_obj.get("type")
        if q_type not in ("factoid", "list"):
            continue

        category = 4 if q_type == "factoid" else 6

        qid = q_obj.get("id")
        question_text = q_obj.get("body")
        snippets = q_obj.get("snippets", [])
        exact_answer_raw = q_obj.get("exact_answer")

        if not qid or not question_text:
            continue

        snippets_meta: List[Dict[str, Any]] = []
        seen_texts = set()  

        if isinstance(snippets, list):
            for snip in snippets:
                if not isinstance(snip, dict):
                    continue
                snip_text = snip.get("text")
                if not (isinstance(snip_text, str) and snip_text.strip()):
                    continue

                key = normalize_text_for_dedup(snip_text)  
                if key in seen_texts:                      
                    continue                               
                seen_texts.add(key)                        

                snippets_meta.append({
                    "text": snip_text.strip(),
                    "document": snip.get("document"),
                    "beginSection": snip.get("beginSection"),
                    "endSection": snip.get("endSection"),
                    "offsetInBeginSection": snip.get("offsetInBeginSection"),
                    "offsetInEndSection": snip.get("offsetInEndSection"),
                })

        if not snippets_meta:
            continue

        answer_str = flatten_bioasq_answer(exact_answer_raw)
        if not answer_str:
            continue

        grouped_records.append({
            "question_id": str(qid),
            "question": question_text.strip(),
            "answer_str": answer_str,
            "snippets": snippets_meta,
            "category": category,
        })

    print(f"[Transform] Total grouped records: {len(grouped_records)}")
    return grouped_records



def build_chunks(records: List[Dict[str, Any]], chunk_size: int, num_chunks: int, seed: int):
    total_len = len(records)
    print(f"[Dataset] Total valid grouped records intended for chunking: {total_len}")
    if total_len == 0:
        print("Warning: No records available to chunk.")
        return []

    indices = list(range(total_len))
    random.seed(seed)
    random.shuffle(indices)

    max_possible_chunks = total_len // chunk_size
    if max_possible_chunks == 0:
        print(f"Warning: Not enough records ({total_len}) for chunk_size={chunk_size}.")
        return []

    if num_chunks > max_possible_chunks:
        print(f"Warning: Requested {num_chunks} chunks, but data size only allows {max_possible_chunks}. Adjusting limit.")
        num_chunks = max_possible_chunks

    samples = []
    for chunk_idx in tqdm(range(num_chunks), desc="Creating BioASQ Chunks"):
        chunk_indices = indices[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]

        all_turns = []
        all_qas = []
        turn_count = 0

        for ridx in chunk_indices:
            item = records[ridx]
            question = item.get("question")
            answer = item.get("answer_str")
            snippets = item.get("snippets") or []
            cat = item.get("category")

            if not question or not answer or not snippets:
                continue

            gold_turn_ids: List[str] = []
            gold_turn_span = {"start": None, "end": None}  # optional

            start_turn_idx = turn_count

            for snip in snippets:
                snip_text = snip.get("text")
                if not (isinstance(snip_text, str) and snip_text.strip()):
                    continue

                for sent in split_into_sentences(snip_text):
                    dia_id = f"{chunk_idx}:{turn_count}"
                    all_turns.append({
                        "speaker": "System",
                        "dia_id": dia_id,
                        "text": sent,
                    })
                    gold_turn_ids.append(dia_id)
                    turn_count += 1

            end_turn_idx = turn_count  # exclusive

            if end_turn_idx > start_turn_idx:
                gold_turn_span["start"] = f"{chunk_idx}:{start_turn_idx}"
                gold_turn_span["end"] = f"{chunk_idx}:{end_turn_idx - 1}"

            all_qas.append({
                "question_id": item.get("question_id"),
                "question": question,
                "answer": answer,
                "category": cat,
                "gold_turn_ids": gold_turn_ids,     
            })

        if not all_turns:
            print(f"Warning: Chunk {chunk_idx} resulted in empty dialogue turns. Skipping.")
            continue

        samples.append({
            "sample_id": chunk_idx,
            "conversation": {
                "speaker_a": "System",
                "speaker_b": "User",
                "session_0": all_turns,
                "session_0_date_time": "2020-01-01 00:00:00",
            },
            "qa": all_qas,
            "event_summary": {},
            "observation": {},
            "session_summary": {},
        })

    print(f"[Dataset] Generated {len(samples)} chunks.")
    return samples


def main():
    p = argparse.ArgumentParser(description="Build LoCoMo-style chunks from local merged BioASQ JSON files.")
    p.add_argument("--input-pattern", type=str, required=True,
                   help="Glob pattern for input JSON files (e.g., 'data/10B*_golden.json').")
    p.add_argument("--chunk-size", type=int, default=20, help="Number of questions per chunk.")
    p.add_argument("--num-chunks", type=int, default=10, help="Max number of chunks to generate.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    p.add_argument("--out", type=str, default=None, help="Output JSON file path.")
    args = p.parse_args()

    if args.out is None:
        base_name = os.path.basename(args.input_pattern).replace("*", "merged").replace(".json", "")
        args.out = f"bioasq_{base_name}_chunked_cs{args.chunk_size}_n{args.num_chunks}.json"

    raw_questions = load_and_merge_local_files(args.input_pattern)

    grouped_records = transform_bioasq_to_grouped_records(raw_questions)

    samples = build_chunks(grouped_records, args.chunk_size, args.num_chunks, args.seed)

    if not samples:
        print("Error: No output samples generated. Check input data and filtering.")
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"[Done] wrote {len(samples)} chunks to {args.out}")

if __name__ == "__main__":
    main()
