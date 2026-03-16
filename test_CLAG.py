from CLAG_memory import LLMController, AgenticMemorySystem
import os
import json
import argparse
import logging
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from openai import OpenAI
from load_dataset import load_locomo_dataset, QA, Turn, Session, Conversation
import nltk
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim
import statistics
from collections import defaultdict
import pickle
import random
from tqdm import tqdm
from utils import calculate_metrics, aggregate_metrics
from datetime import datetime
import time



try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')


try:
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
except Exception as e:
    print(f"Warning: Could not load SentenceTransformer model: {e}")
    sentence_model = None

class advancedMemAgent:
    def __init__(self, model, backend, retrieve_k, temperature_c5, sglang_host="http://localhost", sglang_port=30000):
        self.memory_system = AgenticMemorySystem(
            model_name='all-MiniLM-L6-v2',
            llm_backend=backend,
            llm_model=model,
            sglang_host=sglang_host,
            sglang_port=sglang_port
        )
        self.retriever_llm = LLMController(
            backend=backend, 
            model=model, 
            api_key=None, 
            sglang_host=sglang_host, 
            sglang_port=sglang_port
        )
        self.retrieve_k = retrieve_k
        self.temperature_c5 = temperature_c5

    def add_memory(self, content, time=None):
        self.memory_system.add_note(content, time=time)

    def retrieve_memory(self, content, k=10):

        query_tags = [t.strip() for t in content.split(",") if t.strip()]
        return self.memory_system.find_related_memories_raw(content, k=k, query_tags=query_tags)

    def retrieve_memory_llm(self, memories_text, query):
        prompt = f"""Given the following conversation memories and a question, select the most relevant parts of the conversation that would help answer the question. Include the date/time if available.

                Conversation memories:
                {memories_text}

                Question: {query}

                Return only the relevant parts of the conversation that would help answer this specific question. Format your response as a JSON object with a "relevant_parts" field containing the selected text. 
                If no parts are relevant, do not do any things just return the input.

                Example response format:
                {{"relevant_parts": "2024-01-01: Speaker A said something relevant..."}}"""
            

        response = self.retriever_llm.llm.get_completion(prompt,response_format={"type": "json_schema", "json_schema": {
                            "name": "response",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "relevant_parts": {
                                        "type": "string",
                                    }
                                },
                                "required": ["relevant_parts"],
                                "additionalProperties": False
                            },
                            "strict": True
                        }})

        return response
    
    def generate_query_llm(self, question):
        prompt = f"""Given the following question, generate several keywords, using 'commas' as the separator.

                Question: {question}

                Format your response as a JSON object with a "keywords" field containing the selected text. 

                Example response format:
                {{"keywords": "keyword1, keyword2, keyword3"}}"""
            
            # Get LLM response
        response = self.retriever_llm.llm.get_completion(prompt,response_format={"type": "json_schema", "json_schema": {
                            "name": "response",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "keywords": {
                                        "type": "string",
                                    }
                                },
                                "required": ["keywords"],
                                "additionalProperties": False
                            },
                            "strict": True
                        }})
        print("response:{}".format(response))
        try:
            response = json.loads(response)["keywords"]
        except:
            response = response.strip()
        return response

    def answer_question(self, question: str, category: int, answer: str) -> str:
        """Generate answer for a question given the conversation context."""

        t_start = time.time()
        keywords = self.generate_query_llm(question)
        t_retrieval_start = time.time()
        raw_context = self.retrieve_memory(keywords,k=self.retrieve_k)
        ms = self.memory_system
        retrieval_debug = {
            "search_mode": getattr(ms, "last_search_mode", "global"),
            "search_space_size": getattr(ms, "last_search_space_size", 0),
            "total_memories": getattr(ms, "last_total_memories", 0),
            "candidate_clusters": getattr(ms, "last_candidate_cluster_ids", []),
            "selected_clusters": getattr(ms, "last_selected_cluster_ids", []),
            "searched_clusters": getattr(ms, "last_searched_cluster_ids", []),
            "retrieved_indices": getattr(ms, "last_retrieved_indices", []),
            "retrieved_memory_ids": getattr(ms, "last_retrieved_memory_ids", []),
            "retrieved_memories": getattr(ms, "last_retrieved_memories", []),
            "query_keywords": keywords,
     }
        t_retrieval_end = time.time()
        retrieval_latency_ms = (t_retrieval_end - t_retrieval_start) * 1000.0

        search_space_size = getattr(self.memory_system, "last_search_space_size", 0) or 0
        total_memories = getattr(self.memory_system, "last_total_memories", 0) or 0
        if total_memories > 0:
            search_space_reduction = 1.0 - search_space_size / total_memories
        else:
            search_space_reduction = 0.0
        context = raw_context
        assert category in [1,2,3,4,5,6]
        user_prompt = f"""Context:
                {context}

                Question: {question}

                Answer the question based only on the information provided in the context above."""
        temperature = 0.7
        if category == 5: 
            answer_tmp = list()
            if random.random() < 0.5:
                answer_tmp.append('Not mentioned in the conversation')
                answer_tmp.append(answer)
            else:
                answer_tmp.append(answer)
                answer_tmp.append('Not mentioned in the conversation')
            user_prompt = f"""
                            Based on the context: {context}, answer the following question. {question} 
                            
                            Select the correct answer: {answer_tmp[0]} or {answer_tmp[1]}  Short answer:
                            """
            temperature = self.temperature_c5
        elif category == 2:
            user_prompt = f"""
                            Based on the context: {context}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
                            Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.   

                            Question: {question} Short answer:
                            """
        elif category == 3:
            user_prompt = f"""
                            Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

                            Question: {question} Short answer:
        
                            """
        elif category == 6:
            user_prompt = f"""
                            Based on the context below, answer the question.

                            This is a LIST question:
                            - Extract ALL distinct answer items that are supported by the context.
                            - Output the answers as a semicolon-separated list using EXACT words from the context whenever possible.
                            - Use this exact format: item1 ; item2 ; item3
                            - Do NOT add numbering, commas, bullet points, or extra text.
                            - If there is only one item, output just that single item (no extra words).

                            Context:
                            {context}

                            Question: {question}
                            Answer:

                            """

        else:
            user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

                            Question: {question} Short answer:
                            """
        response = self.memory_system.llm_controller.llm.get_completion(
            user_prompt,response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "answer": {
                                    "type": "string",
                                }
                            },
                            "required": ["answer"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }},temperature=temperature
        )
        
        t_end = time.time()
        end_to_end_latency_ms = (t_end - t_start) * 1000.0
        return (response,
            user_prompt,
            raw_context,
            retrieval_latency_ms, 
            end_to_end_latency_ms,
            search_space_size,
            total_memories,
            search_space_reduction,
            retrieval_debug,
        )


def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger('locomo_eval')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger




def evaluate_dataset(dataset_path: str, model: str, output_path: Optional[str] = None, ratio: float = 1.0, backend: str = "sglang", temperature_c5: float = 0.5, retrieve_k: int = 10, sglang_host: str = "http://localhost", sglang_port: int = 30000, fig_root_dir: Path = None):
    """Evaluate the agent on the LoComo dataset.
    
    Args:
        dataset_path: Path to the dataset file
        model: Name of the model to use
        output_path: Path to save results
        ratio: Ratio of dataset to evaluate
        fig_root_dir: Path to save figures and summaries"""

    total_cluster_searches = 0
    total_global_searches = 0
    # Generate automatic log filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_filename = f"eval_ours_{model}_{backend}_ratio{ratio}_{timestamp}.log"
    log_path = os.path.join(os.path.dirname(__file__), "logs_CLAG", log_filename)
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    logger = setup_logger(log_path)
    logger.info(f"Loading dataset from {dataset_path}")
    
    # Load dataset
    samples = load_locomo_dataset(dataset_path)
    logger.info(f"Loaded {len(samples)} samples")
    
    # Select subset of samples based on ratio
    if ratio < 1.0:
        num_samples = max(1, int(len(samples) * ratio))
        samples = samples[:num_samples]
        logger.info(f"Using {num_samples} samples ({ratio*100:.1f}% of dataset)")
    
    # Store results
    # 🔹 Cluster stats (per-sample) for JSON output
    cluster_stats_by_sample = []  # list of {sample_id, total_memories, num_clusters, cluster_member_counts}
    num_clusters_list = []
    max_cluster_size_list = []
    mean_cluster_size_list = []

    results = []
    all_metrics = []
    all_categories = []
    total_questions = 0
    category_counts = defaultdict(int)
    retrieval_latencies = []
    e2e_latencies = []

    # Evaluate each sample
    i = 0
    error_num = 0
    ds_name = Path(dataset_path).stem 
    memories_dir = os.path.join(
        os.path.dirname(__file__), 
        f"cached_memories__{ds_name}_{backend}_{Path(model).stem}"
    )
    os.makedirs(memories_dir, exist_ok=True)
    allow_categories = [1,2,3,4,5,6]
    for sample_idx, sample in enumerate(tqdm(samples, desc="Samples", total=len(samples))):
        agent = advancedMemAgent(model, backend, retrieve_k, temperature_c5, sglang_host, sglang_port)
        # Create memory cache filename based on sample and session indices
        memory_cache_file = os.path.join(
            memories_dir,
            f"memory_cache_sample_{sample_idx}.pkl"
        )
        retriever_cache_file = os.path.join(
            memories_dir,
            f"retriever_cache_sample_{sample_idx}.pkl"
        )
        retriever_cache_embeddings_file = os.path.join(
            memories_dir,
            f"retriever_cache_embeddings_sample_{sample_idx}.npy"
        )

        # Check if cached memories exist
        if os.path.exists(memory_cache_file):
            logger.info(f"Loading cached memories for sample {sample_idx}")
            # try:
            with open(memory_cache_file, 'rb') as f:
                cached_memories = pickle.load(f)
            # Restore memories to agent
            agent.memory_system.memories = cached_memories
            if os.path.exists(retriever_cache_file):
                print(f"Found retriever cache files:")
                print(f"  - Retriever cache: {retriever_cache_file}")
                print(f"  - Embeddings cache: {retriever_cache_embeddings_file}")
                agent.memory_system.retriever = agent.memory_system.retriever.load(retriever_cache_file,retriever_cache_embeddings_file)
            else:
                print(f"No retriever cache found at {retriever_cache_file}, loading from memory")
                agent.memory_system.retriever = agent.memory_system.retriever.load_from_local_memory(cached_memories, 'all-MiniLM-L6-v2')
            print(agent.memory_system.retriever.corpus)
            logger.info(f"Successfully loaded {len(cached_memories)} memories")
            agent.memory_system.initialize_clusters_if_needed()
            agent.memory_system.consolidate_memories()
            # except Exception as e:
            #     logger.info(f"Error loading cached memories: {e}. Will recreate memories.")
            #     cached_memories = None
        else:
            logger.info(f"No cached memories found for sample {sample_idx}. Creating new memories.")
            cached_memories = None

            for _,turns in sample.conversation.sessions.items():
                for turn in turns.turns:
                    turn_datatime = turns.date_time
                    conversation_tmp = "Speaker "+ turn.speaker + "says : " + turn.text
                    agent.add_memory(conversation_tmp,time=turn_datatime)
            agent.memory_system.initialize_clusters_if_needed()
            agent.memory_system.consolidate_memories()
            memories_to_cache = agent.memory_system.memories
            with open(memory_cache_file, 'wb') as f:
                pickle.dump(memories_to_cache, f)

            agent.memory_system.retriever.save(retriever_cache_file, retriever_cache_embeddings_file)
            logger.info(f"\nSuccessfully cached {len(memories_to_cache)} memories")

            cluster_summary = agent.memory_system.get_cluster_debug_summary()
            summary_path = fig_root_dir / f"cluster_summary_sample_{sample_idx}.txt"
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(cluster_summary)
            logger.info(f"[Sample {sample_idx}] Cluster summary saved to {summary_path}")

            vis_path = fig_root_dir / f"cluster_vis_sample_{sample_idx}.png"
            
            with open(memory_cache_file, 'wb') as f:
                pickle.dump(memories_to_cache, f)

            agent.memory_system.retriever.save(retriever_cache_file,retriever_cache_embeddings_file)
            logger.info(f"\nSuccessfully cached {len(memories_to_cache)} memories")
            
        # 🔹 Collect compact cluster stats for this sample (for output JSON)
        ms = agent.memory_system
        try:
            sample_cluster_stats = ms.get_cluster_stats_compact()
        except AttributeError:
            # Backward compatibility if method name differs
            sample_cluster_stats = {
                "total_memories": len(getattr(ms, "memories", {}) or {}),
                "num_clusters": len(getattr(ms, "clusters", {}) or {}),
                "cluster_member_counts": {
                    cid: len((info.get("members", []) or []))
                    for cid, info in (getattr(ms, "clusters", {}) or {}).items()
                },
            }

        sample_cluster_stats["sample_id"] = sample_idx
        cluster_stats_by_sample.append(sample_cluster_stats)

        ncl = sample_cluster_stats.get("num_clusters", 0) or 0
        tm = sample_cluster_stats.get("total_memories", 0) or 0
        cmc = sample_cluster_stats.get("cluster_member_counts", {}) or {}

        num_clusters_list.append(ncl)
        max_cluster_size_list.append(max(cmc.values(), default=0))
        mean_cluster_size_list.append((tm / ncl) if ncl else 0.0)

        logger.info(f"\nProcessing sample {sample_idx + 1}/{len(samples)}")

        
        for qa in tqdm(sample.qa, desc=f"QA (sample {sample_idx})", total=len(sample.qa), leave=False):
            if int(qa.category) in allow_categories:
                total_questions += 1
                
                category_counts[qa.category] += 1
                raw_response, user_prompt, raw_context, retrieval_latency_ms, e2e_latency_ms,search_space_size, total_memories, search_space_reduction,retrieval_debug = agent.answer_question(
                    qa.question, qa.category, qa.final_answer
                )

                search_mode = getattr(agent.memory_system, "last_search_mode", "global")
                if search_mode == "cluster":
                    total_cluster_searches += 1
                else:
                    total_global_searches += 1

                try:
                    prediction = json.loads(raw_response)["answer"]
                except:
                    prediction = raw_response
                    logger.info(f"Failed to parse prediction as JSON: {prediction}")
                    error_num += 1

                retrieval_latencies.append(retrieval_latency_ms)
                e2e_latencies.append(e2e_latency_ms)


                # Log results
                logger.info(f"\nQuestion {total_questions}: {qa.question}")
                logger.info(f"Prediction: {prediction}")
                logger.info(f"Reference: {qa.final_answer}")
                logger.info(f"User Prompt: {user_prompt}")
                logger.info(f"Category: {qa.category}")
                logger.info(f"Raw Context: {raw_context}")
                logger.info(f"Retrieval latency (ms): {retrieval_latency_ms:.2f}")
                logger.info(f"End-to-end latency (ms): {e2e_latency_ms:.2f}")
                logger.info(f"End-to-end latency (ms): {e2e_latency_ms:.2f}")
                logger.info(
                    f"Search space size: {search_space_size} / {total_memories} "
                    f"(reduction: {search_space_reduction * 100:.2f}%)"
                )
                # Calculate metrics
                metrics = calculate_metrics(prediction, qa.final_answer) if qa.final_answer else {
                    "exact_match": 0, "f1": 0.0, "rouge1_f": 0.0, "rouge2_f": 0.0, 
                    "rougeL_f": 0.0, "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, 
                    "bleu4": 0.0, "bert_f1": 0.0, "meteor": 0.0, "sbert_similarity": 0.0
                }
                
                all_metrics.append(metrics)
                all_categories.append(qa.category)
                
                # Store individual result
                result = {
                    "sample_id": sample_idx,
                    "question": qa.question,
                    "prediction": prediction,
                    "reference": qa.final_answer,
                    "category": qa.category,
                    "metrics": metrics,
                    "retrieval_latency_ms": retrieval_latency_ms,
                    "end_to_end_latency_ms": e2e_latency_ms,
                    "search_space_size": search_space_size,
                    "total_memories": total_memories,
                    "search_space_reduction": search_space_reduction,
                    "retrieval_debug": retrieval_debug,
                }
                results.append(result)
                
                # Log progress
                if total_questions % 10 == 0:
                    logger.info(f"Processed {total_questions} questions")
                if hasattr(agent.memory_system, "cluster_search_count"):
                    total_cluster_searches += agent.memory_system.cluster_search_count
                if hasattr(agent.memory_system, "global_search_count"):
                    total_global_searches += agent.memory_system.global_search_count
    
    # Calculate aggregate metrics
    aggregate_results = aggregate_metrics(all_metrics, all_categories)
    
    def summarize_latency(arr):
        if not arr:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0,
            }
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "count": len(arr),
        }

    # 🔹 Cluster stats aggregated across samples
    aggregate_results["cluster_stats"] = {
        "num_clusters": summarize_latency(num_clusters_list),
        "max_cluster_size": summarize_latency(max_cluster_size_list),
        "mean_cluster_size": summarize_latency(mean_cluster_size_list),
    }


    latency_metrics = {
        "retrieval_latency_ms": summarize_latency(retrieval_latencies),
        "end_to_end_latency_ms": summarize_latency(e2e_latencies),
    }


    # Prepare final results
    final_results = {
        "model": model,
        "dataset": dataset_path,
        "total_questions": total_questions,
        "category_distribution": {
            str(cat): count for cat, count in category_counts.items()
        },
        "aggregate_metrics": aggregate_results,
        "latency_metrics" : latency_metrics,
        "cluster_stats_by_sample": cluster_stats_by_sample,
        "individual_results": results,
        "search_routing_stats": {
        "cluster_search_count": total_cluster_searches,
        "global_search_count": total_global_searches
    }
    }
    logger.info(f"Error number: {error_num}")
    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    
    # Log summary
    logger.info("\nEvaluation Summary:")
    logger.info(f"Total questions evaluated: {total_questions}")
    logger.info("\nCategory Distribution:")
    for category, count in sorted(category_counts.items()):
        logger.info(f"Category {category}: {count} questions ({count/total_questions*100:.1f}%)")
    
    logger.info("\nAggregate Metrics:")
    for split_name, metrics in aggregate_results.items():
        logger.info(f"\n{split_name.replace('_', ' ').title()}:")
        for metric_name, stats in metrics.items():
            logger.info(f"  {metric_name}:")
            for stat_name, value in stats.items():
                logger.info(f"    {stat_name}: {value:.4f}")
    
    return final_results

def main():
    parser = argparse.ArgumentParser(description="Evaluate text-only agent on LoComo dataset")
    parser.add_argument("--dataset", type=str, default="data/locomo10.json",
                      help="Path to the dataset file")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                      help="OpenAI model to use")
    parser.add_argument("--output", type=str, default=None,
                      help="Path to save evaluation results")
    parser.add_argument("--ratio", type=float, default=1.0,
                      help="Ratio of dataset to evaluate (0.0 to 1.0)")
    parser.add_argument("--backend", type=str, default="sglang",
                      help="Backend to use (openai, ollama, or sglang)")
    parser.add_argument("--temperature_c5", type=float, default=0.5,
                      help="Temperature for the model")
    parser.add_argument("--retrieve_k", type=int, default=10,
                      help="Retrieve k")
    parser.add_argument("--sglang_host", type=str, default="http://localhost",
                      help="SGLang server host (for sglang backend)")
    parser.add_argument("--sglang_port", type=int, default=30000,
                      help="SGLang server port (for sglang backend)")
    args = parser.parse_args()
    run_name = Path(args.output).stem if args.output else \
           f"{Path(args.dataset).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    fig_root_dir = Path("results_CLAG") / "figures" / Path(args.model).stem / run_name
    fig_root_dir.mkdir(parents=True, exist_ok=True)
    
    if args.ratio <= 0.0 or args.ratio > 1.0:
        raise ValueError("Ratio must be between 0.0 and 1.0")
    
    # Convert relative path to absolute path
    dataset_path = os.path.join(os.path.dirname(__file__), args.dataset)
    if args.output:
        output_path = os.path.join(os.path.dirname(__file__), args.output)
    else:
        output_path = None
    
    evaluate_dataset(dataset_path, args.model, output_path, args.ratio, args.backend, args.temperature_c5, args.retrieve_k, args.sglang_host, args.sglang_port,fig_root_dir)

if __name__ == "__main__":
    main()
