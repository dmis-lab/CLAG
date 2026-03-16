from ast import Str
from typing import List, Dict, Optional, Literal, Any, Union
import json
from datetime import datetime
import uuid
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import os
from abc import ABC, abstractmethod
from transformers import AutoModel, AutoTokenizer
from nltk.tokenize import word_tokenize
import pickle
from pathlib import Path
from litellm import completion
import requests
import json as json_lib
import time
import torch
import re
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt 

def simple_tokenize(text):
    return word_tokenize(text)

class BaseLLMController(ABC):
    @abstractmethod
    def get_completion(self, prompt: str) -> str:
        """Get completion from LLM"""
        pass

class OpenAIController(BaseLLMController):
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        try:
            from openai import OpenAI
            self.model = model
            if api_key is None:
                api_key = os.getenv('OPENAI_API_KEY')
            if api_key is None:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("OpenAI package not found. Install it with: pip install openai")
    
    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt}
            ],
            response_format=response_format,
            temperature=temperature,
            max_tokens=1000
        )
        return response.choices[0].message.content

class OllamaController(BaseLLMController):
    def __init__(self, model: str = "llama2"):
        from ollama import chat
        self.model = model
    
    def _generate_empty_value(self, schema_type: str, schema_items: dict = None) -> Any:
        if schema_type == "array":
            return []
        elif schema_type == "string":
            return ""
        elif schema_type == "object":
            return {}
        elif schema_type == "number":
            return 0
        elif schema_type == "boolean":
            return False
        return None

    def _generate_empty_response(self, response_format: dict) -> dict:
        if "json_schema" not in response_format:
            return {}
            
        schema = response_format["json_schema"]["schema"]
        result = {}
        
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                result[prop_name] = self._generate_empty_value(prop_schema["type"], 
                                                            prop_schema.get("items"))
        
        return result

    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        try:
            response = completion(
                model="ollama_chat/{}".format(self.model),
                messages=[
                    {"role": "system", "content": "You must respond with a JSON object."},
                    {"role": "user", "content": prompt}
                ],
                response_format=response_format,
            )
            return response.choices[0].message.content
        except Exception as e:
            empty_response = self._generate_empty_response(response_format)
            return json.dumps(empty_response)

class SGLangController(BaseLLMController):
    def __init__(self, model: str = "llama2", sglang_host: str = "http://localhost", sglang_port: int = 30000):
        self.model = model
        self.sglang_host = sglang_host
        self.sglang_port = sglang_port
        self.base_url = f"{sglang_host}:{sglang_port}"
    
    def _generate_empty_value(self, schema_type: str, schema_items: dict = None) -> Any:
        if schema_type == "array":
            return []
        elif schema_type == "string":
            return ""
        elif schema_type == "object":
            return {}
        elif schema_type == "number" or schema_type == "integer":
            return 0
        elif schema_type == "boolean":
            return False
        return None

    def _generate_empty_response(self, response_format: dict) -> dict:
        if "json_schema" not in response_format:
            return {}
            
        schema = response_format["json_schema"]["schema"]
        result = {}
        
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                result[prop_name] = self._generate_empty_value(prop_schema["type"], 
                                                            prop_schema.get("items"))
        
        return result

    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        try:
            # Extract JSON schema from response_format and convert to string format
            json_schema = response_format.get("json_schema", {}).get("schema", {})
            json_schema_str = json.dumps(json_schema)
            
            # Prepare SGLang request with correct format
            payload = {
                "text": prompt,
                "sampling_params": {
                    "temperature": temperature,
                    "max_new_tokens": 1000,
                    "json_schema": json_schema_str  # SGLang expects JSON schema as string
                }
            }
            
            # Make request to SGLang server
            response = requests.post(
                f"{self.base_url}/generate",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                # SGLang returns the generated text in 'text' field
                generated_text = result.get("text", "")
                return generated_text
            else:
                print(f"SGLang server returned status {response.status_code}: {response.text}")
                raise Exception(f"SGLang server error: {response.status_code}")
                
        except Exception as e:
            print(f"SGLang completion error: {e}")
            empty_response = self._generate_empty_response(response_format)
            return json.dumps(empty_response)

class LiteLLMController(BaseLLMController):
    """LiteLLM controller for universal LLM access including Ollama and SGLang"""
    def __init__(self, model: str, api_base: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key or "EMPTY"
    
    def _generate_empty_value(self, schema_type: str, schema_items: dict = None) -> Any:
        if schema_type == "array":
            return []
        elif schema_type == "string":
            return ""
        elif schema_type == "object":
            return {}
        elif schema_type == "number":
            return 0
        elif schema_type == "boolean":
            return False
        return None

    def _generate_empty_response(self, response_format: dict) -> dict:
        if "json_schema" not in response_format:
            return {}
            
        schema = response_format["json_schema"]["schema"]
        result = {}
        
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                result[prop_name] = self._generate_empty_value(prop_schema["type"], 
                                                            prop_schema.get("items"))
        
        return result

    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        try:
            # Prepare completion arguments
            completion_args = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You must respond with a JSON object."},
                    {"role": "user", "content": prompt}
                ],
                "response_format": response_format,
                "temperature": temperature
            }
            
            # Add API base and key if provided
            if self.api_base:
                completion_args["api_base"] = self.api_base
            if self.api_key:
                completion_args["api_key"] = self.api_key
                
            response = completion(**completion_args)
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"LiteLLM completion error: {e}")
            empty_response = self._generate_empty_response(response_format)
            return json.dumps(empty_response)

class LLMController:
    """LLM-based controller for memory metadata generation"""
    def __init__(self, 
                 backend: Literal["openai", "ollama", "sglang"] = "sglang",
                 model: str = "gpt-4", 
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000):
        if backend == "openai":
            self.llm = OpenAIController(model, api_key)
        elif backend == "ollama":
            # Use LiteLLM to control Ollama with JSON output
            ollama_model = f"ollama/{model}" if not model.startswith("ollama/") else model
            self.llm = LiteLLMController(
                model=ollama_model, 
                api_base="http://localhost:11434", 
                api_key="EMPTY"
            )
        elif backend == "sglang":
            # Direct SGLang API calls (better performance, no proxy)
            self.llm = SGLangController(model, sglang_host, sglang_port)
        else:
            raise ValueError("Backend must be 'openai', 'ollama', or 'sglang'")

class MemoryNote:
    """Basic memory unit with metadata"""
    def __init__(self, 
                 content: str,
                 id: Optional[str] = None,
                 keywords: Optional[List[str]] = None,
                 links: Optional[Dict] = None,
                 importance_score: Optional[float] = None,
                 retrieval_count: Optional[int] = None,
                 timestamp: Optional[str] = None,
                 last_accessed: Optional[str] = None,
                 context: Optional[str] = None, 
                 evolution_history: Optional[List] = None,
                 category: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 llm_controller: Optional[LLMController] = None,
                 cluster_id: Optional[str] = None
                 ):
        
        self.content = content
        
        # Generate metadata using LLM if not provided and controller is available
        if llm_controller and any(param is None for param in [keywords, context, category, tags]):
            analysis = self.analyze_content(content, llm_controller)
            print("analysis", analysis)
            keywords = keywords or analysis["keywords"]
            context = context or analysis["context"]
            tags = tags or analysis["tags"]
        
        # Set default values for optional parameters
        self.id = id or str(uuid.uuid4())
        self.keywords = keywords or []
        self.links = links or []
        self.importance_score = importance_score or 1.0
        self.retrieval_count = retrieval_count or 0
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time
        
        # Handle context that can be either string or list
        self.context = context or "General"
        if isinstance(self.context, list):
            self.context = " ".join(self.context)  # Convert list to string by joining
            
        self.evolution_history = evolution_history or []
        self.category = category or "Uncategorized"
        self.tags = tags or []
        self.cluster_id = cluster_id

    @staticmethod
    def analyze_content(content: str, llm_controller: LLMController) -> Dict:            
        """Analyze content to extract keywords, context, and other metadata"""
        prompt = """Generate a structured analysis of the following content by:
            1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
            2. Extracting core themes and contextual elements
            3. Creating relevant categorical tags

            Format the response as a JSON object:
            {
                "keywords": [
                    // several specific, distinct keywords that capture key concepts and terminology
                    // Order from most to least important
                    // Don't include keywords that are the name of the speaker or time
                    // At least three keywords, but don't be too redundant.
                ],
                "context": 
                    // one sentence summarizing:
                    // - Main topic/domain
                    // - Key arguments/points
                    // - Intended audience/purpose
                ,
                "tags": [
                    // several broad categories/themes for classification
                    // Include domain, format, and type tags
                    // At least three tags, but don't be too redundant.
                ]
            }

            Content for analysis:
            """ + content
        try:
            response = llm_controller.llm.get_completion(prompt,response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "keywords": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "context": {
                                    "type": "string",
                                },
                                "tags": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                            },
                            "required": ["keywords", "context", "tags"],
                            "additionalProperties": False
                        },
                        "strict": True
                }
            })
            
            try:
                # Clean the response in case there's extra text
                response_cleaned = response.strip()
                # Try to find JSON content if wrapped in other text
                if not response_cleaned.startswith('{'):
                    start_idx = response_cleaned.find('{')
                    if start_idx != -1:
                        response_cleaned = response_cleaned[start_idx:]
                if not response_cleaned.endswith('}'):
                    end_idx = response_cleaned.rfind('}')
                    if end_idx != -1:
                        response_cleaned = response_cleaned[:end_idx+1]
                
                analysis = json.loads(response_cleaned)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error in analyze_content: {e}")
                print(f"Raw response: {response}")
                analysis = {
                    "keywords": [],
                    "context": "General",
                    "tags": []
                }
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing content: {str(e)}")
            print(f"Raw response: {response}")
            return {
                "keywords": [],
                "context": "General",
                "category": "Uncategorized",
                "tags": []
            }

class HybridRetriever:
    """Hybrid retrieval system combining BM25 and semantic search."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', alpha: float = 0.5):
        """Initialize the hybrid retriever.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
            alpha: Weight for combining BM25 and semantic scores (0 = only BM25, 1 = only semantic)
        """
        self.model = SentenceTransformer(model_name)
        self.alpha = alpha
        self.bm25 = None
        self.corpus = []
        self.embeddings = None
        self.document_ids = {}  # Map document content to its index
        
    
    def save(self, retriever_cache_file: str, retriever_cache_embeddings_file: str):
        """Save retriever state to disk"""
        
        # Save embeddings using numpy
        if self.embeddings is not None:
            np.save(retriever_cache_embeddings_file, self.embeddings)
            
        # Save everything else using pickle
        state = {
            'alpha': self.alpha,
            'bm25': self.bm25,
            'corpus': self.corpus,
            'document_ids': self.document_ids,
            'model_name': 'all-MiniLM-L6-v2'  # Default value for model name
        }
        
        # Try to get the actual model name if possible
        try:
            state['model_name'] = self.model.get_config_dict()['model_name']
        except (AttributeError, KeyError):
            pass
            
        with open(retriever_cache_file, 'wb') as f:
            pickle.dump(state, f)
            
    @classmethod
    def load(cls, retriever_cache_file: str, retriever_cache_embeddings_file: str):
        """Load retriever state from disk"""
        # Load the pickled state
        with open(retriever_cache_file, 'rb') as f:
            state = pickle.load(f)
            
        # Create new instance
        retriever = cls(model_name=state['model_name'], alpha=state['alpha'])
        retriever.bm25 = state['bm25']
        retriever.corpus = state['corpus']
        retriever.document_ids = state.get('document_ids', {})
        
        # Load embeddings from numpy file if it exists
        if retriever_cache_embeddings_file.exists():
            retriever.embeddings = np.load(retriever_cache_embeddings_file)
            
        return retriever
    
    @classmethod
    def load_from_local_memory(cls, memories: Dict, model_name: str, alpha: float) -> bool:
        """Load retriever state from memory"""
        all_docs = [", ".join(m.keywords) for m in memories.values()] #[m.content for m in memories.values()]
        retriever = cls(model_name, alpha)
        retriever.add_documents(all_docs)
        return retriever
    
    def add_documents(self, documents: List[str]) -> bool:
        """One-time Add documents to both BM25 and semantic index"""
        if not documents:
            return
            
        # Tokenize for BM25
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # Create embeddings
        self.embeddings = self.model.encode(documents)
        self.corpus = documents
        doc_idx = 0
        for document in documents:
            self.document_ids[document] = doc_idx
            doc_idx += 1

        return True

    def add_document(self, document: str) -> bool:
        """Add a single document to the retriever.
        
        Args:
            document: Text content to add
            
        Returns:
            bool: True if document was added, False if it was already present
        """
        # Check if document already exists
        if document in self.document_ids:
            return False
            
        # Add to corpus and get index
        doc_idx = len(self.corpus)
        self.corpus.append(document)
        self.document_ids[document] = doc_idx
        
        # Update BM25
        if self.bm25 is None:
            # First document, initialize BM25
            tokenized_corpus = [simple_tokenize(document)]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            # Add to existing BM25
            tokenized_doc = simple_tokenize(document)
            self.bm25.add_document(tokenized_doc)
        
        # Update embeddings
        doc_embedding = self.model.encode([document], convert_to_tensor=True)
        if self.embeddings is None:
            self.embeddings = doc_embedding
        else:
            self.embeddings = torch.cat([self.embeddings, doc_embedding])
            
        return True
        
    def retrieve(self, query: str, k: int = 5) -> List[int]:
        """Retrieve documents using hybrid scoring"""
        if not self.corpus:
            return []
            
        # Get BM25 scores
        tokenized_query = query.lower().split()
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))
        
        # Normalize BM25 scores if they exist
        if len(bm25_scores) > 0:
            bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-6)
        
        # Get semantic scores
        query_embedding = self.model.encode([query])[0]
        semantic_scores = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Combine scores
        hybrid_scores = self.alpha * bm25_scores + (1 - self.alpha) * semantic_scores
        
        # Get top k indices
        k = min(k, len(self.corpus))
        top_k_indices = np.argsort(hybrid_scores)[-k:][::-1]
        return top_k_indices.tolist()

class SimpleEmbeddingRetriever:
    """Simple retrieval system using only text embeddings."""
    
    _model_cache = {}

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', model: SentenceTransformer = None):
       
        if model is not None:
            self.model = model
        else:
            if model_name in SimpleEmbeddingRetriever._model_cache:
                self.model = SimpleEmbeddingRetriever._model_cache[model_name]
            else:
                self.model = SentenceTransformer(model_name, device='cuda')
                SimpleEmbeddingRetriever._model_cache[model_name] = self.model

        self.corpus = []
        self.embeddings = None


        self.document_ids = {}

        
    def add_documents(self, documents: List[str]):
        """Add documents to the retriever."""
        # Reset if no existing documents
        if not self.corpus:
            self.corpus = documents
            # print("documents", documents, len(documents))
            self.embeddings = self.model.encode(documents)
            self.document_ids = {doc: idx for idx, doc in enumerate(documents)}
        else:
            # Append new documents
            start_idx = len(self.corpus)
            self.corpus.extend(documents)
            new_embeddings = self.model.encode(documents)
            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
            for idx, doc in enumerate(documents):
                self.document_ids[doc] = start_idx + idx
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, float]]:
        """Search for similar documents using cosine similarity.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of dicts with document text and score
        """
        if not self.corpus:
            return []
        # print("corpus", len(self.corpus), self.corpus)
        # Encode query
        query_embedding = self.model.encode([query])[0]
        
        # Calculate cosine similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        # Get top k results
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
            
        return top_k_indices
        
    def save(self, retriever_cache_file: str, retriever_cache_embeddings_file: str):
        """Save retriever state to disk"""
        # Save embeddings using numpy
        if self.embeddings is not None:
            np.save(retriever_cache_embeddings_file, self.embeddings)
        
        # Save other attributes
        state = {
            'corpus': self.corpus,
            'document_ids': self.document_ids
        }
        with open(retriever_cache_file, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, retriever_cache_file: str, retriever_cache_embeddings_file: str):
        """Load retriever state from disk"""
        print(f"Loading retriever from {retriever_cache_file} and {retriever_cache_embeddings_file}")
        
        # Load embeddings
        if os.path.exists(retriever_cache_embeddings_file):
            print(f"Loading embeddings from {retriever_cache_embeddings_file}")
            self.embeddings = np.load(retriever_cache_embeddings_file)
            print(f"Embeddings shape: {self.embeddings.shape}")
        else:
            print(f"Embeddings file not found: {retriever_cache_embeddings_file}")
        
        # Load other attributes
        if os.path.exists(retriever_cache_file):
            print(f"Loading corpus from {retriever_cache_file}")
            with open(retriever_cache_file, 'rb') as f:
                state = pickle.load(f)
                self.corpus = state['corpus']
                self.document_ids = state['document_ids']
                print(f"Loaded corpus with {len(self.corpus)} documents")
        else:
            print(f"Corpus file not found: {retriever_cache_file}")
            
        return self

    @classmethod
    def load_from_local_memory(cls, memories: Dict, model_name: str) -> 'SimpleEmbeddingRetriever':
        """Load retriever state from memory"""
        # Create documents combining content and metadata for each memory
        all_docs = []
        for m in memories.values():
            metadata_text = f"{m.context} {' '.join(m.keywords)} {' '.join(m.tags)}"
            doc = f"{m.content} , {metadata_text}"
            all_docs.append(doc)
            
        # Create and initialize retriever
        retriever = cls(model_name)
        retriever.add_documents(all_docs)
        return retriever

class AgenticMemorySystem:
    """Memory management system with embedding-based retrieval"""
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm_backend: str = "sglang",
                 llm_model: str = "gpt-4o-mini",
                 evo_threshold: int = 100,
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000):
        self.memories = {}  # id -> MemoryNote
        self.embedding_model = SentenceTransformer(model_name, device='cuda')
        self.retriever = SimpleEmbeddingRetriever(model_name, model=self.embedding_model)
        self.llm_controller = LLMController(llm_backend, llm_model, api_key, api_base, sglang_host, sglang_port)
        self.evolution_system_prompt = '''
                                You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
                                Analyze the the new memory note according to keywords and context, also with their several nearest neighbors memory.
                                Make decisions about its evolution.  

                                The new memory context:
                                {context}
                                content: {content}
                                keywords: {keywords}

                                The nearest neighbors memories:
                                {nearest_neighbors_memories}

                                Based on this information, determine:
                                1. Should this memory be evolved? Consider its relationships with other memories.
                                2. What specific actions should be taken (strengthen, update_neighbor)?
                                   2.1 If choose to strengthen the connection, which memory should it be connected to? Can you give the updated tags of this memory?
                                   2.2 If choose to update_neighbor, you can update the context and tags of these memories based on the understanding of these memories. If the context and the tags are not updated, the new context and tags should be the same as the original ones. Generate the new context and tags in the sequential order of the input neighbors.
                                Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.
                                Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
                                The number of neighbors is {neighbor_number}.
                                Return your decision in JSON format with the following structure:
                                {{
                                    "should_evolve": True or False,
                                    "actions": ["strengthen", "update_neighbor"],
                                    "suggested_connections": ["neighbor_memory_ids"],
                                    "tags_to_update": ["tag_1",..."tag_n"], 
                                    "new_context_neighborhood": ["new context",...,"new context"],
                                    "new_tags_neighborhood": [["tag_1",...,"tag_n"],...["tag_1",...,"tag_n"]],
                                }}
                                '''
        self.evo_cnt = 0 
        self.evo_threshold = evo_threshold

        self.clusters: Dict[str, Dict] = {}
        self.cluster_retrievers: Dict[str, SimpleEmbeddingRetriever] = {}
        self.last_search_space_size = None
        self.last_total_memories = None
        self.last_search_mode = "global"  # "cluster" or "global"
        self.cluster_centroids: Dict[str, np.ndarray] = {}
        self.init_cluster_min_memories = 100   
        self.init_n_clusters = 3     
        self.cluster_split_threshold = 300     
        self.routing_top_k = 5                
        self.clusters_initialized = False
        self.force_top3 = False
        


    def get_cluster_debug_summary(self, max_members_per_cluster: int = 3) -> str:

        lines = []
        lines.append(f"Total Memories: {len(self.memories)}")
        lines.append(f"Total Clusters: {len(self.clusters)}")
        lines.append("")

        for cid, info in self.clusters.items():
            summary = info.get('summary', 'No summary')
            tags = list(info.get('tags', []))
            members = info.get('members', [])

            lines.append(f"[Cluster ID]: {cid}")
            lines.append(f"  - Summary: {summary}")
            lines.append(f"  - Tags: {tags}")
            lines.append(f"  - Member Count: {len(members)}")
            lines.append(f"  - Member Contents Sample (Top {max_members_per_cluster}):")

            for i, mid in enumerate(members[:max_members_per_cluster]):
                mem = self.memories.get(mid)
                if mem:
                    preview = mem.content[:100].replace("\n", " ")
                    lines.append(f"    {i+1}. [{mid[:8]}] {preview}...")

            lines.append("-" * 50)

        return "\n".join(lines)


    def get_cluster_stats_compact(self) -> Dict[str, Any]:
        """Return compact per-sample cluster stats for JSON logging.
        Includes only counts (no text snippets) to keep output small.
        """
        cluster_member_counts: Dict[str, int] = {}
        if self.clusters:
            for cid, info in self.clusters.items():
                members = info.get("members", []) or []
                cluster_member_counts[cid] = len(members)

        return {
            "total_memories": len(self.memories),
            "num_clusters": len(cluster_member_counts),
            "cluster_member_counts": cluster_member_counts,
        }


    def select_clusters_for_query(
    self,
    query: str,
    query_tags: List[str],
    top_n: int = 10,
    candidate_k: int = 5,
) -> List[str]:
        if not self.clusters or not self.cluster_centroids:
            return []
        if self.force_top3:
            forced = sorted(self.clusters.keys())[:3]
            self.last_candidate_cluster_ids = list(forced)
            self.last_selected_cluster_ids = list(forced)
            return forced


        meta = " ".join(query_tags or [])
        q_text = f"{query} {meta}"
        model = self.retriever.model
        try:
            q_emb = model.encode([q_text])[0]
        except Exception as e:
            print(f"[select_clusters_for_query_agentic] encode error: {e}")
            return []


        cid_dists = []
        for cid, center in self.cluster_centroids.items():
            c = np.array(center)
            diff = q_emb - c
            dist = float(np.dot(diff, diff))
            cid_dists.append((cid, dist))

        if not cid_dists:
            return []

        cid_dists.sort(key=lambda x: x[1])
        candidate = cid_dists[:candidate_k]          # [(cid, dist), ...]
        candidate_cids = [cid for cid, _ in candidate]


        cluster_descriptions = []
        for cid, dist in candidate:
            info = self.clusters.get(cid, {})
            member_ids = info.get("members", [])
            cluster_tags = list(info.get("tags", []))
            summary = info.get("summary", "")


            examples = []
            rep_ids = info.get("rep_ids") or []
            if rep_ids:
                for rid in rep_ids[:3]:
                    m = self.memories.get(rid)
                    if m:
                        snippet = (m.content or "")[:160].replace("\n", " ")
                        examples.append(f"- {snippet}")
            else:
                for mid in member_ids[:3]:
                    m = self.memories.get(mid)
                    if m:
                        snippet = (m.content or "")[:160].replace("\n", " ")
                        examples.append(f"- {snippet}")

            example_block = "\n".join(examples)

            cluster_descriptions.append({
                "cluster_id": cid,
                "distance": dist,         
                "cluster_tags": cluster_tags,
                "summary": summary,
                "examples": example_block,
            })

        prompt = (
            "You are an AI memory router that selects the most relevant memory clusters for a given query.\n"
            "You will be given several candidate clusters. Each cluster has:\n"
            "- cluster_id\n"
            "- one-sentence summary\n"
            "- optional tags\n"
            "- one or more representative memory examples\n\n"
            "Your task:\n"
            "1. Analyze the user query and query_tags.\n"
            "2. For each candidate cluster, judge how relevant it is.\n"
            f"3. Decide how many clusters are actually needed. You should return between 0 and {top_n} clusters.\n"
            "   - If one cluster is definitely sufficient for answering the query, return just that one.\n"
            "   - Include additional clusters if they are needed for answering the query.\n"
            "4. If none of the clusters are meaningfully related, return an empty list.\n\n"
            "Return ONLY JSON with this format:\n"
            "{\n"
            '  "selected_clusters": ["cluster_id_1", "cluster_id_2"]\n'
            "}\n"
            "If no cluster is relevant, return:\n"
            "{\n"
            '  "selected_clusters": []\n'
            "}\n\n"
            f"User query: {query}\n"
            f"Query tags: {query_tags}\n\n"
            "Candidate clusters:\n"
        )

        for cd in cluster_descriptions:
            prompt += (
                f"- cluster_id: {cd['cluster_id']}\n"
                f"  summary: {cd['summary']}\n"
                f"  cluster_tags: {cd['cluster_tags']}\n"
                f"  representative_memories:\n"
                f"{cd['examples']}\n"
            )


        try:
            response = self.llm_controller.llm.get_completion(
                prompt,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "cluster_selection_response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "selected_clusters": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            },
                            "required": ["selected_clusters"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                }
            )

            response_cleaned = response.strip()
            if not response_cleaned.startswith("{"):
                start = response_cleaned.find("{")
                if start != -1:
                    response_cleaned = response_cleaned[start:]
            if not response_cleaned.endswith("}"):
                end = response_cleaned.rfind("}")
                if end != -1:
                    response_cleaned = response_cleaned[:end+1]

            data = json.loads(response_cleaned)
            selected = data.get("selected_clusters", [])
        except Exception as e:
            print(f"[select_clusters_for_query_agentic] error: {e}")
            print("Raw response:", response if "response" in locals() else None)
            return []


        valid = []
        for cid in selected:
            if cid in candidate_cids and cid in self.clusters:
                valid.append(cid)
            if len(valid) >= top_n:
                break

        self.last_candidate_cluster_ids = list(candidate_cids)
        self.last_selected_cluster_ids = list(valid)


        return valid





    def initialize_clusters_if_needed(self):


        if len(self.memories) < self.init_cluster_min_memories:
            return
        if self.clusters:
            self.clusters_initialized = True
            return

        self.cluster_memories_kmeans(max_clusters=self.init_n_clusters)

        self._build_cluster_profiles_with_llm()

        self.clusters_initialized = True
        self._rebuild_cluster_retrievers()

    def parse_cluster_profile(self, raw: str):

        if isinstance(raw, dict):
            return raw

        text = str(raw).strip()

        try:
            return json.loads(text)
        except Exception:
            pass

        summary = ""
        m = re.search(r'"summary"\s*:\s*"([^"]*)"', text)
        if m:
            summary = m.group(1)

        tags = []
        m = re.search(r'"tags"\s*:\s*\[(.*)', text, re.DOTALL)
        if m:
            tag_block = m.group(1)
            candidates = re.findall(r'"([^"]+)"', tag_block)
            seen = set()
            for t in candidates:
                if t in seen:
                    continue
                seen.add(t)
                tags.append(t)
                if len(tags) >= 5:
                    break

        if not summary and not tags:
            raise ValueError("Could not parse summary/tags from LLM response")

        return {
            "summary": summary,
            "tags": tags,
        }
    def _rebuild_cluster_retrievers(self):
        self.cluster_retrievers = {}

        try:
            model_name = self.retriever.model.get_config_dict().get("model_name", "all-MiniLM-L6-v2")
        except Exception:
            model_name = "all-MiniLM-L6-v2"

        for cid, info in self.clusters.items():
            member_ids = info.get("members", [])
            docs = []
            for mid in member_ids:
                m = self.memories.get(mid)
                if not m:
                    continue
                meta = f"{m.context} {' '.join(m.keywords)} {' '.join(m.tags)}"
                docs.append(m.content + " , " + meta)

            if not docs:
                continue

            retr = SimpleEmbeddingRetriever(model_name, model=self.retriever.model)
            retr.add_documents(docs)
            self.cluster_retrievers[cid] = retr


    def _build_cluster_profiles_with_llm(self, top_m: int = 3):
        if not self.clusters or not self.cluster_centroids:
            return

        model = self.retriever.model

        for cid, info in self.clusters.items():
            member_ids = info.get("members", [])
            if not member_ids:
                continue

            docs = []
            mids = []
            for mid in member_ids:
                m = self.memories.get(mid)
                if not m:
                    continue
                meta = f"{m.context} {' '.join(m.keywords)} {' '.join(m.tags)}"
                docs.append(m.content + " , " + meta)
                mids.append(mid)

            if not docs:
                continue

            emb = model.encode(docs)
            center = np.array(self.cluster_centroids[cid])

            dists = np.sum((emb - center) ** 2, axis=1)
            order = np.argsort(dists)
            top_indices = order[: min(top_m, len(order))]

            samples = []
            for idx in top_indices:
                m = self.memories[mids[idx]]
                samples.append(
                    f"- content: {m.content}\n"
                    f"  context: {m.context}\n"
                    f"  tags: {m.tags}\n"
                )
            samples_text = "\n".join(samples)

            prompt = f"""
                You are a memory clustering assistant.

                Below are several memory snippets that belong to the SAME cluster:

                {samples_text}

                Your task:
                1. Write ONE short sentence summary that best describes the main topic of this cluster.
                2. Return EXACTLY 3 tags.
                - Each tag must be a single word.
                - Do NOT repeat the same tag.

                Return ONLY a JSON object with the following KEYS (this is a schema, not the actual content):

                {{
                    "summary": "...your one-sentence summary here...",
                    "tags": ["tag_1", "tag_2", "tag_3"]
                }}
                """

            try:
                resp = self.llm_controller.llm.get_completion(
                    prompt,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "cluster_profile",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "summary": {"type": "string"},
                                    "tags": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                },
                                "required": ["summary", "tags"],
                                "additionalProperties": False,
                            },
                            "strict": True,
                        },
                    },
                    temperature=0.3,
                )

                data = self.parse_cluster_profile(resp)

                summary = data.get("summary")
                raw_tags = data.get("tags", [])

                if isinstance(summary, str) and summary.strip():
                    info["summary"] = summary.strip()

                clean_tags = []
                seen = set()
                if isinstance(raw_tags, (list, set, tuple)):
                    for t in raw_tags:
                        if not isinstance(t, str):
                            continue
                        t = t.strip()
                        if not t:
                            continue
                        t = t.split()[0] 
                        if t in seen:
                            continue
                        seen.add(t)
                        clean_tags.append(t)
                        if len(clean_tags) >= 3:
                            break

                if clean_tags:
                    info["tags"] = clean_tags

            except Exception as e:
                try:
                    full_text = str(resp)
                except Exception:
                    full_text = "<unprintable>"
                print(
                    f"[cluster profile LLM error] cid={cid}, err={e}, "
                    f"full_text={full_text}"
                )
                continue



    def cluster_memories_kmeans(self, max_clusters: int = 10):

        if not self.memories:
            return

        mem_ids = list(self.memories.keys())
        docs = []
        for mid in mem_ids:
            m = self.memories[mid]
            meta = " ".join(m.keywords + m.tags)
            text = f"{m.content} {m.context} {meta}"
            docs.append(text)

        n_docs = len(docs)
        if n_docs == 0:
            return

        embeddings = self.retriever.model.encode(docs)

        n_clusters = min(max_clusters, n_docs)

        self.cluster_centroids = {}

        if n_clusters <= 1:
            cid = "cluster_0"
            cluster_tags = set()
            for mid in mem_ids:
                m = self.memories[mid]
                cluster_tags.update(m.tags)

            summary = "General cluster of all memories."

            self.clusters = {
                cid: {
                    "members": mem_ids,
                    "tags": cluster_tags,
                    "summary": summary,
                }
            }
            for mid in mem_ids:
                self.memories[mid].cluster_id = cid

            center = np.mean(embeddings, axis=0)
            self.cluster_centroids[cid] = center
            return

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        centers = kmeans.cluster_centers_   # (n_clusters, dim)

        from collections import defaultdict, Counter
        label_to_members = defaultdict(list)
        for mid, lab in zip(mem_ids, labels):
            label_to_members[int(lab)].append(mid)

        self.clusters = {}
        for lab, mids in label_to_members.items():
            cid = f"cluster_{lab}"

            tag_counter = Counter()
            contexts = []
            for mid in mids:
                m = self.memories[mid]
                tag_counter.update(m.tags)
                if m.context:
                    contexts.append(m.context)

            top_tags = [t for t, _ in tag_counter.most_common(5)]
            cluster_tags = set(top_tags)

            if top_tags:
                summary = f"Memories related to: {', '.join(top_tags)}"
            elif contexts:
                summary = contexts[0][:120]
            else:
                summary = "Clustered memories with similar semantic content."

            self.clusters[cid] = {
                "members": mids,
                "tags": cluster_tags,
                "summary": summary,
            }


            self.cluster_centroids[cid] = centers[lab]


        for cid, info in self.clusters.items():
            for mid in info["members"]:
                if mid in self.memories:
                    self.memories[mid].cluster_id = cid


    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        note = MemoryNote(content=content, llm_controller=self.llm_controller, timestamp=time, **kwargs)

        pred_cid = None
        new_emb = None

        # (1) Predict-only cluster routing (NO side effects)
        if self.clusters and self.cluster_centroids:
            meta = " ".join(note.keywords + note.tags)
            text_for_cluster = f"{note.content} {note.context} {meta}"
            new_emb = self.retriever.model.encode([text_for_cluster])[0]
            pred_cid = self.route_new_memory_with_llm(note, new_emb) 


        cid_for_evo = pred_cid if pred_cid else None   
        evo_label, note = self.process_memory(note, cluster_id=cid_for_evo)


        self.memories[note.id] = note
        base_doc = (
            "content:" + note.content + " context:" + note.context
            + " keywords: " + ", ".join(note.keywords)
            + " tags: " + ", ".join(note.tags)
        )
        self.retriever.add_documents([base_doc])

        # (4) Commit cluster assignment AFTER evolve
        if self.clusters and self.cluster_centroids:
            if pred_cid and pred_cid in self.clusters:
                # existing cluster
                self.clusters[pred_cid]["members"].append(note.id)
                note.cluster_id = pred_cid
                retr = self.cluster_retrievers.get(pred_cid)
                if retr is not None:
                    meta2 = f"{note.context} {' '.join(note.keywords)} {' '.join(note.tags)}"
                    retr.add_documents([note.content + " , " + meta2])
            else:

                meta2 = " ".join(note.keywords + note.tags)
                text2 = f"{note.content} {note.context} {meta2}"
                new_emb2 = self.retriever.model.encode([text2])[0]
                self._create_new_cluster(note, new_emb2)

        self.memories[note.id] = note  

        if evo_label is True:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()

        self.initialize_clusters_if_needed()
        return note.id

    def route_new_memory_with_llm(
    self,
    note: MemoryNote,
    new_emb: np.ndarray,
    sim_threshold: float = 0.1,
) -> Optional[str]:
        if not self.clusters or not self.cluster_centroids:
            return None

        dists: list[tuple[float, str]] = []
        for cid, center in self.cluster_centroids.items():
            center_vec = np.array(center)
            diff = new_emb - center_vec
            dist = float(np.dot(diff, diff))
            dists.append((dist, cid))

        if not dists:
            return None

        dists.sort(key=lambda x: x[0])
        top_k = min(self.routing_top_k, len(dists))
        candidate = dists[:top_k]
        candidate_cids = [cid for _, cid in candidate]

        candidates_desc = []
        for _, cid in candidate:
            info = self.clusters.get(cid, {})
            summary = info.get("summary", "")
            tags = list(info.get("tags", []))
            size = len(info.get("members", []))

            examples = []
            rep_ids = info.get("rep_ids") or []
            if rep_ids:
                for rid in rep_ids[:3]:
                    m = self.memories.get(rid)
                    if m:
                        snippet = (m.content or "")[:160].replace("\n", " ")
                        examples.append(f"- {snippet}")
            else:
                for mid in info.get("members", [])[:3]:
                    m = self.memories.get(mid)
                    if m:
                        snippet = (m.content or "")[:160].replace("\n", " ")
                        examples.append(f"- {snippet}")

            example_block = "\n".join(examples)
            candidates_desc.append(
                f"- cluster_id: {cid}\n"
                f"  size: {size}\n"
                f"  summary: {summary}\n"
                f"  tags: {tags}\n"
                f"  representative_memories:\n"
                f"{example_block}\n"
            )
        candidates_text = "\n".join(candidates_desc)

        prompt = f"""
            You are a memory routing assistant.

            A new memory has arrived:
            - Content: {note.content}
            - Context: {note.context}
            - Tags: {note.tags}

            Here are candidate clusters (pre-selected by vector similarity) that might relate to this memory:
            {candidates_text}

            Your task:
            1. Analyze the topics and contexts of the candidate clusters provided above.
            2. Select the single `cluster_id` that exhibits the highest semantic relevance and thematic alignment with the new memory.
            3. You MUST choose exactly one cluster_id from the candidate list.

            Return ONLY a JSON object:
            {{
            "choice": "cluster_1"
            }}
        """

        try:
            resp = self.llm_controller.llm.get_completion(
                prompt,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "route_choice",
                        "schema": {
                            "type": "object",
                            "properties": {"choice": {"type": "string"}},
                            "required": ["choice"],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    },
                },
                temperature=0.2,
            )
            text = resp.strip()
            if not text.startswith("{"):
                text = text[text.find("{"):]
            if not text.endswith("}"):
                text = text[: text.rfind("}") + 1]
            data = json.loads(text)
            choice = data.get("choice", "").strip()
        except Exception as e:
            print(f"[route_new_memory_with_llm] error: {e}")
            choice = ""


        if not choice or choice not in candidate_cids or choice not in self.clusters:
            return None


        center_vec = np.array(self.cluster_centroids.get(choice))
        if center_vec is None:
            return None

        q = new_emb.astype(np.float32)
        c = center_vec.astype(np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-8)
        c_norm = c / (np.linalg.norm(c) + 1e-8)
        cos_sim = float(np.dot(q_norm, c_norm))


        try:
            log_dir = os.path.join(os.path.dirname(__file__), "logs_CLAG")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "cluster_route_similarity.txt")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(
                    f"\n[route_new_memory] note_id={note.id}, "
                    f"choice={choice}, cos_sim={cos_sim:.4f}, threshold={sim_threshold}\n"
                )
        except Exception as e:
            print(f"[Routing Similarity Log Error] {e}")

        if cos_sim < sim_threshold:
            return None


        return choice


    def _create_new_cluster(self, note: MemoryNote, new_emb: np.ndarray) -> str:
        cid = f"cluster_new_{len(self.clusters)}"


        summary = f"Cluster around: {', '.join(note.tags) or note.context[:50]}"
        tags = set(note.tags)

        self.clusters[cid] = {
            "members": [note.id],
            "tags": tags,
            "summary": summary,
        }
        self.cluster_centroids[cid] = np.array(new_emb)
        note.cluster_id = cid


        try:
            model_name = self.retriever.model.get_config_dict()['model_name']
        except (AttributeError, KeyError):
            model_name = 'all-MiniLM-L6-v2'

        retr = SimpleEmbeddingRetriever(model_name, model=self.retriever.model)
        meta = f"{note.context} {' '.join(note.keywords)} {' '.join(note.tags)}"
        retr.add_documents([note.content + " , " + meta])
        self.cluster_retrievers[cid] = retr

        return cid    

    def _split_cluster(self, cid: str):
 
        info = self.clusters.get(cid)
        if not info:
            return

        member_ids = info.get("members", [])
        if len(member_ids) <= self.cluster_split_threshold:
            return 

        print(f"[split_cluster] splitting {cid} with {len(member_ids)} members")


        docs = []
        mids = []
        for mid in member_ids:
            m = self.memories.get(mid)
            if not m:
                continue
            meta = f"{m.context} {' '.join(m.keywords)} {' '.join(m.tags)}"
            docs.append(m.content + " , " + meta)
            mids.append(mid)

        if len(docs) < 2:
            return

        model = self.retriever.model
        emb = model.encode(docs)


        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(emb)
        centers = kmeans.cluster_centers_


        cid_a = f"{cid}_a"
        cid_b = f"{cid}_b"

        members_a = [mids[i] for i, lab in enumerate(labels) if lab == 0]
        members_b = [mids[i] for i, lab in enumerate(labels) if lab == 1]


        def collect_tags(members):
            tags = set()
            for mid in members:
                m = self.memories.get(mid)
                if m:
                    tags.update(m.tags)
            return tags

        tags_a = collect_tags(members_a)
        tags_b = collect_tags(members_b)

        self.clusters[cid_a] = {
            "members": members_a,
            "tags": tags_a,
            "summary": info.get("summary", "") + " (subcluster A)",
        }
        self.clusters[cid_b] = {
            "members": members_b,
            "tags": tags_b,
            "summary": info.get("summary", "") + " (subcluster B)",
        }

        self.cluster_centroids[cid_a] = centers[0]
        self.cluster_centroids[cid_b] = centers[1]


        del self.clusters[cid]
        if cid in self.cluster_centroids:
            del self.cluster_centroids[cid]
        if cid in self.cluster_retrievers:
            del self.cluster_retrievers[cid]


        for mid in members_a:
            if mid in self.memories:
                self.memories[mid].cluster_id = cid_a
        for mid in members_b:
            if mid in self.memories:
                self.memories[mid].cluster_id = cid_b


        try:
            model_name = self.retriever.model.get_config_dict()['model_name']
        except (AttributeError, KeyError):
            model_name = 'all-MiniLM-L6-v2'

        for new_cid, members in [(cid_a, members_a), (cid_b, members_b)]:
            retr = SimpleEmbeddingRetriever(model_name, model=self.retriever.model)

            docs = []
            for mid in members:
                m = self.memories.get(mid)
                if not m:
                    continue
                meta = f"{m.context} {' '.join(m.keywords)} {' '.join(m.tags)}"
                docs.append(m.content + " , " + meta)
            if docs:
                retr.add_documents(docs)
                self.cluster_retrievers[new_cid] = retr

    def consolidate_memories(self):

        if not self.memories or not self.clusters:
            return


        new_clusters = {}

        for cid, info in list(self.clusters.items()):
            member_ids = info.get("members", [])
            if len(member_ids) <= self.cluster_split_threshold:

                new_clusters[cid] = info
                continue


            print(f"[consolidate] splitting cluster {cid} (size={len(member_ids)})")

            docs = []
            mids = []
            for mid in member_ids:
                m = self.memories.get(mid)
                if not m:
                    continue
                meta = f"{m.context} {' '.join(m.keywords)} {' '.join(m.tags)}"
                doc = m.content + " , " + meta
                docs.append(doc)
                mids.append(mid)

            if len(docs) < 2:

                new_clusters[cid] = info
                continue

            model = self.retriever.model
            emb = model.encode(docs)

            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(emb)
            centers = kmeans.cluster_centers_


            cid_a = f"{cid}_a"
            cid_b = f"{cid}_b"

            members_a = [mids[i] for i, lab in enumerate(labels) if lab == 0]
            members_b = [mids[i] for i, lab in enumerate(labels) if lab == 1]

            def collect_tags(members):
                tags = set()
                for mid in members:
                    m = self.memories.get(mid)
                    if m:
                        tags.update(m.tags)
                return tags

            tags_a = collect_tags(members_a)
            tags_b = collect_tags(members_b)

            new_clusters[cid_a] = {
                "members": members_a,
                "tags": tags_a,
                "summary": info.get("summary", "") + " (subcluster A)",
            }
            new_clusters[cid_b] = {
                "members": members_b,
                "tags": tags_b,
                "summary": info.get("summary", "") + " (subcluster B)",
            }


            for mid in members_a:
                if mid in self.memories:
                    self.memories[mid].cluster_id = cid_a
            for mid in members_b:
                if mid in self.memories:
                    self.memories[mid].cluster_id = cid_b


        self.clusters = new_clusters


        self.cluster_retrievers = {}
        self.cluster_centroids = {}

        try:
            model_name = self.retriever.model.get_config_dict()['model_name']
        except (AttributeError, KeyError):
            model_name = 'all-MiniLM-L6-v2'

        model = self.retriever.model

        for cid, info in self.clusters.items():
            member_ids = info.get("members", [])
            docs = []
            emb_list = []
            for mid in member_ids:
                m = self.memories.get(mid)
                if not m:
                    continue
                meta = f"{m.context} {' '.join(m.keywords)} {' '.join(m.tags)}"
                doc = m.content + " , " + meta
                docs.append(doc)


                emb_list.append(model.encode([doc])[0])

            if not docs or not emb_list:
                continue


            retr = SimpleEmbeddingRetriever(model_name, model=self.retriever.model)
            retr.add_documents(docs)
            self.cluster_retrievers[cid] = retr


            emb_array = np.vstack(emb_list)          # (num_members, dim)
            center = np.mean(emb_array, axis=0)      # (dim,)
            self.cluster_centroids[cid] = center


            dists = np.sum((emb_array - center) ** 2, axis=1)  


            order = np.argsort(dists)
            max_reps = 3  
            rep_indices = order[:min(max_reps, len(order))]

            rep_ids = []
            for idx in rep_indices:
                if 0 <= idx < len(member_ids):
                    rep_ids.append(member_ids[idx])


            if rep_ids:
                info["rep_ids"] = rep_ids              
                info["rep_id"] = rep_ids[0]           


        self._build_cluster_profiles_with_llm()


    
    def process_memory(self, note: MemoryNote, cluster_id: Optional[str] = None) -> bool:
        """Process a memory note and return an evolution label"""
        neighbor_memory, indices = self.find_related_memories(note.content, k=5,cluster_id=cluster_id)
        prompt_memory = self.evolution_system_prompt.format(context=note.context, content=note.content, keywords=note.keywords, nearest_neighbors_memories=neighbor_memory,neighbor_number=len(indices))
        print("prompt_memory", prompt_memory)
        response = self.llm_controller.llm.get_completion(
            prompt_memory,response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "should_evolve": {
                                    "type": "boolean",
                                },
                                "actions": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "suggested_connections": {
                                    "type": "array",
                                    "items": {
                                        "type": "integer"
                                    }
                                },
                                "new_context_neighborhood": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "tags_to_update": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "new_tags_neighborhood": {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "items": {
                                            "type": "string"
                                        }
                                    }
                                }
                            },
                            "required": ["should_evolve","actions","suggested_connections","tags_to_update","new_context_neighborhood","new_tags_neighborhood"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }}
        )
        try:
            print("response", response, type(response))
            # Clean the response in case there's extra text
            response_cleaned = response.strip()
            # Try to find JSON content if wrapped in other text
            if not response_cleaned.startswith('{'):
                start_idx = response_cleaned.find('{')
                if start_idx != -1:
                    response_cleaned = response_cleaned[start_idx:]
            if not response_cleaned.endswith('}'):
                end_idx = response_cleaned.rfind('}')
                if end_idx != -1:
                    response_cleaned = response_cleaned[:end_idx+1]
            
            response_json = json.loads(response_cleaned)
            print("response_json", response_json, type(response_json))
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw response: {response}")
            # Return default values for failed parsing
            return False, note
        should_evolve = response_json["should_evolve"]
        if should_evolve:
            actions = response_json["actions"]
            for action in actions:
                if action == "strengthen":
                    suggest_connections = response_json["suggested_connections"]
                    new_tags = response_json["tags_to_update"]
                    note.links.extend(suggest_connections)
                    note.tags = new_tags
                elif action == "update_neighbor":
                    new_context_neighborhood = response_json["new_context_neighborhood"]
                    new_tags_neighborhood = response_json["new_tags_neighborhood"]
                    noteslist = list(self.memories.values())
                    notes_id = list(self.memories.keys())
                    print("indices", indices)
                    # if slms output less than the number of neighbors, use the sequential order of new tags and context.
                    for i in range(min(len(indices), len(new_tags_neighborhood))):
                        # find some memory
                        tag = new_tags_neighborhood[i]
                        if i < len(new_context_neighborhood):
                            context = new_context_neighborhood[i]
                        else:
                            context = noteslist[indices[i]].context
                        memorytmp_idx = indices[i]
                        notetmp = noteslist[memorytmp_idx]
                        # add tag to memory
                        notetmp.tags = tag
                        notetmp.context = context
                        self.memories[notes_id[memorytmp_idx]] = notetmp
        return should_evolve,note

    def find_related_memories(
        self,
        query: str,
        k: int = 5,
        cluster_id: Optional[str] = None,
    ) -> tuple[str, list[int]]:

        if not self.memories:
            return "", []

        all_memories = list(self.memories.values())
        all_ids = list(self.memories.keys())


        if not cluster_id or cluster_id not in self.clusters or not self.cluster_retrievers:
            indices = self.retriever.search(query, k)
            memory_str = ""
            for i in indices:
                i = int(i)
                if i < 0 or i >= len(all_memories):
                    continue
                m = all_memories[i]
                memory_str += (
                    "memory index:" + str(i) +
                    "\t talk start time:" + m.timestamp +
                    "\t memory content: " + m.content +
                    "\t memory context: " + m.context +
                    "\t memory keywords: " + str(m.keywords) +
                    "\t memory tags: " + str(m.tags) + "\n"
                )
            return memory_str, indices


        cid = cluster_id
        cluster_info = self.clusters.get(cid)
        retr = self.cluster_retrievers.get(cid, None)

        if cluster_info is None or retr is None:

            indices = self.retriever.search(query, k)
            memory_str = ""
            for i in indices:
                i = int(i)
                if i < 0 or i >= len(all_memories):
                    continue
                m = all_memories[i]
                memory_str += (
                    "memory index:" + str(i) +
                    "\t talk start time:" + m.timestamp +
                    "\t memory content: " + m.content +
                    "\t memory context: " + m.context +
                    "\t memory keywords: " + str(m.keywords) +
                    "\t memory tags: " + str(m.tags) + "\n"
                )
            return memory_str, indices

        member_ids = cluster_info["members"]  


        local_indices = retr.search(query, k)


        indices: list[int] = []
        for li in local_indices:
            li = int(li)
            if li < 0 or li >= len(member_ids):
                continue
            mid = member_ids[li]          
            if mid not in self.memories:
                continue
            try:
                gi = all_ids.index(mid)   
                indices.append(gi)
            except ValueError:
                continue


        memory_str = ""
        for i in indices:
            i = int(i)
            if i < 0 or i >= len(all_memories):
                continue
            m = all_memories[i]
            memory_str += (
                "memory index:" + str(i) +
                "\t talk start time:" + m.timestamp +
                "\t memory content: " + m.content +
                "\t memory context: " + m.context +
                "\t memory keywords: " + str(m.keywords) +
                "\t memory tags: " + str(m.tags) + "\n"
            )

        return memory_str, indices
    def find_related_memories_raw(self, query: str, k: int = 5, query_tags=None):
        if not self.memories:
            self.last_total_memories = 0
            self.last_search_space_size = 0
            self.last_search_mode = "global"
            self.last_candidate_cluster_ids = []
            self.last_selected_cluster_ids = []
            self.last_searched_cluster_ids = []
            self.last_retrieved_indices = []
            self.last_retrieved_memory_ids = []
            self.last_retrieved_memories = []
            return ""

        all_memories = list(self.memories.values())
        all_ids = list(self.memories.keys())
        total_memories = len(all_memories)

        search_space_size = total_memories
        search_mode = "global"
        cluster_ids = []


        id_to_clusters = {}
        if self.clusters:
            for cid, info in self.clusters.items():
                for mid in info.get("members", []):
                    id_to_clusters.setdefault(mid, []).append(cid)


        if isinstance(query_tags, str):
            query_tags = [t.strip() for t in query_tags.split(",") if t.strip()]
        if query_tags is None:
            query_tags = []


        if self.clusters and self.cluster_centroids:
            try:
                cluster_ids = self.select_clusters_for_query(
                    query=query,
                    query_tags=query_tags,
                    top_n=10,
                    candidate_k=self.routing_top_k
                )
            except Exception:
                cluster_ids = []


        if not hasattr(self, "last_candidate_cluster_ids"):
            self.last_candidate_cluster_ids = []
        if not hasattr(self, "last_selected_cluster_ids"):
            self.last_selected_cluster_ids = []


        if (not self.clusters) or (not self.cluster_retrievers) or (not query_tags) or (not cluster_ids):
            indices = self.retriever.search(query, k)
            search_mode = "global"
            search_space_size = total_memories
            cluster_ids = []  
        else:
            indices = []
            search_mode = "cluster"
            search_space_size = 0

            for cid in cluster_ids:
                cluster = self.clusters.get(cid)
                if cluster is None:
                    continue
                member_ids = cluster.get("members", [])
                retr = self.cluster_retrievers.get(cid)

                if not retr or not member_ids:
                    continue

                search_space_size += len(member_ids)
                local_indices = retr.search(query, k)

                for li in local_indices:
                    li = int(li)
                    if li < 0 or li >= len(member_ids):
                        continue
                    mid = member_ids[li]
                    if mid not in self.memories:
                        continue
                    try:
                        gi = all_ids.index(mid)
                        indices.append(gi)
                    except ValueError:
                        continue


            uniq = []
            seen = set()
            for gi in indices:
                if gi not in seen:
                    seen.add(gi)
                    uniq.append(gi)

            if not uniq:
                indices = self.retriever.search(query, k)
                search_mode = "global"
                search_space_size = total_memories
                cluster_ids = []
            else:
                indices = uniq[:k]


        safe_indices = []
        for i in indices:
            i = int(i)
            if 0 <= i < len(all_memories):
                safe_indices.append(i)

        retrieved_ids = [all_ids[i] for i in safe_indices]

        retrieved_meta = []
        for mid in retrieved_ids:
            m = self.memories.get(mid)
            if not m:
                continue
            retrieved_meta.append({
                "memory_id": mid,
                "timestamp": getattr(m, "timestamp", ""),
                "content": getattr(m, "content", ""),
                "context": getattr(m, "context", ""),
                "keywords": list(getattr(m, "keywords", []) or []),
                "tags": list(getattr(m, "tags", []) or []),
                "clusters": id_to_clusters.get(mid, []),
            })

        self.last_total_memories = total_memories
        self.last_search_space_size = search_space_size
        self.last_search_mode = search_mode

        self.last_searched_cluster_ids = list(cluster_ids)
        self.last_retrieved_indices = safe_indices
        self.last_retrieved_memory_ids = retrieved_ids
        self.last_retrieved_memories = retrieved_meta


        memory_str = ""
        for i in safe_indices:
            current_memory = all_memories[i]
            memory_str += (
                f"talk start time:{current_memory.timestamp}\t"
                f"memory content: {current_memory.content}\t"
                f"memory context: {current_memory.context}\t"
                f"memory keywords: {str(current_memory.keywords)}\t"
                f"memory tags: {str(current_memory.tags)}\n"
            )

            links = getattr(current_memory, "links", []) or []
            added = 0
            for neighbor_idx in links:
                try:
                    neighbor_idx = int(neighbor_idx)
                except (ValueError, TypeError):
                    continue
                if neighbor_idx < 0 or neighbor_idx >= total_memories:
                    continue

                neighbor_mem = all_memories[neighbor_idx]
                memory_str += (
                    f"talk start time:{neighbor_mem.timestamp}\t"
                    f"memory content: {neighbor_mem.content}\t"
                    f"memory context: {neighbor_mem.context}\t"
                    f"memory keywords: {str(neighbor_mem.keywords)}\t"
                    f"memory tags: {str(neighbor_mem.tags)}\n"
                )
                added += 1
                if added >= k:
                    break

        return memory_str

