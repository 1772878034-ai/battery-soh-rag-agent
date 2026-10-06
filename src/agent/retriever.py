import time
from functools import lru_cache

import numpy as np
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import EnsembleRetriever
from langchain_text_splitters import CharacterTextSplitter
from sentence_transformers import CrossEncoder

from . import config


@lru_cache(maxsize=1)
def load_chunks():
    """话题级切分：按段落切，每段一个 chunk，返回带稳定 cid 的 chunk。

    day3/day4/day5 教程脚本使用 300/30 粗切分，多个话题被合并进同一 chunk；
    Agent 与评估链路改用段落级切分以定位证据（详见 README 失败案例一节）。
    """
    documents = TextLoader(str(config.KNOWLEDGE_PATH), encoding="utf-8").load()
    splitter = CharacterTextSplitter(separator="\n\n", chunk_size=120, chunk_overlap=0)
    chunks = splitter.split_documents(documents)
    for cid, chunk in enumerate(chunks):
        chunk.metadata["cid"] = cid
        chunk.metadata["source"] = config.KNOWLEDGE_PATH.name
    return chunks


@lru_cache(maxsize=1)
def embeddings():
    kwargs = {"model_name": config.EMBED_MODEL, "model_kwargs": {"trust_remote_code": True}}
    if config.LOCAL_MODEL_CACHE.exists():
        kwargs["cache_folder"] = str(config.LOCAL_MODEL_CACHE)
    return HuggingFaceEmbeddings(**kwargs)


@lru_cache(maxsize=1)
def _cross_encoder():
    return CrossEncoder(config.RERANK_MODEL)


class HybridRerankRetriever:
    """BM25 + FAISS 混合召回（RRF 融合），BGE CrossEncoder 重排。

    CrossEncoder 不可用时退化为 MiniLM 余弦重排，保证离线可跑。
    """

    def __init__(self, fetch_k=5, top_k=3, weights=(0.5, 0.5)):
        self.fetch_k = fetch_k
        self.top_k = top_k
        chunks = load_chunks()

        faiss_db = FAISS.load_local(
            str(config.FAISS_DIR), embeddings(), allow_dangerous_deserialization=True
        )
        faiss_retriever = faiss_db.as_retriever(search_kwargs={"k": fetch_k})
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = fetch_k

        self.ensemble = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever], weights=list(weights)
        )
        try:
            _cross_encoder()
            self.reranker_name = config.RERANK_MODEL
        except Exception:
            self.reranker_name = "fallback-minilm-cosine"

    def _rerank(self, query, candidates):
        if self.reranker_name == config.RERANK_MODEL:
            scores = _cross_encoder().predict([(query, d.page_content) for d in candidates])
            return [float(s) for s in scores]
        query_vec = np.array(embeddings().embed_query(query))
        doc_vecs = np.array(embeddings().embed_documents([d.page_content for d in candidates]))
        scores = doc_vecs @ query_vec / (
            np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(query_vec) + 1e-8
        )
        return [float(s) for s in scores]

    def search(self, query, top_k=None):
        top_k = top_k or self.top_k
        t0 = time.perf_counter()
        raw = self.ensemble.invoke(query)
        t_recall = time.perf_counter()

        cid_by_content = {c.page_content: c.metadata["cid"] for c in load_chunks()}
        seen, candidates = set(), []
        for rank, doc in enumerate(raw):
            if doc.page_content in seen:
                continue
            seen.add(doc.page_content)
            doc.metadata["cid"] = cid_by_content.get(doc.page_content, -1)
            candidates.append({"doc": doc, "ensemble_rank": rank + 1})

        scores = self._rerank(query, [c["doc"] for c in candidates]) if candidates else []
        for c, score in zip(candidates, scores):
            c["rerank_score"] = round(score, 4)
        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        t_rerank = time.perf_counter()

        return {
            "query": query,
            "reranker": self.reranker_name,
            "candidates": candidates,
            "ranked": candidates[:top_k],
            "timings": {
                "recall_ms": round((t_recall - t0) * 1000, 1),
                "rerank_ms": round((t_rerank - t_recall) * 1000, 1),
            },
        }


@lru_cache(maxsize=1)
def get_retriever():
    return HybridRerankRetriever()
