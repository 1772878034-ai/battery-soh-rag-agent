"""按 agent 链路的话题级切分重建 FAISS 索引。

知识库文本有改动后必须重跑，否则索引里是旧文本（README 失败案例之一）。
"""
from langchain_community.vectorstores import FAISS

from agent.retriever import embeddings, load_chunks
from agent import config


def main():
    chunks = load_chunks()
    db = FAISS.from_documents(chunks, embeddings())
    db.save_local(str(config.FAISS_DIR))
    print(f"已重建 FAISS 索引：{config.FAISS_DIR}，共 {len(chunks)} 个话题 chunk")
    for chunk in chunks:
        print(f"  cid={chunk.metadata['cid']} len={len(chunk.page_content)} {chunk.page_content[:28]}...")


if __name__ == "__main__":
    main()
