from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# 1.加载知识库文档
loader = TextLoader("battery_info.txt", encoding="utf-8")
documents = loader.load()

# 2.文档切分
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
split_docs = text_splitter.split_documents(documents)

# 3.嵌入模型 + FAISS向量检索
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"trust_remote_code": True},
    cache_folder="./model"
)
faiss_db = FAISS.from_documents(split_docs, embedding)
faiss_retriever = faiss_db.as_retriever(search_kwargs={"k":2})

# 4.BM25关键词检索
bm25_retriever = BM25Retriever.from_documents(split_docs)
bm25_retriever.k = 2

# 5.混合检索器：RRF倒数排序融合
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.5, 0.5]
)

# 6.本地DistilG PT2大模型
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./model", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="./model", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    temperature=0.3
)
llm = HuggingFacePipeline(pipeline=pipe)

# RAG问答函数
def rag_ask(question):
    docs = ensemble_retriever.invoke(question)
    context = "\n".join([d.page_content for d in docs])
    prompt = f"""基于下面电池知识库内容回答问题。
知识库：{context}
问题：{question}
回答："""
    return llm.invoke(prompt)

if __name__ == "__main__":
    print("===混合检索RAG（BM25+FAISS RRF融合）===")
    res = rag_ask("锂电池循环次数增加，SOH会怎么变化？")
    print(res)