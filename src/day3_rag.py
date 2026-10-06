from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# 1.加载锂电池知识库
loader = TextLoader("battery_info.txt", encoding="utf-8")
documents = loader.load()

# 2.文本切分
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
split_docs = text_splitter.split_documents(documents)

# 3.加载开源嵌入模型，去掉报错参数
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"trust_remote_code": True},
    cache_folder="./model"
)

# 4.存入FAISS向量数据库
vectordb = FAISS.from_documents(documents=split_docs, embedding=embedding)
vectordb.save_local("./faiss_db")

# 5.检索器
retriever = vectordb.as_retriever(search_kwargs={"k":2})

# 6.加载轻量大模型做问答
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir="./model",
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir="./model",
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    temperature=0.3
)
llm = HuggingFacePipeline(pipeline=pipe)

# 7.RAG问答函数
def rag_ask(question):
    docs = retriever.invoke(question)
    context = "\n".join([d.page_content for d in docs])
    prompt = f"""基于下面电池资料回答问题：
{context}
问题：{question}
回答："""
    output = llm.invoke(prompt)
    return output

# 测试提问
if __name__ == "__main__":
    print("===锂电池知识库RAG问答===")
    res = rag_ask("锂电池循环次数增加，SOH会怎么变化？")
    print(res)