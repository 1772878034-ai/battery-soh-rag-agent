import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def _model_path(local_dir, repo_id):
    # 本地已下载则直接用本地（离线、无符号链接），否则回退到在线仓库 ID
    return local_dir if os.path.exists(local_dir) else repo_id

# ---------------------- 页面配置 ----------------------
st.set_page_config(page_title="锂电池SOH预测 | RAG智能问答 Demo", layout="wide")
st.title("锂电池SOH预测 | RAG智能问答 Demo")

# 文件路径
mat_path = "data/B0005.mat"
txt_path = "data/battery_info.txt"
model_path = "data/lstm_battery_model.h5"
faiss_path = "data/faiss_db"

# ---------------------- 加载LLM Qwen2.5‑1.5B‑Instruct 正确使用chat_template ----------------------
@st.cache_resource
def load_llm():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    model_name = _model_path(
        "data/hf_models/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto"
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        temperature=0.2,
        do_sample=True,
        return_full_text=False,   # 只返回模型生成部分，不要输入prompt
    )
    return HuggingFacePipeline(pipeline=pipe), tokenizer

llm, tokenizer = load_llm()

# ---------------------- 双标签页 ----------------------
tab1, tab2 = st.tabs(["🔍 RAG智能问答", "📈 LSTM电池SOH预测"])

# ==================================================
# 标签页1：RAG智能问答 BM25+FAISS混合检索 + BGE重排 + LLM生成
# ==================================================
with tab1:
    st.subheader("基于BM25+FAISS混合检索 + BGE重排 + LLM答案生成")

    @st.cache_resource
    def build_retriever():
        loader = TextLoader(txt_path, encoding="utf-8")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        split_docs = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        faiss_db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        faiss_retriever = faiss_db.as_retriever(search_kwargs={"k": 3})

        bm25_retriever = BM25Retriever.from_documents(split_docs)
        bm25_retriever.k = 3

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5]
        )

        rerank_model = HuggingFaceCrossEncoder(model_name=_model_path(
            "data/hf_models/bge-reranker-base", "BAAI/bge-reranker-base"))
        compressor = CrossEncoderReranker(model=rerank_model, top_n=2)
        compression_retriever = ContextualCompressionRetriever(
            base_retriever=ensemble_retriever,
            base_compressor=compressor
        )
        return compression_retriever

    retriever = build_retriever()
    query = st.text_input("请输入关于锂电池的问题：", value="什么是电池的SOH")
    if st.button("回答", key="rag_btn") and query:
        with st.spinner("检索知识库、重排、LLM生成回答..."):
            docs = retriever.invoke(query)
            context = "\n".join([d.page_content for d in docs]) if docs else "暂无相关资料"

            # ✅ 使用Qwen官方chat_template，不要手写prompt字符串
            messages = [
                {"role": "system", "content": "你是锂电池领域专家，严格依据提供的知识库回答用户问题。知识库没有信息就直接回复【知识库暂无相关信息】，禁止编造内容。回答简洁清晰。"},
                {"role": "user", "content": f"知识库：{context}\n问题：{query}"}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            answer = llm.invoke(prompt)

            st.markdown("### 回答")
            st.write(answer)
            st.markdown("### 检索到的参考文档片段")
            for idx, doc in enumerate(docs):
                st.write(f"{idx+1}. {doc.page_content}")

# ==================================================
# 标签页2：LSTM时序预测（与day2训练代码逻辑完全一致）
# ==================================================
with tab2:
    st.subheader("LSTM时序预测 | NASA锂电池容量衰减预测")
    st.caption("加载NASA B0005电池循环数据集，调用训练完成的LSTM模型，预测电池容量变化趋势")
    if st.button("加载数据 & 开始预测", key="lstm_btn"):
        with st.spinner("加载数据并运行LSTM预测..."):
            mat = scipy.io.loadmat(mat_path)
            data = mat['B0005'][0][0]
            cycles = data['cycle'][0]
            cap_list = []
            cycle_id = []

            # 和day1/day2完全一致：电流积分计算容量
            for i, cyc in enumerate(cycles):
                cyc_type = cyc['type'][0]
                if cyc_type == 'discharge':
                    t = cyc['data'][0][0]['Time'][0]
                    I = cyc['data'][0][0]['Current_measured'][0]
                    dt = np.diff(t)
                    cap = np.sum(-I[1:] * dt) / 3600
                    cap_list.append(cap)
                    cycle_id.append(i)

            cap_array = np.array(cap_list).reshape(-1, 1)
            cycle_arr = np.array(cycle_id)

            model = load_model(model_path)
            scaler = MinMaxScaler(feature_range=(0, 1))
            cap_scaled = scaler.fit_transform(cap_array)

            # 训练时time_step=5，保持一致
            time_step = 5
            X, y = [], []
            for i in range(time_step, len(cap_scaled)):
                X.append(cap_scaled[i - time_step:i, 0])
                y.append(cap_scaled[i, 0])
            X, y = np.array(X), np.array(y)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))

            y_pred_scaled = model.predict(X, verbose=0)
            y_pred = scaler.inverse_transform(y_pred_scaled)
            y_true = scaler.inverse_transform(y.reshape(-1, 1))

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(cycle_arr[time_step:], y_true, 'b-', label='真实容量', linewidth=1.5)
            ax.plot(cycle_arr[time_step:], y_pred, 'r--', label='LSTM预测容量', linewidth=2)
            ax.set_xlabel('循环次数')
            ax.set_ylabel('电池容量 (Ah)')
            ax.set_title('锂电池循环容量衰减曲线：真实值 vs LSTM预测值')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            st.success("预测完成！实线为NASA实测容量，虚线为LSTM模型预测结果，可以观察电池随充放电循环的衰减趋势。")

# ==================================================
# 底部：项目成果展示
# ==================================================
st.markdown("---")
st.markdown("### 项目效果展示")
col1, col2 = st.columns(2)
with col1:
    st.image("assets/web_lstm_pred_page.png", caption="LSTM预测页面", use_container_width=True)
with col2:
    st.image("assets/web_rag_soh_page.png", caption="RAG问答页面", use_container_width=True)
