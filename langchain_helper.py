from langchain_community.document_loaders import BiliBiliLoader
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()

# 使用开源的嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def create_db_from_bilibili_video_url(video_url: str) -> FAISS:
    """
    从B站视频URL创建向量数据库
    
    Args:
        video_url: B站视频URL，例如："https://www.bilibili.com/video/BV1iphCzLE1c"
    
    Returns:
        FAISS向量数据库对象
    """
    # 从环境变量获取B站认证信息
    sessdata = os.getenv("BILIBILI_SESSDATA")
    bili_jct = os.getenv("BILIBILI_JCT")
    buvid3 = os.getenv("BILIBILI_BUVID3")
    
    if not all([sessdata, bili_jct, buvid3]):
        raise ValueError("请设置B站认证信息环境变量: BILIBILI_SESSDATA, BILIBILI_JCT, BILIBILI_BUVID3")
    
    # 初始化BiliBiliLoader - 视频URL作为列表传入
    loader = BiliBiliLoader(
        video_urls=[video_url],  # 这里填写视频URL:cite[2]
        sessdata=sessdata,
        bili_jct=bili_jct,
        buvid3=buvid3
    )
    
    # 加载视频字幕
    transcript = loader.load()

    # 分割文本
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    # 创建向量数据库
    db = FAISS.from_documents(docs, embeddings)
    return db

def get_response_from_query(db, query, k=4):
    """
    基于向量数据库查询获取回答
    
    Args:
        db: FAISS向量数据库
        query: 查询问题
        k: 返回的最相似文档数量
    
    Returns:
        response: 模型生成的回答
        docs: 相关的文档片段
    """
    # 相似度搜索
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    # 使用DeepSeek聊天模型
    llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        openai_api_base="https://api.deepseek.com",
        temperature=0.7,
        max_tokens=1000,
    )

    # 创建提示模板
    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that can answer questions about videos 
        based on the video's transcript.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.Use a document format other than Markdown, and pay attention to line breaks.
        
        """,
    )
    
    # 创建链
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # 运行链
    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs

# 使用示例
if __name__ == "__main__":
    # 在这里填写你的B站视频URL
    video_url = "https://www.bilibili.com/video/BV1iphCzLE1c"  # 替换为你的视频URL
    
    # 创建向量数据库
    db = create_db_from_bilibili_video_url(video_url)
    
    # 示例查询
    query = "这个视频的主要内容是什么？"
    response, relevant_docs = get_response_from_query(db, query)
    
    print("回答:", response)
    print("\n相关文档片段:", relevant_docs)