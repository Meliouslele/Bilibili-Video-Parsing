import streamlit as st
import langchain_helper1 as lch1
import textwrap
import os

st.title("B站助理")

with st.sidebar:
    with st.form(key='my_form'):
        bilibili_url = st.sidebar.text_area(  # 修改变量名从 youtube_url 改为 bilibili_url
            label="B站视频链接是多少?",
            max_chars=100  # 增加最大字符数，因为B站URL可能较长
        )
        query = st.sidebar.text_area(
            label="询问我关于视频的内容?",
            max_chars=100,  # 增加最大字符数
            key="query"
        )
        # 修改为DeepSeek API密钥输入
        deepseek_api_key = st.sidebar.text_input(
            label="DeepSeek API Key",  # 修改标签
            key="langchain_search_api_key_deepseek",  # 修改key
            max_chars=100,
            type="password"
        )
        # 更新链接到DeepSeek
        "[Get a DeepSeek API key](https://platform.deepseek.com/)"
        "[View the source code](https://github.com/rishabkumar7/pets-name-langchain/tree/main)"
        submit_button = st.form_submit_button(label='Submit')

if query and bilibili_url:  # 修改变量名
    if not deepseek_api_key:  # 修改检查的API密钥变量
        st.info("Please add your DeepSeek API key to continue.")  # 更新提示信息
        st.stop()
    else:
        # 设置环境变量，供langchain_helper1使用
        os.environ["DEEPSEEK_API_KEY"] = deepseek_api_key
        
        # 确保函数名正确，应该是create_db_from_bilibili_video_url而不是create_db_from_youtube_video_url
        db = lch1.create_db_from_bilibili_video_url(bilibili_url)  # 修改函数名和参数
        response, docs = lch1.get_response_from_query(db, query)
        st.subheader("Answer:")
        st.text(textwrap.fill(response, width=85))