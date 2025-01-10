# app.py
import os
import streamlit as st
from dotenv import load_dotenv

# LangChain v0.3
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# .envファイルからOPENAI_API_KEYを読み込む
load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")

# --- 1段階目: 回答生成用のLLM -------------------------------------------------- #
def generate_response(user_query: str) -> str:
    """
    こども向けのプログラミング学習を想定した優しい回答を
    第一段階として生成する関数
    """
    system_prompt = (
        "あなたは小学生向けプログラミング教室の優しいチューターです。"
        "以下のルールを守りながら、子供向けにわかりやすく応答してください。"
    )
    # ChatOpenAIのインスタンス (最新のLangChain 0.3準拠)
    chat = ChatOpenAI(
        temperature=0.3,
        openai_api_key=openai_api_key
    )
    response = chat([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query)
    ])
    return response.content

# --- 2段階目: 回答監督用のLLM -------------------------------------------------- #
def supervise_response(generated_answer: str) -> str:
    """
    1段階目の回答をレビューし、子ども向けに不適切な表現がないかをチェックし、
    必要に応じて修正して最終回答を返す関数
    """
    system_prompt = (
        "あなたは小学生向けの指導アシスタントとして、下記の回答をレビューしてください。"
        "もし問題があれば修正し、最終的に子ども向けにわかりやすく安全な形で回答してください。"
    )
    chat = ChatOpenAI(
        temperature=0,
        openai_api_key=openai_api_key
    )
    final_response = chat([
        SystemMessage(content=system_prompt),
        HumanMessage(content=generated_answer)
    ])
    return final_response.content

# --- Streamlit アプリ -------------------------------------------------------- #
def main():
    st.title("こどもプログラミング教室向けチャットアプリ")

    # チャットの履歴をセッションに保存
    if "history" not in st.session_state:
        st.session_state["history"] = []

    # ユーザー入力欄
    user_input = st.text_input("質問を入力してください", value="", max_chars=200)

    if st.button("送信"):
        if user_input.strip():
            # 1. 回答生成
            generated = generate_response(user_input)
            # 2. 監督モデルでレビュー
            supervised = supervise_response(generated)

            # 履歴に追加
            st.session_state["history"].append({"role": "user", "content": user_input})
            st.session_state["history"].append({"role": "assistant", "content": supervised})

    # チャット履歴を表示
    for msg in st.session_state["history"]:
        if msg["role"] == "user":
            st.markdown(f"**ユーザー**: {msg['content']}")
        else:
            st.markdown(f"**アシスタント**: {msg['content']}")

if __name__ == "__main__":
    main()
