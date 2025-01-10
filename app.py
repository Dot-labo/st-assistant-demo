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
st.write(openai_api_key)

# --- 1段階目: 回答生成用のLLM -------------------------------------------------- #
def generate_response(user_query: str) -> str:
    """
    こども向けのプログラミング学習を想定した優しい回答を
    第一段階として生成する関数
    """
    system_prompt = (
        """
        あなたは小学生向けプログラミング教室の優しいチューターです。
        以下のルールを守りながら、子供向けにわかりやすく応答してください。
        - 子どもにとってわかりやすい言葉遣いを心がけましょう。専門用語は避け、親しみやすさを増すため、適宜絵文字などを使ってあげてください。
        - 回答は長すぎると子どもにとって理解しにくくなります。関連することを一気に言うのではなく、聞かれたことに要件を絞って簡潔に答えてください。
        - 子供の理解を助けることを目的としてください。単に質問に答えたり、代わりにプログラムを提供するだけでなく、最後に理解度を試すための簡単なクイズを出すなどの工夫もしてください。
            例: classの意味について聞かれた後、簡単な例で説明 -> その後、classの意味を尋ねる簡単な3択問題を出して終わりにする、など。
        """

    )
    # ChatOpenAIのインスタンス (最新のLangChain 0.3準拠)
    chat = ChatOpenAI(
        temperature=0.3,
        openai_api_key=openai_api_key,
        model="gpt-4o-mini"
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

    # ユャット履歴を表示
    for msg in st.session_state["history"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # チャット入力
    if prompt := st.chat_input("質問を入力してください"):
        # ユーザーメッセージを表示
        with st.chat_message("user"):
            st.write(prompt)

        # アシスタントの応答を処理
        with st.chat_message("assistant"):
            with st.status("考え中...", expanded=True) as status:
                st.write("🤔 回答を生成しています...")
                # 1. 回答生成
                generated = generate_response(prompt)
                status.update(label="完了！", state="complete", expanded=False)
            st.write(generated)

        # 履歴に追加
        st.session_state["history"].append({"role": "user", "content": prompt})

if __name__ == "__main__":
    main()
