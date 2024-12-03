import streamlit as st
import langchain_openai
from langchain.evaluation.qa import QAEvalChain

def generate_response(
    uploaded_file,
    openai_api_key,
    query_text,
    response_text
):
    if uploaded_file is None:
        st.error("Please upload a valid file before submitting.")
        return None  # Exit the function early
    else:
        documents = [uploaded_file.read().decode()]
    from langchain_text_splitters import CharacterTextSplitter
    splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )
    text = splitter.create_documents(documents)

    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import OpenAIEmbeddings

    db = FAISS.from_documents(text, embedding=OpenAIEmbeddings(openai_api_key=openai_api_key))
    retriever = db.as_retriever()

    real_qa = [{"question":query_text, "answer":response_text}]

    from langchain.chains import RetrievalQA
    from langchain_openai import OpenAI
    qna_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=openai_api_key),
        chain_type="stuff",
        retriever=retriever,
        input_key="question"
    )

    predictions = qna_chain.apply(real_qa)

    eval_chain = QAEvalChain.from_llm(
        llm=OpenAI(openai_api_key=openai_api_key)
    )


    graded_outputs = eval_chain.evaluate(
        real_qa,
        predictions,
        question_key="question",
        prediction_key="result",
        answer_key="answer"
    )

    response = {
        "predictions": predictions,
        "graded_outputs": graded_outputs
    }

    return response

result = []

st.set_page_config(
    page_title="Evaluate a RAG app"
)
st.title("Evaluate a RAG app")

with st.expander("Evaluation basis:"):
    st.write(
        """
        To evaluate the repsonse of the LLM we will first ask the question from
        LLM, and then check it with right answer, we will get to know whether 
        the LLM has produced accurate results or is just hallucinating.
        """
    )


uploaded_file = st.file_uploader(
    "## Upload a .txt document :",
    type="txt"
)
query_text = st.text_input(
    "Enter a question you have already fact-checked:",
    placeholder="Write your question here",
    disabled=not uploaded_file
)

response_text = st.text_input(
    "Enter the real answer to the question:",
    placeholder="Write the confirmed answer here",
    disabled=not uploaded_file
)

with st.form("My form", clear_on_submit=True):
    openai_api_key = st.text_input(
        "## Enter the OpenAI API key: ",
        type="password",
        disabled=not (uploaded_file and query_text)
    )
    submitted = st.form_submit_button(
        "SUBMIT",
        disabled=not (uploaded_file and query_text)
    )

    if submitted:
        if not uploaded_file:
            st.error("Please upload a file before submitting.")
        elif not openai_api_key.startswith("sk-"):
            st.error("Please enter a valid OpenAI API key.")
        else:
            with st.spinner("Good things come to those who wait...."):
                response = generate_response(
                    uploaded_file,
                    openai_api_key,
                    query_text,
                    response_text
                )
                if response is not None:
                    result.append(response)

#         response = generate_response(query_text=query_text, uploaded_file=uploaded_file, openai_api_key=openai_api_key, response_text=response_text)
#         result.append(response)
#         del openai_api_key
#
if len(result):
    st.write("Question")
    st.info(response["predictions"][0]["question"])
    st.write("Real answer")
    st.info(response["predictions"][0]["answer"])
    st.write("Answer provided by the AI App")
    st.info(response["predictions"][0]["result"])
    st.write("Therefore, the AI App answer was")
    st.info(response["graded_outputs"][0]["results"])