import time

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
from fastapi import FastAPI
from groq import Groq

app = FastAPI()


client = Groq(
    api_key="API_KEY_OF_YOU",
)
# load index
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cuda'}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
)
retriever = FAISS.load_local(folder_path="./db_faiss", embeddings=embeddings, allow_dangerous_deserialization=True)

history_chat = []


def contextualized_query(query, history_chat):
    """
        Contextualized query input question
    :param query:
    :param history_chat: history chat between llm and user
    :return:
    """

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an assistant responsible for rephrasing user queries to be clearer and more"
                           " concise. Ensure that the rephrased query retains the original intent and does not "
                           "introduce any new assumptions or information. You're just restating the user's question, "
                           "not answering the user's question. Just return the final result without any "
                           "further explanation"
            },
            {
                "role": "user",
                "content": ("Rephrase the following user query to make it clearer and more concise, while preserving "
                            "the original intent. If the conversation history is empty, focus solely on improving the "
                            "clarity and conciseness of the current query without adding any assumptions, fabricated "
                            "information, or additional context. \n"
                            "'Conversation history': {} \n"
                            "'User's current query': {}").format(history_chat, query)
            }
        ],
        model="llama3-8b-8192",
    )

    answer_contextualized = chat_completion.choices[0].message.content
    return answer_contextualized


@app.post("/QnA")
async def qna(input: str):
    question_contextualized = contextualized_query(input, history_chat)
    print('[INFO] question_contextualized: ', question_contextualized)
    docs = retriever.similarity_search(question_contextualized)
    context_RAG = ""
    for ix, doc in enumerate(docs):
        context_RAG += f"{ix + 1}: {str(docs)}\n"

    time.sleep(5)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an assistant responsible for providing accurate and detailed responses to user "
                           "queries. Use the provided context from the RAG process, the conversation history, and the "
                           "rephrased user query to generate a response. If the conversation history is empty, base "
                           "your response on the context from RAG and the rephrased query. Ensure your answer is "
                           "precise, relevant, and directly addresses the user's intent without introducing any new "
                           "assumptions or fabricated information. Just return the final result without any further "
                           "explanation"
            },
            {
                "role": "user",
                "content": ("Provide a detailed and accurate response based on the following context from RAG, the "
                            "rephrased query, and the conversation history. If the conversation history is empty, use "
                            "the context from RAG and the rephrased query to formulate your response. Avoid introducing"
                            "assumptions or fabricated information. Make sure your answer is relevant and directly "
                            " addresses the user's question. \n"
                            "'RAG Context': {} \n"
                            "'Conversation history': {} \n"
                            "'Rephrased user query': {}").format(context_RAG, history_chat, question_contextualized)
            }
        ],
        model="llama3-8b-8192",
    )

    answer = chat_completion.choices[0].message.content

    history_chat.extend([
        {
            "role": "user",
            "content": input
        },
        {
            "role": "assistant",
            "content": answer
        }]
    )

    return answer
