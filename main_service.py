import uvicorn

if __name__ == "__main__":
    uvicorn.run("rag_gen_langchain:app", host="0.0.0.0", port=8780)
