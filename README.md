# RAG chatbot
Xây dựng chatbot đơn giản sử dụng các thư viện, tools hỗ trợ như langchain, ...
## 1. Langchain
- Sử dụng thư viện langchain để xây dựng RAG Applications 
- Pipeline build chatbox:
![Photo](./docs/pipeline-rag-chatbot.png)
- Mô tả: 
  + Store lịch sử đoạn chat: LLM hiểu rõ được bối cảnh
  + (query, conversation history) -> LLM -> rephrased query -> retriever: LLM biểu đạt lại câu
hỏi của người dùng trước khi cho vào Retrieval
  
### 1.1 Vectorstore
- Hiện tại đang bóc tách thông tin từ trang web sử dụng tools của langchain
- Sử dụng FAISS lưu trữ các vetor
```
python build_faiss.py
```

### 1.2 API Chatbot
- Sử dụng model Llama3 của Groq
- Sử dụng FastApi làm giao diện
```bash
python main_service.py
```

## TODO
- xây dựng chatbot sử dụng function calling 
