from langchain_groq import ChatGroq
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain  # type: ignore
from langchain_classic.chains.combine_documents import create_stuff_documents_chain  # type: ignore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from flipkart.config import Config
from flipkart.retrieval import get_retriever


class RAGChainBuilder:
    def __init__(self, vector_store, retrieval_strategy: str = "hybrid"):
        self.vector_store = vector_store
        self.retrieval_strategy = retrieval_strategy
        self.model = ChatGroq(model=Config.RAG_MODEL, temperature=0.5)
        self.history_store = {}

    def _get_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.history_store:
            self.history_store[session_id] = ChatMessageHistory()
        return self.history_store[session_id]

    def build_chain(self):
        retriever = get_retriever(self.vector_store, strategy=self.retrieval_strategy, k=3)

        context_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Given the chat history and user question, rewrite it as a standalone question."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are ShopSmart AI, a product recommendation assistant.

INSTRUCTIONS:
1. Answer ONLY using the product information provided in CONTEXT below.
2. When discussing a product, always include: Product Name, Price (₹), Rating (/5).
3. If comparing products, use a clear format with pros and cons for each.
4. If the question is about a product NOT in the context, respond: "I don't have information about that product in my database."
5. NEVER invent or guess prices, ratings, or features.
6. Keep responses concise — 2-3 paragraphs maximum.

CONTEXT:
{context}

QUESTION: {input}""",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(self.model, retriever, context_prompt)

        question_answer_chain = create_stuff_documents_chain(self.model, qa_prompt)

        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        return RunnableWithMessageHistory(
            rag_chain,
            self._get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
