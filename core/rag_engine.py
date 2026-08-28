import os

from langchain_mistralai import ChatMistralAI

from langchain_core.prompts import (
    ChatPromptTemplate
)

from langchain_core.output_parsers import (
    StrOutputParser
)

from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda
)

from langchain_chroma import Chroma

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_core.documents import (
    Document
)


CHROMA_DIR = os.getenv(
    "CHROMA_DIR",
    "vector_db"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2"
)

_embeddings = None


def get_embeddings():

    global _embeddings

    if _embeddings is None:

        print(
            "Loading embedding model..."
        )

        _embeddings = HuggingFaceEmbeddings(

            model_name=EMBEDDING_MODEL,

            model_kwargs={
                "device": "cpu"
            }
        )

        print(
            "Embedding model loaded."
        )

    return _embeddings


def get_llm():

    return ChatMistralAI(

        model=os.getenv(
            "MISTRAL_MODEL",
            "mistral-small-latest"
        ),

        mistral_api_key=os.getenv(
            "MISTRAL_API_KEY"
        ),

        temperature=0.3,

        timeout=120,

        max_retries=2
    )


def format_docs(
    docs
):

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


def build_vector_store(
    transcript: str
):

    print(
        "Building vector store..."
    )

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=800,

        chunk_overlap=80
    )

    chunks = splitter.split_text(
        transcript
    )

    if not chunks:

        raise ValueError(
            "Transcript is empty."
        )

    docs = [

        Document(

            page_content=chunk,

            metadata={
                "chunk_index": i
            }
        )

        for i, chunk
        in enumerate(chunks)
    ]

    # Unique collection for each meeting
    collection_name = (
        f"meeting_"
        f"{abs(hash(transcript)) % 10000000000}"
    )

    vector_store = Chroma.from_documents(

        documents=docs,

        embedding=get_embeddings(),

        collection_name=collection_name,

        persist_directory=CHROMA_DIR
    )

    print(
        "Vector store created successfully."
    )

    return vector_store


def get_retriever(
    vector_store,
    k: int = 4
):

    return vector_store.as_retriever(

        search_type="similarity",

        search_kwargs={
            "k": k
        }
    )


def build_rag_chain(
    transcript: str
):

    vector_store = build_vector_store(
        transcript
    )

    retriever = get_retriever(
        vector_store,
        k=4
    )

    prompt = ChatPromptTemplate.from_messages(

        [

            (
                "system",

                """
You are an expert meeting assistant.

Answer the user's question based ONLY
on the meeting transcript context.

If the answer is not found in the context,
say:

"I could not find this information
in the meeting transcript."

Be concise and precise.

Context:

{context}
"""
            ),

            (
                "human",
                "{question}"
            )
        ]
    )

    rag_chain = (

        {
            "context":
                retriever
                | RunnableLambda(format_docs),

            "question":
                RunnablePassthrough()
        }

        | prompt

        | get_llm()

        | StrOutputParser()
    )

    return rag_chain


def ask_question(
    rag_chain,
    question: str
) -> str:

    print(
        f"Question: {question}"
    )

    answer = rag_chain.invoke(
        question
    )

    print(
        f"Answer: {answer}"
    )

    return answer