import os

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
    )


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )

    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:

    llm = get_llm()

    # Prompt for summarizing each transcript chunk
    map_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Summarize this portion of a meeting transcript "
            "concisely. Focus on the important topics, discussions, "
            "decisions, and information."
        ),
        (
            "human",
            "{text}"
        ),
    ])

    map_chain = map_prompt | llm | StrOutputParser()

    # Split transcript into smaller chunks
    chunks = split_transcript(transcript)

    if not chunks:
        return "No transcript available to summarize."

    # Summarize each chunk
    chunk_summaries = []

    for chunk in chunks:
        summary = map_chain.invoke({
            "text": chunk
        })
        chunk_summaries.append(summary)

    # Combine all partial summaries
    combined = "\n\n".join(chunk_summaries)

    # Final summary prompt
    combined_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting summarizer.

Combine the partial meeting summaries below into one
professional final meeting summary.

Include:
- Main topics discussed
- Important points
- Key decisions
- Important conclusions

Use concise bullet points.
Do not add information that is not present in the summaries."""
        ),
        (
            "human",
            "{text}"
        ),
    ])

    combined_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | combined_prompt
        | llm
        | StrOutputParser()
    )

    return combined_chain.invoke(combined)


def generate_title(transcript: str) -> str:

    llm = get_llm()

    title_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            (
                "system",
                """Based on the meeting transcript, generate a short
professional meeting title.

Requirements:
- Maximum 8 words
- Clear and descriptive
- Professional
- Only return the title
- Do not add quotes or explanations"""
            ),
            (
                "human",
                "{text}"
            ),
        ])
        | llm
        | StrOutputParser()
    )

    # Use only the first 2000 characters for title generation
    return title_chain.invoke(transcript[:2000])