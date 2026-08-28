import os
import re

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def get_llm():

    return ChatMistralAI(

        model=os.getenv(
            "MISTRAL_MODEL",
            "mistral-small-latest"
        ),

        mistral_api_key=os.getenv(
            "MISTRAL_API_KEY"
        ),

        temperature=0.2,

        timeout=120,

        max_retries=2
    )


def extract_section(
    text: str,
    section_name: str,
    next_sections: list
) -> str:

    if next_sections:

        next_pattern = "|".join(
            re.escape(x)
            for x in next_sections
        )

        pattern = (
            rf"(?is)"
            rf"{re.escape(section_name)}\s*:?"
            rf"(.*?)"
            rf"(?=\n\s*(?:{next_pattern})\s*:|\Z)"
        )

    else:

        pattern = (
            rf"(?is)"
            rf"{re.escape(section_name)}\s*:?"
            rf"(.*)"
        )

    match = re.search(
        pattern,
        text
    )

    if match:

        return match.group(1).strip()

    return ""


def analyze_meeting(
    transcript: str
) -> dict:

    if not transcript.strip():

        return {
            "title": "Untitled Meeting",
            "summary": "No transcript available.",
            "action_items": "No action items found.",
            "key_decisions": "No key decisions found.",
            "open_questions": "No open questions found."
        }

    # Prevent excessively large requests.
    max_chars = int(
        os.getenv(
            "MAX_ANALYSIS_CHARS",
            "50000"
        )
    )

    transcript_text = transcript[
        :max_chars
    ]

    prompt = ChatPromptTemplate.from_messages(

        [

            (
                "system",

                """
You are an expert meeting analyst.

Analyze the meeting transcript.

Return EXACTLY these five sections:

TITLE:
A short professional meeting title.
Maximum 8 words.

SUMMARY:
Concise bullet points covering:
- Main topics
- Important points
- Conclusions
- Decisions

ACTION ITEMS:
A numbered list.

For every action item include:
- Task
- Owner if mentioned
- Deadline if mentioned

If there are no action items:
No action items found.

KEY DECISIONS:
A numbered list of important decisions.

If there are no decisions:
No key decisions found.

OPEN QUESTIONS:
A numbered list of unresolved questions
or follow-up topics.

If there are no open questions:
No open questions found.

Use ONLY information from the transcript.
Do not invent information.
""",
            ),

            (
                "human",
                "{text}"
            )
        ]
    )

    chain = (
        prompt
        | get_llm()
        | StrOutputParser()
    )

    result = chain.invoke(
        {
            "text": transcript_text
        }
    )

    title = extract_section(
        result,
        "TITLE",
        [
            "SUMMARY",
            "ACTION ITEMS",
            "KEY DECISIONS",
            "OPEN QUESTIONS"
        ]
    )

    summary = extract_section(
        result,
        "SUMMARY",
        [
            "ACTION ITEMS",
            "KEY DECISIONS",
            "OPEN QUESTIONS"
        ]
    )

    action_items = extract_section(
        result,
        "ACTION ITEMS",
        [
            "KEY DECISIONS",
            "OPEN QUESTIONS"
        ]
    )

    decisions = extract_section(
        result,
        "KEY DECISIONS",
        [
            "OPEN QUESTIONS"
        ]
    )

    questions = extract_section(
        result,
        "OPEN QUESTIONS",
        []
    )

    return {

        "title": title
        or "Meeting Analysis",

        "summary": summary
        or "No summary generated.",

        "action_items": action_items
        or "No action items found.",

        "key_decisions": decisions
        or "No key decisions found.",

        "open_questions": questions
        or "No open questions found."
    }


# Compatibility functions
def generate_title(
    transcript: str
) -> str:

    return analyze_meeting(
        transcript
    )["title"]


def summarize(
    transcript: str
) -> str:

    return analyze_meeting(
        transcript
    )["summary"]