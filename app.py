import json
import os
import re
import tempfile
from typing import Any

import numpy as np
import requests
import streamlit as st
from pypdf import PdfReader


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3")
PDF_MODEL = os.getenv("PDF_MODEL", "llama3.2")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")


st.set_page_config(
    page_title="Interní AI asistent KÚSK",
    page_icon=":material/smart_toy:",
    layout="wide",
)


def load_data(path: str = "data.json") -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return {"error": "Soubor data.json nebyl nalezen."}
    except json.JSONDecodeError as error:
        return {"error": f"Soubor data.json není platný JSON: {error}"}


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        if len(token) > 2
    }


def flatten_knowledge_base(data: Any, prefix: str = "znalostní báze") -> list[dict[str, str]]:
    if isinstance(data, dict):
        chunks = []
        for key, value in data.items():
            chunks.extend(flatten_knowledge_base(value, f"{prefix} / {key}"))
        return chunks

    if isinstance(data, list):
        chunks = []
        for index, value in enumerate(data, start=1):
            label = value.get("jmeno", index) if isinstance(value, dict) else index
            chunks.extend(flatten_knowledge_base(value, f"{prefix} / {label}"))
        return chunks

    return [{"title": prefix, "content": str(data)}]


def retrieve_knowledge_context(question: str, data: dict[str, Any], k: int = 4) -> list[dict[str, str]]:
    chunks = flatten_knowledge_base(data)
    question_tokens = tokenize(question)

    scored_chunks = []
    for chunk in chunks:
        searchable = f"{chunk['title']} {chunk['content']}"
        score = len(question_tokens & tokenize(searchable))
        scored_chunks.append((score, chunk))

    matches = [chunk for score, chunk in sorted(scored_chunks, key=lambda item: item[0], reverse=True) if score > 0]
    return matches[:k] if matches else chunks[:k]


def call_ollama_chat(model: str, messages: list[dict[str, str]]) -> str:
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={"model": model, "messages": messages, "stream": False},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def call_ollama_generate(model: str, prompt: str) -> str:
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["response"]


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_bytes)
        temp_path = temp_file.name

    try:
        reader = PdfReader(temp_path)
        text_parts = [
            page_text
            for page in reader.pages
            if (page_text := page.extract_text())
        ]
        return "\n".join(text_parts)
    finally:
        os.remove(temp_path)


def split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - overlap

    return chunks


def get_embedding(text: str) -> list[float]:
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBEDDING_MODEL, "prompt": text},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["embedding"]


def build_embedding_index(chunks: list[str]) -> np.ndarray:
    return np.array([get_embedding(chunk) for chunk in chunks], dtype="float32")


def search_similar_chunks(
    question: str,
    chunks: list[str],
    index: np.ndarray,
    k: int = 4,
) -> list[str]:
    question_vector = np.array([get_embedding(question)], dtype="float32")
    distances = np.linalg.norm(index - question_vector, axis=1)
    indices = np.argsort(distances)[: min(k, len(chunks))]
    return [chunks[int(i)] for i in indices]


def answer_from_pdf(question: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""
Jsi interní dokumentový asistent.
Použij pouze informace z poskytnutého kontextu.
Odpovídej česky, stručně a věcně.
Pokud odpověď v kontextu není, napiš: "Tohle v dokumentu nemohu spolehlivě najít."

KONTEXT:
{context}

OTÁZKA:
{question}
"""
    return call_ollama_generate(PDF_MODEL, prompt)


def init_state() -> None:
    defaults = {
        "company_messages": [],
        "company_last_answer": None,
        "company_sources": [],
        "pdf_chunks": None,
        "pdf_index": None,
        "pdf_signature": None,
        "pdf_last_answer": None,
        "pdf_sources": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_company_chat() -> None:
    st.session_state.company_messages = []
    st.session_state.company_last_answer = None
    st.session_state.company_sources = []


def reset_pdf() -> None:
    st.session_state.pdf_chunks = None
    st.session_state.pdf_index = None
    st.session_state.pdf_signature = None
    st.session_state.pdf_last_answer = None
    st.session_state.pdf_sources = []


def render_sidebar() -> None:
    with st.sidebar:
        st.title("Nastavení")
        st.caption("Vše běží lokálně přes Ollama.")
        st.write(f"Chat model: `{CHAT_MODEL}`")
        st.write(f"PDF model: `{PDF_MODEL}`")
        st.write(f"Embeddings: `{EMBEDDING_MODEL}`")
        st.divider()

        if st.button("Vymazat interní chat", use_container_width=True):
            reset_company_chat()
            st.rerun()

        if st.button("Vymazat PDF dokument", use_container_width=True):
            reset_pdf()
            st.rerun()


def render_demo_overview() -> None:
    with st.expander("Jak ukázat RAG při prezentaci"):
        col_search, col_context, col_answer = st.columns(3)

        with col_search:
            st.markdown("**1. Vyhledání**")
            st.write("Aplikace najde části znalostní báze nebo PDF, které se nejvíc podobají dotazu.")

        with col_context:
            st.markdown("**2. Kontext**")
            st.write("Do modelu se posílají jen vybrané úryvky, ne celý dokument ani volné hádání.")

        with col_answer:
            st.markdown("**3. Odpověď**")
            st.write("Uživatel vidí odpověď a může si rozbalit zdroje, ze kterých vznikla.")


def render_company_assistant(data: dict[str, Any]) -> None:
    st.header("Interní asistent")
    st.caption("Dotazy nad interní znalostní bází organizace.")

    if "error" in data:
        st.warning(data["error"])

    for message in st.session_state.company_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Na co se chcete zeptat?", key="company_prompt"):
        st.session_state.company_messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        context_chunks = retrieve_knowledge_context(prompt, data)
        context = "\n\n".join(
            f"ZDROJ: {chunk['title']}\nOBSAH: {chunk['content']}"
            for chunk in context_chunks
        )

        system_context = f"""
Jsi profesionální interní asistent na Krajském úřadě.
Mluv vždy česky a odpovídej věcně.

Relevantní části znalostní báze:
{context}

Pravidla:
1. Odpovídej pouze podle relevantních částí znalostní báze.
2. Pokud uživatel žádá přesné znění pravidla, cituj relevantní text.
3. Pokud informaci v poskytnutém kontextu nemáš, řekni to a doporuč ověření u odpovědného oddělení.
"""

        messages = [
            {"role": "system", "content": system_context},
            *st.session_state.company_messages,
        ]

        with st.chat_message("assistant"):
            with st.spinner("Připravuji odpověď..."):
                try:
                    answer = call_ollama_chat(CHAT_MODEL, messages)
                except requests.RequestException as error:
                    st.error(f"Nepodařilo se spojit s Ollama: {error}")
                    return

            st.markdown(answer)

        st.session_state.company_messages.append({"role": "assistant", "content": answer})
        st.session_state.company_last_answer = answer
        st.session_state.company_sources = context_chunks
        st.rerun()

    if st.session_state.company_last_answer:
        with st.expander("Použitý kontext ze znalostní báze"):
            for index, chunk in enumerate(st.session_state.company_sources, start=1):
                st.markdown(f"**Zdroj {index}: {chunk['title']}**")
                st.info(chunk["content"])

        st.download_button(
            "Stáhnout poslední odpověď",
            data=st.session_state.company_last_answer,
            file_name="odpoved_interni_asistent.txt",
            mime="text/plain",
        )


def render_pdf_assistant() -> None:
    st.header("Dokumentový asistent")
    st.caption("Nahrajte PDF a ptejte se na jeho obsah.")

    uploaded_file = st.file_uploader("PDF dokument", type=["pdf"])

    if uploaded_file is not None:
        pdf_bytes = uploaded_file.getvalue()
        signature = (uploaded_file.name, len(pdf_bytes))

        if st.session_state.pdf_signature != signature:
            with st.spinner("Zpracovávám PDF a vytvářím vyhledávací index..."):
                try:
                    text = extract_text_from_pdf(pdf_bytes)
                    if not text.strip():
                        st.error("Z PDF se nepodařilo načíst žádný text.")
                        reset_pdf()
                        return

                    chunks = split_text(text)
                    st.session_state.pdf_chunks = chunks
                    st.session_state.pdf_index = build_embedding_index(chunks)
                    st.session_state.pdf_signature = signature
                    st.session_state.pdf_last_answer = None
                    st.session_state.pdf_sources = []
                except (requests.RequestException, ValueError) as error:
                    st.error(f"PDF se nepodařilo zpracovat: {error}")
                    return

        st.success(f"PDF je připravené: {uploaded_file.name}")

    question = st.text_input("Zeptejte se na dokument")
    is_ready = st.session_state.pdf_chunks is not None and st.session_state.pdf_index is not None

    if question and is_ready:
        with st.spinner("Hledám odpověď v dokumentu..."):
            try:
                sources = search_similar_chunks(
                    question,
                    st.session_state.pdf_chunks,
                    st.session_state.pdf_index,
                )
                answer = answer_from_pdf(question, sources)
            except requests.RequestException as error:
                st.error(f"Nepodařilo se spojit s Ollama: {error}")
                return

        st.session_state.pdf_last_answer = answer
        st.session_state.pdf_sources = sources

    elif question and not is_ready:
        st.info("Nejdřív nahrajte PDF dokument.")

    if st.session_state.pdf_last_answer:
        st.subheader("Odpověď")
        st.markdown(st.session_state.pdf_last_answer)

        with st.expander("Použité části dokumentu"):
            for index, chunk in enumerate(st.session_state.pdf_sources, start=1):
                st.markdown(f"**Úryvek {index}**")
                st.info(chunk[:700] + ("..." if len(chunk) > 700 else ""))

        st.download_button(
            "Stáhnout poslední odpověď",
            data=st.session_state.pdf_last_answer,
            file_name="odpoved_pdf_asistent.txt",
            mime="text/plain",
        )


init_state()
render_sidebar()

st.title("Interní AI asistent KÚSK")
st.write("Jeden lokální asistent pro interní znalostní bázi i dotazy nad PDF dokumenty.")
render_demo_overview()

tab_company, tab_pdf = st.tabs(["Interní znalostní báze", "PDF dokumenty"])

with tab_company:
    render_company_assistant(load_data())

with tab_pdf:
    render_pdf_assistant()
