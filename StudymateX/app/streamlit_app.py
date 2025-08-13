import sys
import os

# Ensure the project root is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
import streamlit as st
import tempfile
from pathlib import Path
import os
import json
from dotenv import load_dotenv

from ingest.pdf_ingest import process_uploaded_files
from ingest.chunker import make_chunks
from embedding.embed_index import EmbeddingIndex
from retriever.retriever import Retriever
from llm.groq_client import GroqLLM

load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE_WORDS", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP_WORDS", 100))
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH", 32))
TOP_K = int(os.getenv("TOP_K", 3))

st.set_page_config(page_title="StudyMate - Llama 3.1 8B", layout="wide")
st.title("📚 StudyMate — PDF Q&A")

# Session state
if "index" not in st.session_state:
    st.session_state.index = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "quiz" not in st.session_state:
    st.session_state.quiz = None  # structure: {questions: [...], answers: [...], score: int}
if "quiz_state" not in st.session_state:
    st.session_state.quiz_state = {"current": 0, "selected": {}, "submitted": False}

# Sidebar controls for a polished, presentation-ready experience
with st.sidebar:
    st.subheader("Quiz generator")
    with st.expander("Build a quiz from your PDFs"):
        num_q = st.number_input("Number of questions", 3, 25, 5)
        num_choices = st.selectbox("Choices per question", [3, 4, 5], index=1)
        gen_quiz = st.button("Generate quiz", type="primary", disabled=st.session_state.index is None)
        if gen_quiz:
            if st.session_state.index is None:
                st.warning("Please upload and process PDFs first.")
            else:
                try:
                    # Sample top chunks to seed quiz topics
                    # We pick random chunks from the metadatas in the index
                    metas = getattr(st.session_state.index, "metadatas", [])
                    if not metas:
                        st.warning("No chunks available to generate a quiz.")
                    else:
                        sample_topics = []
                        step = max(1, len(metas) // max(10, num_q))
                        for i, m in enumerate(metas[::step][: num_q * 2]):
                            # Use the chunk text as a topic seed
                            sample_topics.append(m.get("text", ""))
                        # Ask LLM to produce a JSON quiz
                        llm = GroqLLM()
                        quiz_prompt = (
                            "You are a strict quiz generator. Create a multiple-choice quiz based ONLY on the provided context.\n"
                            "Rules:\n"
                            "- Every question must be answerable from the context.\n"
                            "- Keep questions concise, clear, and factual.\n"
                            "- Provide exactly one correct answer and other plausible distractors.\n"
                            "- Return ONLY valid JSON with this exact structure:\n"
                            "{\n"
                            '  "questions": [\n'
                            "    {\n"
                            '      "question": "string",\n'
                            '      "choices": ["string1", "string2", ...],\n'
                            '      "answer_index": 0\n'
                            "    }\n"
                            "  ]\n"
                            "}\n"
                            f"- Choices per question: exactly {num_choices} choices.\n"
                            f"- Number of questions: exactly {num_q}.\n"
                            "- Do not include any markdown formatting or explanations.\n"
                            "- Ensure answer_index is a valid integer (0 to {num_choices-1}).\n\n"
                            f"Context:\n{chr(10).join(sample_topics[:20])}\n"
                        )
                        raw = llm.generate(quiz_prompt, temperature=0.1, max_tokens=1200)
                            
                        # Clean and parse JSON
                        data = None
                        try:
                            # Remove markdown code blocks if present
                            raw_clean = raw.strip()
                            if raw_clean.startswith("```json"):
                                raw_clean = raw_clean[7:]
                            if raw_clean.startswith("```"):
                                raw_clean = raw_clean[3:]
                            if raw_clean.endswith("```"):
                                raw_clean = raw_clean[:-3]
                            
                            data = json.loads(raw_clean.strip())
                            
                            # Validate quiz structure
                            if not isinstance(data, dict) or "questions" not in data:
                                st.error("Invalid quiz format: missing 'questions' field")
                                raise ValueError("Invalid format")
                                
                            qs = data["questions"]
                            valid = []
                            
                            for q in qs:
                                if not isinstance(q, dict):
                                    continue
                                    
                                question = q.get("question", "").strip()
                                choices = q.get("choices", [])
                                ans_idx = q.get("answer_index")
                                
                                # Validate all required fields
                                if not question:
                                    continue
                                    
                                if not isinstance(choices, list) or len(choices) != num_choices:
                                    continue
                                    
                                if not all(isinstance(c, str) and c.strip() for c in choices):
                                    continue
                                    
                                if not isinstance(ans_idx, int) or ans_idx < 0 or ans_idx >= len(choices):
                                    continue
                                    
                                valid.append({
                                    "question": question,
                                    "choices": choices,
                                    "answer_index": ans_idx
                                })
                            
                            if not valid:
                                st.error("Quiz generation failed: no valid questions generated. Try reducing the number of questions or ensure your PDF has sufficient content.")
                            else:
                                st.session_state.quiz = {"questions": valid}
                                st.session_state.quiz_state = {"current": 0, "selected": {}, "submitted": False}
                                st.success(f"Generated {len(valid)} valid questions.")
                                
                        except json.JSONDecodeError as e:
                            st.error(f"Failed to parse quiz response: {str(e)[:100]}...")
                            st.write("Raw response:", raw[:200] + "..." if len(raw) > 200 else raw)
                        except ValueError as e:
                            st.error(str(e))
                except Exception as e:
                    st.error(f"Quiz generation error: {e}")

    cols = st.columns(2)
    with cols[0]:
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()
    with cols[1]:
        chat_md = "\n\n".join([
            (f"**User:** {m['content']}" if m.get('role') == 'user' else f"**Assistant:** {m.get('content','')}")
            for m in st.session_state.messages
        ])
        st.download_button("Download chat", data=chat_md, file_name="studymate_chat.md", mime="text/markdown")

    st.subheader("Settings")
    top_k_ui = st.slider("Top K passages", min_value=1, max_value=10, value=TOP_K, help="How many chunks to retrieve")
    temperature_ui = st.slider("Creativity (temperature)", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    min_score_ui = st.slider("Minimum context score", min_value=0.0, max_value=1.0, value=0.2, step=0.05, help="If best match is below this, the assistant will admit uncertainty")
    system_prompt_ui = st.text_area("System instructions", value="You are a helpful, concise study assistant. Cite sources.", height=100)

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    tmpdir = Path(tempfile.gettempdir()) / "studymate_uploads"
    tmpdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for uf in uploaded_files:
        p = tmpdir / uf.name
        p.write_bytes(uf.getbuffer())
        paths.append(p)

    corpus = process_uploaded_files(paths)

    metadatas, texts = [], []
    for doc in corpus:
        chunks = make_chunks(doc["text"], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        for c in chunks:
            metadatas.append({"source": doc["path"], "chunk_id": c["chunk_id"], "text": c["text"]})
            texts.append(c["text"])

    emb_index = EmbeddingIndex(model_name=EMBED_MODEL, normalize=True)
    embeddings = emb_index.embed_texts(texts, batch_size=BATCH_SIZE)
    emb_index.build_index(embeddings, metadatas)
    st.session_state.index = emb_index
    st.success("✅ PDF processed and indexed.")
    st.info(f"Embedded {len(texts)} chunks with model '{EMBED_MODEL}'.")

# Quiz display section
if st.session_state.quiz and st.session_state.quiz.get("questions"):
    st.header("📊 Quiz")
    questions = st.session_state.quiz["questions"]
    current = st.session_state.quiz_state["current"]
    selected = st.session_state.quiz_state["selected"]
    
    # Check if all questions have been answered
    if len(selected) == len(questions):
        # Show final results
        st.subheader("🎯 Quiz Results")
        
        correct_count = 0
        for i, question in enumerate(questions):
            user_answer = selected.get(i)
            correct_answer = question["answer_index"]
            
            st.write(f"**Question {i+1}:** {question['question']}")
            
            for j, choice in enumerate(question["choices"]):
                if j == correct_answer:
                    st.write(f"✅ **{choice}** (Correct)")
                elif j == user_answer and j != correct_answer:
                    st.write(f"❌ **{choice}** (Your answer - incorrect)")
                elif j == user_answer and j == correct_answer:
                    st.write(f"✅ **{choice}** (Your answer - correct)")
                else:
                    st.write(f"   {choice}")
            
            if user_answer == correct_answer:
                correct_count += 1
                st.success("Correct!")
            else:
                st.error(f"Incorrect. Correct answer: {question['choices'][correct_answer]}")
            st.write("---")
        
        score = (correct_count / len(questions)) * 100
        st.success(f"**Final Score: {correct_count}/{len(questions)} ({score:.1f}%)**")
        
        if st.button("Retake Quiz"):
            st.session_state.quiz = None
            st.session_state.quiz_state = {"current": 0, "selected": {}, "submitted": False}
            st.rerun()
    
    elif current < len(questions):
        question = questions[current]
        st.subheader(f"Question {current + 1} of {len(questions)}")
        st.write(question["question"])
        
        # Display choices as radio buttons
        choice = st.radio(
            "Select your answer:",
            question["choices"],
            key=f"q_{current}",
            index=selected.get(current)
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Previous", disabled=current == 0):
                st.session_state.quiz_state["current"] = max(0, current - 1)
                st.rerun()
        
        with col2:
            if st.button("Next", disabled=current == len(questions) - 1):
                st.session_state.quiz_state["current"] = min(len(questions) - 1, current + 1)
                st.rerun()
        
        with col3:
            if st.button("Submit All Answers", disabled=len(selected) < len(questions)):
                st.session_state.quiz_state["submitted"] = True
                st.rerun()
                
        if choice is not None:
            selected[current] = question["choices"].index(choice)
            st.info("Answer saved! Continue to next question or submit when complete.")
        
        # Show progress
        st.progress(len(selected) / len(questions))
        st.write(f"Progress: {len(selected)}/{len(questions)} questions answered")

# Display previous messages (old messages appear above new ones)
for msg in st.session_state.messages:
    role = msg.get("role", "assistant")
    content = msg.get("content", "")
    citations = msg.get("citations")
    avatar = "🧑‍🎓" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)
        # Show citations under assistant replies
        if role == "assistant" and citations:
            with st.expander("Sources"):
                for c in citations:
                    st.markdown(f"- {c}")

# Chat input at the bottom
question = st.chat_input("Ask a question")
if question:
    if st.session_state.index is None:
        st.warning("Please upload and process PDFs first.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                retriever = Retriever(st.session_state.index)
                hits = retriever.retrieve(question, top_k=top_k_ui)

                # Filter by min score if desired
                filtered = [h for h in hits if h["score"] >= min_score_ui]
                if not filtered and hits:
                    filtered = [hits[0]]  # always keep best one as fallback

                context = "\n\n".join([
                    f"Source: {h['meta']['source']} (chunk {h['meta']['chunk_id']})\n{h['meta']['text']}"
                    for h in filtered
                ])
                if not context:
                    answer = "I couldn't find relevant content in the uploaded PDFs to answer that confidently."
                    citations = None
                else:
                    prompt = (
                        f"System: {system_prompt_ui}\n\n"
                        "Answer the question using only the following context. If the answer isn't in the context, say you don't know.\n\n"
                        f"Context:\n{context}\n\nQuestion: {question}"
                    )
                    try:
                        llm = GroqLLM()
                        answer = llm.generate(prompt, temperature=temperature_ui)
                    except Exception as e:
                        answer = f"Error generating answer: {e}"
                        citations = None
                    # Build citations list
                    citations = [
                        f"{h['meta']['source']} • chunk {h['meta']['chunk_id']}"
                        for h in filtered
                    ]
            st.markdown(answer)
        # Save assistant reply with citations
        st.session_state.messages.append({"role": "assistant", "content": answer, "citations": citations})
