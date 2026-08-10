import os
import tempfile
from pathlib import Path
from typing import Literal

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pydantic import BaseModel, Field


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="AI Resume Screening Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Change this path if your Qwen model is stored elsewhere.
MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5


# ============================================================
# 2. PAGE STYLING
# ============================================================

st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    .hero {
        padding: 1.5rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #111827, #243b53);
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.2rem;
    }

    .hero p {
        margin-top: 0.5rem;
        color: #dbeafe;
        font-size: 1rem;
    }

    .metric-card {
        padding: 1rem;
        border-radius: 14px;
        background: white;
        border: 1px solid #e5e7eb;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .candidate-card {
        padding: 1.2rem;
        border-radius: 16px;
        background: white;
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
        box-shadow: 0 3px 12px rgba(0,0,0,0.05);
    }

    .small-muted {
        color: #6b7280;
        font-size: 0.9rem;
    }

    .stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <h1>📄 AI Resume Screening Assistant</h1>
    <p>LangChain + RAG + FAISS + LLM powered candidate screening</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 4. LOAD MODELS
# ============================================================

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


@st.cache_resource
def load_llm():
    tokenizer =tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True,
        return_full_text=False,
    )

    return HuggingFacePipeline(
        pipeline=hf_pipeline
    )


# ============================================================
# 5. OUTPUT SCHEMA
# ============================================================

class ResumeEvaluation(BaseModel):
    match_score: int = Field(
        description="Score from 0 to 100"
    )

    matching_skills: list[str] = Field(
        description="Skills explicitly present in both JD and resume"
    )

    missing_skills: list[str] = Field(
        description="JD skills not found in the resume"
    )

    recommendation: Literal[
        "Hire",
        "Shortlist",
        "Reject"
    ]


parser = PydanticOutputParser(
    pydantic_object=ResumeEvaluation
)


# ============================================================
# 6. PROMPT
# ============================================================

structured_prompt = ChatPromptTemplate.from_template("""
You are a resume screening assistant.

Compare the Job Description with the Resume Context.

Use ONLY the resume context.

IMPORTANT:

1. matching_skills:
   Include ONLY skills explicitly present
   in BOTH the Job Description and Resume Context.

2. missing_skills:
   Include ONLY skills present in the Job Description
   but NOT present in the Resume Context.

3. A skill MUST NOT appear in both lists.

4. Do not invent information.

5. match_score must be an integer between 0 and 100.

6. recommendation must be exactly one of:
   "Hire"
   "Shortlist"
   "Reject"

Return ONLY the requested structured output.

Job Description:
{job_description}

Resume Context:
{context}

{format_instructions}
""")


# ============================================================
# 7. HELPER FUNCTIONS
# ============================================================

def format_docs(docs):
    return "\n\n".join(
        doc.page_content for doc in docs
    )


def load_uploaded_resumes(uploaded_files):
    """
    Load all uploaded PDFs and add resume metadata.
    """
    documents = []

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_dir = Path(temp_dir)

        for uploaded_file in uploaded_files:

            file_path = temp_dir / uploaded_file.name

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyPDFLoader(str(file_path))
            docs = loader.load()

            for doc in docs:
                doc.metadata["resume_file"] = uploaded_file.name
                doc.metadata["candidate_name"] = Path(
                    uploaded_file.name
                ).stem

            documents.extend(docs)

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    return splitter.split_documents(documents)


def create_vector_store(documents, embedding_model):
    return FAISS.from_documents(
        documents=documents,
        embedding=embedding_model
    )


def evaluate_candidates(job_description, vector_store, documents, llm):
    """
    Evaluate each uploaded resume separately.

    FAISS is used as the vector database.
    Each candidate gets a candidate-specific retrieval
    so candidates are not mixed together.
    """

    candidate_names = list(dict.fromkeys(
        doc.metadata["resume_file"]
        for doc in documents
    ))

    results = {}
    failed = []

    candidate_parser = PydanticOutputParser(
        pydantic_object=ResumeEvaluation
    )

    candidate_chain = (
        {
            "context": lambda x: x["context"],
            "job_description": lambda x: x["job_description"],
            "format_instructions": lambda _: (
                candidate_parser.get_format_instructions()
            )
        }
        | structured_prompt
        | llm
        | candidate_parser
    )

    for candidate_name in candidate_names:

        candidate_docs = [
            doc for doc in documents
            if doc.metadata["resume_file"] == candidate_name
        ]

        # Candidate-specific FAISS retrieval
        candidate_store = FAISS.from_documents(
            documents=candidate_docs,
            embedding=load_embedding_model()
        )

        candidate_retriever = candidate_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": min(TOP_K, len(candidate_docs))}
        )

        try:
            relevant_docs = candidate_retriever.invoke(
                job_description
            )

            resume_context = format_docs(relevant_docs)

            result = candidate_chain.invoke({
                "job_description": job_description,
                "context": resume_context
            })

            results[candidate_name] = result

        except Exception as e:
            failed.append({
                "candidate": candidate_name,
                "error": str(e)
            })

    return results, failed


# ============================================================
# 8. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Screening Settings")

    st.markdown(
        "Upload resumes and provide a Job Description "
        "to screen candidates using RAG."
    )

    st.divider()

    st.caption("Current LLM")
    st.code("Qwen 0.5B Instruct", language="text")

    st.caption("Embeddings")
    st.code("all-MiniLM-L6-v2", language="text")

    st.caption("Vector Database")
    st.code("FAISS", language="text")

    st.divider()

    st.info(
        "The screening pipeline uses only the uploaded "
        "resume content as retrieval context."
    )


# ============================================================
# 9. INPUTS
# ============================================================

st.subheader("1️⃣ Upload Resumes")

uploaded_files = st.file_uploader(
    "Upload one or more resume PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    help="You can upload multiple candidate resumes."
)

if uploaded_files:
    st.success(
        f"{len(uploaded_files)} resume(s) uploaded"
    )

    cols = st.columns(min(len(uploaded_files), 4))

    for i, file in enumerate(uploaded_files):
        with cols[i % len(cols)]:
            st.markdown(
                f"📄 **{file.name}**"
            )


st.subheader("2️⃣ Job Description")

job_description = st.text_area(
    "Enter the Job Description",
    height=220,
    placeholder=(
        "Example:\n"
        "We are looking for a Data Scientist with Python, "
        "Machine Learning, SQL, Pandas, Scikit-learn, "
        "TensorFlow and Docker."
    )
)


# ============================================================
# 10. EVALUATE BUTTON
# ============================================================

st.divider()

evaluate_button = st.button(
    "🚀 Evaluate Candidates",
    type="primary"
)


if evaluate_button:

    if not uploaded_files:
        st.warning("Please upload at least one resume PDF.")

    elif not job_description.strip():
        st.warning("Please enter a Job Description.")

    else:

        try:

            with st.spinner(
                "Loading models and preparing the resumes..."
            ):

                embedding_model = load_embedding_model()
                llm = load_llm()

                documents = load_uploaded_resumes(
                    uploaded_files
                )

                chunked_documents = split_documents(
                    documents
                )

                vector_store = create_vector_store(
                    chunked_documents,
                    embedding_model
                )

            st.success(
                f"Processed {len(uploaded_files)} resumes "
                f"into {len(chunked_documents)} chunks."
            )

            with st.spinner(
                "Retrieving relevant information and evaluating candidates..."
            ):

                candidate_results, failed_candidates = (
                    evaluate_candidates(
                        job_description,
                        vector_store,
                        chunked_documents,
                        llm
                    )
                )

            if not candidate_results:
                st.error(
                    "No candidate could be evaluated. "
                    "Please check the model and resume files."
                )

            else:

                # Rank candidates
                ranked_candidates = sorted(
                    candidate_results.items(),
                    key=lambda x: x[1].match_score,
                    reverse=True
                )

                # ====================================================
                # SUMMARY
                # ====================================================

                st.subheader("📊 Screening Summary")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Resumes Uploaded",
                        len(uploaded_files)
                    )

                with col2:
                    st.metric(
                        "Successfully Evaluated",
                        len(candidate_results)
                    )

                with col3:
                    st.metric(
                        "Best Match",
                        f"{ranked_candidates[0][1].match_score}/100"
                    )

                # ====================================================
                # BEST CANDIDATE
                # ====================================================

                best_name, best_result = ranked_candidates[0]

                st.subheader("🏆 Best Candidate")

                st.markdown(
                    f"""
                    <div class="candidate-card">
                        <h3>🥇 {best_name}</h3>
                        <p><b>Match Score:</b> {best_result.match_score}/100</p>
                        <p><b>Recommendation:</b> {best_result.recommendation}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # ====================================================
                # RANKING TABLE
                # ====================================================

                st.subheader("📈 Candidate Ranking")

                ranking_data = []

                for rank, (name, result) in enumerate(
                    ranked_candidates,
                    start=1
                ):

                    ranking_data.append({
                        "Rank": rank,
                        "Candidate": name,
                        "Match Score": result.match_score,
                        "Recommendation": result.recommendation
                    })

                st.dataframe(
                    ranking_data,
                    use_container_width=True,
                    hide_index=True
                )

                # ====================================================
                # DETAILED RESULTS
                # ====================================================

                st.subheader("🔍 Candidate Evaluations")

                for rank, (name, result) in enumerate(
                    ranked_candidates,
                    start=1
                ):

                    with st.expander(
                        f"#{rank}  {name}  —  "
                        f"{result.match_score}/100  "
                        f"• {result.recommendation}",
                        expanded=(rank == 1)
                    ):

                        score_col, rec_col = st.columns(2)

                        with score_col:
                            st.metric(
                                "Match Score",
                                f"{result.match_score}/100"
                            )

                        with rec_col:
                            st.metric(
                                "Recommendation",
                                result.recommendation
                            )

                        st.markdown("### ✅ Matching Skills")

                        if result.matching_skills:
                            for skill in result.matching_skills:
                                st.markdown(
                                    f"- {skill}"
                                )
                        else:
                            st.write("No matching skills identified.")

                        st.markdown("### ❌ Missing Skills")

                        if result.missing_skills:
                            for skill in result.missing_skills:
                                st.markdown(
                                    f"- {skill}"
                                )
                        else:
                            st.write("No missing skills identified.")

                # ====================================================
                # FAILED CANDIDATES
                # ====================================================

                if failed_candidates:

                    st.warning(
                        f"{len(failed_candidates)} candidate(s) "
                        "could not be evaluated by the current local LLM."
                    )

                    with st.expander(
                        "View failed candidates"
                    ):

                        for item in failed_candidates:
                            st.write(
                                f"**{item['candidate']}**"
                            )

        except Exception as e:

            st.error(
                "An error occurred while running the screening pipeline."
            )

            with st.expander("Technical details"):
                st.exception(e)


# ============================================================
# 11. FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Resume Screening Assistant • "
    "LangChain • RAG • FAISS • Local LLM"
)
