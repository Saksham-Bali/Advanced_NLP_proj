"""
backend.py — ContractGuard FastAPI Backend
==========================================
Wires together:
  • PDF/DOCX clause extraction
  • Fine-tuned LLaMA 3.1 8B (QLoRA) for risk classification
  • FAISS + BGE embeddings for RAG-powered chat
  • Two endpoints the existing frontend expects:
      POST /analyze  → multipart form upload  → clause risk results
      POST /ask      → JSON question          → grounded RAG answer

Run:
    uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
"""

# ─────────────────────────────────────────────────────────────
#  Standard library
# ─────────────────────────────────────────────────────────────
import json
import logging
import os
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

# ─────────────────────────────────────────────────────────────
#  FastAPI
# ─────────────────────────────────────────────────────────────
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─────────────────────────────────────────────────────────────
#  ML / RAG
# ─────────────────────────────────────────────────────────────
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ─────────────────────────────────────────────────────────────
#  Clause extraction  (clauseextraction.py — your pipeline)
# ─────────────────────────────────────────────────────────────
from clauseextraction import extract_clauses_from_pdf


# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION  — edit the three paths, everything else is fine
# ═══════════════════════════════════════════════════════════════

# Path to your saved LoRA adapter directory (the qlora_output folder)
# If None the server starts in "classification-disabled" mode and returns
# placeholder risk levels.  Set this before going live.
ADAPTER_PATH: Optional[str] = "./qlora_output"          # e.g. "./qlora_output"

# Base model ID — must match whatever you fine-tuned on
MODEL_ID: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Your clause dataset — used to build the RAG index
DATASET_PATH: str = "./final_test_updated.json"

# Where the FAISS index will be saved/loaded from
INDEX_PATH: str = "./legal_rag_index"

# BGE embedding model — same one used in legal_rag.ipynb
EMBED_MODEL: str = "BAAI/bge-base-en-v1.5"
BGE_QUERY_PREFIX = "Represent this legal clause for retrieval: "

TOP_K: int = 3                              # clauses to retrieve per question

MAX_SEQ_LEN: int = 8192
MAX_NEW_TOKENS_CLASSIFY: int = 150
MAX_NEW_TOKENS_CHAT: int = 400

# ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("contractguard")


# ═══════════════════════════════════════════════════════════════
#  LAZY MODEL LOADING
#  The model is heavy (~8 GB in 4-bit). Load it once on first use.
# ═══════════════════════════════════════════════════════════════

_model = None
_tokenizer = None
_model_ready = False


def _load_model():
    """Load base LLaMA model + LoRA adapter. Called once, cached globally."""
    global _model, _tokenizer, _model_ready

    if _model_ready:
        return

    if ADAPTER_PATH is None:
        log.warning(
            "ADAPTER_PATH is not set. Classification will use a placeholder. "
            "Set ADAPTER_PATH in the config section at the top of backend.py."
        )
        _model_ready = True
        return

    log.info("Loading tokenizer from %s …", MODEL_ID)
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    _tokenizer.pad_token = _tokenizer.eos_token
    _tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    log.info("Loading base model in 4-bit …")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    log.info("Applying LoRA adapter from %s …", ADAPTER_PATH)
    _model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    _model.eval()

    _model_ready = True
    log.info("Model ready.")


# ═══════════════════════════════════════════════════════════════
#  FAISS RAG INDEX
# ═══════════════════════════════════════════════════════════════

@dataclass
class LegalChunk:
    chunk_id:             str
    raw_clause:           str
    licensor_risk:        str
    licensee_risk:        str
    licensor_explanation: str
    licensee_explanation: str
    clause_index:         int
    enriched_text:        str


_faiss_index = None
_all_chunks: List[LegalChunk] = []
_embedder: Optional[SentenceTransformer] = None
_rag_ready = False


def _build_enriched_text(clause: str, licensor_risk: str, licensee_risk: str) -> str:
    return (
        f"Licensor Risk: {licensor_risk}. "
        f"Licensee Risk: {licensee_risk}. "
        f"Clause: {clause}"
    )


def _load_dataset_as_chunks(dataset_path: str) -> List[LegalChunk]:
    with open(dataset_path, "r") as f:
        data = json.load(f)

    chunks = []
    for entry in data:
        if entry.get("status") != "success":
            continue
        parties  = entry.get("parties", {})
        licensor = parties.get("Licensor", {})
        licensee = parties.get("Licensee", {})
        enriched = _build_enriched_text(
            entry["clause_text"],
            licensor.get("risk", "Unknown"),
            licensee.get("risk", "Unknown"),
        )
        chunks.append(LegalChunk(
            chunk_id             = f"clause_{entry['clause_index']}",
            raw_clause           = entry["clause_text"],
            licensor_risk        = licensor.get("risk", "Unknown"),
            licensee_risk        = licensee.get("risk", "Unknown"),
            licensor_explanation = licensor.get("explanation", ""),
            licensee_explanation = licensee.get("explanation", ""),
            clause_index         = entry["clause_index"],
            enriched_text        = enriched,
        ))
    log.info("Loaded %d clauses from dataset.", len(chunks))
    return chunks


def _init_rag():
    """Build or load the FAISS index. Called once at startup."""
    global _faiss_index, _all_chunks, _embedder, _rag_ready

    if _rag_ready:
        return

    _embedder = SentenceTransformer(EMBED_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")

    index_file = Path(INDEX_PATH) / "index.faiss"
    chunks_file = Path(INDEX_PATH) / "chunks.pkl"

    if index_file.exists() and chunks_file.exists():
        log.info("Loading existing FAISS index from %s …", INDEX_PATH)
        _faiss_index = faiss.read_index(str(index_file))
        with open(chunks_file, "rb") as f:
            _all_chunks = pickle.load(f)
        log.info("Index loaded: %d vectors.", _faiss_index.ntotal)
    else:
        if not Path(DATASET_PATH).exists():
            log.warning(
                "Dataset not found at %s — RAG will be unavailable. "
                "Place final_test_updated.json next to backend.py.",
                DATASET_PATH,
            )
            _rag_ready = True
            return

        log.info("Building FAISS index …")
        _all_chunks = _load_dataset_as_chunks(DATASET_PATH)
        texts = [c.enriched_text for c in _all_chunks]
        embeddings = _embedder.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        dim = embeddings.shape[1]
        _faiss_index = faiss.IndexFlatIP(dim)
        _faiss_index.add(embeddings)

        Path(INDEX_PATH).mkdir(parents=True, exist_ok=True)
        faiss.write_index(_faiss_index, str(index_file))
        with open(chunks_file, "wb") as f:
            pickle.dump(_all_chunks, f)
        log.info("Index saved: %d vectors, dim=%d.", _faiss_index.ntotal, dim)

    _rag_ready = True


def _retrieve(query: str, role: str, top_k: int = TOP_K) -> List[dict]:
    """Retrieve top-k clauses relevant to a query."""
    if not _rag_ready or _faiss_index is None or _embedder is None:
        return []

    prefixed = BGE_QUERY_PREFIX + query
    query_vec = _embedder.encode(
        [prefixed], normalize_embeddings=True
    ).astype(np.float32)

    scores, indices = _faiss_index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = _all_chunks[idx]
        if role == "Licensor":
            risk, explanation = chunk.licensor_risk, chunk.licensor_explanation
        else:
            risk, explanation = chunk.licensee_risk, chunk.licensee_explanation

        results.append({
            "clause":       chunk.raw_clause,
            "risk":         risk,
            "explanation":  explanation,
            "clause_index": chunk.clause_index,
            "similarity":   round(float(score), 4),
        })

    return results


# ═══════════════════════════════════════════════════════════════
#  CLAUSE EXTRACTION  — delegates to clauseextraction.py
# ═══════════════════════════════════════════════════════════════

def extract_clauses_from_upload(filename: str, file_bytes: bytes) -> List[dict]:
    """
    Write the uploaded bytes to a temp file, run the full 4-stage
    extraction pipeline from clauseextraction.py, and return the
    enriched clause list.

    Each item in the returned list is a dict with (at minimum):
        clause_id         (int)   — sequential id assigned by the extractor
        text              (str)   — full clause text
        section_header    (str)   — nearest section heading (e.g. "DEFINITIONS")
        page              (int)   — page number in the source PDF
        is_subpart        (bool)  — True if this is a lettered/roman subpart
        is_decimal_section(bool)  — True if this is a "2.1 …" decimal subsection
        parent_clause_id  (int|None) — clause_id of the parent, for sub-clauses
        party_refs        (list)  — party terms found in the text
        token_count       (int)

    Only PDF is supported by clauseextraction.py.
    """
    ext = Path(filename).suffix.lower()
    if ext != ".pdf":
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{ext}'. "
                   "clauseextraction.py only supports PDF.",
        )

    # pdfplumber needs a real file path, not an in-memory buffer
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        result = extract_clauses_from_pdf(tmp_path)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Clause extraction failed: {exc}",
        ) from exc
    finally:
        os.unlink(tmp_path)   # clean up temp file regardless of outcome

    clauses = result.get("clauses", [])
    if not clauses:
        raise HTTPException(
            status_code=422,
            detail="No clauses could be identified in this document. "
                   "The PDF may be scanned or have no extractable text.",
        )

    log.info(
        "Extractor: %d pages, %d clauses from '%s'",
        result["total_pages"], len(clauses), filename,
    )
    return clauses


# ═══════════════════════════════════════════════════════════════
#  CLASSIFICATION PROMPTS & INFERENCE
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT_CLASSIFY = (
    "You are a legal risk analyst specializing in licensing agreements. "
    "Your task is to analyze a contract clause and assess its risk level "
    "from a specific party's perspective. Focus on the practical consequences "
    "for the party — what they stand to lose, what obligations they must perform, "
    "and how protected they are if things go wrong. As a rough guide: low risk "
    "implies little to no liability for the party, and high risk implies exposure.\n\n"
    "Respond with exactly two lines:\n"
    "  Risk Level: <Low | High>\n"
    "  Explanation: <one or two sentences explaining why>\n\n"
    "Do not add any other text."
)


def _build_classification_prompt(clause_text: str, role: str) -> str:
    user_content = (
        f"Party role: {role}\n\n"
        f"Contract clause:\n{clause_text}\n\n"
        "What is the risk level of this clause for the above party?"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_CLASSIFY},
        {"role": "user",   "content": user_content},
    ]
    return _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _parse_model_output(raw_output: str) -> tuple[str, str]:
    """
    Parse the model's two-line output into (risk_level, explanation).
    Falls back gracefully if the format is off.
    """
    risk_level  = "High"
    explanation = raw_output.strip()

    for line in raw_output.splitlines():
        line = line.strip()
        if line.lower().startswith("risk level:"):
            raw_risk = line.split(":", 1)[1].strip()
            if "low" in raw_risk.lower():
                risk_level = "Low"
            elif "high" in raw_risk.lower():
                risk_level = "High"
            else:
                log.warning("Unexpected risk label — defaulting to High. Raw: %s", raw_risk)
                risk_level = "High"
        elif line.lower().startswith("explanation:"):
            explanation = line.split(":", 1)[1].strip()

    return risk_level, explanation


def _placeholder_classify(clause_text: str, role: str) -> tuple[str, str]:
    """
    Used when ADAPTER_PATH is not set.
    Returns a deterministic placeholder so the pipeline still runs end-to-end.
    """
    text_lower = clause_text.lower()
    if any(w in text_lower for w in ["indemnif", "liabil", "damages", "penalty"]):
        risk = "High"
        exp  = f"[Placeholder] Clause contains liability/indemnification language — High risk for {role}."
    elif any(w in text_lower for w in ["terminat", "notice", "warrant"]):
        risk = "Medium"
        exp  = f"[Placeholder] Clause involves rights/obligations that create moderate risk for {role}."
    else:
        risk = "Low"
        exp  = f"[Placeholder] Clause appears administrative — Low risk for {role}."
    return risk, exp


def classify_clause(clause_text: str, role: str) -> tuple[str, str]:
    """Run the fine-tuned model on one clause. Returns (risk_level, explanation)."""
    if not _model_ready or _model is None or _tokenizer is None:
        return _placeholder_classify(clause_text, role)

    prompt = _build_classification_prompt(clause_text, role)
    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS_CLASSIFY,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=_tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_output = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return _parse_model_output(raw_output)


# ═══════════════════════════════════════════════════════════════
#  RAG CHAT PROMPTS & GENERATION
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT_CHAT = """You are a legal AI assistant helping analyze a contract.
You will be given:
  1. A summary of the initial risk analysis of the contract clauses.
  2. Reference clauses retrieved from a legal database relevant to the question.
  3. The conversation history so far.

Rules:
  - Always analyze from the perspective of the {role}.
  - Ground your answers in the retrieved reference clauses wherever possible.
  - If the references don't cover the question, say so rather than guessing.
  - Be concise and specific. Avoid generic legal disclaimers."""


def _format_initial_analysis(results: list, question: str = "") -> str:
    import re
    # Only include high-risk clauses + any clause numbers explicitly mentioned
    mentioned = {int(n) for n in re.findall(r"clause\s*(\d+)", question, re.I)}
    lines = []
    for r in results:
        is_high      = "high" in r["model_output"].lower()
        is_mentioned = r["clause_index"] in mentioned
        if not (is_high or is_mentioned):
            continue
        first_line = r["model_output"].split("\n")[0].strip()
        snippet    = r["clause_text"][:80].replace("\n", " ")
        lines.append(f"  Clause {r['clause_index']}: {first_line} | \"{snippet}...\"")
    if not lines:
        lines.append("  No high-risk clauses identified.")
    return "\n".join(lines)


def _format_retrieved_context(retrieved: list, role: str) -> str:
    if not retrieved:
        return "No closely matching reference clauses found."
    blocks = []
    for i, ctx in enumerate(retrieved, 1):
        blocks.append(
            f"[Reference {i}]\n"
            f"  Risk for {role}: {ctx['risk']}\n"
            f"  Why: {ctx['explanation']}\n"
            f"  Clause text: {ctx['clause'][:300]}"
            f"{'...' if len(ctx['clause']) > 300 else ''}"
        )
    return "\n\n".join(blocks)


def _generate_chat_response(
    user_question: str,
    role: str,
    conversation_history: list,
    initial_results: list,
    retrieved: list,
) -> str:
    """Build prompt and generate one RAG chat turn."""
    if not _model_ready or _model is None or _tokenizer is None:
        # Graceful degradation — return a text-only answer based on retrieved context
        if retrieved:
            refs = "\n".join(
                f"• [{r['risk']} for {role}] {r['clause'][:120]}…"
                for r in retrieved
            )
            return (
                f"Based on retrieved reference clauses (model not loaded):\n\n{refs}\n\n"
                "Load the fine-tuned model by setting ADAPTER_PATH in backend.py "
                "for AI-generated answers."
            )
        return "No relevant clauses found and model is not loaded."

    analysis_summary = _format_initial_analysis(initial_results, user_question)
    system_content   = (
        SYSTEM_PROMPT_CHAT.format(role=role) + "\n\n"
        f"CONTRACT ANALYSIS SUMMARY ({role} perspective):\n{analysis_summary}"
    )

    retrieved_block    = _format_retrieved_context(retrieved, role)
    augmented_user_msg = (
        f"RELEVANT REFERENCE CLAUSES (retrieved for this question):\n"
        f"{retrieved_block}\n\n"
        f"QUESTION:\n{user_question}"
    )

    messages = [{"role": "system", "content": system_content}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": augmented_user_msg})

    prompt = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = _tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN
    )
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS_CHAT,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=_tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ═══════════════════════════════════════════════════════════════
#  SERVER-SIDE SESSION STATE
#  Single-user local tool — we keep the last analysis in memory.
#  For multi-user production, replace with Redis or a DB.
# ═══════════════════════════════════════════════════════════════

_session = {
    "initial_results":     [],    # output of analyze_all_clauses()
    "role":                None,  # "Licensor" or "Licensee"
    "conversation_history": [],   # [{role, content}, …]
}


# ═══════════════════════════════════════════════════════════════
#  FASTAPI APPLICATION
# ═══════════════════════════════════════════════════════════════

app = FastAPI(
    title="ContractGuard API",
    description="Legal clause risk analysis + RAG-powered Q&A",
    version="1.0.0",
)

# Allow the local frontend (file:// or localhost) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """
    Initialise the RAG index at startup.
    Model loading is deferred to first request to avoid blocking startup.
    """
    log.info("Initialising RAG index …")
    _init_rag()
    # Load the model in the background on startup so the first request isn't slow.
    # Comment this out on machines with < 16 GB RAM — use lazy loading instead.
    log.info("Model loading skipped at startup — will be loaded manually.")
    log.info("ContractGuard ready.")


# ─────────────────────────────────────────────────────────────
#  POST /analyze
#  Receives: multipart form — file (PDF/DOCX), role (str)
#  Returns:  { clauses: [{clause_index, clause_text, risk_level, explanation}] }
# ─────────────────────────────────────────────────────────────

class ClauseResult(BaseModel):
    clause_index: int
    clause_text:  str
    risk_level:   str
    explanation:  str


class AnalyzeResponse(BaseModel):
    clauses: List[ClauseResult]


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    role: str        = Form("Licensee"),
):
    """
    Step 1 — Upload a PDF/DOCX, extract clauses, classify each one.

    The frontend sends:
        Content-Type: multipart/form-data
        Fields: file, role
    """
    # ── Ensure model is loaded ──────────────────────────────────
    _load_model()

    # ── Read uploaded file ─────────────────────────────────────
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    log.info("Received file: %s  role: %s", file.filename, role)

    # ── Extract clauses ────────────────────────────────────────
    # Returns List[dict] with keys: clause_id, text, section_header,
    # is_subpart, is_decimal_section, parent_clause_id, party_refs, …
    clause_dicts = extract_clauses_from_upload(file.filename or "upload.pdf", file_bytes)
    log.info("Extracted %d clauses.", len(clause_dicts))

    # ── Classify each clause ───────────────────────────────────
    initial_results = []
    clause_response = []

    for clause in clause_dicts:
        cid   = clause["clause_id"]
        ctext = clause["text"]
        # Pass the section heading as extra context so the classifier
        # knows which part of the contract it's looking at (e.g. "INDEMNIFICATION")
        section = clause.get("section_header", "")
        context_text = f"[Section: {section}]\n{ctext}" if section else ctext

        log.info("Classifying clause %d/%d …", cid, len(clause_dicts))
        risk_level, explanation = classify_clause(context_text, role)

        initial_results.append({
            "clause_index": cid,
            "clause_text":  ctext,
            "role":         role,
            "model_output": f"Risk Level: {risk_level}\nExplanation: {explanation}",
        })

        clause_response.append(ClauseResult(
            clause_index=cid,
            clause_text=ctext,
            risk_level=risk_level,
            explanation=explanation,
        ))

    # ── Store in session for follow-up chat ────────────────────
    _session["initial_results"]      = initial_results
    _session["role"]                  = role
    _session["conversation_history"]  = []   # reset conversation on new upload

    return AnalyzeResponse(clauses=clause_response)


# ─────────────────────────────────────────────────────────────
#  POST /ask
#  Receives: { question: str, role: str, domain: str }
#  Returns:  { answer: str, sources: [str, …] }
# ─────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    role:     str = "Licensee"
    domain:   str = "General"


class AskResponse(BaseModel):
    answer:  str
    sources: List[str]


@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest):
    """
    Step 2 — RAG-powered follow-up question answering.

    Retrieves relevant clauses from the FAISS index,
    builds a grounded prompt, and generates a response with the fine-tuned model.
    Conversation history is maintained across calls within the same session
    (i.e., after the same /analyze upload).
    """
    if not _session["initial_results"]:
        raise HTTPException(
            status_code=400,
            detail="No analysis in session. Please run /analyze first.",
        )

    # ── RAG retrieval ─────────────────────────────────────────
    retrieved = _retrieve(body.question, body.role, top_k=TOP_K)
    log.info(
        "Retrieved %d clauses for question: %s…",
        len(retrieved), body.question[:60],
    )

    # ── Generate response ──────────────────────────────────────
    answer = _generate_chat_response(
        user_question        = body.question,
        role                 = body.role,
        conversation_history = _session["conversation_history"],
        initial_results      = _session["initial_results"],
        retrieved            = retrieved,
    )

    # ── Update conversation history (store clean question, not RAG-augmented) ─
    _session["conversation_history"] = _session["conversation_history"] + [
        {"role": "user",      "content": body.question},
        {"role": "assistant", "content": answer},
    ]

    # ── Return answer + source snippets ───────────────────────
    sources = [r["clause"][:200] for r in retrieved]

    return AskResponse(answer=answer, sources=sources)


# ─────────────────────────────────────────────────────────────
#  GET /health   — sanity check
# ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":      "ok",
        "model_ready": _model_ready,
        "rag_ready":   _rag_ready,
        "index_size":  _faiss_index.ntotal if _faiss_index else 0,
        "adapter_path": ADAPTER_PATH,
    }


# ─────────────────────────────────────────────────────────────
#  Dev entrypoint
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=False)
