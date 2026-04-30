# ContractGuard — Contract Risk Analysis via Context-Conditioned LLM Adaptation

**Aarav Singh Luthra · Devyansh Choudhary · Saksham Bali · Uday Sodhi**

A system for clause-level legal risk classification and grounded Q&A over uploaded contracts, built on a QLoRA fine-tuned LLaMA 3.1 8B Instruct model with a FAISS-based RAG module.

---

## Overview

ContractGuard takes a PDF contract as input, extracts individual clauses, and classifies each one as **Low** or **High** risk from either a **Licensor** or **Licensee** perspective. Users can then ask follow-up questions about the contract, with answers grounded in a pre-built knowledge base of annotated CUAD clauses.

The system has two workstreams:
- **Dataset Construction** — a custom role-conditioned annotation pipeline over CUAD License Agreements, producing 4,075 labelled clause-role examples
- **Inference Pipeline** — a deployed FastAPI backend with clause extraction, fine-tuned classification, RAG retrieval, and a standalone HTML frontend

---

## Repository Structure

```
├── backend.py                  # FastAPI server — /analyze, /ask, /health endpoints
├── clauseextraction.py         # 4-stage deterministic PDF clause segmentation pipeline
├── NLP_pipeline.ipynb          # Google Colab notebook — runs the full system end-to-end
├── index.html                  # Standalone frontend — role selection, results, chat UI
├── final_test_updated.json     # RAG knowledge base — 500 annotated CUAD clauses
├── qlora_output/               # LoRA adapter weights (downloaded from HuggingFace)
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── legal_rag_index/            # FAISS index (auto-built on first run)
    ├── index.faiss
    └── chunks.pkl
```

---

## Setup and Running

### Requirements

- Google Colab with T4 GPU (recommended)
- HuggingFace account with access to `meta-llama/Meta-Llama-3.1-8B-Instruct`
- ngrok account (free) for browser access

### Steps

**1. Open `NLP_pipeline.ipynb` in Google Colab**

Set runtime to T4 GPU: Runtime → Change runtime type → T4 GPU

**2. Run Cell 2** — installs all dependencies (~5 min)

**3. Run Cell 3** — verifies numpy 1.26.4 and CUDA availability

**4. Run Cell 4** — upload these four files:
- `backend.py`
- `clauseextraction.py`
- `final_test_updated.json`
- `index.html`

**5. Run Cell 5** — HuggingFace login (paste your token)

**6. Run Cell 6** — downloads LoRA adapter from `the-noble1/legalllm`

**7. Run Cell 7** — patches `backend.py` with the adapter path

**8. Run Cell 8 (sub-cell 1)** — initialises the FAISS RAG index directly (~2–3 min on first run, loads from disk after that)

**9. Run Cell 8 (sub-cell 2)** — starts the FastAPI server on port 8000

**10. Run Cell 9** — loads the fine-tuned model manually (bypasses PEFT meta-tensor issue)

**11. Run Cell 10** — verifies model and RAG are ready

**12. Run Cell 11** — creates ngrok tunnel, patches the URL into `index.html`, and downloads it

**13. Open the downloaded `index.html` in your browser**

---

## Usage

1. Select your role — **Licensee** or **Licensor**
2. Upload a PDF contract
3. Click **Analyze Contract** — clauses are extracted and classified (allow ~2 min for a typical contract)
4. Expand any clause card to see the full text and risk explanation
5. Use the chat box at the bottom to ask follow-up questions

---

## Model Details

| Component | Details |
|---|---|
| Base model | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| Fine-tuning method | QLoRA (4-bit NF4, double quantisation) |
| LoRA config | r=8, alpha=16, 4 attention layers (q/k/v/o_proj) |
| Trainable parameters | 6,815,744 (0.0848% of total) |
| Training | 3 epochs, lr=2×10⁻⁴, moderate High-risk oversampling |
| Task | Binary risk classification (Low / High) per clause per role |

---

## Dataset

| Property | Details |
|---|---|
| Source | CUAD License Agreements |
| Annotation model | GPT-OSS 120B via Groq API with structured rubric prompting |
| Total examples | 4,075 clause-role pairs |
| Label schema | Low / High (Medium merged into High due to class imbalance) |
| Roles | Licensor, Licensee (labelled independently per clause) |

---

## Evaluation Results

| Model | Accuracy | Macro F1 | Low F1 | High F1 |
|---|---|---|---|---|
| Zero-shot baseline | 0.644 | 0.572 | 0.748 | 0.395 |
| Fine-tuned (overall) | 0.815 | 0.750 | 0.880 | 0.630 |
| Fine-tuned (Licensee) | 0.794 | 0.770 | 0.840 | 0.700 |
| Fine-tuned (Licensor) | 0.835 | 0.690 | 0.900 | 0.470 |

Test set: 826 clauses (597 Low, 229 High)

---

## RAG Module

- **Knowledge base**: 500 annotated CUAD clauses with per-party risk labels and explanations
- **Embedding model**: `BAAI/bge-base-en-v1.5`
- **Index**: FAISS `IndexFlatIP` (cosine similarity on L2-normalised vectors)
- **Retrieval**: Top-3 clauses per question, role-specific framing applied at retrieval time
- **Enriched text format**: `"Licensor Risk: X. Licensee Risk: Y. Clause: [text]"` embedded per chunk

---

## Known Limitations

- PDF only (no DOCX support)
- Scoped to License Agreements — generalisation to other contract types not evaluated
- High-risk recall is weaker from the Licensor perspective due to class imbalance (only 18.7% of Licensor examples are High risk)
- Fine-tuned model produces structured but less fluent conversational chat responses
- Single-user in-memory session — not suitable for concurrent multi-user deployment

---

## GitHub Repository

[https://github.com/SakshamBali/Advanced_NLP_proj](https://github.com/SakshamBali/Advanced_NLP_proj)
