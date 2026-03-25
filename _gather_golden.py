"""Batch-extract chunk IDs for golden test set queries."""
import sys, os, json, logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.WARNING)

from src.retrieval.pipeline import retrieve
from src.retrieval.models import RetrievalConfig

config = RetrievalConfig(top_k_broad=20, top_k_final=10, reranker_type="none")

queries = [
    # exact_lookup
    "FVTR_MECH_01",
    "FVTR_FUNC_13",
    "FVTR_FKT_14",
    "FVTR_HVT_01",
    "HW-IRS_DIM_VI_275",
    "HW-IRS_DIM_VI_526",
    "FVTR_OPT_01",
    "FVTR_REL_01",
    # scoped_semantic
    "DIM-V thermal test results",
    "PAM analogue output accuracy",
    "HW-IRS_DIM_VI_382 threshold specification",
    "DIM-V wetting current measurements",
    "PAM high voltage isolation results",
    "HwIRS DIM input channel requirements",
    "DIM-V FVTR power consumption",
    "PAM PWM output frequency range",
    "FVTR_MECH_09 dimension compliance",
    "HwIRS galvanic isolation requirements",
    # unconstrained_semantic
    "What is the thermal protection approach?",
    "How is latent failure detection implemented?",
    "What are the power consumption limits?",
    "Describe the high voltage test procedure",
    "What safety standards are referenced?",
    "How does conformal coating protect the board?",
    "What is the expected service life and maintainability?",
    "How are analogue inputs measured and calibrated?",
    "What are the environmental operating conditions?",
    # acronym_heavy
    "DIM LFD implementation",
    "PAM PWM ESD protection",
    "MTBF MTTR reliability",
    "TOP EEPROM storage requirements",
    "PSU DC voltage derating",
    "HW IRS CAN interface",
    # structured_content
    "DIM-V test equipment calibration table",
    "PAM analogue current output measurement table",
    "DIM-V high voltage test circuit definitions",
    "PAM board variant configuration",
    "DIM-V input threshold specification table",
    "PAM PWM output oscilloscope captures",
    # cross_reference
    "Which tests verify HW-IRS_DIM_VI_381?",
    "What requirements does FVTR_FKT_05 trace to?",
    "HWADD:TOP:0012 architecture definition",
    "Which tests cover HW-IRS_PAM_93?",
    "FVTR_FUNC_06 input circuit requirements",
    "What does FVTSR_PAM_0002 trace to?",
]

results = {}
for q in queries:
    r = retrieve(q, config)
    chunks = []
    for sc in r.scored_chunks[:5]:
        chunks.append({
            "chunk_id": sc.chunk_id,
            "doc_id": sc.doc_id,
            "chunk_type": sc.chunk_type,
            "score": round(sc.score, 4),
            "heading": sc.metadata.get("heading", ""),
            "module": sc.metadata.get("module_name", ""),
            "test_case_id": sc.metadata.get("test_case_id", ""),
            "text_preview": sc.text[:150].replace("\n", " "),
        })
    results[q] = {
        "strategy": r.strategy,
        "n_chunks": len(r.scored_chunks),
        "top_chunks": chunks,
    }

out_path = "_golden_candidates.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"Wrote {len(results)} queries to {out_path}")
