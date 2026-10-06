from prometheus_client import Counter, Histogram

rag_requests_total = Counter(
    "rag_requests_total",
    "Total number of RAG requests",
)

rag_errors_total = Counter(
    "rag_errors_total",
    "Total number of RAG errors",
)

rag_retrieval_latency_seconds = Histogram(
    "rag_retrieval_latency_seconds",
    "Time spent retrieving documents",
)

rag_llm_latency_seconds = Histogram(
    "rag_llm_latency_seconds",
    "Time spent generating the LLM response",
)