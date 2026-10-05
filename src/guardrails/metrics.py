from prometheus_client import Counter, Histogram


guardrail_decisions_total = Counter(
    "guardrail_decisions_total",
    "Total number of guardrail decisions",
    ["guardrail", "decision"],
)

guardrail_latency_seconds = Histogram(
    "guardrail_latency_seconds",
    "Guardrail execution latency",
    ["guardrail"],
)