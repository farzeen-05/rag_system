"""
WHY MONITORING IN KUBERNETES?
Pods die and restart. Requests fail. Models slow down.
Without metrics, you're flying blind.

We use Prometheus format — the standard for Kubernetes monitoring.
"""

from prometheus_client import Counter, Histogram, Gauge
import time

# Counters: only go up (total requests, total errors)
RAG_QUERIES_TOTAL = Counter(
    "rag_queries_total",
    "Total RAG queries",
    ["status"]  # Labels: success/error
)

# Histograms: track distributions (response time percentiles)
RAG_QUERY_LATENCY = Histogram(
    "rag_query_latency_seconds",
    "RAG query latency in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

# Gauges: can go up and down (current vector count)
VECTOR_STORE_SIZE = Gauge(
    "vector_store_total_chunks",
    "Total chunks in vector store"
)

RETRIEVAL_SCORE_AVG = Gauge(
    "rag_avg_retrieval_score",
    "Average similarity score of retrieved chunks"
)


def track_query(func):
    """Decorator to automatically track query metrics."""
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            RAG_QUERIES_TOTAL.labels(status="success").inc()
            
            if hasattr(result, 'retrieval_scores') and result.retrieval_scores:
                avg_score = sum(result.retrieval_scores) / len(result.retrieval_scores)
                RETRIEVAL_SCORE_AVG.set(avg_score)
            
            return result
        except Exception as e:
            RAG_QUERIES_TOTAL.labels(status="error").inc()
            raise
        finally:
            RAG_QUERY_LATENCY.observe(time.time() - start)
    return wrapper