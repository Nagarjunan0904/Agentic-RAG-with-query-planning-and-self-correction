from typing import TypedDict, Literal, Optional, List


class AgentState(TypedDict):
    query:          str
    strategy:       Literal['SEMANTIC', 'KEYWORD', 'STRUCTURED']
    retrieved_docs: List[dict]
    reranked_docs:  List[dict]
    answer:         str
    confidence:     float
    retry_count:    int
    retry_reason:   Optional[str]
    latency_ms:     dict
    query_vector:   Optional[List[float]]   # set by self_evaluator (HyDE) for retry
