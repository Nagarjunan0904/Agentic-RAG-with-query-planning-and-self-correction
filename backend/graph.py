from langgraph.graph import StateGraph, END
from backend.agents import (query_planner, vector_retriever,
    bm25_retriever, sql_agent, reranker, generator, self_evaluator)
from backend.agents.state import AgentState


def route_query(state: AgentState) -> str:
    return state['strategy']


def should_retry(state: AgentState) -> str:
    if state['confidence'] < 0.65 and state['retry_count'] < 2:
        return 'retry'
    return 'end'


graph = StateGraph(AgentState)
graph.add_node('planner',   query_planner.run)
graph.add_node('vector',    vector_retriever.run)
graph.add_node('bm25',      bm25_retriever.run)
graph.add_node('sql',       sql_agent.run)
graph.add_node('reranker',  reranker.run)
graph.add_node('generator', generator.run)
graph.add_node('evaluator', self_evaluator.run)

graph.set_entry_point('planner')
graph.add_conditional_edges('planner', route_query,
    {'SEMANTIC': 'vector', 'KEYWORD': 'bm25', 'STRUCTURED': 'sql'})
graph.add_edge('vector',    'reranker')
graph.add_edge('bm25',      'reranker')
graph.add_edge('sql',       'reranker')
graph.add_edge('reranker',  'generator')
graph.add_edge('generator', 'evaluator')
graph.add_conditional_edges('evaluator', should_retry,
    {'retry': 'reranker', 'end': END})

app = graph.compile()
