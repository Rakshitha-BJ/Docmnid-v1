# TODO: Improve RAG Metrics

## Tasks
- [x] Upgrade embedding model in embedder.py to 'all-MiniLM-L12-v2'
- [x] Add embedding caching in embedder.py
- [x] Increase top_k to 15 in retriever.py
- [x] Reduce maxOutputTokens to 256 in llm_client.py
- [x] Add response caching in llm_client.py
- [x] Refine prompt in prompt_templates.py for better relevance and faithfulness
- [ ] Run eval.py to test improvements
- [ ] Iterate based on results
