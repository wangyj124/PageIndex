## Tree Search Examples
This tutorial provides a basic example of how to perform retrieval using the PageIndex tree.

### Basic LLM Tree Search Example
A simple strategy is to use an LLM agent to conduct tree search. Here is a basic tree search prompt.

```python
prompt = f"""
你将获得一个查询，以及一份文档的树结构。
你的任务是找出所有可能包含答案的节点。

查询：{query}

文档树结构：{PageIndex_Tree}

请按以下 JSON 格式回复：
{{
  "thinking": <你对哪些节点相关的判断理由>,
  "node_list": [node_id1, node_id2, ...]
}}
"""
```
<callout>
In our dashboard and retrieval API, we use a combination of LLM tree search and value function-based Monte Carlo Tree Search ([MCTS](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)). More details will be released soon.
</callout>

### Integrating User Preference or Expert Knowledge
Unlike vector-based RAG where integrating expert knowledge or user preference requires fine-tuning the embedding model, in PageIndex, you can incorporate user preferences or expert knowledge by simply adding knowledge to the LLM tree search prompt. Here is an example pipeline.


#### 1. Preference Retrieval

When a query is received, the system selects the most relevant user preference or expert knowledge snippets from a database or a set of domain-specific rules. This can be done using keyword matching, semantic similarity, or LLM-based relevance search.

#### 2. Tree Search with Preference
Integrating preference into the tree search prompt.

**Enhanced Tree Search with Expert Preference Example**

```python
prompt = f"""
你将获得一个问题，以及一份文档的树结构。
你的任务是找出所有可能包含答案的节点。

问题：{query}

文档树结构：{PageIndex_Tree}

相关章节的专家知识：{Preference}

请按以下 JSON 格式回复：
{{
  "thinking": <你对哪些节点相关的判断理由>,
  "node_list": [node_id1, node_id2, ...]
}}
"""
```

**Example Expert Preference**
> If the query mentions EBITDA adjustments, prioritize Item 7 (MD&A) and footnotes in Item 8 (Financial Statements) in 10-K reports.



By integrating user or expert preferences, node search becomes more targeted and effective, leveraging both the document structure and domain-specific insights.

## 💬 Help & Community
Contact us if you need any advice on conducting document searches for your use case.

- 🤝 [Join our Discord](https://discord.gg/VuXuf29EUj)  
- 📨 [Leave us a message](https://ii2abc2jejf.typeform.com/to/tK3AXl8T)
