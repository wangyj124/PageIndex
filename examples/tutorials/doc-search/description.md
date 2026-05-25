
## Document Search by Description

For documents that don't have metadata, you can use LLM-generated descriptions to help with document selection. This is a lightweight approach that works best with a small number of documents.


### Example Pipeline


#### PageIndex Tree Generation
Upload all documents into PageIndex to get their `doc_id` and tree structure.

#### Description Generation

Generate a description for each document based on its PageIndex tree structure and node summaries.
```python
prompt = f"""
你将获得一份文档的目录树结构。
你的任务是为该文档生成一句话描述，使其容易与其他文档区分开。

文档树结构：{PageIndex_Tree}

直接返回描述，不要输出任何其他内容。
"""
```

#### Search with LLM

Use an LLM to select relevant documents by comparing the user query against the generated descriptions.

Below is a sample prompt for document selection based on their descriptions:

```python
prompt = f""" 
你将获得一个文档列表，其中包含文档 ID、文件名和描述。你的任务是选出可能包含与用户问题相关信息的文档。

问题：{query}

文档列表：[
    {
        "doc_id": "xxx",
        "doc_name": "xxx",
        "doc_description": "xxx"
    }
]

回复格式：
{{
    "thinking": "<你选择这些文档的理由>",
    "answer": <相关 doc_id 组成的 Python 列表>，例如 ['doc_id1', 'doc_id2']。如果没有相关文档，则返回 []。
}}

只返回 JSON 结构，不要输出其他内容。
"""
```

#### Retrieve with PageIndex

Use the PageIndex `doc_id` of the retrieved documents to perform further retrieval via the PageIndex retrieval API.



## 💬 Help & Community
Contact us if you need any advice on conducting document searches for your use case.

- 🤝 [Join our Discord](https://discord.gg/VuXuf29EUj)  
- 📨 [Leave us a message](https://ii2abc2jejf.typeform.com/to/meB40zV0)
