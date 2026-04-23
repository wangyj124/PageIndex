# PageIndex-MJ

面向长文档的通用层级索引工具，支持 PDF/Markdown 建树、混合文档树构建、检索辅助与结构化信息提取。

## 项目概述

本项目用于把长文档转换为适合大模型理解和处理的树形结构，适用于合同、制度、报告、技术文档等层级明确但篇幅较长的资料。

仓库当前保留了上游 PageIndex 的文档建树与检索思路，并在此基础上补充了更贴近业务落地的增强能力，包括：

- 标准 PDF 建树
- Markdown 建树
- PDF 的 Markdown + JSON 混合建树
- 基于 `workspace` 的索引缓存与复用
- 合同字段抽取
- FastAPI 服务封装
- 白盒多 Agent 合同抽取示例

如果你第一次接触这个仓库，可以把它理解为一个“长文档结构化索引与抽取工具箱”：先建树，再复用索引做检索或抽取。

## 开源来源与致谢

本项目基于以下开源项目进行二次开发与场景增强：

1. [`VectifyAI/PageIndex`](https://github.com/VectifyAI/PageIndex)
   项目的主体能力来源于上游 PageIndex，包括文档树构建、Markdown 建树、检索接口和基础客户端能力。
2. `opendataloader-pdf`
   当前仓库的混合建树链路依赖该开源包先将 PDF 转为 Markdown 和 JSON，再结合两类结果重建更稳定的文档树。

本仓库不是对上游的简单镜像，而是在上游基础上面向中文长文档处理和合同抽取场景做了增强、封装和示例补充。README 中关于“本项目能力”的描述，默认均指当前仓库的二次开发结果。

## 本项目的核心改造

相较于上游开源仓库，当前版本的核心变化主要集中在以下几个方向：

### 1. 新增 PDF 混合建树链路

在标准 PDF 建树之外，仓库新增了基于 Markdown + JSON 的混合建树流程。其核心思路是：

- 先用 `opendataloader-pdf` 将 PDF 转换为 Markdown 和 JSON
- 利用 Markdown 标题结构与 JSON 页级内容进行对齐
- 对层级进行重建与修正
- 生成更适合后续抽取和追踪的树结构

这条链路已经接入：

- CLI 入口 `python -m pageindex.cli --pdf_path ... --md-hybrid`
- `PageIndexClient.index(..., strategy="hybrid")`
- 合同抽取服务流程

### 2. 引入基于 workspace 的缓存与稳定标识

当前仓库增加了文档缓存和稳定身份标识机制：

- 基于文件内容计算 `source_sha256`
- 生成稳定的 `doc_id`
- 生成与树结构绑定的 `tree_id`
- 使用 `workspace` 持久化缓存文档索引结果

这样可以避免重复建树，并让后续抽取、调试和白盒流程复用同一份文档索引结果。

### 3. 增加合同字段抽取与 schema 校验

在上游“建树/检索”能力之上，当前仓库新增了合同字段抽取能力：

- 使用 schema 驱动字段抽取
- 支持并发字段提取
- 统一返回 `status`、`value`、`evidence`、`pages`、`confidence` 等字段
- 对抽取结果执行 schema 一致性校验，避免结果字段缺失或漂移

这部分能力适合作为“文档建树之后的结构化抽取层”继续扩展。

### 4. 增加白盒多 Agent 示例与服务封装

仓库在示例和服务层做了进一步封装：

- 增加 `examples/contract_extraction_demo.py`，演示合同抽取最短路径
- 增加 `examples/contract_extraction_whitebox_demo.py`，演示白盒多 Agent 抽取流程
- 增加 `service.py`，统一封装建树、抽取和结果落盘逻辑
- 增加 `api.py`，提供异步任务化的 FastAPI 接口

这使得当前仓库从上游偏研究/演示风格，演进为更适合业务集成和二次开发的项目形态。

### 5. 补充测试覆盖

仓库当前已经围绕以下能力补充测试：

- CLI 入口
- 客户端缓存与混合建树
- Markdown 处理
- 混合建树流程
- 合同抽取服务
- API 接口
- workspace 持久化

## 功能特性

- 支持 PDF 标准建树
- 支持 Markdown 建树
- 支持 PDF 混合建树
- 支持文档结构缓存和复用
- 支持按页追踪的结构化输出
- 支持合同字段抽取
- 支持白盒多 Agent 示例
- 支持 FastAPI 服务化接入

## 项目结构

下面是阅读和使用这个仓库时最值得优先关注的入口：

- `pageindex/`
  核心包，包含客户端、建树、混合建树、抽取、缓存和检索逻辑。
- `api.py`
  FastAPI 服务入口，负责文件上传、后台任务调度和任务状态查询。
- `service.py`
  对外的服务封装层，串联“建树 -> 抽取 -> 落盘”流程。
- `examples/`
  示例脚本目录，包括合同抽取和白盒多 Agent 示例。
- `sample_data/`
  示例 schema 和样例数据。
- `tests/`
  测试目录，可用于快速理解仓库的真实支持能力和调用方式。
- `run_pageindex.py`
  与 CLI 兼容的入口脚本，便于保留旧调用方式。

## 安装部署

### 1. 环境要求

- Python 3.10 及以上
- 建议使用虚拟环境
- 需要可用的大模型 API Key

### 2. 安装依赖

基础依赖安装：

```bash
pip install -r requirements.txt
```

如果需要启动 API 服务，建议额外安装：

```bash
pip install uvicorn
```

如果需要运行白盒多 Agent 或 Agent 检索示例，额外安装：

```bash
pip install openai-agents
```

### 3. 配置 `.env`

在仓库根目录创建 `.env` 文件，例如：

```env
OPENAI_API_KEY=your_openai_api_key
```

当前仓库默认通过 LiteLLM 适配模型调用，最常用的是配置 `OPENAI_API_KEY`。如果你的运行环境已通过系统环境变量注入，也可以不创建 `.env` 文件。

## 使用方法

### 1. 标准 PDF 建树

适用于直接对 PDF 生成树结构：

```bash
python -m pageindex.cli --pdf_path path/to/your/document.pdf
```

默认输出到：

```text
artifacts/results/<文档名>_structure.json
```

兼容旧入口的写法：

```bash
python run_pageindex.py --pdf_path path/to/your/document.pdf
```

### 2. Markdown 建树

适用于原始 Markdown 文档：

```bash
python -m pageindex.cli --md_path path/to/your/document.md
```

说明：

- 该模式基于 Markdown 标题层级建树
- 更适合原生 Markdown 文档
- 如果 Markdown 是由 PDF 粗糙转换而来，通常更推荐使用混合建树

### 3. PDF 混合建树

适用于需要更稳定层级恢复和页级对齐的 PDF：

```bash
python -m pageindex.cli --pdf_path path/to/your/document.pdf --md-hybrid
```

这条命令会：

- 先将 PDF 转成 Markdown 和 JSON
- 再基于两种中间结果生成最终树结构

如果你已经手头有一对 Markdown + JSON 文件，也可以直接走 Markdown 混合建树：

```bash
python -m pageindex.cli --md_path path/to/your/document.md --json_path path/to/your/document.json --md-hybrid
```

### 4. 常用可选参数

```bash
python -m pageindex.cli --help
```

常见参数包括：

- `--output-dir`：指定输出目录
- `--model`：覆盖默认模型
- `--if-add-node-summary`：是否生成节点摘要
- `--if-add-doc-description`：是否生成文档摘要
- `--if-add-node-text`：是否保留节点正文
- `--if-add-node-id`：是否输出节点 ID

### 5. 合同抽取 Demo

运行最短路径合同抽取示例：

```bash
python examples/contract_extraction_demo.py
```

说明：

- 示例脚本会读取仓库 `pdf/` 目录下的第一个 PDF 文件
- 使用 `sample_data/schemas/contract_fields_xt_full.json` 作为字段 schema
- 默认在 `artifacts/contract_workspace/` 下复用或生成索引缓存

### 6. 白盒多 Agent 合同抽取 Demo

运行带中间过程展示的白盒示例：

```bash
python examples/contract_extraction_whitebox_demo.py
```

说明：

- 该示例依赖 `openai-agents`
- 默认读取 `pdf/` 目录下的第一个 PDF
- 默认在 `artifacts/whitebox_contract_workspace/` 下读写缓存

### 7. Python 中使用客户端

最常见的编程式调用方式如下：

```python
from pageindex import PageIndexClient

client = PageIndexClient(workspace="artifacts/workspace")
doc_id = client.index("path/to/document.pdf", strategy="hybrid")
tree_id = client.get_tree_id(doc_id)

print(doc_id, tree_id)
```

如果只需要标准建树，可以把 `strategy` 改为默认值，或直接省略。

## API 使用

### 1. 启动服务

```bash
uvicorn api:app --reload
```

默认接口文档地址：

- `http://127.0.0.1:8000/docs`

当前 API 采用“两阶段异步任务”模式：

1. 先上传 PDF，异步构建文档树，拿到 `doc_id`
2. 再基于 `doc_id` 和动态 `schema_def` 发起结构化抽取

这样可以复用共享 `workspace` 中已构建好的文档索引，避免重复上传和重复建树。

另外，当前接口统一采用标准响应结构：

```json
{
  "code": 200,
  "message": "响应提示信息",
  "data": {}
}
```

说明：

- 成功响应统一为 HTTP `200`
- 业务数据统一位于 `data` 字段
- 常见错误也会保持同样的包裹结构，例如 `400/404/422/429/500`
- 当前服务一次只处理一个活跃任务；如果已有 `pending` 或 `processing` 任务，新请求会收到 `429`

### 2. 第一步：上传 PDF 并异步建树

接口：

- `POST /api/v1/upload_and_build`

请求方式：

- `multipart/form-data`
- 字段名：`file`
- 仅支持 `.pdf`

示例命令：

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/upload_and_build" ^
  -H "accept: application/json" ^
  -F "file=@pdf/your_document.pdf;type=application/pdf"
```

返回示例：

```json
{
  "code": 200,
  "message": "文件已接收，建树任务已提交到后台。",
  "data": {
    "task_id": "b7c2f5c5f1f6496f9858a2dbe4b9b4e0"
  }
}
```

建树任务完成后，可从任务状态返回的 `data` 中读取：

- `doc_id`
- `tree_id`
- `source_file`
- `workspace_dir`
- `output_dir`

### 3. 第二步：基于 `doc_id` 发起动态 schema 抽取

接口：

- `POST /api/v1/extract`

请求体字段：

- `doc_id`：上一步建树完成后得到的文档 ID
- `schema_def`：动态抽取 schema
- `require_evidence`：是否输出带证据溯源的结构化结果，默认 `false`

`schema_def` 当前支持两种风格：

- 现有字段列表风格：`{"fields": [{"name": "...", "description": "..."}]}`
- 标准 JSON Schema 顶层 `properties` 风格

如果 `require_evidence=true`，建议使用标准 JSON Schema 顶层 `properties` 风格。服务端会自动把结果整理为包含 `value/page_number/section_title/original_quote` 的结构。

示例命令：

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/extract" ^
  -H "accept: application/json" ^
  -H "Content-Type: application/json" ^
  -d "{\"doc_id\":\"doc-build-demo\",\"schema_def\":{\"type\":\"object\",\"properties\":{\"party_a\":{\"type\":\"string\",\"description\":\"甲方\"},\"party_b\":{\"type\":\"string\",\"description\":\"乙方\"}}},\"require_evidence\":true}"
```

返回示例：

```json
{
  "code": 200,
  "message": "动态 Schema 抽取任务已提交到后台。",
  "data": {
    "task_id": "58b40f91b2fd4c79a55b8879947b0d23"
  }
}
```

### 4. 查询任务状态与进度

接口：

- `GET /api/v1/task/{task_id}`

示例命令：

```bash
curl "http://127.0.0.1:8000/api/v1/task/b7c2f5c5f1f6496f9858a2dbe4b9b4e0"
```

查询成功时，返回结构同样为：

```json
{
  "code": 200,
  "message": "任务状态查询成功",
  "data": {
    "task_id": "b7c2f5c5f1f6496f9858a2dbe4b9b4e0",
    "task_type": "extraction",
    "status": "processing"
  }
}
```

其中 `data.status` 的任务基础状态通常包含：

- `pending`
- `processing`
- `completed`
- `failed`

`data` 中的任务元信息通常包含：

- `task_id`
- `task_type`
- `created_at`
- `workspace_dir`
- `output_dir`
- `error`

其中：

- 建树任务的 `task_type` 为 `build_tree`
- 抽取任务的 `task_type` 为 `extraction`

当抽取任务处于 `processing` 状态时，`data` 中还会返回：

```json
{
  "code": 200,
  "message": "任务状态查询成功",
  "data": {
    "task_id": "processing-task",
    "task_type": "extraction",
    "status": "processing",
    "progress": {
      "current": 5,
      "total": 20,
      "percent": 25.0
    }
  }
}
```

任务完成后，`data` 中常见返回字段包括：

- 输出文件路径
- `doc_id`
- `tree_id`
- `result_status`
- `require_evidence`
- `extracted_count`
- `total_count`
- 错误信息（如果失败）

### 5. 常见错误响应

当前 API 已统一错误返回格式，README 中最容易用到的几类如下。

参数错误或文件类型错误：

```json
{
  "code": 400,
  "message": "仅支持上传 .pdf 文件",
  "data": null
}
```

任务不存在：

```json
{
  "code": 404,
  "message": "任务不存在",
  "data": null
}
```

请求参数校验失败：

```json
{
  "code": 422,
  "message": "请求参数校验失败",
  "data": [
    {
      "type": "missing",
      "loc": ["body", "doc_id"],
      "msg": "Field required"
    }
  ]
}
```

系统忙碌时：

```json
{
  "code": 429,
  "message": "系统当前正忙，一次只能处理一个任务，请稍后再试",
  "data": null
}
```

## 示例输出说明

### 1. 建树输出

CLI 默认输出一个 `*_structure.json` 文件，常见字段包括：

- `doc_name`
- `doc_description`
- `structure`

其中 `structure` 中的节点通常包含：

- `title`
- `node_id`
- `start_page` / `end_page`
- `summary`
- `text`
- `nodes`

标准 PDF 建树与混合建树的字段细节可能略有差异，但整体都围绕“层级结构 + 页码范围 + 可选摘要”展开。

### 2. 结构化抽取输出

抽取任务会在对应任务输出目录下生成一个 `*_extraction.json` 文件。

- 当通过 API 调用时，结果文件名通常为 `<doc_id>_extraction.json`
- 当通过示例脚本调用时，结果文件名可能按源文件名命名

结果文件常见字段包括：

- `status`
- `doc_id`
- `tree_id`
- `require_evidence`
- `extraction_result`

当 `require_evidence=false` 时，`extraction_result` 中每个字段通常包含：

- `status`
- `value`
- `evidence`
- `pages`
- `confidence`
- `reason`

当 `require_evidence=true` 时，`extraction_result` 中每个字段会被整理为更适合前端或审阅链路消费的结构，通常包含：

- `value`
- `page_number`
- `section_title`
- `original_quote`
- `status`
- `confidence`
- `reason`

这类结构适合继续对接前端、审批流或下游业务系统。

需要注意：

- API 自身返回的是标准响应包裹结构，任务详情在 `data` 里
- 真正的抽取结果文件仍然保存在任务输出目录中，而不是直接内嵌在提交接口响应里

## 适用场景与限制

### 适用场景

- 中文合同与协议抽取
- 长篇报告和制度文档建树
- 需要页码追踪的文档分析
- 需要先建索引再做结构化抽取的业务流程

### 当前限制

- API 当前为轻量级内存任务状态管理，适合本地开发和小规模使用
- 当前服务端限制同时只能存在 1 个活跃任务
- 示例脚本默认依赖仓库内 `pdf/` 目录中的样例文件
- 混合建树依赖 `opendataloader-pdf`
- 白盒多 Agent 示例依赖额外的 `openai-agents`
- 最终效果仍然受文档质量、标题层级、OCR 结果和模型能力影响

### 后续扩展方向

- 接入更完善的任务队列与持久化状态存储
- 扩展更多文档抽取 schema
- 增加更完整的检索问答示例
- 增强复杂扫描件、非规范目录和多语言文档的适配能力

## License

本仓库使用 [MIT License](LICENSE)。

如果你基于本项目继续开发，建议同时保留对上游 `VectifyAI/PageIndex` 和当前二次开发工作的说明。
