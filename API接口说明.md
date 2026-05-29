# API 接口说明

本文档基于当前运行中的服务与最新代码整理，重点说明接口调用方式、返回规则、任务状态流转与常见错误。

## 1. 基本信息

- 服务地址：`http://10.67.75.27:8000`
- Swagger 文档：`http://10.67.75.27:8000/docs`
- OpenAPI 定义：`http://10.67.75.27:8000/openapi.json`
- 接口版本前缀：`/api/v1`

当前 API 采用异步任务模式，主要分两步：

1. 上传文档并发起建树任务
2. 基于 `doc_id` 发起结构化抽取任务

## 2. 通用规则

### 2.1 响应结构

所有接口统一返回如下结构：

```json
{
  "code": 200,
  "message": "响应提示信息",
  "data": {}
}
```

字段说明：

- `code`：业务状态码
- `message`：提示信息
- `data`：业务数据，可能为对象、数组或 `null`

### 2.2 成功与失败判定

- 成功提交任务或成功查询任务时，HTTP 状态码通常为 `200`
- 参数错误、任务不存在、容量限制等情况会返回对应 HTTP 状态码，同时响应体仍保持统一结构

### 2.3 当前容量限制

当前服务端一次只允许存在 1 个活跃任务。

活跃任务定义：

- `pending`
- `processing`

如果已有活跃任务，新请求会返回 `429`。

### 2.4 支持的上传文件类型

建树上传接口当前支持：

- `.pdf`
- `.doc`
- `.docx`

如果上传 `.doc/.docx`，服务端会在后台先自动转换为 PDF，再基于转换后的 PDF 建树。Linux 部署时，Word 转 PDF 默认调用远程服务 `http://10.8.2.63:8000/convert`，可通过环境变量 `PAGEINDEX_WORD_TO_PDF_CONVERT_URL` 覆盖。

## 3. 接口总览

### 3.1 上传并建树

- 方法：`POST`
- 路径：`/api/v1/upload_and_build`
- Content-Type：`multipart/form-data`

### 3.2 发起抽取

- 方法：`POST`
- 路径：`/api/v1/extract`
- Content-Type：`application/json`

### 3.3 查询任务状态

- 方法：`GET`
- 路径：`/api/v1/task/{task_id}`

## 4. 上传并建树

### 4.1 接口说明

上传 PDF 或 Word 文档，并异步发起建树任务。

服务端处理逻辑：

1. 接收文件并写入任务目录
2. 如果是 `.doc/.docx`，先转为 PDF
3. 基于 PDF 构建文档树
4. 将结果缓存到共享 `workspace`

Linux 部署说明：

- 默认转换接口：`http://10.8.2.63:8000/convert`
- 覆盖转换接口：设置 `PAGEINDEX_WORD_TO_PDF_CONVERT_URL`
- 覆盖请求超时：设置 `PAGEINDEX_WORD_TO_PDF_CONVERT_TIMEOUT`，单位秒，默认 `120`

向量兜底检索说明：

- `pageindex/config.yaml` 中的 `embedding_batch_size` 控制单次 embedding 请求最多文本条数
- `embedding_request_token_budget` 控制单次 embedding 请求的估算总 token，默认 `8192`
- 大文档会按上述参数分批构建向量索引，避免把全部 chunks 一次性提交给 embedding 服务

### 4.2 请求参数

表单字段：

- `file`：必填，上传文件本体

### 4.3 curl 示例

上传 PDF：

```bash
curl -X POST "http://10.67.75.27:8000/api/v1/upload_and_build" ^
  -H "accept: application/json" ^
  -F "file=@pdf/your_document.pdf;type=application/pdf"
```

上传 DOCX：

```bash
curl -X POST "http://10.67.75.27:8000/api/v1/upload_and_build" ^
  -H "accept: application/json" ^
  -F "file=@docs/your_document.docx;type=application/vnd.openxmlformats-officedocument.wordprocessingml.document"
```

### 4.4 成功返回示例

```json
{
  "code": 200,
  "message": "文件已接收，建树任务已提交到后台。",
  "data": {
    "task_id": "b7c2f5c5f1f6496f9858a2dbe4b9b4e0"
  }
}
```

### 4.5 任务完成后可获得的关键字段

通过任务查询接口可以拿到：

- `doc_id`
- `tree_id`
- `source_file`
- `workspace_dir`
- `output_dir`
- `result_status`

说明：

- 如果输入是 PDF，`source_file` 通常就是原始 PDF 路径
- 如果输入是 `.doc/.docx`，`source_file` 通常会变成转换后的 PDF 路径

## 5. 发起结构化抽取

### 5.1 接口说明

基于已有 `doc_id` 和动态 `schema_def` 发起结构化抽取任务。

前提条件：

- 必须先完成建树
- 必须拿到有效的 `doc_id`

### 5.2 请求体

```json
{
  "doc_id": "doc-build-demo",
  "schema_def": {
    "type": "object",
    "properties": {
      "party_a": {
        "type": "string",
        "description": "甲方"
      },
      "party_b": {
        "type": "string",
        "description": "乙方"
      }
    }
  },
  "require_evidence": true,
  "long_context_mode": true
}
```

字段说明：

- `doc_id`：必填，建树成功后返回的文档 ID
- `schema_def`：必填，动态抽取 schema
- `require_evidence`：可选，默认 `false`
- `long_context_mode`：可选布尔值，默认 `false`；为 `true` 时直接使用完整分页原文抽取

### 5.3 schema_def 支持格式

当前支持两种风格：

1. 字段列表风格

```json
{
  "fields": [
    {
      "name": "party_a",
      "description": "甲方"
    }
  ]
}
```

2. 标准 JSON Schema 顶层 `properties` 风格

```json
{
  "type": "object",
  "properties": {
    "party_a": {
      "type": "string",
      "description": "甲方"
    }
  }
}
```

建议：

- 如果需要更规范的接口对接，优先使用标准 JSON Schema 风格
- 如果 `require_evidence=true`，建议使用标准 JSON Schema 顶层 `properties` 风格

### 5.4 require_evidence 规则

当 `require_evidence=false` 时，抽取结果更接近内部字段抽取结构，通常包含：

- `status`
- `value`：命中字段所在的完整合同条款原文；如果多个条款共同支持结果，按条款逐条返回
- `evidence`：支撑判断的核心原文片段，可使用省略号压缩上下文
- `pages`
- `confidence`
- `reason`

当 `require_evidence=true` 时，服务端会对结果重新整理，字段通常包含：

- `value`
- `page_number`
- `section_title`
- `original_quote`：核心原文片段，可使用省略号压缩上下文，不返回完整条款全文
- `status`
- `confidence`
- `reason`

当 `long_context_mode=true` 时，服务端仅从建树阶段生成的独立分页原文产物读取全文，并绕过树摘要和检索链路；缺少该产物的历史文档需要重新上传并完成建树。若同时启用 `require_evidence`，`section_title` 固定返回空字符串。

长上下文调用使用服务端 `pageindex/config.yaml` 中的 `long_context_model`，部署时应将其配置为可容纳完整文档上下文的模型。

### 5.5 curl 示例

```bash
curl -X POST "http://10.67.75.27:8000/api/v1/extract" ^
  -H "accept: application/json" ^
  -H "Content-Type: application/json" ^
  -d "{\"doc_id\":\"doc-build-demo\",\"schema_def\":{\"type\":\"object\",\"properties\":{\"party_a\":{\"type\":\"string\",\"description\":\"甲方\"},\"party_b\":{\"type\":\"string\",\"description\":\"乙方\"}}},\"require_evidence\":true,\"long_context_mode\":true}"
```

### 5.6 成功返回示例

```json
{
  "code": 200,
  "message": "动态 Schema 抽取任务已提交到后台。",
  "data": {
    "task_id": "58b40f91b2fd4c79a55b8879947b0d23"
  }
}
```

## 6. 查询任务状态

### 6.1 接口说明

根据 `task_id` 查询任务状态、进度和产物路径。

### 6.2 请求示例

```bash
curl "http://10.67.75.27:8000/api/v1/task/b7c2f5c5f1f6496f9858a2dbe4b9b4e0"
```

### 6.3 成功返回示例

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

### 6.4 任务状态枚举

- `pending`：任务已接收，等待执行
- `processing`：任务正在执行
- `completed`：任务已完成
- `failed`：任务执行失败

### 6.5 task_type 枚举

- `build_tree`：建树任务
- `extraction`：抽取任务

### 6.6 processing 状态下的进度结构

当抽取任务处于 `processing` 状态时，`data.progress` 会返回进度信息：

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

字段说明：

- `current`：已完成字段数
- `total`：总字段数
- `percent`：完成百分比

### 6.7 completed 状态下常见字段

建树任务完成后，`data` 中常见字段包括：

- `task_id`
- `task_type`
- `status`
- `doc_id`
- `tree_id`
- `source_file`
- `workspace_dir`
- `output_dir`
- `completed_at`
- `result_status`

抽取任务完成后，`data` 中常见字段包括：

- `task_id`
- `task_type`
- `status`
- `doc_id`
- `output_path`
- `workspace_dir`
- `output_dir`
- `result_status`
- `require_evidence`
- `long_context_mode`
- `extracted_count`
- `total_count`
- `completed_at`

## 7. 结果文件规则

### 7.1 建树结果

建树结果主要以缓存和文档树形式保存在共享 `workspace` 与任务输出目录中。

如果输入是 Word 文档，服务端会在任务输出目录中记录转换日志，并使用转换后的 PDF 完成索引。

### 7.2 抽取结果

抽取任务完成后，会生成结果文件：

- 文件名通常为：`<doc_id>_extraction.json`

结果文件常见顶层字段：

- `status`
- `doc_id`
- `tree_id`
- `require_evidence`
- `long_context_mode`
- `extraction_result`

注意：

- 提交接口本身只返回 `task_id`
- 真正的抽取结果需要通过任务查询拿到 `output_path`，或直接读取结果文件

## 8. 常见错误响应

### 8.1 文件类型不支持

```json
{
  "code": 400,
  "message": "仅支持上传 .pdf、.doc、.docx 文件",
  "data": null
}
```

### 8.2 任务不存在

```json
{
  "code": 404,
  "message": "任务不存在",
  "data": null
}
```

### 8.3 参数校验失败

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

### 8.4 系统忙碌

```json
{
  "code": 429,
  "message": "系统当前正忙，一次只能处理一个任务，请稍后再试",
  "data": null
}
```

### 8.5 服务内部错误

```json
{
  "code": 500,
  "message": "服务器内部错误: ...",
  "data": null
}
```

## 9. 推荐调用顺序

推荐接入流程如下：

1. 调用 `POST /api/v1/upload_and_build`
2. 从返回的 `data.task_id` 轮询 `GET /api/v1/task/{task_id}`
3. 当建树任务状态变成 `completed` 后，读取 `data.doc_id`
4. 调用 `POST /api/v1/extract`
5. 从返回的 `data.task_id` 轮询 `GET /api/v1/task/{task_id}`
6. 当抽取任务状态变成 `completed` 后，读取 `data.output_path` 与结果文件

## 10. 集成注意事项

- 接口当前无鉴权字段，内网使用时建议通过网关或反向代理补充访问控制
- 当前服务一次只处理一个活跃任务，前端或调用方应做好排队或重试
- 处理 Word 文档时，需要保证服务器具备对应平台的转换环境
- 如果前端希望展示实时进度，应优先轮询任务查询接口
- 如果前端只关心最终结果，应在任务 `completed` 后读取 `output_path`
