import json

from pageindex.contract_extraction import extract_contract_fields, normalize_schema


class StubClient:
    retrieve_model = "openai/test-model"

    def __init__(self):
        self.page_requests = []

    def get_document_structure(self, doc_id):
        return json.dumps(
            [
                {
                    "title": "价格条款",
                    "summary": "本章节描述合同总价和付款安排",
                    "start_page": 4,
                    "end_page": 5,
                    "nodes": [],
                }
            ],
            ensure_ascii=False,
        )

    def get_page_content(self, doc_id, pages):
        self.page_requests.append(pages)
        return json.dumps(
            [
                {
                    "page": 4,
                    "content": "合同总价为人民币壹佰万元整。",
                }
            ],
            ensure_ascii=False,
        )


def test_normalize_schema_accepts_fields_wrapper():
    fields = normalize_schema(
        {
            "fields": [
                {
                    "name": "contract_amount",
                    "description": "合同总金额",
                    "type": "string",
                    "required": True,
                }
            ]
        }
    )
    assert fields[0].name == "contract_amount"
    assert fields[0].required is True


def test_extract_contract_fields_retries_invalid_confidence(monkeypatch):
    responses = iter(
        [
            '{"pages":[4],"reason":"金额通常在价格条款页"}',
            '{"status":"found","value":"100万元","evidence":"合同总价为人民币壹佰万元整。","pages":[4],"confidence":"Certain","reason":null}',
            '{"status":"found","value":"100万元","evidence":"合同总价为人民币壹佰万元整。","pages":[4],"confidence":"High","reason":null}',
        ]
    )

    async def fake_llm_acompletion(model, prompt):
        return next(responses)

    monkeypatch.setattr("pageindex.contract_extraction.llm_acompletion", fake_llm_acompletion)

    client = StubClient()
    result = extract_contract_fields(
        client,
        "doc-1",
        [
            {
                "name": "contract_amount",
                "description": "合同总金额",
                "type": "string",
            }
        ],
        max_concurrency=1,
        retries=1,
    )

    assert result["contract_amount"]["status"] == "found"
    assert result["contract_amount"]["confidence"] == "High"
    assert result["contract_amount"]["pages"] == [4]
    assert client.page_requests == ["4"]


def test_extract_contract_fields_not_found_forces_low_confidence(monkeypatch):
    responses = iter(
        [
            '{"pages":[4],"reason":"签署信息可能在尾页"}',
            '{"status":"not_found","value":"2024-01-01","evidence":"not used","pages":[4],"confidence":"High","reason":"未在提供页中找到明确签订日期"}',
        ]
    )

    async def fake_llm_acompletion(model, prompt):
        return next(responses)

    monkeypatch.setattr("pageindex.contract_extraction.llm_acompletion", fake_llm_acompletion)

    client = StubClient()
    result = extract_contract_fields(
        client,
        "doc-2",
        [
            {
                "name": "signing_date",
                "description": "合同签订日期",
                "type": "date",
            }
        ],
        max_concurrency=1,
    )

    assert result["signing_date"] == {
        "status": "not_found",
        "value": "",
        "evidence": "",
        "pages": [4],
        "confidence": "Low",
        "reason": "未在提供页中找到明确签订日期",
    }
