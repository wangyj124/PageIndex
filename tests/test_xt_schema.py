import json
from pathlib import Path

from pageindex.contract_extraction import normalize_schema


FULL_SCHEMA_PATH = Path("sample_data/schemas/contract_fields_xt_full.json")


def test_xt_full_schema_is_normalizable_and_complete():
    schema = json.loads(FULL_SCHEMA_PATH.read_text(encoding="utf-8"))
    fields = normalize_schema(schema)

    assert len(fields) == 45
    assert len({field.name for field in fields}) == 45


def test_xt_full_schema_distinguishes_duplicate_chinese_labels():
    schema = json.loads(FULL_SCHEMA_PATH.read_text(encoding="utf-8"))
    field_map = {field["name"]: field for field in schema["fields"]}

    assert field_map["pricing_technical_service_fee"]["label_cn"] == "技术服务费"
    assert field_map["payment_technical_service_fee"]["label_cn"] == "技术服务费"
    assert field_map["pricing_technical_service_fee"]["focus_cn"] == "合同价格"
    assert field_map["payment_technical_service_fee"]["focus_cn"] == "付款方式"


def test_xt_full_schema_populates_instruction_from_remarks():
    schema = json.loads(FULL_SCHEMA_PATH.read_text(encoding="utf-8"))
    field_map = {field["name"]: field for field in schema["fields"]}

    assert field_map["advance_payment"]["instruction"] == "确认付款比例，付款条件"
    assert field_map["delivery_location"]["instruction"] == "工厂EXW交货？现场交货？\n码头、道路情况"
    assert field_map["insurer"]["instruction"] == "是否有指定"
    assert field_map["performance_penalty"]["instruction"] == "出力、热耗、排放、噪音、震动等"
