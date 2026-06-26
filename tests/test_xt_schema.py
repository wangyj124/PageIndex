import json
from pathlib import Path

from pageindex.contract_extraction import normalize_schema


FULL_SCHEMA_PATH = Path("sample_data/schemas/contract_fields_xt_full.json")


def test_xt_full_schema_is_normalizable_and_complete():
    schema = json.loads(FULL_SCHEMA_PATH.read_text(encoding="utf-8"))
    fields = normalize_schema(schema)

    assert len(fields) == 49
    assert len({field.name for field in fields}) == 49


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


def test_xt_full_schema_marks_price_summary_fields_as_key_info():
    schema = json.loads(FULL_SCHEMA_PATH.read_text(encoding="utf-8"))
    field_map = {field["name"]: field for field in schema["fields"]}

    assert field_map["project_name"]["value_return_mode"] == "key_info"
    assert field_map["unit_configuration"]["value_return_mode"] == "key_info"
    assert field_map["customer_name"]["value_return_mode"] == "key_info"
    assert field_map["contract_total_price"]["value_return_mode"] == "key_info"
    assert field_map["equipment_total_price"]["value_return_mode"] == "key_info"
    assert field_map["pricing_technical_service_fee"]["value_return_mode"] == "key_info"
    assert field_map["overseas_training_and_design_liaison_fee"]["value_return_mode"] == "key_info"
    assert field_map["provisional_amount"]["value_return_mode"] == "key_info"
    assert field_map["misc_freight_and_insurance_fee"]["value_return_mode"] == "key_info"
    assert "value_return_mode" not in field_map["payment_technical_service_fee"]


def test_xt_full_schema_key_info_descriptions_do_not_repeat_return_mode_prompt():
    schema = json.loads(FULL_SCHEMA_PATH.read_text(encoding="utf-8"))
    field_map = {field["name"]: field for field in schema["fields"]}

    key_info_fields = [
        "project_name",
        "unit_configuration",
        "customer_name",
        "contract_total_price",
        "equipment_total_price",
        "pricing_technical_service_fee",
        "overseas_training_and_design_liaison_fee",
        "provisional_amount",
        "misc_freight_and_insurance_fee",
    ]
    assert all("只需提取关键信息" not in field_map[name]["description"] for name in key_info_fields)
    assert field_map["project_name"]["description"].startswith("为合同所服务的项目名称")
    assert field_map["misc_freight_and_insurance_fee"]["description"].startswith("一般指运保费")
