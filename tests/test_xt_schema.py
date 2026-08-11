import json
from pathlib import Path

from pageindex.contract_extraction import normalize_schema


FULL_SCHEMA_PATH = Path("sample_data/schemas/contract_fields_xt_full.json")

EXPECTED_SHEET2_FIELDS = {
    "合同价格": ["合同总价", "设备总价", "技术服务费", "国外技术培训、设联会费", "杂运费（含保险费）"],
    "付款方式": ["预付款", "交货款", "验收款", "质保金", "技术服务费", "运输及保险费", "出国预付款", "出国实际款", "培训费"],
    "交货": ["交货地点", "收货单位", "设备交货计划", "设备提前交货处理", "设备迟交货处理", "技术资料邮寄地址", "技术资料交付", "资料迟交货处理"],
    "质保期": ["性能验收试验", "初步验收证书", "最终验收", "质保期"],
    "罚款": ["性能罚款", "迟交货罚款", "迟交资料罚款", "可靠性运行延误", "服务罚款", "罚款封顶", "付款罚款"],
    "保险": ["保险", "保险公司"],
    "保函": ["履约保函", "预付款保函"],
    "技术服务": ["服务内容", "服务人员", "服务周期和报价"],
    "合同生效": ["合同生效"],
    "其他关注点": ["供应商短名单", "价格表与技术协议范围对比", "监造和监理等相关规定", "汽轮机精装或散装发运"],
    "合同基础信息": ["项目名称", "卖方合同号", "买方合同号", "机组数", "合同甲方", "所属集团", "电厂地址", "合同签订日期", "合同生效日期"],
}


def test_xt_full_schema_is_normalizable_and_complete():
    schema = json.loads(FULL_SCHEMA_PATH.read_text(encoding="utf-8"))
    fields = normalize_schema(schema)

    assert len(fields) == 54
    assert len({field.name for field in fields}) == 54


def test_xt_full_schema_matches_all_sheet2_fields():
    schema = json.loads(FULL_SCHEMA_PATH.read_text(encoding="utf-8"))
    actual = {(field["focus_cn"], field["label_cn"]) for field in schema["fields"]}
    expected = {
        (focus, label)
        for focus, labels in EXPECTED_SHEET2_FIELDS.items()
        for label in labels
    }

    assert len(expected) == 54
    assert actual == expected


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
    assert field_map["insurer"]["instruction"] == "重点判断是否明确指定，并提取被指定机构的完整名称。"
    assert field_map["performance_penalty"]["instruction"] == "出力、热耗、排放、噪音、震动等"


def test_xt_full_schema_contains_updated_contract_fields():
    schema = json.loads(FULL_SCHEMA_PATH.read_text(encoding="utf-8"))
    field_map = {field["name"]: field for field in schema["fields"]}

    assert field_map["power_plant_address"]["label_cn"] == "电厂地址"
    assert field_map["power_plant_address"]["focus_cn"] == "合同基础信息"
    assert field_map["seller_contract_number"]["label_cn"] == "卖方合同号"
    assert field_map["buyer_contract_number"]["label_cn"] == "买方合同号"
    assert field_map["unit_configuration"]["label_cn"] == "机组数"
    assert field_map["customer_name"]["label_cn"] == "合同甲方"
    assert field_map["group_affiliation"]["label_cn"] == "所属集团"
    assert field_map["contract_signing_date"]["type"] == "date"
    assert field_map["contract_effective_date"]["type"] == "date"
    assert "provisional_amount" not in field_map
    assert "签字页" in field_map["power_plant_address"]["description"]
    assert "技术培训费" in field_map["training_payment"]["description"]
    assert "支付方式" in field_map["training_payment"]["description"]
    assert "完整名称" in field_map["insurer"]["description"]
    assert "开立或出具机构" in field_map["advance_payment_bond"]["description"]


def test_xt_full_schema_applies_sheet2_summary_and_original_modes():
    schema = json.loads(FULL_SCHEMA_PATH.read_text(encoding="utf-8"))
    fields = normalize_schema(schema)
    field_map = {field.name: field for field in fields}
    summary_fields = {
        "project_name",
        "seller_contract_number",
        "buyer_contract_number",
        "unit_configuration",
        "customer_name",
        "group_affiliation",
        "power_plant_address",
        "contract_signing_date",
        "contract_effective_date",
        "contract_total_price",
        "equipment_total_price",
        "pricing_technical_service_fee",
        "overseas_training_and_design_liaison_fee",
        "misc_freight_and_insurance_fee",
    }

    assert {name for name, field in field_map.items() if field.value_return_mode == "key_info"} == summary_fields
    assert sum(field.value_return_mode == "full_clause" for field in fields) == 40
    assert field_map["payment_technical_service_fee"].value_return_mode == "full_clause"


def test_xt_full_schema_key_info_descriptions_do_not_repeat_return_mode_prompt():
    schema = json.loads(FULL_SCHEMA_PATH.read_text(encoding="utf-8"))
    field_map = {field["name"]: field for field in schema["fields"]}

    key_info_fields = [
        "project_name",
        "seller_contract_number",
        "buyer_contract_number",
        "unit_configuration",
        "customer_name",
        "group_affiliation",
        "power_plant_address",
        "contract_signing_date",
        "contract_effective_date",
        "contract_total_price",
        "equipment_total_price",
        "pricing_technical_service_fee",
        "overseas_training_and_design_liaison_fee",
        "misc_freight_and_insurance_fee",
    ]
    assert all("只需提取关键信息" not in field_map[name]["description"] for name in key_info_fields)
    assert field_map["project_name"]["description"].startswith("为合同所服务的项目名称")
    assert field_map["misc_freight_and_insurance_fee"]["description"].startswith("一般指运保费")


def test_xt_full_schema_summary_fields_have_generalized_instructions():
    schema = json.loads(FULL_SCHEMA_PATH.read_text(encoding="utf-8"))
    summary_fields = [field for field in schema["fields"] if field.get("value_return_mode") == "key_info"]

    assert len(summary_fields) == 14
    assert all(str(field.get("instruction", "")).strip() for field in summary_fields)
    assert all("潮州深能甘露" not in field["instruction"] for field in summary_fields)


def test_xt_price_summary_fields_require_explicit_wan_amounts():
    schema = json.loads(FULL_SCHEMA_PATH.read_text(encoding="utf-8"))
    field_map = {field["name"]: field for field in schema["fields"]}
    amount_fields = [
        field_map["overseas_training_and_design_liaison_fee"],
        field_map["misc_freight_and_insurance_fee"],
    ]

    assert all("只能返回合同原文明示的具体" in field["instruction"] for field in amount_fields)
    assert all("数字+万元" in field["instruction"] for field in amount_fields)
    assert all("status=not_found、value=未找到" in field["instruction"] for field in amount_fields)
    assert all("禁止" in field["instruction"] and "推算" in field["instruction"] for field in amount_fields)
