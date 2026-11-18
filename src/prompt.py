"""Canonical MDR SaMD prompt and structured-response parser.
--> holds the one canonical natural-language prompt and checks the structure of model answers without changing the original text."""

from __future__ import annotations

import json
import re


PROMPT_VERSION = "mdr_samd_risk_classification_v1"


PERSONA_AND_SCOPE = """Identify and scope
You are a regulatory expert specializing in medical device classification. You receive a description of a software-based medical device (SaMD) and must return the single most likely risk class under EU MDR 2017/745."""

CLASSIFICATION_SYSTEM = """Classification system
Classify the device as exactly one of the following:
NON_MEDICAL_DEVICE | CLASS_I | CLASS_IIA | CLASS_IIB | CLASS_III | INSUFFICIENT_INFORMATION"""

RULES = [
    '1. Use applicable EU guidelines, including the MDR and, where relevant, the IVDR, and reference them explicitly.',
    '2. Maintain clarity by avoiding unnecessary speculation and ensuring responses are well-supported.',
    '3. First determine whether the software-based product qualifies as a medical device based on the information provided. If it does not qualify as a medical device, assign it to the "Non-Medical Device" category.',
    '4. If two risk classes are equally defensible, assign the software-based medical device to the higher-risk class and document the ambiguity in the rationale field by: (a) quoting or precisely identifying the specific text, claim, or missing information that makes the classification unclear; and (b) stating what additional information or clarification would be needed to resolve it and reach an unambiguous single-class assignment.',
    '5. If the information provided is not sufficient or clear enough to determine whether the product is a medical device or to determine its risk class, assign it to the "Not Defined" category and provide your reasoning in the rationale field by specifying: (a) the specific information that is absent or unclear and why it is necessary for classification; (b) the one or two most plausible classifications given what is known; and (c) the concrete questions or data points that, if answered, would allow a definitive single-class assignment.',
    '6. Use only the information provided in the device description. Do not assume unstated functionality, intended purpose, certification, approval, or regulatory status.',
    '7. Provide sources to substantiate the rationale for the classification. If you draw only on your pretrained knowledge, write exactly:"No external resources were used; reasoning based on prior knowledge."',
]

OUTPUT_FORMAT = """Required output format
Respond only in valid JSON:
{
"classification": "NON_MEDICAL_DEVICE | CLASS_I | CLASS_IIA | CLASS_IIB | CLASS_III | INSUFFICIENT_INFORMATION",
"rationale": "Concise explanation based only on the supplied description",
"applicable_regulation": "MDR | IVDR | NOT_APPLICABLE | INSUFFICIENT_INFORMATION",
"applicable_provision": "Exact article, annex, rule, or NOT_APPLICABLE",
"missing_information": ["Required information, or an empty list"]
}"""

ALLOWED_CLASSIFICATIONS = {
    "NON_MEDICAL_DEVICE", "CLASS_I", "CLASS_IIA", "CLASS_IIB", "CLASS_III",
    "INSUFFICIENT_INFORMATION",
}
ALLOWED_REGULATIONS = {"MDR", "IVDR", "NOT_APPLICABLE", "INSUFFICIENT_INFORMATION"}


def assemble_system_prompt(rule_lines: list[str]) -> str:
    parts = [PERSONA_AND_SCOPE, CLASSIFICATION_SYSTEM]
    if rule_lines:
        parts.append("Follow these rules:\n" + "\n".join(rule_lines))
    parts.append(OUTPUT_FORMAT)
    return "\n\n".join(parts)


SYSTEM_PROMPT = assemble_system_prompt(RULES)
USER_TEMPLATE = 'Device description: "{device_description}"'


def build_chat_messages(device_description: str, system_prompt: str = SYSTEM_PROMPT) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": USER_TEMPLATE.format(device_description=device_description)},
    ]


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _find_balanced_json(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escape:
                escape = False
            elif character == "\\":
                escape = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


_REQUIRED_RESPONSE_FIELDS = {
    "classification", "rationale", "applicable_regulation",
    "applicable_provision", "missing_information",
}
NO_EXTERNAL_RESOURCES_STRING = "No external resources were used; reasoning based on prior knowledge."


def _empty_parse(text: str, error: str) -> dict[str, object]:
    return {
        "parse_ok": False,
        "parse_success": False,
        "schema_valid": False,
        "parse_error": error,
        "validation_issues": ["malformed_json"],
        "json_repaired": False,
        "extraneous_text_present": False,
        "required_fields_present": False,
        "classification_valid": False,
        "regulation_valid": False,
        "rationale_present": False,
        "applicable_provision_present": False,
        "missing_information_valid_type": False,
        "citation_present": False,
        "no_external_resources_statement_present": False,
        "classification": "",
        "rationale": "",
        "applicable_regulation": "",
        "applicable_provision": "",
        "missing_information": "",
        "missing_information_raw": None,
        "extra_fields": [],
        "raw": text,
    }


def parse_response(text: str) -> dict[str, object]:
    """Extract JSON while preserving invalid values and exposing validation flags."""
    candidates: list[tuple[str, bool]] = []
    stripped = text.strip()
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        candidates.append((fence_match.group(1).strip(), True))
    balanced = _find_balanced_json(text)
    if balanced:
        candidates.append((balanced, balanced.strip() != stripped))
    candidates.append((stripped, False))

    last_error = "no JSON object found"
    for candidate, extraneous in candidates:
        repaired_candidate = _TRAILING_COMMA_RE.sub(r"\1", candidate)
        for attempt, repaired in ((candidate, False), (repaired_candidate, repaired_candidate != candidate)):
            try:
                obj = json.loads(attempt)
            except json.JSONDecodeError as exc:
                last_error = f"JSONDecodeError: {exc}"
                continue
            if not isinstance(obj, dict):
                last_error = "top-level JSON value is not an object"
                continue

            missing_fields = sorted(_REQUIRED_RESPONSE_FIELDS - set(obj))
            extra_fields = sorted(set(obj) - _REQUIRED_RESPONSE_FIELDS)
            classification = str(obj.get("classification", "")).strip()
            regulation = str(obj.get("applicable_regulation", "")).strip()
            rationale_value = obj.get("rationale", "")
            provision_value = obj.get("applicable_provision", "")
            rationale = rationale_value if isinstance(rationale_value, str) else str(rationale_value)
            provision = provision_value if isinstance(provision_value, str) else str(provision_value)
            missing_raw = obj.get("missing_information")
            missing_type_valid = isinstance(missing_raw, list)
            missing_rendered = (
                "; ".join(str(item) for item in missing_raw)
                if missing_type_valid
                else ("" if missing_raw is None else str(missing_raw))
            )

            classification_valid = classification in ALLOWED_CLASSIFICATIONS
            regulation_valid = regulation in ALLOWED_REGULATIONS
            rationale_present = isinstance(rationale_value, str) and bool(rationale_value.strip())
            provision_present = isinstance(provision_value, str) and bool(provision_value.strip())
            required_present = not missing_fields
            issues: list[str] = []
            if missing_fields:
                issues.append("missing_required_fields:" + ",".join(missing_fields))
            if not classification_valid:
                issues.append("invalid_classification")
            if not regulation_valid:
                issues.append("invalid_regulation")
            if not rationale_present:
                issues.append("missing_or_invalid_rationale")
            if not provision_present:
                issues.append("missing_or_invalid_applicable_provision")
            if not missing_type_valid:
                issues.append("invalid_missing_information_type")
            if repaired:
                issues.append("json_syntax_repaired")
            if extraneous:
                issues.append("extraneous_text")

            schema_valid = (
                required_present
                and classification_valid
                and regulation_valid
                and rationale_present
                and provision_present
                and missing_type_valid
            )
            no_external = NO_EXTERNAL_RESOURCES_STRING in rationale
            citation = provision not in {"", "NOT_APPLICABLE", "INSUFFICIENT_INFORMATION"}
            return {
                "parse_ok": schema_valid,
                "parse_success": True,
                "schema_valid": schema_valid,
                "parse_error": "" if schema_valid else "; ".join(issues),
                "validation_issues": issues,
                "json_repaired": repaired,
                "extraneous_text_present": extraneous,
                "required_fields_present": required_present,
                "classification_valid": classification_valid,
                "regulation_valid": regulation_valid,
                "rationale_present": rationale_present,
                "applicable_provision_present": provision_present,
                "missing_information_valid_type": missing_type_valid,
                "citation_present": citation,
                "no_external_resources_statement_present": no_external,
                "classification": classification,
                "rationale": rationale,
                "applicable_regulation": regulation,
                "applicable_provision": provision,
                "missing_information": missing_rendered,
                "missing_information_raw": missing_raw,
                "extra_fields": extra_fields,
                "raw": text,
            }
    return _empty_parse(text, last_error)
