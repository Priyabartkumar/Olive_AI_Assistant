import re

BLOCKED_PATTERNS = [
    r"\b(hack|hacking)\b.*\b(computer|system|server|network|account|password)\b",
    r"\b(make|create|build|synthesize)\b.*\b(bomb|explosive|weapon|drug|meth)\b",
    r"\b(steal|phish|phishing)\b.*\b(identity|credential|password|account|card)\b",
    r"\bmalware\b",
    r"\bdos attack\b",
    r"\bddos\b",
    r"\b(bypass|break into)\b.*\b(security|firewall|system)\b",
    r"\bjailbreak\b.*\b(prompt|ai|model|instruction)\b",
    r"\bignore\b.*\b(previous|prior|above)\b.*\b(instruction|rule|prompt)\b",
    r"\b(kill|murder|harm|hurt)\b.*\b(person|people|someone|human)\b",
    r"\bself[- ]?harm\b",
    r"\bsuicide\b.*\b(method|how|way)\b",
]

BLOCKED_OUTPUT_PATTERNS = [
    r"(?i)(step[- ]?by[- ]?step|instructions?)\b.*\b(hack|exploit|attack|bomb|weapon)",
    r"(?i)here('s| is) how (to|you can) (hack|steal|attack|bypass|break)",
    r"(?i)(sudo|rm -rf|chmod|format c:).*",
]

SENSITIVE_TOPICS = [
    "race", "religion", "gender", "ethnicity", "sexuality",
    "politics", "disability", "immigration", "mental illness",
]

INPUT_BLOCKED_RESPONSE = (
    "I appreciate you reaching out, but I'm not comfortable going there. "
    "That topic touches on something that could be harmful, and I'd rather "
    "keep our conversation in a positive space. Can I help you with something else?"
)

OUTPUT_BLOCKED_RESPONSE = (
    "Hmm, I started to answer but realized my response wasn't heading in a great direction. "
    "Let me steer us back — is there something else I can help you with?"
)

SENSITIVITY_DISCLAIMER = (
    "\n\n*I want to be thoughtful here — this is a nuanced topic and I've tried to "
    "share a balanced perspective. Let me know if you'd like to explore it further.*"
)


def check_input(text: str) -> tuple[bool, str]:
    lower = text.lower()
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, lower):
            return False, INPUT_BLOCKED_RESPONSE
    return True, ""


def check_output(response: str) -> tuple[bool, str]:
    for pattern in BLOCKED_OUTPUT_PATTERNS:
        if re.search(pattern, response):
            return False, OUTPUT_BLOCKED_RESPONSE
    return True, response


def flag_sensitive(text: str) -> bool:
    lower = text.lower()
    return any(topic in lower for topic in SENSITIVE_TOPICS)


def apply_guardrails(user_input: str, model_response: str) -> str:
    safe_input, blocked_msg = check_input(user_input)
    if not safe_input:
        return blocked_msg

    safe_output, filtered_response = check_output(model_response)
    if not safe_output:
        return filtered_response

    if flag_sensitive(user_input):
        filtered_response += SENSITIVITY_DISCLAIMER

    return filtered_response
