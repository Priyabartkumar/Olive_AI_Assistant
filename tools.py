import re
import math
import datetime

from duckduckgo_search import DDGS


TOOL_DEFINITIONS = [
    {
        "name": "calculator",
        "description": "Evaluate a math expression. Use for arithmetic, unit conversions, percentages.",
        "trigger_patterns": [
            r"\b(calculate|compute|what is|how much|solve)\b.*[\d+\-*/^%]",
            r"\d+\s*[+\-*/^%]\s*\d+",
            r"\b(sqrt|sin|cos|tan|log|pow)\b",
            r"\b(convert|conversion)\b.*\b(to|into)\b",
        ],
    },
    {
        "name": "datetime",
        "description": "Get current date, time, timezone info, or date calculations.",
        "trigger_patterns": [
            r"\b(what|current|today).*(time|date|day|month|year)\b",
            r"\b(how many days|days until|days since|days between)\b",
            r"\b(what day|which day)\b",
        ],
    },
    {
        "name": "web_search",
        "description": "Search the web for current information, news, facts.",
        "trigger_patterns": [
            r"\b(search|look up|find|google|latest|current|recent|news|today)\b",
            r"\b(what happened|who won|score|weather|price|stock)\b",
        ],
    },
]


def detect_tool(user_message: str) -> str | None:
    lower = user_message.lower()
    for tool in TOOL_DEFINITIONS:
        for pattern in tool["trigger_patterns"]:
            if re.search(pattern, lower):
                return tool["name"]
    return None


def run_calculator(expression: str) -> str:
    cleaned = re.sub(r"[^0-9+\-*/().%^ ]", "", expression)
    cleaned = cleaned.replace("^", "**")
    try:
        safe_dict = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "log": math.log, "log10": math.log10,
            "pi": math.pi, "e": math.e, "pow": pow,
        }
        result = eval(cleaned, {"__builtins__": {}}, safe_dict)
        return f"The result is: **{result}**"
    except Exception as e:
        return f"I couldn't calculate that — could you rephrase the math expression? (Error: {e})"


def run_datetime(query: str) -> str:
    now = datetime.datetime.now()
    lower = query.lower()

    if "time" in lower:
        return f"It's currently **{now.strftime('%I:%M %p')}** on **{now.strftime('%A, %B %d, %Y')}**."
    if "date" in lower or "today" in lower:
        return f"Today is **{now.strftime('%A, %B %d, %Y')}**."
    if "day" in lower and ("week" in lower or "what" in lower):
        return f"Today is **{now.strftime('%A')}**."
    if "year" in lower:
        return f"The current year is **{now.year}**."
    if "month" in lower:
        return f"The current month is **{now.strftime('%B')}**."

    return f"The current date and time is **{now.strftime('%A, %B %d, %Y at %I:%M %p')}**."


def run_web_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=5))
            if results:
                lines = []
                for r in results:
                    lines.append(f"- **{r['title']}** ({r.get('date', 'recent')})\n  {r.get('body', '')}\n  Source: {r.get('source', 'unknown')}")
                return "Here's the latest I found:\n\n" + "\n".join(lines)

            results = list(ddgs.text(query, max_results=5))
            if results:
                lines = []
                for r in results:
                    lines.append(f"- **{r['title']}**\n  {r.get('body', '')}")
                return "Here's what I found:\n\n" + "\n".join(lines)

        return (
            "I searched but couldn't find a quick answer for that. "
            "Could you try rephrasing your question?"
        )
    except Exception:
        return (
            "I tried to search the web but ran into a connection issue. "
            "Let me answer from what I know instead."
        )


def execute_tool(tool_name: str, user_message: str) -> str | None:
    if tool_name == "calculator":
        nums = re.findall(r"[\d+\-*/().^%\s]+", user_message)
        expression = max(nums, key=len) if nums else user_message
        return run_calculator(expression.strip())
    elif tool_name == "datetime":
        return run_datetime(user_message)
    elif tool_name == "web_search":
        clean_query = re.sub(
            r"\b(search|look up|find|google|can you|please|for me)\b", "",
            user_message, flags=re.IGNORECASE,
        ).strip()
        return run_web_search(clean_query)
    return None
