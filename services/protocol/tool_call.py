"""OpenAI function tool calling over the ChatGPT web conversation endpoint.

The web conversation payload has no `tools` field, so the tool protocol is
compiled into a system message (available tools plus a one-line JSON reply
convention) and model replies are strictly parsed back into OpenAI tool_calls.

Tool results are replayed as user messages prefixed with `[TOOL RESULT ...]`:
the authenticated upstream returns an empty reply for `author.role == "tool"`
frames, while the user-framed form is answered correctly.
"""
from __future__ import annotations

import json
import re
import secrets
import string
from typing import Any, TypedDict

from services.protocol.conversation import message_text

TOOL_CALL_REPLY_FORMAT = '{"tool_calls":[{"name":"<tool>","arguments":{}}]}'
TOOL_RESULT_PREFIX = "[TOOL RESULT"
MAX_PARSE_LENGTH = 8000
DEFAULT_MAX_STREAM_BUFFER = 1200
CALL_ID_LENGTH = 22
CALL_ID_ALPHABET = string.ascii_letters + string.digits
GENUI_PATTERN = re.compile("\ue200[^\ue201]*\ue201|[\ue200\ue201]")
FENCE_PATTERN = re.compile(r"^```(?:json)?\s*\n?([\s\S]*?)\n?\s*```$")


class ToolDefinition(TypedDict, total=False):
    name: str
    description: str
    parameters: dict[str, Any]


class ParsedToolCall(TypedDict):
    name: str
    arguments: str


def normalize_tool_definitions(tools: object) -> list[ToolDefinition]:
    """Accept `{type, function:{name,...}}` and shorthand `{type, name, ...}` function tools."""
    if not isinstance(tools, list):
        return []
    output: list[ToolDefinition] = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type", "function") != "function":
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        name = function.get("name") or tool.get("name")
        if not isinstance(name, str) or not name:
            continue
        definition: ToolDefinition = {"name": name}
        description = function.get("description", tool.get("description"))
        if isinstance(description, str):
            definition["description"] = description
        parameters = function.get("parameters", tool.get("parameters"))
        if isinstance(parameters, dict):
            definition["parameters"] = parameters
        output.append(definition)
    return output


def tool_calling_enabled(body: dict[str, Any]) -> bool:
    return bool(normalize_tool_definitions(body.get("tools"))) and body.get("tool_choice") != "none"


def build_tool_protocol_system_message(tools: list[ToolDefinition], tool_choice: object = None) -> str:
    tool_lines = []
    for tool in tools:
        schema = json.dumps(tool.get("parameters") or {}, ensure_ascii=False)
        params = "no arguments" if schema == "{}" else schema
        description = f" -> {tool['description']}" if tool.get("description") else ""
        tool_lines.append(f"- {tool['name']}({params}){description}")

    # Framed as "the client executes, you only request": authenticated models refuse wording that
    # claims the tools are connected to them, since they know their real tool list.
    lines = [
        "[FUNCTION CALLING PROTOCOL]",
        "",
        "This conversation runs inside a client application that can execute the functions listed below. "
        "You cannot execute them yourself. Instead, when a function is needed, you REQUEST it by replying with JSON; "
        "the application executes it and sends you the result in a later message.",
        "",
        "Functions:",
        *tool_lines,
        "",
        "To request functions, reply with ONLY this single line of raw JSON and nothing else:",
        TOOL_CALL_REPLY_FORMAT,
        "",
        "Request a function whenever the answer depends on data or actions it provides (private, internal, or live data). "
        "Never say you lack access to that data and never answer it from memory or web search - request the function instead.",
        "",
        f'Function results arrive in user messages that start with "{TOOL_RESULT_PREFIX} <function> <call id>]". '
        "Treat them as authoritative and answer from them.",
        "",
        "If no function is needed, answer normally in plain text.",
    ]
    if tool_choice == "required":
        lines += ["", "For this request you MUST request a function. A plain-text answer is invalid."]
    elif isinstance(tool_choice, dict):
        function = tool_choice.get("function") if isinstance(tool_choice.get("function"), dict) else {}
        name = function.get("name")
        if isinstance(name, str) and name:
            lines += ["", f'For this request you MUST request the function "{name}". No other function and no plain-text answer is acceptable.']
    return "\n".join(lines)


def _tool_call_name(call: dict[str, Any]) -> str:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = call.get("name", function.get("name"))
    return name if isinstance(name, str) else ""


def _tool_call_arguments(call: dict[str, Any]) -> object:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    arguments = call.get("arguments", function.get("arguments"))
    if not isinstance(arguments, str):
        return arguments if arguments is not None else {}
    stripped = arguments.strip()
    if not stripped:
        return {}
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return stripped


def prepare_tool_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rewrite assistant tool_calls and tool results into text frames the web endpoint answers."""
    call_names: dict[str, str] = {}
    output: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        tool_calls = message.get("tool_calls")
        if role == "assistant" and isinstance(tool_calls, list) and tool_calls:
            calls = [call for call in tool_calls if isinstance(call, dict)]
            for call in calls:
                if isinstance(call.get("id"), str):
                    call_names[call["id"]] = _tool_call_name(call)
            payload = {"tool_calls": [{"name": _tool_call_name(call), "arguments": _tool_call_arguments(call)} for call in calls]}
            output.append({"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)})
        elif role == "tool":
            call_id = message.get("tool_call_id") if isinstance(message.get("tool_call_id"), str) else ""
            name = message.get("name") if isinstance(message.get("name"), str) and message.get("name") else call_names.get(call_id) or "tool"
            header = " ".join(part for part in (TOOL_RESULT_PREFIX, name, call_id) if part)
            output.append({"role": "user", "content": f"{header}]\n{message_text(message.get('content'))}"})
        else:
            output.append(message)
    return output


def strip_genui_markers(text: str) -> str:
    return GENUI_PATTERN.sub("", text)


def _json_candidate(text: str) -> str | None:
    trimmed = text.strip()
    if not trimmed:
        return None
    fence = FENCE_PATTERN.match(trimmed)
    if fence:
        return fence.group(1).strip()
    if "\n" not in trimmed or (trimmed.startswith("{") and trimmed.endswith("}")):
        return trimmed
    return None


def try_parse_tool_call(text: str) -> list[ParsedToolCall] | None:
    """Return parsed calls when the whole reply is the tool-call JSON convention, else None."""
    text = strip_genui_markers(text or "")
    if not text or len(text) > MAX_PARSE_LENGTH:
        return None
    candidate = _json_candidate(text)
    if candidate is None:
        return None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    calls = parsed.get("tool_calls") if isinstance(parsed, dict) else None
    if not isinstance(calls, list) or not calls:
        return None
    output: list[ParsedToolCall] = []
    for call in calls:
        if not isinstance(call, dict):
            return None
        name = _tool_call_name(call)
        if not name:
            return None
        arguments = _tool_call_arguments(call)
        if isinstance(arguments, str):
            output.append({"name": name, "arguments": arguments})
            continue
        if not isinstance(arguments, dict):
            arguments = {"value": arguments}
        output.append({"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)})
    return output


def new_call_id() -> str:
    return "call_" + "".join(secrets.choice(CALL_ID_ALPHABET) for _ in range(CALL_ID_LENGTH))


def to_openai_tool_calls(parsed: list[ParsedToolCall]) -> list[dict[str, Any]]:
    return [{"id": new_call_id(), "type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}} for call in parsed]


class ToolCallStreamState:
    """Turns cumulative reply text into stream deltas, holding back text that may still become a tool call.

    A reply is held only while it opens with `{` or a ```json fence and stays under `max_buffer`;
    anything else is provably plain text and streams through unchanged.
    """

    def __init__(self, max_buffer: int = DEFAULT_MAX_STREAM_BUFFER) -> None:
        self.max_buffer = max_buffer
        self.mode = "deciding"
        self.sent_length = 0

    def _flush(self, text: str) -> list[dict[str, Any]]:
        self.mode = "text"
        if len(text) <= self.sent_length:
            return []
        delta = text[self.sent_length:]
        self.sent_length = len(text)
        return [{"content": delta}]

    def _emit_tool_calls(self, parsed: list[ParsedToolCall]) -> list[dict[str, Any]]:
        self.mode = "tool_calls"
        calls = to_openai_tool_calls(parsed)
        return [{"tool_calls": [{"index": index, **call}]} for index, call in enumerate(calls)]

    def feed(self, text: str) -> list[dict[str, Any]]:
        if self.mode == "tool_calls":
            return []
        if self.mode == "text":
            return self._flush(text)
        trimmed = text.strip()
        if not trimmed:
            return []
        if not (trimmed.startswith("{") or trimmed.startswith("```json")) or len(text) > self.max_buffer:
            return self._flush(text)
        parsed = try_parse_tool_call(text)
        return self._emit_tool_calls(parsed) if parsed else []

    def finish(self, text: str) -> list[dict[str, Any]]:
        if self.mode != "deciding":
            return [] if self.mode == "tool_calls" else self._flush(text)
        parsed = try_parse_tool_call(text)
        return self._emit_tool_calls(parsed) if parsed else self._flush(text)

    def finish_reason(self) -> str:
        return "tool_calls" if self.mode == "tool_calls" else "stop"
