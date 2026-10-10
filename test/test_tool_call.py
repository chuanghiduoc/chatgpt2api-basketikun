from __future__ import annotations

import json
import unittest
from unittest import mock

from services.protocol import openai_v1_chat_complete
from services.protocol.chat_completion_cache import chat_completion_cache
from services.protocol.tool_call import (
    ToolCallStreamState,
    build_tool_protocol_system_message,
    normalize_tool_definitions,
    prepare_tool_messages,
    tool_calling_enabled,
    try_parse_tool_call,
)

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
TOOL_CALL_REPLY = '{"tool_calls":[{"name":"get_weather","arguments":{"city":"Hanoi"}}]}'


def chunk_deltas(chunks: list[dict]) -> list[dict]:
    return [chunk["choices"][0]["delta"] for chunk in chunks if chunk.get("choices")]


class ParseToolCallTests(unittest.TestCase):
    def test_parses_single_line_json(self) -> None:
        self.assertEqual(try_parse_tool_call(TOOL_CALL_REPLY), [{"name": "get_weather", "arguments": '{"city": "Hanoi"}'}])

    def test_parses_fenced_json_block(self) -> None:
        parsed = try_parse_tool_call("```json\n" + TOOL_CALL_REPLY + "\n```")
        self.assertEqual(parsed[0]["name"], "get_weather")

    def test_parses_multiple_calls_and_string_arguments(self) -> None:
        text = '{"tool_calls":[{"name":"a","arguments":"{\\"x\\":1}"},{"name":"b"}]}'
        self.assertEqual(try_parse_tool_call(text), [{"name": "a", "arguments": '{"x": 1}'}, {"name": "b", "arguments": "{}"}])

    def test_accepts_openai_function_shape(self) -> None:
        text = '{"tool_calls":[{"function":{"name":"a","arguments":"{}"}}]}'
        self.assertEqual(try_parse_tool_call(text), [{"name": "a", "arguments": "{}"}])

    def test_rejects_plain_text_and_prose_around_json(self) -> None:
        self.assertIsNone(try_parse_tool_call("The weather is sunny."))
        self.assertIsNone(try_parse_tool_call("Sure!\n" + TOOL_CALL_REPLY))
        self.assertIsNone(try_parse_tool_call('{"answer": 42}'))
        self.assertIsNone(try_parse_tool_call('{"tool_calls": []}'))
        self.assertIsNone(try_parse_tool_call('{"tool_calls":[{"arguments":{}}]}'))
        self.assertIsNone(try_parse_tool_call("{not json"))

    def test_strips_genui_markers(self) -> None:
        self.assertIsNotNone(try_parse_tool_call("\ue200x\ue201" + TOOL_CALL_REPLY))


class ToolDefinitionTests(unittest.TestCase):
    def test_normalizes_standard_and_shorthand_forms(self) -> None:
        tools = normalize_tool_definitions([
            WEATHER_TOOL,
            {"type": "function", "name": "ping"},
            {"type": "function", "function": {"description": "no name"}},
            {"type": "web_search"},
        ])
        self.assertEqual([tool["name"] for tool in tools], ["get_weather", "ping"])

    def test_tool_calling_enabled_respects_tool_choice_none(self) -> None:
        self.assertTrue(tool_calling_enabled({"tools": [WEATHER_TOOL]}))
        self.assertFalse(tool_calling_enabled({"tools": [WEATHER_TOOL], "tool_choice": "none"}))
        self.assertFalse(tool_calling_enabled({"tools": []}))
        self.assertFalse(tool_calling_enabled({}))

    def test_protocol_message_lists_tools_and_tool_choice(self) -> None:
        tools = normalize_tool_definitions([WEATHER_TOOL])
        message = build_tool_protocol_system_message(tools, "auto")
        self.assertIn("get_weather(", message)
        self.assertIn("[TOOL RESULT", message)
        self.assertNotIn("MUST request", message)
        self.assertIn("MUST request a function", build_tool_protocol_system_message(tools, "required"))
        named = build_tool_protocol_system_message(tools, {"type": "function", "function": {"name": "get_weather"}})
        self.assertIn('MUST request the function "get_weather"', named)


class PrepareToolMessagesTests(unittest.TestCase):
    def test_converts_assistant_tool_calls_and_tool_results(self) -> None:
        messages = prepare_tool_messages([
            {"role": "user", "content": "Weather in Hanoi?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"Hanoi"}'}}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"temp_c":31}'},
        ])
        self.assertEqual(messages[0], {"role": "user", "content": "Weather in Hanoi?"})
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(json.loads(messages[1]["content"]), {"tool_calls": [{"name": "get_weather", "arguments": {"city": "Hanoi"}}]})
        self.assertEqual(messages[2], {"role": "user", "content": '[TOOL RESULT get_weather call_1]\n{"temp_c":31}'})

    def test_tool_result_name_prefers_explicit_name(self) -> None:
        messages = prepare_tool_messages([{"role": "tool", "name": "lookup", "content": [{"type": "text", "text": "ok"}]}])
        self.assertEqual(messages[0]["content"], "[TOOL RESULT lookup]\nok")

    def test_keeps_multimodal_user_content_untouched(self) -> None:
        content = [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]
        self.assertIs(prepare_tool_messages([{"role": "user", "content": content}])[0]["content"], content)


class ToolCallStreamStateTests(unittest.TestCase):
    def test_plain_text_flushes_immediately_and_keeps_streaming(self) -> None:
        state = ToolCallStreamState()
        self.assertEqual(state.feed("Hel"), [{"content": "Hel"}])
        self.assertEqual(state.feed("Hello"), [{"content": "lo"}])
        self.assertEqual(state.finish("Hello"), [])
        self.assertEqual(state.finish_reason(), "stop")

    def test_buffers_json_until_complete_then_emits_tool_calls(self) -> None:
        state = ToolCallStreamState()
        self.assertEqual(state.feed('{"tool_calls":[{"name":"get_'), [])
        emits = state.feed(TOOL_CALL_REPLY)
        self.assertEqual(len(emits), 1)
        call = emits[0]["tool_calls"][0]
        self.assertEqual(call["index"], 0)
        self.assertTrue(call["id"].startswith("call_"))
        self.assertEqual(call["function"], {"name": "get_weather", "arguments": '{"city": "Hanoi"}'})
        self.assertEqual(state.feed(TOOL_CALL_REPLY + "\n"), [])
        self.assertEqual(state.finish(TOOL_CALL_REPLY), [])
        self.assertEqual(state.finish_reason(), "tool_calls")

    def test_json_that_is_not_a_tool_call_is_flushed_at_finish(self) -> None:
        state = ToolCallStreamState()
        self.assertEqual(state.feed('{"answer": 42}'), [])
        self.assertEqual(state.finish('{"answer": 42}'), [{"content": '{"answer": 42}'}])
        self.assertEqual(state.finish_reason(), "stop")

    def test_oversized_json_buffer_is_flushed_as_text(self) -> None:
        state = ToolCallStreamState(max_buffer=10)
        self.assertEqual(state.feed('{"a":"0123456789"}'), [{"content": '{"a":"0123456789"}'}])


class ChatCompletionToolCallTests(unittest.TestCase):
    def setUp(self) -> None:
        chat_completion_cache.clear()

    def body(self, **extra) -> dict:
        return {"model": "auto", "messages": [{"role": "user", "content": "Weather in Hanoi?"}], "tools": [WEATHER_TOOL], **extra}

    def test_non_stream_returns_tool_calls_and_injects_protocol(self) -> None:
        captured = {}

        def fake_collect_text(_backend, request):
            captured["messages"] = request.messages
            return TOOL_CALL_REPLY

        with (
            mock.patch("services.protocol.openai_v1_chat_complete.text_backend", return_value=object()),
            mock.patch("services.protocol.openai_v1_chat_complete.collect_text", side_effect=fake_collect_text),
        ):
            response = openai_v1_chat_complete.handle(self.body())

        choice = response["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertIsNone(choice["message"]["content"])
        self.assertEqual(choice["message"]["tool_calls"][0]["function"]["name"], "get_weather")
        system_texts = [m["content"] for m in captured["messages"] if m["role"] == "system"]
        self.assertTrue(any("get_weather(" in text for text in system_texts))
        self.assertFalse(any(openai_v1_chat_complete.TOOL_UNAVAILABLE_SYSTEM_MESSAGE == text for text in system_texts))

    def test_non_stream_plain_text_answer_is_returned_as_content(self) -> None:
        with (
            mock.patch("services.protocol.openai_v1_chat_complete.text_backend", return_value=object()),
            mock.patch("services.protocol.openai_v1_chat_complete.collect_text", return_value="It is sunny."),
        ):
            response = openai_v1_chat_complete.handle(self.body())
        self.assertEqual(response["choices"][0]["message"], {"role": "assistant", "content": "It is sunny."})
        self.assertEqual(response["choices"][0]["finish_reason"], "stop")

    def test_tool_requests_bypass_the_completion_cache(self) -> None:
        calls = []

        def fake_collect_text(_backend, _request):
            calls.append(1)
            return TOOL_CALL_REPLY

        with (
            mock.patch("services.protocol.openai_v1_chat_complete.text_backend", return_value=object()),
            mock.patch("services.protocol.openai_v1_chat_complete.collect_text", side_effect=fake_collect_text),
        ):
            first = openai_v1_chat_complete.handle(self.body())
            second = openai_v1_chat_complete.handle(self.body())
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(first["choices"][0]["message"]["tool_calls"][0]["id"], second["choices"][0]["message"]["tool_calls"][0]["id"])

    def test_stream_emits_tool_call_deltas_and_finish_reason(self) -> None:
        pieces = ['{"tool_calls":[{"name":', '"get_weather","arguments":{"city":"Hanoi"}}]}']
        with (
            mock.patch("services.protocol.openai_v1_chat_complete.text_backend", return_value=object()),
            mock.patch("services.protocol.openai_v1_chat_complete.stream_text_deltas", return_value=iter(pieces)),
        ):
            chunks = list(openai_v1_chat_complete.handle(self.body(stream=True)))
        deltas = chunk_deltas(chunks)
        self.assertEqual(deltas[0]["role"], "assistant")
        self.assertEqual(deltas[0]["tool_calls"][0]["function"]["name"], "get_weather")
        self.assertFalse(any(delta.get("content") for delta in deltas))
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")

    def test_stream_plain_text_streams_content(self) -> None:
        with (
            mock.patch("services.protocol.openai_v1_chat_complete.text_backend", return_value=object()),
            mock.patch("services.protocol.openai_v1_chat_complete.stream_text_deltas", return_value=iter(["It is ", "sunny."])),
        ):
            chunks = list(openai_v1_chat_complete.handle(self.body(stream=True)))
        deltas = chunk_deltas(chunks)
        self.assertEqual(deltas[0], {"role": "assistant", "content": "It is "})
        self.assertEqual("".join(delta.get("content") or "" for delta in deltas), "It is sunny.")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")

    def test_tool_choice_none_keeps_existing_unavailable_behaviour(self) -> None:
        captured = {}

        def fake_collect_text(_backend, request):
            captured["messages"] = request.messages
            return "ok"

        with (
            mock.patch("services.protocol.openai_v1_chat_complete.text_backend", return_value=object()),
            mock.patch("services.protocol.openai_v1_chat_complete.collect_text", side_effect=fake_collect_text),
        ):
            openai_v1_chat_complete.handle(self.body(tool_choice="none"))
        self.assertEqual(captured["messages"][0]["content"], openai_v1_chat_complete.TOOL_UNAVAILABLE_SYSTEM_MESSAGE)


if __name__ == "__main__":
    unittest.main()
