"""LLM provider abstraction — Anthropic (primary), OpenAI (optional), and Human (console)."""
from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict
    type: str = "tool_use"


@dataclass
class LLMResponse:
    content: list  # list[TextBlock | ToolUseBlock]
    stop_reason: str
    usage: dict  # input_tokens, output_tokens


@runtime_checkable
class LLMProvider(Protocol):
    async def send_message(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        tool_choice: dict | str,
        max_tokens: int,
    ) -> LLMResponse: ...


def _blocks_to_anthropic_content(content: Any) -> Any:
    """Convert content blocks (dataclasses or dicts) to Anthropic-compatible dicts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        result = []
        for block in content:
            if isinstance(block, TextBlock):
                result.append({"type": "text", "text": block.text})
            elif isinstance(block, ToolUseBlock):
                result.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
            elif isinstance(block, dict):
                result.append(block)
            else:
                result.append(block)
        return result
    return content


def _serialize_messages_for_anthropic(messages: list[dict]) -> list[dict]:
    """Ensure all message content is in Anthropic-compatible format."""
    result = []
    for msg in messages:
        result.append(
            {"role": msg["role"], "content": _blocks_to_anthropic_content(msg["content"])}
        )
    return result


class AnthropicProvider:
    def __init__(
        self, model: str = "claude-sonnet-4-6", api_key: str | None = None
    ) -> None:
        import anthropic

        self.model = model
        self.client = anthropic.AsyncAnthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    async def send_message(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        tool_choice: dict | str = "auto",
        max_tokens: int = 2048,
    ) -> LLMResponse:
        if isinstance(tool_choice, str):
            tc: dict = {"type": tool_choice if tool_choice in ("auto", "any") else "auto"}
        else:
            tc = tool_choice

        anthropic_messages = _serialize_messages_for_anthropic(messages)

        response = await self.client.messages.create(
            model=self.model,
            system=system,
            messages=anthropic_messages,
            tools=tools,  # type: ignore[arg-type]
            tool_choice=tc,  # type: ignore[arg-type]
            max_tokens=max_tokens,
        )

        content: list[TextBlock | ToolUseBlock] = []
        for block in response.content:
            if block.type == "text":
                content.append(TextBlock(text=block.text))
            elif block.type == "tool_use":
                content.append(
                    ToolUseBlock(id=block.id, name=block.name, input=block.input)
                )

        return LLMResponse(
            content=content,
            stop_reason=response.stop_reason,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )


class OpenAIProvider:
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None) -> None:
        import openai

        self.model = model
        self.client = openai.AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY")
        )

    async def send_message(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        tool_choice: dict | str = "auto",
        max_tokens: int = 2048,
    ) -> LLMResponse:
        # Convert tools from Anthropic format to OpenAI format
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            }
            for t in tools
        ]

        tc_str = "required" if tool_choice == "any" else "auto"

        # Build OpenAI message list
        openai_messages: list[dict] = [{"role": "system", "content": system}]
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if isinstance(content, str):
                openai_messages.append({"role": role, "content": content})
            elif isinstance(content, list):
                # Detect if this is an assistant message with tool calls
                tool_use_blocks = [
                    b for b in content if isinstance(b, (ToolUseBlock,)) or
                    (isinstance(b, dict) and b.get("type") == "tool_use")
                ]
                tool_result_blocks = [
                    b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"
                ]
                text_blocks = [
                    b for b in content if isinstance(b, (TextBlock,)) or
                    (isinstance(b, dict) and b.get("type") == "text")
                ]

                if tool_result_blocks:
                    for block in tool_result_blocks:
                        openai_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": block["tool_use_id"],
                                "content": block.get("content", ""),
                            }
                        )
                elif tool_use_blocks:
                    tc_list = []
                    for block in tool_use_blocks:
                        if isinstance(block, ToolUseBlock):
                            bid, bname, binput = block.id, block.name, block.input
                        else:
                            bid, bname, binput = block["id"], block["name"], block["input"]
                        tc_list.append(
                            {
                                "id": bid,
                                "type": "function",
                                "function": {
                                    "name": bname,
                                    "arguments": json.dumps(binput),
                                },
                            }
                        )
                    text = " ".join(
                        (b.text if isinstance(b, TextBlock) else b.get("text", ""))
                        for b in text_blocks
                    )
                    openai_messages.append(
                        {
                            "role": "assistant",
                            "content": text or None,
                            "tool_calls": tc_list,
                        }
                    )
                else:
                    text = " ".join(
                        (b.text if isinstance(b, TextBlock) else b.get("text", ""))
                        for b in text_blocks
                    )
                    openai_messages.append({"role": role, "content": text})
            else:
                openai_messages.append({"role": role, "content": str(content)})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,  # type: ignore[arg-type]
            tools=openai_tools if openai_tools else None,  # type: ignore[arg-type]
            tool_choice=tc_str if openai_tools else None,  # type: ignore[arg-type]
            max_tokens=max_tokens,
        )

        message = response.choices[0].message
        content_blocks: list[TextBlock | ToolUseBlock] = []

        if message.content:
            content_blocks.append(TextBlock(text=message.content))

        if message.tool_calls:
            for tc_call in message.tool_calls:
                try:
                    input_dict = json.loads(tc_call.function.arguments)
                except Exception:
                    input_dict = {}
                content_blocks.append(
                    ToolUseBlock(
                        id=tc_call.id,
                        name=tc_call.function.name,
                        input=input_dict,
                    )
                )

        stop_reason = (
            "tool_use"
            if response.choices[0].finish_reason == "tool_calls"
            else "end_turn"
        )
        usage: dict = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }

        return LLMResponse(
            content=content_blocks, stop_reason=stop_reason, usage=usage
        )


class HumanProvider:
    """Interactive human provider — console input drives the tool-calling loop."""

    def __init__(self) -> None:
        self._system_shown = False

    async def send_message(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        tool_choice: dict | str,
        max_tokens: int,
    ) -> LLMResponse:
        if not self._system_shown:
            print("\n" + "=" * 60)
            print("SYSTEM PROMPT")
            print("=" * 60)
            print(system)
            print("=" * 60 + "\n")
            self._system_shown = True

        pack_names = self._extract_pack_names(messages)
        self._display_latest(messages)

        while True:
            try:
                raw = input("human> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                if pack_names:
                    print(f"  (Interrupted — auto-picking {pack_names[0]})")
                    return self._make_response(
                        "pick_card",
                        {"card_name": pack_names[0], "reasoning": "Interrupted."},
                    )
                raise

            if not raw:
                continue

            result = self._parse(raw, pack_names)
            if result is not None:
                return result
            print("  Unknown command. Type '?' for help.")

    # ── Display helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _display_latest(messages: list[dict]) -> None:
        """Print the content of the last user message."""
        for msg in reversed(messages):
            if msg["role"] != "user":
                continue
            content = msg["content"]
            if isinstance(content, str):
                print("\n" + content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        text = block.get("content", "")
                        if text:
                            print(f"\n{text}")
            return

    # ── Pack name extraction ───────────────────────────────────────────────────

    @staticmethod
    def _extract_pack_names(messages: list[dict]) -> list[str]:
        """Return ordered card names from the most recent pack listing in messages."""
        for msg in reversed(messages):
            content = msg.get("content")
            if isinstance(content, str) and "Current pack" in content:
                names = HumanProvider._parse_pack_names(content)
                if names:
                    return names
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        text = block.get("content", "")
                        if "Current pack" in text:
                            names = HumanProvider._parse_pack_names(text)
                            if names:
                                return names
        return []

    @staticmethod
    def _parse_pack_names(text: str) -> list[str]:
        """Parse numbered card names from a formatted pack listing. Handles DFCs."""
        names: list[str] = []
        pending: str | None = None
        await_face2 = False

        for line in text.splitlines():
            s = line.strip()
            m = re.match(r"^(\d+)\.\s+\[[A-Z?]\]\s+(.+)$", s)
            if m:
                if pending is not None:
                    names.append(pending)
                raw = m.group(2).strip()
                pending = re.sub(r"\s+(\{[^}]+\})+$", "", raw).strip()
                await_face2 = False
            elif s == "//" and pending is not None:
                await_face2 = True
            elif await_face2 and pending is not None:
                m2 = re.match(r"^\[[A-Z?]\]\s+(.+)$", s)
                if m2:
                    raw2 = m2.group(1).strip()
                    face2 = re.sub(r"\s+(\{[^}]+\})+$", "", raw2).strip()
                    pending = pending + " // " + face2
                    await_face2 = False

        if pending is not None:
            names.append(pending)
        return names

    # ── Command parsing ────────────────────────────────────────────────────────

    def _parse(self, raw: str, pack_names: list[str]) -> LLMResponse | None:
        parts = raw.split(None, 1)
        cmd = parts[0].lower()
        rest = parts[1].strip() if len(parts) > 1 else ""

        # Bare number → pick by position
        if raw.isdigit():
            return self._pick_by_index(int(raw) - 1, pack_names)

        if cmd in ("pick", "p"):
            target = rest
            if target.isdigit():
                return self._pick_by_index(int(target) - 1, pack_names)
            if not target:
                print("  Usage: pick <name or number>")
                return None
            reasoning = self._ask_reasoning(target)
            return self._make_response("pick_card", {"card_name": target, "reasoning": reasoning})

        if cmd in ("view", "v", "pack", "vp"):
            return self._make_response("view_current_pack", {})

        if cmd in ("my", "picks"):
            valid = {"color", "type", "cmc", "pick_order"}
            group = rest if rest in valid else ("pick_order" if rest not in ("order",) else "pick_order")
            return self._make_response("view_my_picks", {"group_by": group})

        if cmd in ("lookup", "l", "search", "find"):
            if not rest:
                print("  Usage: lookup <card name>")
                return None
            return self._make_response("lookup_card", {"card_name": rest})

        if cmd in ("move", "mv"):
            move_parts = rest.rsplit(None, 1)
            if len(move_parts) < 2:
                print("  Usage: move <card name> deck|side")
                return None
            card_name, dest_raw = move_parts[0], move_parts[1].lower()
            dest = "sideboard" if dest_raw in ("side", "sideboard", "sb") else "deck"
            return self._make_response("move_card", {"card_name": card_name, "destination": dest})

        if cmd in ("note", "n"):
            if not rest:
                print("  Usage: note <text>")
                return None
            return self._make_response("add_note", {"note": rest})

        if cmd == "add":
            return self._parse_land_cmd("add_basic_land", rest)

        if cmd in ("remove", "rm"):
            return self._parse_land_cmd("remove_basic_land", rest)

        if cmd in ("done", "finalize", "finish"):
            notes = rest or ""
            return self._make_response("finalize_deck", {"notes": notes})

        if cmd in ("help", "h", "?"):
            self._print_help(pack_names)
            return None

        return None

    def _parse_land_cmd(self, tool_name: str, rest: str) -> LLMResponse | None:
        """Parse 'add|remove <land_type> [count]' into a tool response."""
        _LAND_ALIASES = {
            "plains": "Plains", "island": "Island", "swamp": "Swamp",
            "mountain": "Mountain", "forest": "Forest",
            "w": "Plains", "u": "Island", "b": "Swamp", "r": "Mountain", "g": "Forest",
        }
        parts = rest.split()
        if not parts:
            verb = "add" if tool_name == "add_basic_land" else "remove"
            print(f"  Usage: {verb} <land_type> [count]")
            return None

        # Last token may be a count
        count = 1
        if len(parts) >= 2 and parts[-1].isdigit():
            count = int(parts[-1])
            land_raw = " ".join(parts[:-1]).lower().strip()
        else:
            land_raw = " ".join(parts).lower().strip()

        land_type = _LAND_ALIASES.get(land_raw)
        if land_type is None:
            print(
                f"  Unknown land type '{land_raw}'. "
                f"Use: plains, island, swamp, mountain, forest"
            )
            return None
        return self._make_response(tool_name, {"land_type": land_type, "count": count})

    def _pick_by_index(self, idx: int, pack_names: list[str]) -> LLMResponse | None:
        if not pack_names:
            print("  No pack cards found — type the card name instead.")
            return None
        if not (0 <= idx < len(pack_names)):
            print(f"  Invalid number. Pack has {len(pack_names)} card(s).")
            return None
        name = pack_names[idx]
        reasoning = self._ask_reasoning(name)
        return self._make_response("pick_card", {"card_name": name, "reasoning": reasoning})

    @staticmethod
    def _ask_reasoning(card_name: str) -> str:
        raw = input(f"  Reasoning for {card_name!r} (Enter to skip): ").strip()
        return raw or f"Picked {card_name}."

    # ── Response factory ───────────────────────────────────────────────────────

    @staticmethod
    def _make_response(tool_name: str, tool_input: dict) -> LLMResponse:
        block = ToolUseBlock(
            id=f"human_{uuid.uuid4().hex[:8]}",
            name=tool_name,
            input=tool_input,
        )
        return LLMResponse(
            content=[block],
            stop_reason="tool_use",
            usage={"input_tokens": 0, "output_tokens": 0},
        )

    @staticmethod
    def _print_help(pack_names: list[str]) -> None:
        pack_info = (
            f"  Pack cards: {', '.join(f'{i+1}={n}' for i, n in enumerate(pack_names))}\n"
            if pack_names else ""
        )
        print(f"""
{pack_info}  Commands:
    <N>                          pick card number N
    pick <N or name>             pick a card (prompts for reasoning)
    view  (or v)                 view current pack with full card text
    my [color|type|cmc|order]    view your drafted cards
    lookup <name>  (or l)        look up any card in the set
    move <name> deck|side        move card between deck and sideboard
    note <text>  (or n)          add a strategic note

  Deckbuilding:
    add <land_type> [count]      add basic land(s) to deck
    remove <land_type> [count]   remove basic land(s) from deck
    done [notes]                 finalize deck (also: finalize, finish)
    ?  help                      show this help
""")
