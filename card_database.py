"""Card database — Scryfall + MTGJSON data management."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

USER_AGENT = "MTG-Draft-LLM-Harness/1.0 (contact: llm-draft)"
SCRYFALL_BULK_URL = "https://api.scryfall.com/bulk-data/oracle-cards"
SCRYFALL_SEARCH_URL = "https://api.scryfall.com/cards/search"
MTGJSON_URL = "https://mtgjson.com/api/v5/{set_code}.json"
CACHE_MAX_AGE_DAYS = 7


@dataclass
class Card:
    name: str
    mana_cost: str
    cmc: float
    type_line: str
    oracle_text: str
    colors: list[str]
    color_identity: list[str]
    rarity: str
    power: str | None
    toughness: str | None
    set_code: str
    collector_number: str
    scryfall_id: str
    card_faces: list[dict] | None
    keywords: list[str]


class CardDatabase:
    def __init__(self, cache_dir: str = "~/.mtg-draft-harness") -> None:
        self.cache_dir = Path(os.path.expanduser(cache_dir))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._oracle_cards: dict[str, Any] = {}  # name -> scryfall data
        self._set_cards: dict[str, list[Card]] = {}  # set_code -> cards
        self._mtgjson_cache: dict[str, Any] = {}  # set_code -> mtgjson data
        self._mtgjson_uuid_to_card: dict[str, Card] = {}  # uuid -> Card

    def _is_stale(self, path: Path, max_age_days: int = CACHE_MAX_AGE_DAYS) -> bool:
        if not path.exists():
            return True
        age_seconds = time.time() - path.stat().st_mtime
        return age_seconds > max_age_days * 86400

    async def _download_json(self, url: str) -> Any:
        headers = {"User-Agent": USER_AGENT}
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            return resp.json()

    async def _ensure_oracle_bulk(self) -> None:
        cache_path = self.cache_dir / "scryfall_oracle_cards.json"
        if not self._is_stale(cache_path):
            logger.debug("Using cached Scryfall oracle bulk data")
            with open(cache_path, encoding="utf-8") as f:
                data = json.load(f)
        else:
            logger.info("Downloading Scryfall oracle bulk data...")
            meta = await self._download_json(SCRYFALL_BULK_URL)
            download_url = meta["download_uri"]
            async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
                resp = await client.get(download_url, headers={"User-Agent": USER_AGENT})
                resp.raise_for_status()
                data = resp.json()
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            logger.info(f"Cached {len(data)} oracle cards")

        for card in data:
            self._oracle_cards[card["name"]] = card

    async def _ensure_set_scryfall(self, set_code: str) -> None:
        set_dir = self.cache_dir / "sets" / set_code
        set_dir.mkdir(parents=True, exist_ok=True)
        cache_path = set_dir / "scryfall_cards.json"

        if cache_path.exists():
            logger.debug(f"Using cached Scryfall set data for {set_code}")
            with open(cache_path, encoding="utf-8") as f:
                cards_data = json.load(f)
        else:
            logger.info(f"Downloading Scryfall set data for {set_code}...")
            cards_data: list[dict] = []
            url: str | None = (
                f"{SCRYFALL_SEARCH_URL}?q=set:{set_code.lower()}+is:booster&page=1"
            )
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                while url:
                    resp = await client.get(url, headers={"User-Agent": USER_AGENT})
                    resp.raise_for_status()
                    page = resp.json()
                    cards_data.extend(page.get("data", []))
                    if page.get("has_more"):
                        url = page.get("next_page")
                        await asyncio.sleep(0.1)
                    else:
                        url = None
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cards_data, f)
            logger.info(f"Cached {len(cards_data)} cards for set {set_code}")

        cards = [self._parse_scryfall_card(c) for c in cards_data]
        self._set_cards[set_code] = cards

    def _parse_scryfall_card(self, data: dict) -> Card:
        card_faces = data.get("card_faces")
        oracle_text = data.get("oracle_text", "")
        mana_cost = data.get("mana_cost", "")

        if card_faces and not oracle_text:
            oracle_text = " // ".join(f.get("oracle_text", "") for f in card_faces)
        if card_faces and not mana_cost:
            mana_cost = card_faces[0].get("mana_cost", "")

        return Card(
            name=data["name"],
            mana_cost=mana_cost,
            cmc=float(data.get("cmc", 0.0)),
            type_line=data.get("type_line", ""),
            oracle_text=oracle_text,
            colors=data.get("colors", []),
            color_identity=data.get("color_identity", []),
            rarity=data.get("rarity", "common"),
            power=data.get("power"),
            toughness=data.get("toughness"),
            set_code=data.get("set", "").upper(),
            collector_number=data.get("collector_number", ""),
            scryfall_id=data.get("id", ""),
            card_faces=card_faces,
            keywords=data.get("keywords", []),
        )

    async def _ensure_mtgjson(self, set_code: str) -> None:
        set_dir = self.cache_dir / "sets" / set_code
        set_dir.mkdir(parents=True, exist_ok=True)
        cache_path = set_dir / "mtgjson.json"

        if not self._is_stale(cache_path):
            logger.debug(f"Using cached MTGJSON data for {set_code}")
            with open(cache_path, encoding="utf-8") as f:
                data = json.load(f)
        else:
            logger.info(f"Downloading MTGJSON data for {set_code}...")
            try:
                url = MTGJSON_URL.format(set_code=set_code)
                data = await self._download_json(url)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                logger.info(f"Cached MTGJSON data for {set_code}")
            except Exception as e:
                logger.warning(f"Failed to download MTGJSON for {set_code}: {e}")
                data = {}

        self._mtgjson_cache[set_code] = data

    def _build_uuid_mapping(self, set_code: str) -> None:
        """Map MTGJSON card UUIDs to Scryfall Card objects."""
        mtgjson_data = self._mtgjson_cache.get(set_code, {})
        mtgjson_set = mtgjson_data.get("data", {})
        if not mtgjson_set:
            return

        # Build lookup: (name, collector_number) -> Card
        set_cards_by_name_num: dict[tuple[str, str], Card] = {}
        set_cards_by_name: dict[str, Card] = {}
        for card in self._set_cards.get(set_code, []):
            set_cards_by_name_num[(card.name, card.collector_number)] = card
            set_cards_by_name[card.name] = card

        for mtgjson_card in mtgjson_set.get("cards", []):
            uuid = mtgjson_card.get("uuid")
            name = mtgjson_card.get("name")
            number = str(mtgjson_card.get("number", ""))

            if not uuid or not name:
                continue

            # Prefer exact name+number match, fall back to name-only
            scryfall_card = set_cards_by_name_num.get(
                (name, number)
            ) or set_cards_by_name.get(name)
            if scryfall_card:
                self._mtgjson_uuid_to_card[uuid] = scryfall_card

    async def ensure_set_data(self, set_code: str) -> None:
        set_code = set_code.upper()
        await self._ensure_oracle_bulk()
        await self._ensure_set_scryfall(set_code)
        await self._ensure_mtgjson(set_code)
        self._build_uuid_mapping(set_code)

    def get_card_by_name(self, name: str, set_code: str) -> Card | None:
        name_lower = name.lower()
        for card in self._set_cards.get(set_code.upper(), []):
            if card.name.lower() == name_lower:
                return card
        return None

    def get_cards_for_set(self, set_code: str) -> list[Card]:
        return self._set_cards.get(set_code.upper(), [])

    def search_cards(self, query: str, set_code: str) -> list[Card]:
        query_lower = query.lower()
        return [
            c
            for c in self._set_cards.get(set_code.upper(), [])
            if query_lower in c.name.lower() or query_lower in c.oracle_text.lower()
        ]

    def get_set_mechanics(self, set_code: str) -> list[str]:
        keywords: set[str] = set()
        for card in self._set_cards.get(set_code.upper(), []):
            keywords.update(card.keywords)
        return sorted(keywords)

    def format_card_for_llm(self, card: Card) -> str:
        rarity_map = {"common": "C", "uncommon": "U", "rare": "R", "mythic": "M"}
        rarity_initial = rarity_map.get(card.rarity, "?")

        if card.card_faces:
            faces_text: list[str] = []
            for face in card.card_faces:
                face_mana = face.get("mana_cost", "")
                face_name = face.get("name", card.name)
                face_type = face.get("type_line", "")
                face_power = face.get("power")
                face_toughness = face.get("toughness")
                face_oracle = face.get("oracle_text", "")

                header = f"[{rarity_initial}] {face_name}"
                if face_mana:
                    header += f" {face_mana}"

                type_str = f"    {face_type}"
                if face_power is not None and face_toughness is not None:
                    type_str += f" ({face_power}/{face_toughness})"

                lines = [header, type_str]
                if face_oracle:
                    for line in face_oracle.split("\n"):
                        lines.append(f"    {line}")
                faces_text.append("\n".join(lines))

            return "\n    //\n".join(faces_text)

        header = f"[{rarity_initial}] {card.name}"
        if card.mana_cost:
            header += f" {card.mana_cost}"

        type_str = f"    {card.type_line}"
        if card.power is not None and card.toughness is not None:
            type_str += f" ({card.power}/{card.toughness})"

        lines = [header, type_str]
        if card.oracle_text:
            for line in card.oracle_text.split("\n"):
                lines.append(f"    {line}")

        return "\n".join(lines)
