"""17Lands ratings — GIH WR data for bot logic and evaluation (never exposed to LLM)."""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

USER_AGENT = "MTG-Draft-LLM-Harness/1.0"
SEVENTEEN_LANDS_URL = (
    "https://www.17lands.com/card_ratings/data?expansion={set_code}&format=PremierDraft"
)
FALLBACK_RATINGS = {"mythic": 0.62, "rare": 0.58, "uncommon": 0.54, "common": 0.52}
CACHE_MAX_AGE_HOURS = 24


class Ratings:
    def __init__(self, cache_dir: str = "~/.mtg-draft-harness") -> None:
        self.cache_dir = Path(os.path.expanduser(cache_dir))
        self._ratings: dict[str, dict[str, float]] = {}  # set_code -> {name -> WR}

    def _is_stale(self, path: Path) -> bool:
        if not path.exists():
            return True
        age_seconds = time.time() - path.stat().st_mtime
        return age_seconds > CACHE_MAX_AGE_HOURS * 3600

    async def ensure_ratings(self, set_code: str) -> None:
        set_code = set_code.upper()
        set_dir = self.cache_dir / "sets" / set_code
        set_dir.mkdir(parents=True, exist_ok=True)
        cache_path = set_dir / "17lands_ratings.json"

        if not self._is_stale(cache_path):
            logger.debug(f"Using cached 17Lands ratings for {set_code}")
            with open(cache_path, encoding="utf-8") as f:
                self._ratings[set_code] = json.load(f)
            return

        logger.info(f"Downloading 17Lands ratings for {set_code}...")
        try:
            url = SEVENTEEN_LANDS_URL.format(set_code=set_code)
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url, headers={"User-Agent": USER_AGENT})
                resp.raise_for_status()
                data = resp.json()

            ratings: dict[str, float] = {}
            for card in data:
                name = card.get("name")
                wr = card.get("ever_drawn_win_rate")
                if name and wr is not None:
                    ratings[name] = float(wr)

            self._ratings[set_code] = ratings
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(ratings, f)
            logger.info(f"Cached {len(ratings)} ratings for {set_code}")

        except Exception as e:
            logger.warning(f"Failed to download 17Lands ratings for {set_code}: {e}")
            if cache_path.exists():
                with open(cache_path, encoding="utf-8") as f:
                    self._ratings[set_code] = json.load(f)
                logger.info(f"Using stale cached ratings for {set_code}")
            else:
                logger.warning(f"No ratings available for {set_code}; using fallback")
                self._ratings[set_code] = {}

    def get_rating(self, card_name: str, set_code: str) -> float | None:
        return self._ratings.get(set_code.upper(), {}).get(card_name)

    def get_all_ratings(self, set_code: str) -> dict[str, float]:
        return dict(self._ratings.get(set_code.upper(), {}))
