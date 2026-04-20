"""Stochastic training data generator with curriculum ordering."""

import hashlib, random, time
from typing import Optional

class TrainingCasino:
    TABLES = {
        "deadband_p0": ["P0: validate before scoring", "P0: reject if confidence < 0.3", "P0: content must be >= 10 chars"],
        "tile_ops": ["validate, score, dedup, store, search, rank", "tiles are discrete knowledge units", "tile pipeline has 14 steps"],
        "i2i": ["I2I protocol: git-based messaging", "bottles in for-fleet/ directories", "beachcomb every 30 minutes"],
        "constraint_theory": ["Pythagorean triples snap exactly", "zero drift, every machine", "CT is 4% faster than float"],
        "boundary": ["confidence threshold is 0.3", "keyword gating at 0.01 overlap", "temporal freshness: 7-day window"],
    }

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed or int(time.time() * 1000) % (2**32)
        self._rng = random.Random(self.seed)

    def _hash(self, table: str, index: int) -> int:
        return int(hashlib.sha256(f"{table}:{index}:{self.seed}".encode()).hexdigest(), 16)

    def deal(self, table: str, count: int = 1) -> list[str]:
        rows = self.TABLES.get(table, [])
        if not rows:
            return []
        dealt = []
        for i in range(count):
            idx = self._hash(table, i) % len(rows)
            dealt.append(rows[idx])
        return dealt

    def deal_curriculum(self, count: int = 10) -> list[tuple[str, str]]:
        order = list(self.TABLES.keys())
        self._rng.shuffle(order)
        result = []
        for table in order:
            for item in self.deal(table, max(1, count // len(order))):
                result.append((table, item))
        self._rng.shuffle(result)
        return result[:count]

    def weighted_deal(self, weights: dict[str, float], count: int = 5) -> list[tuple[str, str]]:
        tables = list(weights.keys())
        w = [weights.get(t, 1.0) for t in tables]
        result = []
        for _ in range(count):
            table = self._rng.choices(tables, weights=w, k=1)[0]
            items = self.deal(table, 1)
            if items:
                result.append((table, items[0]))
        return result

    @property
    def table_names(self) -> list[str]:
        return list(self.TABLES.keys())

    def table_size(self, table: str) -> int:
        return len(self.TABLES.get(table, []))
