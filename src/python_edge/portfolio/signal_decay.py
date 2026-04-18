"""
signal_decay.py — Signal Decay: конфіг і математика для time-decay alpha score.

Логіка:
    effective_score = raw_score * exp(-age / tau)

де age = кількість календарних днів від signal_last_updated до today.

Два рівні виходу:
    score_floor   — мін effective_score щоб залишитись (soft decay exit)
    hard_cap_days — примусовий вихід незалежно від score (hard expire)

Зауваження:
    - score_floor застосовується тільки якщо raw_score > 0 (long-side signal).
      Якщо raw_score <= 0 або відсутній — позиція не виходить через decay,
      щоб не конфліктувати з логікою short/neutral.
    - hard_cap_days рахується від entry_date, а не від signal_last_updated,
      щоб захистити від "зомбі-позицій" де сигнал постійно оновлюється
      але позиція реально стара.
    - Якщо signal_last_updated відсутній — age вважається від entry_date.

Використання:
    config = DecayConfig.from_env("optimal")
    result = config.evaluate(
        symbol="NVDA",
        raw_score=0.0031,
        signal_last_updated="2026-04-14",
        entry_date="2026-04-01",
        today="2026-04-18",
    )
    # result.reason: None | "decay_exit" | "hard_expire"
    # result.effective_score: float
    # result.signal_age_days: int
    # result.hard_age_days:   int
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import date
from typing import Optional


# ──────────────────────────────────────────────────────────────
# Дефолтні конфіги per-config
# ──────────────────────────────────────────────────────────────
_DEFAULTS: dict[str, dict] = {
    "optimal": {
        "tau_days":       4.0,
        "hard_cap_days":  15,
        "score_floor":    0.05,
        "enabled":        True,
    },
    "aggressive": {
        "tau_days":       3.0,
        "hard_cap_days":  10,
        "score_floor":    0.08,
        "enabled":        True,
    },
}

_DEFAULT_FALLBACK: dict = {
    "tau_days":       4.0,
    "hard_cap_days":  15,
    "score_floor":    0.05,
    "enabled":        True,
}


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _trading_days_between(date_str: str, today_str: str) -> int:
    try:
        d1 = date.fromisoformat(str(date_str)[:10])
        d2 = date.fromisoformat(str(today_str)[:10])
        if d2 <= d1:
            return 0
        total = (d2 - d1).days
        # прибираємо вихідні: ~5/7 від календарних днів
        full_weeks, remainder = divmod(total, 7)
        trading = full_weeks * 5
        # для залишку рахуємо вручну
        current = d1
        for _ in range(remainder):
            current = date.fromordinal(current.toordinal() + 1)
            if current.weekday() < 5:  # 0=Mon..4=Fri
                trading += 1
        return max(0, trading)
    except (ValueError, TypeError):
        return 0


def _parse_float(val: str | None, fallback: float) -> float:
    if val is None or str(val).strip() == "":
        return fallback
    try:
        return float(val)
    except ValueError:
        return fallback


def _parse_int(val: str | None, fallback: int) -> int:
    if val is None or str(val).strip() == "":
        return fallback
    try:
        return int(val)
    except ValueError:
        return fallback


def _parse_bool(val: str | None, fallback: bool) -> bool:
    if val is None:
        return fallback
    v = str(val).strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return fallback


# ──────────────────────────────────────────────────────────────
# DecayResult
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DecayResult:
    """Результат оцінки decay для однієї позиції."""
    reason:           Optional[str]   # None | "decay_exit" | "hard_expire"
    effective_score:  float           # raw_score * exp(-age / tau)
    raw_score:        float
    signal_age_days:  int             # days since signal_last_updated
    hard_age_days:    int             # days since entry_date
    decay_multiplier: float           # exp(-age / tau)
    tau_days:         float
    score_floor:      float
    hard_cap_days:    int


# ──────────────────────────────────────────────────────────────
# DecayConfig
# ──────────────────────────────────────────────────────────────
@dataclass
class DecayConfig:
    """Конфіг і логіка signal decay для одного config."""

    config_name:   str
    tau_days:      float        # half-life (~= tau * ln2 ≈ 0.693 * tau)
    hard_cap_days: int          # примусовий вихід від entry_date
    score_floor:   float        # мін effective_score (0 = вимкнено)
    enabled:       bool         # False = decay повністю вимкнений

    @classmethod
    def from_env(cls, config_name: str) -> "DecayConfig":
        """
        Читає параметри з env-змінних з prefix DECAY_{CONFIG}_.
        Наприклад:
            DECAY_OPTIMAL_TAU_DAYS=4.0
            DECAY_OPTIMAL_HARD_CAP_DAYS=15
            DECAY_OPTIMAL_SCORE_FLOOR=0.05
            DECAY_OPTIMAL_ENABLED=1
        """
        prefix   = f"DECAY_{config_name.upper()}_"
        defaults = _DEFAULTS.get(config_name.lower(), _DEFAULT_FALLBACK)

        tau_days = _parse_float(
            os.getenv(f"{prefix}TAU_DAYS"),
            float(defaults["tau_days"]),
        )
        hard_cap_days = _parse_int(
            os.getenv(f"{prefix}HARD_CAP_DAYS"),
            int(defaults["hard_cap_days"]),
        )
        score_floor = _parse_float(
            os.getenv(f"{prefix}SCORE_FLOOR"),
            float(defaults["score_floor"]),
        )
        enabled = _parse_bool(
            os.getenv(f"{prefix}ENABLED"),
            bool(defaults.get("enabled", True)),
        )

        # Санітація
        tau_days      = max(0.1, tau_days)
        hard_cap_days = max(1, hard_cap_days)
        score_floor   = max(0.0, score_floor)

        return cls(
            config_name=config_name,
            tau_days=tau_days,
            hard_cap_days=hard_cap_days,
            score_floor=score_floor,
            enabled=enabled,
        )

    def compute_effective_score(self, raw_score: float, signal_age_days: int) -> tuple[float, float]:
        """
        Повертає (effective_score, decay_multiplier).
        effective_score = raw_score * exp(-age / tau)
        """
        if self.tau_days <= 0:
            return raw_score, 1.0
        multiplier = math.exp(-signal_age_days / self.tau_days)
        return float(raw_score * multiplier), float(multiplier)

    def evaluate(
        self,
        symbol: str,
        raw_score: float,
        signal_last_updated: Optional[str],
        entry_date: Optional[str],
        today: str,
    ) -> DecayResult:
        """
        Оцінює decay для позиції.

        Повертає DecayResult з reason:
          None          — позиція залишається
          "decay_exit"  — effective_score впав нижче score_floor
          "hard_expire" — позиція тримається >= hard_cap_days від entry_date

        Примітки:
          - Якщо decay вимкнений — завжди повертає reason=None
          - Якщо raw_score <= 0 — score_floor не застосовується
            (уникаємо хибного виходу для neutral/short сигналів)
          - hard_expire завжди активний якщо enabled=True і hard_cap_days > 0
        """
        # Fallback для дат
        anchor = signal_last_updated or entry_date or today
        entry  = entry_date or today

        signal_age_days = _days_between(anchor, today)
        hard_age_days   = _days_between(entry, today)

        effective_score, multiplier = self.compute_effective_score(raw_score, signal_age_days)

        if not self.enabled:
            return DecayResult(
                reason=None,
                effective_score=effective_score,
                raw_score=raw_score,
                signal_age_days=signal_age_days,
                hard_age_days=hard_age_days,
                decay_multiplier=multiplier,
                tau_days=self.tau_days,
                score_floor=self.score_floor,
                hard_cap_days=self.hard_cap_days,
            )

        reason: Optional[str] = None

        # 1. Hard cap (від entry_date, незалежно від score)
        if self.hard_cap_days > 0 and hard_age_days >= self.hard_cap_days:
            reason = "hard_expire"

        # 2. Score floor (тільки якщо raw_score > 0)
        elif self.score_floor > 0 and raw_score > 0:
            if effective_score < self.score_floor:
                reason = "decay_exit"

        return DecayResult(
            reason=reason,
            effective_score=effective_score,
            raw_score=raw_score,
            signal_age_days=signal_age_days,
            hard_age_days=hard_age_days,
            decay_multiplier=multiplier,
            tau_days=self.tau_days,
            score_floor=self.score_floor,
            hard_cap_days=self.hard_cap_days,
        )

    def describe(self, result: DecayResult) -> str:
        """Рядок для логування."""
        return (
            f"decay: raw={result.raw_score:.5f} "
            f"eff={result.effective_score:.5f} "
            f"mult={result.decay_multiplier:.3f} "
            f"sig_age={result.signal_age_days}d "
            f"hard_age={result.hard_age_days}d/"
            f"{self.hard_cap_days}d "
            f"tau={self.tau_days}d "
            f"floor={self.score_floor:.4f}"
        )

    def __repr__(self) -> str:
        return (
            f"DecayConfig({self.config_name!r} "
            f"tau={self.tau_days}d hard_cap={self.hard_cap_days}d "
            f"floor={self.score_floor} enabled={self.enabled})"
        )
