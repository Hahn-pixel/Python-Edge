"""
exit_policy.py — Exit Strategy: конфіг і price-based логіка оцінки позицій.

Підтримує чотири механізми per-config:
  stop_loss_pct      — закрити якщо збиток > X%  (наприклад 0.08 = 8%)
  take_profit_pct    — закрити якщо прибуток > Y% (наприклад 0.20 = 20%)
  trailing_stop      — volatility-scaled: trail_pct = K × ivol_20d,
                       clamped до [trail_min, trail_max]
                       трейлінг від peak_price
  max_hold_days      — примусовий вихід після N днів (0 = вимкнено)

Значення None або 0 = механізм вимкнений.

Використання:
    policy = ExitPolicy.from_env("optimal")
    reason = policy.evaluate(
        symbol="OKLO",
        side=1,
        entry_price=40.0,
        current_price=35.0,
        peak_price=48.0,
        entry_date="2026-04-10",
        today="2026-04-16",
        ivol_20d=0.025,   # денна vol; None = trailing вимкнений
    )
    # reason: None | "stop_loss" | "take_profit" | "trailing_stop" | "max_hold"

Trailing stop розрахунок:
    trail_pct = clamp(K × ivol_20d, trail_min, trail_max)
    Для long:  спрацьовує якщо (peak - current) / peak >= trail_pct
    Для short: спрацьовує якщо (current - peak) / peak >= trail_pct
    де peak_price для short = мінімальна ціна з моменту входу
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from typing import Optional


# ──────────────────────────────────────────────────────────────
# Дефолтні конфіги per-config
# ──────────────────────────────────────────────────────────────
_DEFAULTS: dict[str, dict] = {
    "optimal": {
        "stop_loss_pct":   0.08,
        "take_profit_pct": 0.25,
        "trail_k":         1.5,
        "trail_min":       0.03,
        "trail_max":       0.20,
        "max_hold_days":   30,
    },
    "aggressive": {
        "stop_loss_pct":   0.10,
        "take_profit_pct": 0.30,
        "trail_k":         1.5,
        "trail_min":       0.03,
        "trail_max":       0.20,
        "max_hold_days":   20,
    },
}

_DEFAULT_FALLBACK: dict = {
    "stop_loss_pct":   0.08,
    "take_profit_pct": 0.25,
    "trail_k":         1.5,
    "trail_min":       0.03,
    "trail_max":       0.20,
    "max_hold_days":   30,
}


# ──────────────────────────────────────────────────────────────
# Env helpers
# ──────────────────────────────────────────────────────────────
def _parse_optional_float(val: str | None) -> Optional[float]:
    if val is None or val.strip() in ("", "0"):
        return None
    try:
        f = float(val)
        return f if f > 0 else None
    except ValueError:
        return None


def _parse_optional_int(val: str | None) -> Optional[int]:
    if val is None or val.strip() in ("", "0"):
        return None
    try:
        i = int(val)
        return i if i > 0 else None
    except ValueError:
        return None


def _days_between(date_str: str, today_str: str) -> int:
    try:
        d1 = date.fromisoformat(date_str[:10])
        d2 = date.fromisoformat(today_str[:10])
        return max(0, (d2 - d1).days)
    except (ValueError, TypeError):
        return 0


# ──────────────────────────────────────────────────────────────
# ExitPolicy
# ──────────────────────────────────────────────────────────────
@dataclass
class ExitPolicy:
    """Конфіг і логіка price-based exit для одного config."""

    config_name:     str
    stop_loss_pct:   Optional[float]
    take_profit_pct: Optional[float]
    trail_k:         float             # множник волатильності
    trail_min:       float             # нижня межа trail_pct
    trail_max:       float             # верхня межа trail_pct
    max_hold_days:   Optional[int]

    @classmethod
    def from_env(cls, config_name: str) -> "ExitPolicy":
        """
        Читає параметри з env-змінних з prefix EXIT_{CONFIG}_.
        Наприклад: EXIT_OPTIMAL_STOP_LOSS_PCT=0.08
                   EXIT_OPTIMAL_TRAIL_K=1.5
                   EXIT_OPTIMAL_TRAIL_MIN=0.03
                   EXIT_OPTIMAL_TRAIL_MAX=0.20
        """
        prefix   = f"EXIT_{config_name.upper()}_"
        defaults = _DEFAULTS.get(config_name.lower(), _DEFAULT_FALLBACK)

        def env_float(key: str) -> Optional[float]:
            raw = os.getenv(f"{prefix}{key.upper()}")
            if raw is not None:
                return _parse_optional_float(raw)
            return defaults.get(key.lower())

        def env_float_req(key: str, fallback: float) -> float:
            v = env_float(key)
            return v if v is not None else fallback

        def env_int(key: str) -> Optional[int]:
            raw = os.getenv(f"{prefix}{key.upper()}")
            if raw is not None:
                return _parse_optional_int(raw)
            v = defaults.get(key.lower())
            return int(v) if v else None

        return cls(
            config_name=config_name,
            stop_loss_pct=env_float("stop_loss_pct"),
            take_profit_pct=env_float("take_profit_pct"),
            trail_k=env_float_req("trail_k", _DEFAULT_FALLBACK["trail_k"]),
            trail_min=env_float_req("trail_min", _DEFAULT_FALLBACK["trail_min"]),
            trail_max=env_float_req("trail_max", _DEFAULT_FALLBACK["trail_max"]),
            max_hold_days=env_int("max_hold_days"),
        )

    def compute_trail_pct(self, ivol_20d: Optional[float]) -> Optional[float]:
        """
        Повертає trail_pct = clamp(K × ivol_20d, trail_min, trail_max).
        None якщо ivol_20d недоступний — trailing пропускається.
        """
        if ivol_20d is None or ivol_20d <= 0:
            return None
        raw = self.trail_k * ivol_20d
        return max(self.trail_min, min(self.trail_max, raw))

    def evaluate(
        self,
        symbol: str,
        side: float,
        entry_price: float,
        current_price: float,
        peak_price: float,
        entry_date: str,
        today: str,
        ivol_20d: Optional[float] = None,
    ) -> Optional[str]:
        """
        Перевіряє всі exit-умови для позиції.
        Повертає назву першого спрацьованого механізму або None.

        Порядок пріоритету: stop_loss → trailing_stop → take_profit → max_hold

        side:     +1 = long, -1 = short
        ivol_20d: денна vol для volatility-scaled trailing
        """
        if side == 0.0 or entry_price <= 0 or current_price <= 0:
            return None

        pnl_pct = (current_price - entry_price) / entry_price * float(side)

        # 1. Stop loss
        if self.stop_loss_pct is not None:
            if pnl_pct <= -self.stop_loss_pct:
                return "stop_loss"

        # 2. Volatility-scaled trailing stop
        trail_pct = self.compute_trail_pct(ivol_20d)
        if trail_pct is not None and peak_price > 0:
            if side > 0:
                drawdown_from_peak = (peak_price - current_price) / peak_price
            else:
                drawdown_from_peak = (current_price - peak_price) / peak_price
            if drawdown_from_peak >= trail_pct:
                return "trailing_stop"

        # 3. Take profit
        if self.take_profit_pct is not None:
            if pnl_pct >= self.take_profit_pct:
                return "take_profit"

        # 4. Max hold days
        if self.max_hold_days is not None:
            if _days_between(entry_date, today) >= self.max_hold_days:
                return "max_hold"

        return None

    def describe_trail(self, ivol_20d: Optional[float]) -> str:
        """Рядок для логування."""
        trail_pct = self.compute_trail_pct(ivol_20d)
        if trail_pct is None:
            return "trail=disabled(no_ivol)"
        raw = self.trail_k * ivol_20d if ivol_20d else 0.0
        clamped = abs(trail_pct - raw) > 1e-9
        return (
            f"trail={trail_pct:.2%}"
            f"(K={self.trail_k}x ivol={ivol_20d:.4f} raw={raw:.2%}"
            f"{', CLAMPED' if clamped else ''})"
        )

    def __repr__(self) -> str:
        return (
            f"ExitPolicy({self.config_name!r} "
            f"sl={self.stop_loss_pct} tp={self.take_profit_pct} "
            f"trail_k={self.trail_k} trail=[{self.trail_min:.0%},{self.trail_max:.0%}] "
            f"max_hold={self.max_hold_days}d)"
        )
