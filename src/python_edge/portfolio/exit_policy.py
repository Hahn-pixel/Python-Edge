"""
exit_policy.py — Exit Strategy: конфіг і price-based логіка оцінки позицій.

Підтримує чотири механізми per-config:
  stop_loss_pct      — закрити якщо збиток > X%  (наприклад 0.08 = 8%)
  take_profit_pct    — закрити якщо прибуток > Y% (наприклад 0.20 = 20%)
  trailing_stop_pct  — трейлінг від peak_price     (наприклад 0.10 = 10%)
  max_hold_days      — примусовий вихід після N днів (0 = вимкнено)

Значення None або 0 = механізм вимкнений.

Використання:
    policy = ExitPolicy.from_env("optimal")
    reason = policy.evaluate(
        symbol="OKLO",
        side=1,               # +1 = long, -1 = short
        entry_price=40.0,
        current_price=35.0,
        peak_price=48.0,
        entry_date="2026-04-10",
        today="2026-04-16",
    )
    # reason: None | "stop_loss" | "take_profit" | "trailing_stop" | "max_hold"
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional


# ──────────────────────────────────────────────────────────────
# Дефолтні конфіги per-config
# Можна перевизначити через env-змінні (див. from_env)
# ──────────────────────────────────────────────────────────────
_DEFAULTS: dict[str, dict] = {
    "optimal": {
        "stop_loss_pct":     0.08,   # 8%
        "take_profit_pct":   0.25,   # 25%
        "trailing_stop_pct": 0.12,   # 12% від піку
        "max_hold_days":     30,     # 30 днів
    },
    "aggressive": {
        "stop_loss_pct":     0.10,   # 10%
        "take_profit_pct":   0.30,   # 30%
        "trailing_stop_pct": 0.15,   # 15% від піку
        "max_hold_days":     20,     # 20 днів
    },
}

_DEFAULT_FALLBACK: dict = {
    "stop_loss_pct":     0.08,
    "take_profit_pct":   0.25,
    "trailing_stop_pct": 0.12,
    "max_hold_days":     30,
}


def _parse_optional_float(val: str | None) -> Optional[float]:
    if val is None or val.strip() == "" or val.strip() == "0":
        return None
    try:
        f = float(val)
        return f if f > 0 else None
    except ValueError:
        return None


def _parse_optional_int(val: str | None) -> Optional[int]:
    if val is None or val.strip() == "" or val.strip() == "0":
        return None
    try:
        i = int(val)
        return i if i > 0 else None
    except ValueError:
        return None


def _days_between(date_str: str, today_str: str) -> int:
    """Повертає кількість днів між двома датами у форматі YYYY-MM-DD."""
    try:
        d1 = date.fromisoformat(date_str[:10])
        d2 = date.fromisoformat(today_str[:10])
        return max(0, (d2 - d1).days)
    except (ValueError, TypeError):
        return 0


@dataclass
class ExitPolicy:
    """Конфіг і логіка price-based exit для одного config."""

    config_name:       str
    stop_loss_pct:     Optional[float]  # None = вимкнено
    take_profit_pct:   Optional[float]
    trailing_stop_pct: Optional[float]
    max_hold_days:     Optional[int]

    # ── Фабрика з env-змінних ─────────────────────────────────
    @classmethod
    def from_env(cls, config_name: str) -> "ExitPolicy":
        """
        Читає параметри з env-змінних (з prefix EXIT_{CONFIG}_).
        Якщо змінна не задана — використовує дефолт для config.
        Наприклад: EXIT_OPTIMAL_STOP_LOSS_PCT=0.08
        """
        prefix = f"EXIT_{config_name.upper()}_"
        defaults = _DEFAULTS.get(config_name.lower(), _DEFAULT_FALLBACK)

        def env_float(key: str) -> Optional[float]:
            raw = os.getenv(f"{prefix}{key.upper()}")
            if raw is not None:
                return _parse_optional_float(raw)
            return defaults.get(key.lower())

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
            trailing_stop_pct=env_float("trailing_stop_pct"),
            max_hold_days=env_int("max_hold_days"),
        )

    # ── Оцінка позиції ────────────────────────────────────────
    def evaluate(
        self,
        symbol: str,
        side: float,
        entry_price: float,
        current_price: float,
        peak_price: float,
        entry_date: str,
        today: str,
    ) -> Optional[str]:
        """
        Перевіряє всі exit-умови для позиції.
        Повертає назву першого спрацьованого механізму або None.

        Порядок пріоритету: stop_loss → trailing_stop → take_profit → max_hold

        side: +1 = long, -1 = short
        """
        if side == 0.0 or entry_price <= 0 or current_price <= 0:
            return None

        # Розрахунок PnL% (для long: (current - entry) / entry)
        pnl_pct = (current_price - entry_price) / entry_price * float(side)

        # 1. Stop loss
        if self.stop_loss_pct is not None:
            if pnl_pct <= -self.stop_loss_pct:
                return "stop_loss"

        # 2. Trailing stop від peak_price
        if self.trailing_stop_pct is not None and peak_price > 0:
            # Для long: якщо впав від піку більше ніж trailing_stop_pct
            # Для short: peak_price = найнижча ціна (найвигідніша для шорту)
            if side > 0:
                drawdown_from_peak = (peak_price - current_price) / peak_price
            else:
                # short: peak_price = min price seen
                drawdown_from_peak = (current_price - peak_price) / peak_price
            if drawdown_from_peak >= self.trailing_stop_pct:
                return "trailing_stop"

        # 3. Take profit
        if self.take_profit_pct is not None:
            if pnl_pct >= self.take_profit_pct:
                return "take_profit"

        # 4. Max hold days
        if self.max_hold_days is not None:
            hold_days = _days_between(entry_date, today)
            if hold_days >= self.max_hold_days:
                return "max_hold"

        return None

    def __repr__(self) -> str:
        return (
            f"ExitPolicy({self.config_name!r} "
            f"sl={self.stop_loss_pct} tp={self.take_profit_pct} "
            f"trail={self.trailing_stop_pct} max_hold={self.max_hold_days}d)"
        )
