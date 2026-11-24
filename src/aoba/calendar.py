"""Flexible climate calendar & orbital insolation timing for AOBA.

Supports:
- ``gregorian``: Proleptic Gregorian calendar with standard leap years (365/366 days).
- ``noleap``: Fixed 365-day calendar (February always 28 days).
- ``360_day``: Idealized 360-day calendar (12 months x 30 days).

Also supports perpetual insolation experiments (``insolation_mode="perpetual_date"``)
and optional diurnal-cycle averaging (``diurnal_cycle=False``).

AOBA Formulation Reference: Sec. 1.3, Eqs. (1.1)-(1.4) (Orbital Solar Geometry & Calendar).
"""

from __future__ import annotations

from dataclasses import dataclass
import math


VALID_CALENDARS: tuple[str, ...] = ("gregorian", "noleap", "360_day")
VALID_INSOLATION_MODES: tuple[str, ...] = ("seasonal", "perpetual_date")

_MONTH_DAYS_NOLEAP: tuple[int, ...] = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
_MONTH_DAYS_LEAP: tuple[int, ...] = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


@dataclass(frozen=True)
class CalendarDate:
    """Instantaneous model calendar date and time-of-day."""

    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: int
    doy: int
    days_in_year: int

    @property
    def iso_date(self) -> str:
        return f"{self.year:04d}-{self.month:02d}-{self.day:02d}"

    @property
    def iso_datetime(self) -> str:
        return f"{self.year:04d}-{self.month:02d}-{self.day:02d} {self.hour:02d}:{self.minute:02d}:{self.second:02d}"


def parse_ymd(date_str: str, fallback: tuple[int, int, int] = (2000, 1, 1)) -> tuple[int, int, int]:
    """Parse ``YYYY-MM-DD`` (including negative BCE years like ``-19050-03-20``) safely."""
    s = str(date_str or "").strip()
    if not s:
        return fallback
    sign = -1 if s.startswith("-") else 1
    body = s[1:] if sign < 0 else s
    parts = body.split("-")
    if len(parts) != 3:
        return fallback
    try:
        y, m, d = sign * int(parts[0]), int(parts[1]), int(parts[2])
        m = max(1, min(12, m))
        d = max(1, min(31, d))
        return (y, m, d)
    except ValueError:
        return fallback


def is_leap_year(year: int, calendar_type: str = "gregorian") -> bool:
    """Return True if ``year`` is a leap year under ``calendar_type``."""
    if calendar_type != "gregorian":
        return False
    return (year % 4 == 0) and (year % 100 != 0 or year % 400 == 0)


def days_in_month(year: int, month: int, calendar_type: str = "gregorian") -> int:
    """Number of days in ``(year, month)`` under ``calendar_type``."""
    m = max(1, min(12, int(month)))
    if calendar_type == "360_day":
        return 30
    if is_leap_year(year, calendar_type):
        return _MONTH_DAYS_LEAP[m - 1]
    return _MONTH_DAYS_NOLEAP[m - 1]


def days_in_year(year: int, calendar_type: str = "gregorian") -> int:
    """Total number of days in ``year`` under ``calendar_type``."""
    if calendar_type == "360_day":
        return 360
    return 366 if is_leap_year(year, calendar_type) else 365


def _day_of_year_0based(year: int, month: int, day: int, calendar_type: str) -> int:
    m = max(1, min(12, month))
    d_max = days_in_month(year, m, calendar_type)
    d = max(1, min(d_max, day))
    total = 0
    for mm in range(1, m):
        total += days_in_month(year, mm, calendar_type)
    return total + (d - 1)


def _add_days(year: int, month: int, day: int, delta_days: int, calendar_type: str) -> tuple[int, int, int, int]:
    """Add ``delta_days >= 0`` to ``(year, month, day)`` and return ``(y, m, d, doy_1based)``."""
    doy0 = _day_of_year_0based(year, month, day, calendar_type) + delta_days
    y = year
    while doy0 >= days_in_year(y, calendar_type):
        doy0 -= days_in_year(y, calendar_type)
        y += 1
    while doy0 < 0:
        y -= 1
        doy0 += days_in_year(y, calendar_type)

    rem = doy0
    m = 1
    while m < 12:
        dim = days_in_month(y, m, calendar_type)
        if rem < dim:
            break
        rem -= dim
        m += 1
    return y, m, rem + 1, doy0 + 1


@dataclass
class AobaCalendar:
    """Calendar state & orbital-time mapper for an AOBA simulation."""

    start_date: str = "2000-03-20"
    calendar_type: str = "gregorian"
    insolation_mode: str = "seasonal"
    fixed_insolation_date: str = "2000-03-20"
    diurnal_cycle: bool = True
    seconds_per_day: float = 86400.0

    def __post_init__(self) -> None:
        if self.calendar_type not in VALID_CALENDARS:
            self.calendar_type = "gregorian"
        if self.insolation_mode not in VALID_INSOLATION_MODES:
            self.insolation_mode = "seasonal"
        y, m, d = parse_ymd(self.start_date, (2000, 1, 1))
        d = min(d, days_in_month(y, m, self.calendar_type))
        self.start_year, self.start_month, self.start_day = y, m, d
        self.start_date = f"{y:04d}-{m:02d}-{d:02d}"

        fy, fm, fd = parse_ymd(self.fixed_insolation_date, (2000, 3, 20))
        fd = min(fd, days_in_month(fy, fm, self.calendar_type))
        self.fixed_year, self.fixed_month, self.fixed_day = fy, fm, fd
        self.fixed_insolation_date = f"{fy:04d}-{fm:02d}-{fd:02d}"

    def date_at(self, sim_time_sec: float) -> CalendarDate:
        """Return calendar date/time corresponding to ``sim_time_sec`` elapsed from ``start_date 00:00:00``."""
        spd = max(float(self.seconds_per_day), 1.0)
        t_clamped = max(float(sim_time_sec), 0.0)
        total_days = int(math.floor(t_clamped / spd + 1.0e-9))
        sec_in_day = t_clamped - total_days * spd
        if sec_in_day < 0.0:
            sec_in_day = 0.0
        sec_int = int(round(sec_in_day))
        if sec_int >= int(round(spd)):
            total_days += 1
            sec_int = 0

        y, m, d, doy = _add_days(self.start_year, self.start_month, self.start_day, total_days, self.calendar_type)
        hour = sec_int // 3600
        minute = (sec_int % 3600) // 60
        second = sec_int % 60
        return CalendarDate(
            year=y,
            month=m,
            day=d,
            hour=hour,
            minute=minute,
            second=second,
            doy=doy,
            days_in_year=days_in_year(y, self.calendar_type),
            )

    def decimal_year_at(self, sim_time_sec: float) -> float:
        """Return continuous decimal calendar year (e.g. ``2025.50``) at ``sim_time_sec``."""
        cd = self.date_at(sim_time_sec)
        spd = max(float(self.seconds_per_day), 1.0)
        frac_day = (cd.hour * 3600.0 + cd.minute * 60.0 + cd.second) / spd
        return float(cd.year) + (float(cd.doy - 1) + frac_day) / float(max(cd.days_in_year, 1))

    def days_for_years(self, years: int) -> int:
        """Return exact number of calendar days for ``years`` starting from ``start_date``."""
        total = 0
        y0 = self.start_year
        for k in range(max(int(years), 1)):
            total += days_in_year(y0 + k, self.calendar_type)
        return total

    def days_until(self, end_date_str: str) -> int | None:
        """Return integer days from ``start_date`` to ``end_date_str`` (inclusive of start, exclusive of end), or None."""
        s = str(end_date_str or "").strip()
        if not s:
            return None
        ey, em, ed = parse_ymd(s, (self.start_year, self.start_month, self.start_day))
        ed = min(ed, days_in_month(ey, em, self.calendar_type))
        if (ey, em, ed) <= (self.start_year, self.start_month, self.start_day):
            return 1
        total = 0
        if ey == self.start_year:
            return max(
                1,
                _day_of_year_0based(ey, em, ed, self.calendar_type)
                - _day_of_year_0based(self.start_year, self.start_month, self.start_day, self.calendar_type),
            )
        total += days_in_year(self.start_year, self.calendar_type) - _day_of_year_0based(
            self.start_year, self.start_month, self.start_day, self.calendar_type
        )
        for y in range(self.start_year + 1, ey):
            total += days_in_year(y, self.calendar_type)
        total += _day_of_year_0based(ey, em, ed, self.calendar_type)
        return max(1, total)

    def _year_fraction_from_vernal_equinox(self, year: int, doy_1based: int, frac_day: float) -> float:
        diy = float(days_in_year(year, self.calendar_type))
        if self.calendar_type == "360_day":
            ve_doy0 = 80.0
        elif is_leap_year(year, self.calendar_type):
            ve_doy0 = 79.0  # March 20 in leap year (0-based)
        else:
            ve_doy0 = 78.0  # March 20 in common year (0-based)
        return ((float(doy_1based - 1) + frac_day - ve_doy0) % diy) / diy

    def orbital_times_sec(self, sim_time_sec: float, tropical_year_sec: float) -> tuple[float, float]:
        """Return ``(t_season_sec, t_diurnal_sec)`` relative to Vernal Equinox ``00:00 UTC``."""
        spd = max(float(self.seconds_per_day), 1.0)
        frac_day = (float(sim_time_sec) % spd) / spd
        if self.insolation_mode == "perpetual_date":
            doy_fix = _day_of_year_0based(self.fixed_year, self.fixed_month, self.fixed_day, self.calendar_type) + 1
            frac_yr = self._year_fraction_from_vernal_equinox(self.fixed_year, doy_fix, 0.5)
        else:
            cd = self.date_at(sim_time_sec)
            frac_yr = self._year_fraction_from_vernal_equinox(cd.year, cd.doy, frac_day)
        t_season = frac_yr * float(tropical_year_sec)
        t_diurnal = frac_day * spd
        return t_season, t_diurnal
