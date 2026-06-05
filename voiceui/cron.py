from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from voiceui.logs import log_event
from voiceui.models import CronConfig, CronJobConfig

_MONTH_NAMES = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
_DOW_NAMES = {
    "sun": 0,
    "mon": 1,
    "tue": 2,
    "wed": 3,
    "thu": 4,
    "fri": 5,
    "sat": 6,
}
_ALL_DAYS_OF_MONTH = frozenset(range(1, 32))
_ALL_DAYS_OF_WEEK = frozenset(range(0, 7))
_FIXED_TIMEZONES = {
    "utc": UTC,
    "asia/shanghai": timezone(timedelta(hours=8), name="Asia/Shanghai"),
}


@dataclass(frozen=True, slots=True)
class CronSchedule:
    minutes: frozenset[int]
    hours: frozenset[int]
    days_of_month: frozenset[int]
    months: frozenset[int]
    days_of_week: frozenset[int]

    @classmethod
    def parse(cls, expression: str) -> CronSchedule:
        fields = expression.split()
        if len(fields) != 5:
            raise ValueError(f"Cron schedule must have 5 fields: {expression!r}")

        minute, hour, day_of_month, month, day_of_week = fields
        return cls(
            minutes=_parse_field(minute, minimum=0, maximum=59),
            hours=_parse_field(hour, minimum=0, maximum=23),
            days_of_month=_parse_field(day_of_month, minimum=1, maximum=31),
            months=_parse_field(month, minimum=1, maximum=12, names=_MONTH_NAMES),
            days_of_week=_parse_field(
                day_of_week,
                minimum=0,
                maximum=7,
                names=_DOW_NAMES,
                normalize_seven_to_zero=True,
            ),
        )

    def matches(self, when: datetime) -> bool:
        if when.minute not in self.minutes:
            return False
        if when.hour not in self.hours:
            return False
        if when.month not in self.months:
            return False

        day_of_month_matches = when.day in self.days_of_month
        day_of_week_matches = _cron_day_of_week(when) in self.days_of_week
        day_of_month_is_any = self.days_of_month == _ALL_DAYS_OF_MONTH
        day_of_week_is_any = self.days_of_week == _ALL_DAYS_OF_WEEK
        if not day_of_month_is_any and not day_of_week_is_any:
            return day_of_month_matches or day_of_week_matches
        return day_of_month_matches and day_of_week_matches

    def next_after(self, after: datetime, *, max_days: int = 366) -> datetime:
        cursor = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
        deadline = after + timedelta(days=max_days)
        while cursor <= deadline:
            if self.matches(cursor):
                return cursor
            cursor += timedelta(minutes=1)
        raise ValueError(f"Cron schedule did not match within {max_days} days")


@dataclass(frozen=True, slots=True)
class RuntimeCronJob:
    config: CronJobConfig
    schedule: CronSchedule
    name: str
    timezone: tzinfo | None


class CronScheduler:
    def __init__(
        self,
        config: CronConfig,
        handler: Callable[[CronJobConfig], None],
        *,
        now_fn: Callable[[], datetime] | None = None,
    ):
        self.config = config
        self._jobs = build_runtime_jobs(config)
        self._handler = handler
        self._now_fn = now_fn or datetime.now
        self._poll_seconds = max(0.1, float(config.poll_seconds))
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_run_minutes: dict[str, str] = {}
        self._lock = threading.Lock()

    @property
    def has_jobs(self) -> bool:
        return bool(self._jobs)

    def start(self) -> None:
        if self._thread is not None:
            return
        log_event("cron", "started", log_id="cron.started", jobs=len(self._jobs))
        if not self._jobs:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="voiceui-cron", daemon=True)
        self._thread.start()

    def stop(self, *, timeout_seconds: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_seconds)
            self._thread = None
        log_event("cron", "stopped", log_id="cron.stopped")

    def run_pending(self, now: datetime | None = None) -> list[str]:
        current = now or self._now_fn()
        fired: list[str] = []
        with self._lock:
            for job in self._jobs:
                job_now = _convert_datetime(current, job.timezone)
                minute_key = job_now.replace(second=0, microsecond=0).isoformat()
                if self._last_run_minutes.get(job.name) == minute_key:
                    continue
                if not job.schedule.matches(job_now):
                    continue
                self._last_run_minutes[job.name] = minute_key
                fired.append(job.name)
                self._run_job(job)
        return fired

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.run_pending(self._now_fn())
            self._stop_event.wait(self._poll_seconds)

    def _run_job(self, job: RuntimeCronJob) -> None:
        started = time.monotonic()
        log_event(
            "cron",
            "job_started",
            log_id="cron.job_started",
            job=job.name,
            schedule=job.config.schedule,
        )
        try:
            self._handler(job.config)
        except Exception as exc:
            latency_ms = int((time.monotonic() - started) * 1000)
            log_event(
                "cron",
                "job_failed",
                log_id="cron.job_failed",
                job=job.name,
                latency_ms=latency_ms,
                error=exc,
            )
            return
        latency_ms = int((time.monotonic() - started) * 1000)
        log_event(
            "cron",
            "job_completed",
            log_id="cron.job_completed",
            job=job.name,
            latency_ms=latency_ms,
        )


def build_runtime_jobs(config: CronConfig) -> list[RuntimeCronJob]:
    if not config.enabled:
        return []

    jobs: list[RuntimeCronJob] = []
    seen_names: set[str] = set()
    for index, job in enumerate(config.jobs, start=1):
        if not job.enabled:
            continue
        name = job.id.strip() or f"job_{index}"
        if name in seen_names:
            raise ValueError(f"Duplicate cron job id: {name}")
        seen_names.add(name)
        if not job.schedule.strip():
            raise ValueError(f"Cron job {name!r} is missing schedule")
        if not job.text.strip():
            raise ValueError(f"Cron job {name!r} is missing text")
        jobs.append(
            RuntimeCronJob(
                config=job,
                schedule=CronSchedule.parse(job.schedule),
                name=name,
                timezone=_resolve_timezone(job.timezone),
            )
        )
    return jobs


def _parse_field(
    raw: str,
    *,
    minimum: int,
    maximum: int,
    names: dict[str, int] | None = None,
    normalize_seven_to_zero: bool = False,
) -> frozenset[int]:
    text = raw.strip().lower()
    if not text:
        raise ValueError("Cron field cannot be empty")

    values: set[int] = set()
    for part in text.split(","):
        values.update(
            _parse_field_part(
                part.strip(),
                minimum=minimum,
                maximum=maximum,
                names=names or {},
            )
        )
    if normalize_seven_to_zero:
        values = {0 if value == 7 else value for value in values}
    if not values:
        raise ValueError(f"Cron field cannot be empty: {raw!r}")
    return frozenset(sorted(values))


def _parse_field_part(
    part: str,
    *,
    minimum: int,
    maximum: int,
    names: dict[str, int],
) -> set[int]:
    if not part:
        raise ValueError("Cron field list contains an empty item")
    range_part, step = _split_step(part)
    if range_part in {"*", "?"}:
        start = minimum
        end = maximum
    elif "-" in range_part:
        start_text, end_text = range_part.split("-", 1)
        start = _parse_value(start_text, minimum=minimum, maximum=maximum, names=names)
        end = _parse_value(end_text, minimum=minimum, maximum=maximum, names=names)
    else:
        start = _parse_value(range_part, minimum=minimum, maximum=maximum, names=names)
        end = maximum if "/" in part else start

    if start > end:
        raise ValueError(f"Cron range start must not exceed end: {part!r}")
    return set(range(start, end + 1, step))


def _split_step(part: str) -> tuple[str, int]:
    if "/" not in part:
        return part, 1
    range_part, step_text = part.split("/", 1)
    try:
        step = int(step_text)
    except ValueError as exc:
        raise ValueError(f"Cron step must be an integer: {part!r}") from exc
    if step <= 0:
        raise ValueError(f"Cron step must be positive: {part!r}")
    return range_part, step


def _parse_value(
    raw: str,
    *,
    minimum: int,
    maximum: int,
    names: dict[str, int],
) -> int:
    text = raw.strip().lower()
    if text in names:
        value = names[text]
    else:
        try:
            value = int(text)
        except ValueError as exc:
            raise ValueError(f"Invalid cron value: {raw!r}") from exc
    if value < minimum or value > maximum:
        raise ValueError(f"Cron value {value} outside {minimum}-{maximum}")
    return value


def _cron_day_of_week(when: datetime) -> int:
    return (when.weekday() + 1) % 7


def _resolve_timezone(value: str) -> tzinfo | None:
    name = value.strip()
    if not name:
        return None
    fixed_timezone = _FIXED_TIMEZONES.get(name.lower())
    if fixed_timezone is not None:
        return fixed_timezone
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown cron timezone: {name}") from exc


def _convert_datetime(value: datetime, target_timezone: tzinfo | None) -> datetime:
    if target_timezone is None:
        return value
    if value.tzinfo is None:
        value = value.astimezone()
    return value.astimezone(target_timezone)
