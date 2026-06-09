from __future__ import annotations

import heapq
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import ceil
from typing import Any

from voiceui.logs import log_event


@dataclass(frozen=True, slots=True)
class Reminder:
    id: str
    due_at: datetime
    text: str
    created_at: datetime
    kind: str = "reminder"
    payload: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ParsedReminder:
    due_at: datetime
    delay_seconds: int
    text: str
    kind: str
    label: str


@dataclass(frozen=True, slots=True)
class ParsedScheduledCommand:
    due_at: datetime
    delay_seconds: int
    command_text: str
    label: str


ReminderHandler = Callable[[Reminder], None]

_NUMBER_PATTERN = r"\d+(?:\.\d+)?|[零一二两三四五六七八九十百半]+"
_RELATIVE_TIME_RE = re.compile(
    rf"(?:过|在)?(?P<num>{_NUMBER_PATTERN})\s*个?\s*"
    r"(?P<unit>秒钟?|分钟|分|小时|钟头|天)"
    r"\s*(?:后|以后|之后)"
)
_CLOCK_TIME_RE = re.compile(
    r"(?P<hour>\d{1,2}|[零一二两三四五六七八九十]+)点"
    r"(?:(?P<half>半)|(?P<minute>\d{1,2}|[零一二两三四五六七八九十]+)分?)?"
)
_REMINDER_TERMS = ("闹钟", "提醒", "定时", "叫我", "喊我", "alarm", "timer", "remind")
_CREATE_TERMS = (
    "定",
    "设",
    "设置",
    "提醒我",
    "叫我",
    "喊我",
    "alarm",
    "timer",
    "remind",
)
_CANCEL_TERMS = ("取消", "删除", "关掉", "cancel")
_SOFT_CANCEL_TERMS = ("不用提醒", "不要提醒", "提醒取消", "闹钟取消", "算了")
_STATUS_TERMS = ("还有", "等", "待", "查看", "列表", "状态", "什么反应", "怎么响", "响了以后")
_CN_DIGITS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


class ReminderScheduler:
    def __init__(
        self,
        handler: ReminderHandler,
        *,
        now_fn: Callable[[], datetime] | None = None,
        poll_seconds: float = 0.2,
    ):
        self._handler = handler
        self._now_fn = now_fn or _local_now
        self._poll_seconds = max(0.05, float(poll_seconds))
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._heap: list[tuple[float, int, Reminder]] = []
        self._next_id = 1
        self._sequence = 0

    def schedule_at(
        self,
        due_at: datetime,
        text: str,
        *,
        kind: str = "reminder",
        payload: dict[str, Any] | None = None,
    ) -> Reminder:
        created_at = self._now_fn()
        due_at = _ensure_compatible_datetime(due_at, created_at)
        reminder = Reminder(
            id=f"reminder_{self._next_id}",
            due_at=due_at,
            text=text.strip() or "提醒时间到了。",
            created_at=created_at,
            kind=kind,
            payload=payload,
        )
        self._next_id += 1
        due_timestamp = due_at.timestamp()
        with self._condition:
            self._sequence += 1
            heapq.heappush(self._heap, (due_timestamp, self._sequence, reminder))
            self._ensure_thread_locked()
            self._condition.notify_all()
        delay_seconds = max(0, ceil(due_timestamp - created_at.timestamp()))
        log_event(
            "reminder",
            "scheduled",
            log_id="reminder.scheduled",
            id=reminder.id,
            kind=reminder.kind,
            due_at=reminder.due_at.isoformat(timespec="seconds"),
            delay_seconds=delay_seconds,
            text_len=len(reminder.text),
        )
        return reminder

    def schedule_after(
        self,
        delay_seconds: float,
        text: str,
        *,
        kind: str = "reminder",
    ) -> Reminder:
        delay = max(1.0, float(delay_seconds))
        return self.schedule_at(self._now_fn() + timedelta(seconds=delay), text, kind=kind)

    def cancel_all(self) -> int:
        with self._condition:
            count = len(self._heap)
            self._heap.clear()
            self._condition.notify_all()
        log_event("reminder", "cancelled", log_id="reminder.cancelled", count=count)
        return count

    def pending(self) -> list[Reminder]:
        with self._condition:
            reminders = [item[2] for item in self._heap]
        return sorted(reminders, key=lambda reminder: reminder.due_at.timestamp())

    def run_due(self, now: datetime | None = None) -> list[str]:
        current = _ensure_compatible_datetime(now or self._now_fn(), self._now_fn())
        current_timestamp = current.timestamp()
        due: list[Reminder] = []
        with self._condition:
            while self._heap and self._heap[0][0] <= current_timestamp:
                _due_timestamp, _sequence, reminder = heapq.heappop(self._heap)
                due.append(reminder)
        for reminder in due:
            self._run_reminder(reminder)
        return [reminder.id for reminder in due]

    def stop(self, *, timeout_seconds: float = 5.0) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=timeout_seconds)
            self._thread = None

    def _ensure_thread_locked(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="voiceui-reminder",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            fired = self.run_due(self._now_fn())
            if fired:
                continue
            with self._condition:
                if self._stop_event.is_set():
                    return
                if self._heap:
                    next_due = self._heap[0][0]
                    wait_seconds = max(0.01, next_due - self._now_fn().timestamp())
                    wait_seconds = min(wait_seconds, self._poll_seconds)
                else:
                    wait_seconds = self._poll_seconds
                self._condition.wait(wait_seconds)

    def _run_reminder(self, reminder: Reminder) -> None:
        started = time.monotonic()
        log_event(
            "reminder",
            "triggered",
            log_id="reminder.triggered",
            id=reminder.id,
            kind=reminder.kind,
            due_at=reminder.due_at.isoformat(timespec="seconds"),
        )
        try:
            self._handler(reminder)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            latency_ms = int((time.monotonic() - started) * 1000)
            log_event(
                "reminder",
                "failed",
                log_id="reminder.failed",
                id=reminder.id,
                kind=reminder.kind,
                latency_ms=latency_ms,
                error=exc,
            )
            return
        latency_ms = int((time.monotonic() - started) * 1000)
        log_event(
            "reminder",
            "completed",
            log_id="reminder.completed",
            id=reminder.id,
            kind=reminder.kind,
            latency_ms=latency_ms,
        )


def looks_like_reminder_text(text: str) -> bool:
    normalized = _normalize_text(text)
    return any(term in normalized for term in _REMINDER_TERMS)


def looks_like_reminder_cancel_text(text: str) -> bool:
    normalized = _normalize_text(text)
    if not looks_like_reminder_text(text):
        return False
    if any(term in normalized for term in ("不要忘", "别忘")):
        return False
    return any(term in normalized for term in _CANCEL_TERMS + _SOFT_CANCEL_TERMS)


def looks_like_reminder_status_text(text: str) -> bool:
    normalized = _normalize_text(text)
    if not looks_like_reminder_text(text):
        return False
    return any(term in normalized for term in _STATUS_TERMS)


def looks_like_reminder_create_text(text: str) -> bool:
    normalized = _normalize_text(text)
    if not looks_like_reminder_text(text):
        return False
    if looks_like_reminder_cancel_text(text):
        return False
    return any(term in normalized for term in _CREATE_TERMS)


def parse_reminder_request(
    text: str,
    *,
    now: datetime | None = None,
    max_delay_seconds: int = 30 * 24 * 60 * 60,
) -> ParsedReminder | None:
    if not looks_like_reminder_create_text(text):
        return None
    current = now or _local_now()
    due_at = _extract_relative_due_at(text, current)
    if due_at is None:
        due_at = _extract_clock_due_at(text, current)
    if due_at is None and "现在" in text:
        due_at = current + timedelta(seconds=1)
    if due_at is None:
        return None

    delay_seconds = max(1, ceil(due_at.timestamp() - current.timestamp()))
    if delay_seconds > max_delay_seconds:
        return None
    kind = (
        "alarm"
        if any(term in _normalize_text(text) for term in ("闹钟", "alarm", "timer"))
        else "reminder"
    )
    label = format_delay(delay_seconds)
    return ParsedReminder(
        due_at=due_at,
        delay_seconds=delay_seconds,
        text=_extract_alert_text(text, kind),
        kind=kind,
        label=label,
    )


def parse_scheduled_command(
    text: str,
    *,
    now: datetime | None = None,
    max_delay_seconds: int = 30 * 24 * 60 * 60,
) -> ParsedScheduledCommand | None:
    current = now or _local_now()
    due_at = _extract_relative_due_at(text, current)
    if due_at is None:
        due_at = _extract_clock_due_at(text, current)
    if due_at is None:
        return None

    delay_seconds = max(1, ceil(due_at.timestamp() - current.timestamp()))
    if delay_seconds > max_delay_seconds:
        return None

    command_text = _clean_scheduled_command_text(_remove_time_fragments(text))
    if not command_text:
        return None

    return ParsedScheduledCommand(
        due_at=due_at,
        delay_seconds=delay_seconds,
        command_text=command_text,
        label=format_delay(delay_seconds),
    )


def format_reminder_confirmation(parsed: ParsedReminder) -> str:
    return f"好的，{parsed.label}提醒你。"


def format_pending_reminders(
    reminders: list[Reminder],
    *,
    now: datetime | None = None,
) -> str:
    if not reminders:
        return "当前没有待提醒的闹钟。"
    current = now or _local_now()
    next_reminder = min(reminders, key=lambda reminder: reminder.due_at.timestamp())
    delay_seconds = max(0, ceil(next_reminder.due_at.timestamp() - current.timestamp()))
    label = format_delay(delay_seconds) if delay_seconds > 0 else "马上"
    if len(reminders) == 1:
        return f"还有一个提醒，{label}会响。"
    return f"还有{len(reminders)}个提醒，最近一个{label}会响。"


def format_delay(delay_seconds: int) -> str:
    seconds = max(0, int(delay_seconds))
    if seconds < 60:
        return f"{seconds}秒后"
    if seconds < 3600:
        minutes = max(1, round(seconds / 60))
        return f"{minutes}分钟后"
    if seconds < 86400:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        if minutes:
            return f"{hours}小时{minutes}分钟后"
        return f"{hours}小时后"
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    if hours:
        return f"{days}天{hours}小时后"
    return f"{days}天后"


def _extract_relative_due_at(text: str, now: datetime) -> datetime | None:
    match = _RELATIVE_TIME_RE.search(text)
    if match is None:
        return None
    amount = _parse_number(match.group("num"))
    if amount is None:
        return None
    unit_seconds = _unit_seconds(match.group("unit"))
    if unit_seconds <= 0:
        return None
    delay_seconds = max(1.0, amount * unit_seconds)
    return now + timedelta(seconds=delay_seconds)


def _extract_clock_due_at(text: str, now: datetime) -> datetime | None:
    match = _CLOCK_TIME_RE.search(text)
    if match is None:
        return None
    hour = _parse_number(match.group("hour"))
    if hour is None:
        return None
    minute = 30 if match.group("half") else 0
    if match.group("minute"):
        parsed_minute = _parse_number(match.group("minute"))
        if parsed_minute is None:
            return None
        minute = int(parsed_minute)
    hour = int(hour)
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return None
    normalized = _normalize_text(text)
    if any(term in normalized for term in ("下午", "晚上", "今晚", "傍晚")) and hour < 12:
        hour += 12
    if "中午" in normalized and hour < 11:
        hour += 12
    if "凌晨" in normalized and hour == 12:
        hour = 0
    day_offset = 0
    if "后天" in normalized:
        day_offset = 2
    elif any(term in normalized for term in ("明天", "明早", "明晚")):
        day_offset = 1
    due_at = (now + timedelta(days=day_offset)).replace(
        hour=hour,
        minute=minute,
        second=0,
        microsecond=0,
    )
    if day_offset == 0 and due_at <= now:
        due_at += timedelta(days=1)
    return due_at


def _extract_alert_text(text: str, kind: str) -> str:
    if kind == "alarm":
        return "闹钟时间到了。"
    cleaned = _remove_time_fragments(text)
    for marker in ("提醒我", "叫我", "喊我"):
        index = cleaned.find(marker)
        if index >= 0:
            message = _clean_alert_message(cleaned[index + len(marker) :])
            if message:
                return f"提醒你：{message}。"
    message = _clean_alert_message(cleaned)
    if message and not looks_like_reminder_text(message):
        return f"提醒你：{message}。"
    return "提醒时间到了。"


def _remove_time_fragments(text: str) -> str:
    cleaned = _RELATIVE_TIME_RE.sub("", text)
    cleaned = _CLOCK_TIME_RE.sub("", cleaned)
    for term in ("今天", "明天", "后天", "上午", "下午", "晚上", "今晚", "明早", "明晚"):
        cleaned = cleaned.replace(term, "")
    return cleaned


def _clean_alert_message(text: str) -> str:
    cleaned = text.strip(" ，。！？,.!?;；：:")
    prefixes = (
        "请",
        "帮我",
        "麻烦",
        "你可以",
        "可以",
        "设置",
        "设",
        "定",
        "一个",
        "提醒",
        "闹钟",
        "定时",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip(" ，。！？,.!?;；：:")
                changed = True
    suffixes = ("吗", "么", "吧", "的", "闹钟", "提醒")
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if cleaned.endswith(suffix):
                cleaned = cleaned[: -len(suffix)].strip(" ，。！？,.!?;；：:")
                changed = True
    if cleaned in {"我", "你", "可以", "能不能"}:
        return ""
    return cleaned


def _clean_scheduled_command_text(text: str) -> str:
    cleaned = text.strip(" ，。！？.!?;；：:")
    prefixes = (
        "请",
        "帮我",
        "麻烦",
        "你可以",
        "可以",
        "设置",
        "设",
        "定",
        "定时",
        "预约",
        "到点",
        "之后",
        "以后",
        "后",
        "把",
        "给我",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip(" ，。！？.!?;；：:")
                changed = True
    return cleaned


def _parse_number(raw: str) -> float | None:
    text = raw.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    if text == "半":
        return 0.5
    if "百" in text:
        left, _, right = text.partition("百")
        hundred = _parse_number(left) if left else 1
        rest = _parse_number(right) if right else 0
        if hundred is None or rest is None:
            return None
        return hundred * 100 + rest
    if "十" in text:
        left, _, right = text.partition("十")
        tens = _CN_DIGITS.get(left, 1) if left else 1
        ones = _CN_DIGITS.get(right, 0) if right else 0
        return tens * 10 + ones
    if len(text) == 1:
        return _CN_DIGITS.get(text)
    value = 0
    for char in text:
        digit = _CN_DIGITS.get(char)
        if digit is None:
            return None
        value = value * 10 + digit
    return float(value)


def _unit_seconds(unit: str) -> int:
    if unit.startswith("秒"):
        return 1
    if unit in {"分", "分钟"}:
        return 60
    if unit in {"小时", "钟头"}:
        return 3600
    if unit == "天":
        return 86400
    return 0


def _ensure_compatible_datetime(value: datetime, reference: datetime) -> datetime:
    if value.tzinfo is None and reference.tzinfo is not None:
        return value.replace(tzinfo=reference.tzinfo)
    if value.tzinfo is not None and reference.tzinfo is None:
        return value.replace(tzinfo=None)
    return value


def _normalize_text(text: str) -> str:
    return text.lower().replace(" ", "")


def _local_now() -> datetime:
    return datetime.now().astimezone()
