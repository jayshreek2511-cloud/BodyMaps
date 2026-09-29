"""Plan limits, and the checks that enforce them.

The single seam for "is this account allowed to do this". Endpoints ask
``check_inference`` / ``check_ai_message`` before doing work and record usage
after; nothing else reads ``user_account.plan`` or the ``usage_event`` table.

The limit table below is mirrored in the frontend
(PanTS-Demo/src/helpers/accountProfile.ts, PLAN_LIMITS) so the UI can grey a
locked control out instead of letting you click into a 402. The server copy is
the one that decides — the frontend copy only decides how things look. Keep them
in step; a drifted frontend is a cosmetic bug, a drifted backend is a real one.

``None`` as a limit means unlimited.

Admins are outside the table: they get ``UNLIMITED`` whatever plan their account
is on, because the people running the site have to be able to exercise every
part of it. ``limits_for_user`` is where that happens, and every check goes
through it.

Nothing is charged. Upgrading is a plain column write (``set_plan``) with no
payment step, because pricing hasn't been set — but the limits on the plan you
are on are applied for real.
"""

import uuid
from datetime import timedelta

from sqlalchemy import select

from models.engine import session_scope
from models.job import utcnow
from models.usage_event import KIND_AI_MESSAGE, KIND_INFERENCE, KIND_INTERACTIVE_SESSION, UsageEvent
from models.user import User
from services import role_store

# Quotas are a rolling window rather than a calendar day: "3 a day" that resets
# at a fixed hour hands out 6 runs to anyone who waits until 23:59.
WINDOW = timedelta(hours=24)

DEFAULT_PLAN = "free"

PLAN_LIMITS: dict[str, dict] = {
    # Tier story: free = any signed-up account (T1), pro = the verified tier
    # (T2: email verified + profile complete; see limits_for_user), granted
    # automatically - the ids stay the two the UI already knows.
    "free": {
        "daily_scans": 1,
        "concurrent_scans": 1,
        # Model ids the plan may run. LesionSegmenter is the free model; "None"
        # (view-only) never reaches the server, so it isn't listed.
        "models": ["LesionSegmenter"],
        "postprocessing": False,
        "daily_ai_messages": 10,
        "create_reports": False,
        "retention_days": 7,
        "priority_queue": False,
    },
    "pro": {
        "daily_scans": 10,
        "concurrent_scans": 5,
        "models": None,
        "postprocessing": True,
        "daily_ai_messages": 200,
        "create_reports": True,
        "retention_days": 365,
        "priority_queue": True,
    },
    # Team is Pro for each member; the pooling and shared-library half of the
    # plan copy is not built, so per-member limits are what it grants today.
    "team": {
        "daily_scans": 10,
        "concurrent_scans": 5,
        "models": None,
        "postprocessing": True,
        "daily_ai_messages": 200,
        "create_reports": True,
        "retention_days": 365,
        "priority_queue": True,
    },
    "enterprise": {
        "daily_scans": None,
        "concurrent_scans": None,
        "models": None,
        "postprocessing": True,
        "daily_ai_messages": None,
        "create_reports": True,
        "retention_days": None,
        "priority_queue": True,
    },
}

PLAN_IDS = tuple(PLAN_LIMITS)

# Models that are a postprocessing step rather than a segmentation run. They are
# gated by the "postprocessing" flag and don't count against the scan quota.
POSTPROCESSING_MODELS = frozenset({"ShapeKit"})


# What an admin gets. Spelled out rather than aliased to the enterprise row: an
# admin's access shouldn't quietly change the day someone edits a plan's limits.
UNLIMITED: dict = {
    "daily_scans": None,
    "concurrent_scans": None,
    "models": None,
    "postprocessing": True,
    "daily_ai_messages": None,
    "create_reports": True,
    "retention_days": None,
    "priority_queue": True,
}


def limits_for(plan: str | None) -> dict:
    """The limit dict for a plan id, falling back to free for anything unknown."""
    return PLAN_LIMITS.get(plan or DEFAULT_PLAN, PLAN_LIMITS[DEFAULT_PLAN])


# The tier a free account is lifted to once its email is verified and its
# profile is filled in. Computed on the fly, never written to the plan column:
# no trigger to miss, no backfill, and clearing the profile drops the account
# back on its own.
VERIFIED_PLAN = "pro"


def _qualifies_as_verified(user) -> bool:
    """T2: a verified email plus every profile field filled in."""
    return (
        user is not None
        and user.email_verified_at is not None
        and bool((user.organization or "").strip())
        and bool((user.occupation or "").strip())
        and bool((user.role_description or "").strip())
    )


def is_verified_researcher(user_id: str) -> bool:
    """Whether the account has completed the existing verified-researcher gate."""
    with session_scope() as s:
        user = s.get(User, user_id)
        return _qualifies_as_verified(user)


def limits_for_user(user_id: str) -> tuple[str, dict]:
    """The effective plan of an account, and the limits actually applied.

    Two things can lift an account above its stored plan column:

    - An admin bypasses limits entirely. The reported plan id stays the stored
      one — the role is not a plan, and naming an admin "Enterprise" would be a
      lie about their account.
    - A free account whose email is verified and whose profile is complete IS
      on the verified tier: that promotion is the account's real plan story,
      so it is reported as ``VERIFIED_PLAN`` and every consumer (usage page,
      402 bodies, the upgrade dialog) names it consistently.

    Mirrored in the frontend by helpers/accountProfile.ts (``gatingPlan``).
    """
    with session_scope() as s:
        user = s.get(User, user_id)
        plan = (user.plan if user else None) or DEFAULT_PLAN
        promoted = plan == DEFAULT_PLAN and _qualifies_as_verified(user)
    if role_store.has_role(user_id, role_store.ROLE_ADMIN):
        return plan, UNLIMITED
    if promoted:
        return VERIFIED_PLAN, limits_for(VERIFIED_PLAN)
    return plan, limits_for(plan)


# ---- plan ------------------------------------------------------------------

def get_plan(user_id: str) -> str:
    with session_scope() as s:
        user = s.get(User, user_id)
        return (user.plan if user else None) or DEFAULT_PLAN


def set_plan(user_id: str, plan: str) -> dict | None:
    """Move an account to another plan. Returns the user's public dict, or None
    for an unknown/system account. Raises ValueError on an unknown plan id."""
    if plan not in PLAN_LIMITS:
        raise ValueError(f"Unknown plan {plan!r}")
    with session_scope() as s:
        user = s.get(User, user_id)
        if user is None or user.is_system:
            return None
        user.plan = plan
        s.flush()
        return user.to_public_dict()


# ---- counting --------------------------------------------------------------

def _window_start():
    return utcnow() - WINDOW


def _count_in_window(s, user_id: str, kind: str) -> int:
    return len(list(s.execute(
        select(UsageEvent.id).where(
            UsageEvent.user_id == user_id,
            UsageEvent.kind == kind,
            UsageEvent.created_at >= _window_start(),
        )
    ).scalars()))


def _count_in_flight(s, user_id: str) -> int:
    return len(list(s.execute(
        select(UsageEvent.id).where(
            UsageEvent.user_id == user_id,
            UsageEvent.kind == KIND_INFERENCE,
            UsageEvent.finished_at.is_(None),
        )
    ).scalars()))


def _resets_at(s, user_id: str, kind: str) -> str | None:
    """When the oldest event in the window ages out — i.e. when the next unit of
    quota comes back. None if there's nothing in the window."""
    oldest = s.execute(
        select(UsageEvent.created_at).where(
            UsageEvent.user_id == user_id,
            UsageEvent.kind == kind,
            UsageEvent.created_at >= _window_start(),
        ).order_by(UsageEvent.created_at.asc()).limit(1)
    ).scalar_one_or_none()
    return (oldest + WINDOW).isoformat() if oldest else None


# ---- checks ----------------------------------------------------------------
#
# Each returns None when the action is allowed, or a dict describing the block.
# That dict is the 402 body: `reason` is what the UI switches on, the rest is
# what it needs to write a sentence without hardcoding the numbers.

def check_inference(user_id: str, model: str) -> dict | None:
    plan, limits = limits_for_user(user_id)
    postprocessing = model in POSTPROCESSING_MODELS

    if postprocessing:
        if not limits["postprocessing"]:
            return {
                "reason": "postprocessing", "plan": plan, "feature": model,
                "message": f"{model} postprocessing isn't included in the Free plan.",
            }
        return None

    allowed = limits["models"]
    if allowed is not None and model not in allowed:
        return {
            "reason": "model_locked", "plan": plan, "feature": model,
            "allowed_models": list(allowed),
            "message": f"{model} isn't included in the Free plan.",
        }

    with session_scope() as s:
        max_concurrent = limits["concurrent_scans"]
        if max_concurrent is not None:
            in_flight = _count_in_flight(s, user_id)
            if in_flight >= max_concurrent:
                return {
                    "reason": "concurrent_scans", "plan": plan,
                    "limit": max_concurrent, "used": in_flight,
                    "message": (
                        "The Free plan runs one scan at a time."
                        if max_concurrent == 1
                        else f"Your plan runs {max_concurrent} scans at a time."
                    ),
                }

        daily = limits["daily_scans"]
        if daily is not None:
            used = _count_in_window(s, user_id, KIND_INFERENCE)
            if used >= daily:
                return {
                    "reason": "daily_scans", "plan": plan, "limit": daily, "used": used,
                    "resets_at": _resets_at(s, user_id, KIND_INFERENCE),
                    "message": (
                        "You've used your scan for today."
                        if daily == 1
                        else f"You've used your {daily} scans for today."
                    ),
                }
    return None


def check_assistant(user_id: str | None) -> dict | None:
    """Whether this caller may send an assistant message. None means yes.

    Signed out is a refusal, not a pass: the assistant costs real compute and is
    metered per account, so it needs one — and leaving it open made the daily
    allowance meaningless, since signing out was an unlimited tier.

    The decision lives here rather than in the endpoint so it's testable without
    importing api_blueprint, which drags in the whole nibabel/scipy stack.
    """
    if user_id is None:
        return {"reason": "auth_required", "message": "Sign in to use the assistant."}
    return check_ai_message(user_id)


def check_ai_message(user_id: str) -> dict | None:
    plan, limits = limits_for_user(user_id)
    daily = limits["daily_ai_messages"]
    if daily is None:
        return None
    with session_scope() as s:
        used = _count_in_window(s, user_id, KIND_AI_MESSAGE)
        if used >= daily:
            return {
                "reason": "daily_ai_messages", "plan": plan, "limit": daily, "used": used,
                "resets_at": _resets_at(s, user_id, KIND_AI_MESSAGE),
                "message": f"You've used your {daily} assistant messages for today.",
            }
    return None


# ---- recording -------------------------------------------------------------

def record_inference(user_id: str, session_id: str, model: str) -> None:
    """Open a usage row for a run that was just accepted. Re-running the same
    session id (the frontend reuses it on retry) reopens the existing row rather
    than charging twice."""
    if model in POSTPROCESSING_MODELS:
        return  # gated, but not metered — it's a step on an existing result
    with session_scope() as s:
        existing = s.execute(
            select(UsageEvent).where(
                UsageEvent.kind == KIND_INFERENCE, UsageEvent.ref_id == session_id
            )
        ).scalar_one_or_none()
        if existing is not None:
            existing.finished_at = None
            existing.model = model
            return
        s.add(UsageEvent(
            id=str(uuid.uuid4()), user_id=user_id, kind=KIND_INFERENCE,
            ref_id=session_id, model=model,
        ))


def finish_inference(session_id: str) -> None:
    """Close a run's usage row so it stops counting against concurrency. Called
    when the job reaches a terminal state; a completed run still counts against
    the daily quota (created_at is untouched)."""
    with session_scope() as s:
        event = s.execute(
            select(UsageEvent).where(
                UsageEvent.kind == KIND_INFERENCE, UsageEvent.ref_id == session_id
            )
        ).scalar_one_or_none()
        if event is not None and event.finished_at is None:
            event.finished_at = utcnow()


def record_ai_message(user_id: str) -> None:
    with session_scope() as s:
        s.add(UsageEvent(
            id=str(uuid.uuid4()), user_id=user_id, kind=KIND_AI_MESSAGE,
            finished_at=utcnow(),
        ))


def count_interactive_sessions(user_id: str) -> int:
    with session_scope() as s:
        return _count_in_window(s, user_id, KIND_INTERACTIVE_SESSION)


def record_interactive_session(user_id: str, session_id: str) -> None:
    with session_scope() as s:
        s.add(UsageEvent(
            id=str(uuid.uuid4()), user_id=user_id,
            kind=KIND_INTERACTIVE_SESSION, ref_id=session_id,
            finished_at=utcnow(),
        ))


# ---- reporting -------------------------------------------------------------

def usage_summary(user_id: str) -> dict:
    """What the settings page shows: plan, its limits, and what's been used of
    the metered ones in the current window."""
    plan, limits = limits_for_user(user_id)
    with session_scope() as s:
        return {
            "plan": plan,
            "limits": limits,
            "scans": {
                "used": _count_in_window(s, user_id, KIND_INFERENCE),
                "limit": limits["daily_scans"],
                "in_flight": _count_in_flight(s, user_id),
                "resets_at": _resets_at(s, user_id, KIND_INFERENCE),
            },
            "ai_messages": {
                "used": _count_in_window(s, user_id, KIND_AI_MESSAGE),
                "limit": limits["daily_ai_messages"],
                "resets_at": _resets_at(s, user_id, KIND_AI_MESSAGE),
            },
            "interactive_sessions": {
                "used": _count_in_window(s, user_id, KIND_INTERACTIVE_SESSION),
                "limit": None,
                "resets_at": _resets_at(s, user_id, KIND_INTERACTIVE_SESSION),
            },
        }
