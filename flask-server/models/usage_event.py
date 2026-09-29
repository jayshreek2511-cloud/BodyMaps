"""Per-account usage, for plan limit enforcement.

Deliberately its own table rather than a count over ``job``: the frontend starts
runs through ``/run-inference``, which keeps state in an in-memory dict mirrored
to ``sessions/<id>/job.json`` and never writes a ``job`` row. Counting jobs would
therefore see zero for every run a real user makes.

One row per metered action. ``finished_at`` is what makes concurrency
answerable — a null means the run is still in flight, so "how many scans does
this user have going right now" is a count rather than a guess.
"""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from models.base import db
from models.job import utcnow

# Wire values; plan_store switches on them.
KIND_INFERENCE = "inference"
KIND_AI_MESSAGE = "ai_message"
KIND_INTERACTIVE_SESSION = "interactive_session"


class UsageEvent(db.Model):
    __tablename__ = "usage_event"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    # CASCADE, unlike job.user_id: usage is bookkeeping, not the user's data, so
    # a purged account should take it with them rather than block the delete.
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("user_account.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    kind: Mapped[str] = mapped_column(String(24), nullable=False)
    # The inference session id, so the row can be closed out when the run ends.
    # Null for kinds that finish the instant they start (ai_message).
    ref_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    # Null means still running. Set when the job reaches a terminal state.
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_usage_event_user_kind_created", "user_id", "kind", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<UsageEvent {self.kind} user={self.user_id} ref={self.ref_id}>"
