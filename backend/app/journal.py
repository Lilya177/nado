from sqlalchemy.orm import Session
from models import JournalEntry
from typing import Optional
from fastapi import Request


def log_action(
    db: Session,
    action: str,
    user_id: Optional[int] = None,
    entity: Optional[str] = None,
    entity_id: Optional[int] = None,
    detail: Optional[str] = None,
    request: Optional[Request] = None,
):
    """Записывает действие в журнал. Вызывай после успешных операций."""
    ip = None
    if request:
        forwarded = request.headers.get("X-Forwarded-For")
        ip = forwarded.split(",")[0] if forwarded else request.client.host

    entry = JournalEntry(
        user_id=user_id,
        action=action,
        entity=entity,
        entity_id=entity_id,
        detail=detail,
        ip_address=ip,
    )
    db.add(entry)
    db.commit()
