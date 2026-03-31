from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from database import SessionLocal
import models
from auth import decode_token

bearer_scheme = HTTPBearer()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> models.User:
    payload = decode_token(credentials.credentials)
    user_id: int = payload.get("sub")
    if user_id is None:
        raise HTTPException(status_code=401, detail="Невалидный токен")
    user = db.query(models.User).get(int(user_id))
    if not user:
        raise HTTPException(status_code=401, detail="Пользователь не найден")
    return user


def require_role(*roles: str):
    """Декоратор-зависимость для проверки роли."""
    def role_checker(current_user: models.User = Depends(get_current_user)):
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Доступ запрещён. Требуется роль: {', '.join(roles)}"
            )
        return current_user
    return role_checker


# Готовые зависимости для роутов
require_user = require_role("user", "doctor", "admin")
require_doctor = require_role("doctor", "admin")
require_admin = require_role("admin")
