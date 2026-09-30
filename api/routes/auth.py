"""Authentication, user registration, and profile endpoints."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.dependencies import (
    get_client_ip,
    get_current_user,
    get_db,
)
from api.schemas import (
    ErrorResponse,
    TokenResponse,
    UserLoginRequest,
    UserProfileResponse,
    UserRegisterRequest,
)
from src.database.models import User
from src.security.audit import AuditAction, record_audit_event
from src.security.auth import (
    JWT_EXPIRE_MINUTES,
    ROLE_PERMISSIONS,
    UserRole,
    create_access_token,
    hash_password,
    verify_password,
)

logger = logging.getLogger("chequesense.api.auth")

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=UserProfileResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new banking user",
    description="Creates a new user record with securely bcrypt-hashed password and assigned RBAC role.",
    responses={
        400: {"model": ErrorResponse, "description": "Username or email already exists / invalid role"},
    },
)
def register_user(
    request: Request,
    payload: UserRegisterRequest,
    db: Session = Depends(get_db),
) -> UserProfileResponse:
    # 1. Validate Role
    try:
        assigned_role = UserRole(payload.role.upper())
    except ValueError:
        valid_roles = [r.value for r in UserRole]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role '{payload.role}'. Supported roles: {valid_roles}",
        )

    # 2. Check duplicate username
    stmt_username = select(User).where(User.username == payload.username)
    if db.scalars(stmt_username).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Username '{payload.username}' is already registered.",
        )

    # 3. Check duplicate email
    stmt_email = select(User).where(User.email == payload.email)
    if db.scalars(stmt_email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Email '{payload.email}' is already registered.",
        )

    # 4. Hash password with bcrypt (cost factor 12) - Never store plaintext
    hashed_pwd = hash_password(payload.password)

    new_user = User(
        username=payload.username,
        email=payload.email,
        hashed_password=hashed_pwd,
        role=assigned_role.value,
        is_active=True,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    client_ip = get_client_ip(request)
    logger.info("New user registered: %s (role: %s, ip: %s)", new_user.username, new_user.role, client_ip)

    permissions = [p.value for p in ROLE_PERMISSIONS.get(assigned_role, set())]
    return UserProfileResponse(
        id=new_user.id,
        username=new_user.username,
        email=new_user.email,
        role=new_user.role,
        is_active=new_user.is_active,
        permissions=sorted(permissions),
        created_at=new_user.created_at,
    )


@router.post(
    "/login",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Authenticate user and obtain JWT token",
    description="Verifies user credentials, issues a signed JWT access token, and writes an audit event.",
    responses={
        401: {"model": ErrorResponse, "description": "Invalid username or password"},
        403: {"model": ErrorResponse, "description": "Account is inactive/suspended"},
    },
)
def login_user(
    request: Request,
    payload: UserLoginRequest,
    db: Session = Depends(get_db),
) -> TokenResponse:
    # 1. Lookup user by username
    stmt = select(User).where(User.username == payload.username)
    user = db.scalars(stmt).first()

    # 2. Verify password
    if not user or not verify_password(payload.password, user.hashed_password):
        logger.warning("Failed login attempt for username: %s", payload.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 3. Check active state
    if not user.is_active:
        logger.warning("Login attempt on inactive account: %s", payload.username)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive. Contact banking administrator.",
        )

    # 4. Generate JWT access token with role claim
    token_data = {
        "sub": user.username,
        "user_id": user.id,
        "role": user.role,
    }
    access_token = create_access_token(data=token_data)

    client_ip = get_client_ip(request)

    # 5. Record mandatory audit log
    record_audit_event(
        session=db,
        action=AuditAction.LOGIN,
        username=user.username,
        user_id=user.id,
        resource_type="auth",
        resource_id=str(user.id),
        details={"role": user.role, "email": user.email},
        ip_address=client_ip,
    )

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=JWT_EXPIRE_MINUTES * 60,
        role=user.role,
        username=user.username,
    )


@router.get(
    "/me",
    response_model=UserProfileResponse,
    status_code=status.HTTP_200_OK,
    summary="Get current user profile and permissions",
    description="Retrieves authenticated user details and resolved RBAC permissions.",
)
def get_current_user_profile(
    current_user: User = Depends(get_current_user),
) -> UserProfileResponse:
    try:
        user_role = UserRole(current_user.role.upper())
    except ValueError:
        user_role = UserRole.EMPLOYEE

    permissions = [p.value for p in ROLE_PERMISSIONS.get(user_role, set())]
    return UserProfileResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        role=current_user.role,
        is_active=current_user.is_active,
        permissions=sorted(permissions),
        created_at=current_user.created_at,
    )
