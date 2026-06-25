# FastAPI × 클린 아키텍처 실전 가이드

> 그룹 프로젝트에 바로 적용하기 위한 리포트 · 2026년 중반 기준 (FastAPI 0.136.x / Pydantic v2 / SQLAlchemy 2.0 async / Starlette 1.0)

---

## 0. 먼저 읽는 결론 (TL;DR)

그룹 프로젝트에서 "클린 아키텍처를 한다"는 건 다음 한 문장으로 압축됩니다.

> **의존성은 항상 안쪽(비즈니스 규칙)을 향하고, 바깥쪽(FastAPI·DB·외부 API)은 안쪽이 정의한 인터페이스를 구현한다.**

실무에서 이걸 구현하는 가장 현실적인 형태는 **4계층 + 의존성 주입(DI)** 입니다.

```
요청 → [Presentation: Router/Schema] → [Application: UseCase/Service]
        → [Domain: Entity/Interface] ← [Infrastructure: Repository/DB/외부연동]
```

- **Domain**은 FastAPI도 SQLAlchemy도 모릅니다. 순수 파이썬만 있습니다.
- **Infrastructure**가 Domain이 정의한 인터페이스(`Protocol`/`ABC`)를 구현합니다.
- **FastAPI의 `Depends`** 가 이 둘을 런타임에 연결(주입)합니다.

그룹 프로젝트라면 **풀스펙 클린 아키텍처를 다 하지 말고**, 5장의 "현실적 레벨 선택"에서 권장하는 **중간 레벨(Service + Repository 분리)** 부터 시작하는 것을 권합니다. 과설계는 협업 속도를 떨어뜨립니다.

---

## 1. FastAPI 핵심 (현재 기준)

### 1.1 무엇이고 왜 쓰는가

FastAPI는 **Starlette(ASGI 웹 토대) + Pydantic(데이터 검증)** 위에 만들어진 비동기 우선(async-first) 파이썬 웹 프레임워크입니다. 세 가지가 핵심 강점입니다.

1. **타입 힌트 기반 자동 검증·문서화** — 함수 시그니처의 타입만으로 요청/응답 검증과 OpenAPI(`/docs`, `/redoc`) 문서가 자동 생성됩니다.
2. **비동기 I/O** — DB 쿼리, 외부 API 호출 같은 I/O 바운드 작업에서 높은 동시성을 냅니다. (단, CPU 바운드 작업은 `async def` 안에서 돌리면 이벤트 루프를 막으므로 별도 워커로 분리)
3. **내장 의존성 주입(`Depends`)** — 클린 아키텍처를 별도 프레임워크 없이도 구현할 수 있게 해주는 결정적 기능입니다.

### 1.2 버전·환경 (2026년 중반 기준)

| 항목 | 권장 |
|---|---|
| Python | **3.12 또는 3.13** |
| FastAPI | **0.136.x** (`fastapi[standard]` 설치 권장 — uvicorn 등 표준 도구 포함) |
| Pydantic | **v2** (2.9 이상, 현재 2.13.x 대) |
| Starlette | **1.0.0** |
| ORM | **SQLAlchemy 2.0+ (async)** + asyncpg(Postgres) |
| 마이그레이션 | **Alembic** |

`requirements.txt` 최소 예시:

```text
fastapi[standard]==0.136.1
sqlalchemy>=2.0
asyncpg
alembic
pydantic-settings
python-dotenv
```

### 1.3 반드시 알아야 할 두 가지 최신 패턴

**(1) 시작/종료는 `lifespan`으로** (`@app.on_event`는 폐기 예정 — 쓰지 마세요)

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시: DB 풀 생성, 모델 로드 등
    await init_db()
    yield
    # 종료 시: 정리
    await close_db()

app = FastAPI(lifespan=lifespan)
```

**(2) 의존성 주입은 `Annotated` + `Depends`로** (FastAPI 권장 최신 스타일)

```python
from typing import Annotated
from fastapi import Depends

# 재사용 가능한 타입 별칭으로 정의해두면 라우터가 깔끔해집니다
SessionDep = Annotated[AsyncSession, Depends(get_session)]

@router.get("/users/{user_id}")
async def read_user(user_id: int, session: SessionDep):
    ...
```

### 1.4 Pydantic v2 핵심 변경점 (v1과 다름 — 팀원 혼동 주의)

| v1 | v2 (현재) |
|---|---|
| `class Config: orm_mode = True` | `model_config = ConfigDict(from_attributes=True)` |
| `.dict()` | `.model_dump()` |
| `.parse_obj()` | `.model_validate()` |
| `@validator` | `@field_validator` / `@model_validator` |

---

## 2. 클린 아키텍처 핵심

### 2.1 한 줄 정의

**비즈니스 규칙(도메인)을 프레임워크·DB·UI 같은 "세부 구현"으로부터 격리**해서, 세부 구현이 바뀌어도 핵심 로직은 그대로 두는 설계 방식입니다.

### 2.2 동심원과 "의존성 규칙"

```
            ┌─────────────────────────────────────┐
            │   Frameworks & Drivers (가장 바깥)   │  ← FastAPI, SQLAlchemy, Redis, 외부 API
            │   ┌─────────────────────────────┐   │
            │   │   Interface Adapters         │   │  ← Router, Repository 구현체, Schema 변환
            │   │   ┌─────────────────────┐   │   │
            │   │   │   Use Cases (App)    │   │   │  ← 애플리케이션 비즈니스 흐름
            │   │   │   ┌─────────────┐   │   │   │
            │   │   │   │  Entities    │   │   │   │  ← 핵심 도메인 규칙 (가장 안쪽)
            │   │   │   └─────────────┘   │   │   │
            │   │   └─────────────────────┘   │   │
            │   └─────────────────────────────┘   │
            └─────────────────────────────────────┘

   의존성 화살표는 언제나 안쪽(→ 중심)을 향한다.
```

**의존성 규칙(The Dependency Rule)** — 이 한 가지만 지키면 절반은 성공입니다.

> 소스 코드의 의존성은 **오직 안쪽**을 향해야 한다. 안쪽 원은 바깥쪽 원에 대해 **아무것도 알지 못한다.**

즉 `Domain`은 `import fastapi`도, `import sqlalchemy`도 하지 않습니다. 거꾸로 바깥 계층이 안쪽을 import 합니다.

### 2.3 의존성 역전(DIP) — 어떻게 바깥을 안 알 수 있나?

도메인이 DB를 써야 하는데 SQLAlchemy를 모른다는 게 모순처럼 보입니다. 해법은 **인터페이스를 안쪽에 두는 것**입니다.

- **안쪽(Domain)**: "사용자를 저장할 무언가가 필요하다"는 **인터페이스**(`UserRepository`)만 선언.
- **바깥쪽(Infrastructure)**: 그 인터페이스를 SQLAlchemy로 **구현**(`SqlAlchemyUserRepository`).

도메인은 추상(인터페이스)에만 의존하고, 구체적 구현은 런타임에 주입됩니다. 이게 **의존성 역전 원칙(Dependency Inversion Principle)** 이며, FastAPI에서는 `Depends`가 "주입" 역할을 합니다.

### 2.4 얻는 것 / 치르는 비용

**얻는 것**
- **테스트 용이성**: DB 없이 인메모리 가짜 리포지토리로 비즈니스 로직 단위 테스트 가능.
- **교체 가능성**: Postgres → MongoDB, REST → gRPC 전환 시 도메인 코드 무수정.
- **병렬 작업**: 인터페이스만 먼저 합의하면 팀원들이 계층별로 동시 개발 가능.

**치르는 비용**
- 보일러플레이트(파일·클래스 수)가 늘어남.
- 작은 프로젝트엔 과할 수 있음 → 레벨 조절 필요(5장).

---

## 3. FastAPI에 클린 아키텍처 매핑하기

클린 아키텍처의 추상적 4계층을 FastAPI 실무 용어로 옮기면 다음과 같습니다.

| 클린 아키텍처 계층 | FastAPI 프로젝트에서의 실체 | 무엇을 알아도 되나 |
|---|---|---|
| **Entities** (Domain) | 순수 도메인 모델, 도메인 예외, **Repository 인터페이스** | 파이썬 표준 라이브러리만 |
| **Use Cases** (Application) | UseCase / Service 클래스 (비즈니스 흐름 조율) | Domain만 |
| **Interface Adapters** | Router(컨트롤러), Pydantic Schema(DTO), **Repository 구현체** | Application + 프레임워크 |
| **Frameworks & Drivers** | FastAPI 앱, SQLAlchemy 세션/엔진, 외부 SDK, 설정 | 전부 |

**핵심 연결점 = FastAPI `Depends`**: 라우터에서 UseCase를, UseCase에 Repository 구현체를 `Depends`로 주입함으로써 "바깥이 안의 인터페이스를 구현해 끼워 넣는" 구조가 완성됩니다.

---

## 4. 실전 프로젝트 구조

그룹 프로젝트에 바로 복붙 가능한 폴더 구조입니다. **기능(도메인)별 수직 분할**을 기본으로 하되, 공통은 `core`로 뺍니다.

```
project/
├── app/
│   ├── main.py                      # FastAPI 앱 생성, lifespan, 라우터 등록
│   ├── core/                        # 횡단 공통 (프레임워크 영역)
│   │   ├── config.py                # pydantic-settings 기반 환경설정
│   │   ├── database.py              # 엔진/세션 팩토리, get_session
│   │   └── exceptions.py            # 전역 예외 핸들러
│   │
│   ├── domain/                      # ★ 가장 안쪽: 아무것도 import 안 함
│   │   └── user/
│   │       ├── entity.py            # User 도메인 모델 (dataclass)
│   │       ├── exceptions.py        # UserNotFound 등 도메인 예외
│   │       └── repository.py        # UserRepository 인터페이스(Protocol/ABC)
│   │
│   ├── application/                 # ★ 유스케이스: domain만 import
│   │   └── user/
│   │       └── service.py           # RegisterUser, GetUser 등
│   │
│   ├── infrastructure/              # ★ 바깥: domain 인터페이스 구현
│   │   └── user/
│   │       ├── model.py             # SQLAlchemy ORM 모델 (DB 테이블)
│   │       └── repository.py        # SqlAlchemyUserRepository (구현체)
│   │
│   └── presentation/                # ★ 바깥: HTTP 어댑터
│       └── user/
│           ├── schema.py            # Pydantic 요청/응답 DTO
│           ├── router.py            # 엔드포인트
│           └── dependencies.py      # Depends 와이어링
│
├── tests/
│   ├── conftest.py
│   ├── unit/                        # 도메인·유스케이스 (DB 없이)
│   └── integration/                 # 라우터·DB 포함
├── alembic/                         # 마이그레이션
├── requirements.txt
└── .env
```

> **수직 분할(domain/user, application/user …) vs 수평 분할(모든 router를 한 폴더에)?**
> 그룹 프로젝트에서는 **수직 분할을 강력 추천**합니다. 팀원이 "나는 user 기능 담당"이면 `*/user/` 폴더들만 건드리게 되어 **머지 충돌이 극적으로 줄어듭니다.**

---

## 5. 계층별 코드 (전체 흐름 한 바퀴)

사용자 등록(register) 한 기능을 4계층으로 끝까지 구현한 예시입니다. 이 패턴을 그대로 다른 기능에 복제하면 됩니다.

### 5.1 Domain — Entity (순수 파이썬)

```python
# app/domain/user/entity.py
from dataclasses import dataclass

@dataclass
class User:
    id: int | None
    email: str
    hashed_password: str

    def verify_owner(self, user_id: int) -> bool:
        # 도메인 규칙은 여기에. FastAPI/DB를 전혀 모른다.
        return self.id == user_id
```

```python
# app/domain/user/exceptions.py
class UserNotFoundError(Exception): ...
class EmailAlreadyExistsError(Exception): ...
```

### 5.2 Domain — Repository 인터페이스 (의존성 역전의 핵심)

```python
# app/domain/user/repository.py
from typing import Protocol
from app.domain.user.entity import User

class UserRepository(Protocol):
    """도메인이 '필요하다'고 선언만 하는 계약. 구현은 바깥에서."""
    async def add(self, user: User) -> User: ...
    async def get_by_id(self, user_id: int) -> User | None: ...
    async def get_by_email(self, email: str) -> User | None: ...
```

> `Protocol`(구조적 타이핑)을 쓰면 구현체가 명시적으로 상속하지 않아도 되어 결합도가 더 낮습니다. 팀이 익숙하다면 `abc.ABC` + `@abstractmethod`도 무방합니다.

### 5.3 Application — Service / UseCase

```python
# app/application/user/service.py
from app.domain.user.entity import User
from app.domain.user.repository import UserRepository
from app.domain.user.exceptions import EmailAlreadyExistsError, UserNotFoundError

class UserService:
    def __init__(self, repo: UserRepository):
        self._repo = repo                      # 추상에만 의존 (구현 모름)

    async def register(self, email: str, raw_password: str) -> User:
        if await self._repo.get_by_email(email):
            raise EmailAlreadyExistsError(email)
        user = User(id=None, email=email, hashed_password=_hash(raw_password))
        return await self._repo.add(user)

    async def get(self, user_id: int) -> User:
        user = await self._repo.get_by_id(user_id)
        if user is None:
            raise UserNotFoundError(user_id)
        return user

def _hash(pw: str) -> str:
    # 실제로는 passlib/bcrypt 사용
    return "hashed:" + pw
```

### 5.4 Infrastructure — ORM 모델 + Repository 구현체

```python
# app/infrastructure/user/model.py
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase

class Base(DeclarativeBase): ...

class UserModel(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(unique=True, index=True)
    hashed_password: Mapped[str]
```

```python
# app/infrastructure/user/repository.py
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.domain.user.entity import User
from app.domain.user.repository import UserRepository
from app.infrastructure.user.model import UserModel

class SqlAlchemyUserRepository(UserRepository):    # 도메인 계약을 구현
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(self, user: User) -> User:
        row = UserModel(email=user.email, hashed_password=user.hashed_password)
        self._session.add(row)
        await self._session.flush()
        user.id = row.id
        return user

    async def get_by_id(self, user_id: int) -> User | None:
        row = await self._session.get(UserModel, user_id)
        return self._to_entity(row) if row else None

    async def get_by_email(self, email: str) -> User | None:
        stmt = select(UserModel).where(UserModel.email == email)
        row = (await self._session.execute(stmt)).scalar_one_or_none()
        return self._to_entity(row) if row else None

    @staticmethod
    def _to_entity(row: UserModel) -> User:
        # ORM 모델(바깥) → 도메인 엔티티(안쪽) 변환. 경계를 명확히.
        return User(id=row.id, email=row.email, hashed_password=row.hashed_password)
```

### 5.5 Presentation — Schema(DTO) + DI 와이어링 + Router

```python
# app/presentation/user/schema.py
from pydantic import BaseModel, ConfigDict, EmailStr

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)   # v2 방식
    id: int
    email: EmailStr
```

```python
# app/presentation/user/dependencies.py
from typing import Annotated
from fastapi import Depends
from app.core.database import SessionDep
from app.infrastructure.user.repository import SqlAlchemyUserRepository
from app.application.user.service import UserService

def get_user_service(session: SessionDep) -> UserService:
    # 여기서 "바깥 구현체"를 "안쪽 인터페이스" 자리에 끼워 넣는다.
    repo = SqlAlchemyUserRepository(session)
    return UserService(repo)

UserServiceDep = Annotated[UserService, Depends(get_user_service)]
```

```python
# app/presentation/user/router.py
from fastapi import APIRouter, HTTPException, status
from app.presentation.user.schema import UserCreate, UserOut
from app.presentation.user.dependencies import UserServiceDep
from app.domain.user.exceptions import EmailAlreadyExistsError, UserNotFoundError

router = APIRouter(prefix="/v1/users", tags=["users"])

@router.post("", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def register(body: UserCreate, service: UserServiceDep):
    try:
        user = await service.register(body.email, body.password)
    except EmailAlreadyExistsError:
        raise HTTPException(status.HTTP_409_CONFLICT, "이미 가입된 이메일")
    return user

@router.get("/{user_id}", response_model=UserOut)
async def get_user(user_id: int, service: UserServiceDep):
    try:
        return await service.get(user_id)
    except UserNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "사용자 없음")
```

### 5.6 Core — DB 세션 & main

```python
# app/core/database.py
from typing import Annotated, AsyncGenerator
from fastapi import Depends
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

engine = create_async_engine("postgresql+asyncpg://user:pw@localhost/db")
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

SessionDep = Annotated[AsyncSession, Depends(get_session)]
```

```python
# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.presentation.user.router import router as user_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield   # 시작/종료 훅

app = FastAPI(lifespan=lifespan)
app.include_router(user_router)
```

**의존성 흐름 한눈에**: `router → UserServiceDep(Depends) → UserService → UserRepository(Protocol) ← SqlAlchemyUserRepository → SessionDep(Depends) → DB`. 화살표 방향과 무관하게 **컴파일타임 의존성은 모두 안쪽을 향합니다.**

---

## 6. 그룹 프로젝트 적용 가이드 (이게 핵심)

### 6.1 현실적 레벨 선택 — "어디까지 클린하게?"

클린 아키텍처는 **전부 아니면 전무가 아닙니다.** 팀 규모·기간에 맞춰 레벨을 고르세요.

| 레벨 | 구조 | 적합한 상황 |
|---|---|---|
| **L1 미니멀** | `router → service → model` (Repository 없음) | 1~2주 단기, 프로토타입, CRUD 위주 |
| **L2 권장** | `router → service → repository → model` | **대부분의 그룹 프로젝트 ★** |
| **L3 풀스펙** | 위 + `domain/entity` + `Protocol 인터페이스` + 엔티티/ORM 분리 | 장기 유지보수, 학습 목적, DB 교체 가능성 |

대부분의 학교/사이드 그룹 프로젝트는 **L2로 시작**하고, 핵심 도메인 한두 개만 L3로 끌어올리는 게 가성비가 가장 좋습니다. (이 리포트의 5장 코드는 L3 예시이지만, `domain/entity.py`와 `Protocol`을 빼면 그대로 L2가 됩니다.)

### 6.2 병렬 작업을 가능하게 하는 절차 (협업의 핵심)

클린 아키텍처의 가장 큰 협업 이점은 **"인터페이스를 먼저 합의하면 계층을 동시에 짤 수 있다"** 는 점입니다.

1. **0일차 — 경계 합의 회의**: 기능별로 `Repository 인터페이스`와 `Pydantic Schema(요청/응답)`를 **먼저** 함께 정의해 PR로 머지. (코드는 거의 없고 시그니처만)
2. **병렬 개발**: 인터페이스가 고정되면
   - A는 `presentation`(라우터)을 **가짜 서비스**로 개발,
   - B는 `application`(서비스)을 **인메모리 리포지토리**로 개발,
   - C는 `infrastructure`(실제 DB 리포지토리)를 구현.
   서로를 기다리지 않습니다.
3. **통합**: `dependencies.py`의 와이어링만 진짜 구현체로 교체하면 끝.

### 6.3 담당 분배 2가지 방식

- **기능 수직 분배 (추천)**: "A=user, B=post, C=auth" 처럼 도메인 단위. `*/user/` 폴더만 만지므로 충돌 최소.
- **계층 수평 분배**: "A=라우터 전부, B=서비스 전부" — 인터페이스 강제 합의가 필요할 때 학습용으로만.

대부분 **수직 분배 + 공통(core)만 한 명이 관리**가 가장 매끄럽습니다.

### 6.4 테스트 전략 — 인메모리 리포지토리 & `dependency_overrides`

클린 구조의 보상은 테스트에서 나옵니다.

```python
# 단위 테스트: DB 없이 비즈니스 로직만
class FakeUserRepo:                     # Protocol을 만족하는 가짜
    def __init__(self): self._data = {}
    async def add(self, user):
        user.id = len(self._data) + 1
        self._data[user.id] = user
        return user
    async def get_by_id(self, uid): return self._data.get(uid)
    async def get_by_email(self, email):
        return next((u for u in self._data.values() if u.email == email), None)

async def test_register_duplicate_email():
    service = UserService(FakeUserRepo())
    await service.register("a@a.com", "pw")
    with pytest.raises(EmailAlreadyExistsError):
        await service.register("a@a.com", "pw")
```

```python
# 통합 테스트: 라우터는 그대로, 의존성만 교체
from app.main import app
from app.presentation.user.dependencies import get_user_service

app.dependency_overrides[get_user_service] = lambda: UserService(FakeUserRepo())
# 이제 TestClient/httpx로 실제 HTTP 흐름을 DB 없이 검증
```

`dependency_overrides`는 FastAPI가 클린 아키텍처 테스트를 위해 사실상 제공하는 공식 메커니즘입니다.

### 6.5 팀 컨벤션 체크리스트 (PR 리뷰 기준으로 박아두기)

- [ ] `domain/`, `application/` 안에서 `import fastapi`, `import sqlalchemy`가 **없다** (있으면 리젝트)
- [ ] 라우터에는 비즈니스 로직이 없다 — 입력 검증 → 서비스 호출 → 예외 매핑만
- [ ] DB 모델(`UserModel`)이 라우터/서비스로 새어 나가지 않는다 — 경계에서 엔티티/스키마로 변환
- [ ] 도메인 예외 → HTTP 상태 코드 매핑은 **presentation에서만**
- [ ] 새 기능은 기존 `user/` 슬라이스를 복제해 동일 패턴 유지
- [ ] 커밋/브랜치: `feat/user-register` 처럼 기능 단위

---

## 7. 흔한 실수 / 함정

| 실수 | 증상 | 해결 |
|---|---|---|
| 라우터에 비즈니스 로직 작성 | 라우터가 100줄+, 테스트 불가 | 서비스로 이동 |
| ORM 모델을 응답으로 그대로 반환 | DB 스키마 변경이 API를 깨뜨림 | Pydantic 응답 DTO 분리 |
| `async def` 안에서 동기 DB 드라이버/무거운 CPU 작업 | p95 지연 폭증, 이벤트 루프 블로킹 | async 드라이버 사용, CPU 작업은 워커로 |
| Pydantic v1/v2 혼용 | `orm_mode` 등에서 에러 | v2로 통일 (`from_attributes`) |
| 처음부터 L3 풀스펙 | 파일 폭발, 진도 안 나감 | L2로 시작, 필요한 곳만 승급 |
| 인터페이스 합의 없이 병렬 개발 | 통합 시 대규모 충돌 | 0일차 경계 회의 먼저 |
| `@app.on_event` 사용 | 폐기 경고 | `lifespan` 사용 |

---

## 8. 단계별 도입 로드맵 (그룹 프로젝트용)

**1주차 — 토대**
- 레포 생성, 폴더 구조(4장) 스캐폴딩, `core/config.py`·`database.py`, Docker Compose(Postgres) 세팅
- 팀 컨벤션(6.5) 문서화, CI에 ruff/black/pytest 연결

**2주차 — 경계 합의 + 첫 슬라이스**
- 기능별 Repository 인터페이스·Schema를 함께 정의해 머지(6.2)
- `user` 슬라이스 하나를 4계층 전부 구현해 **레퍼런스 패턴**으로 삼음

**3~N주차 — 병렬 확장**
- 각자 담당 도메인 슬라이스를 레퍼런스 복제로 개발
- 단위 테스트(인메모리)부터, 통합 테스트는 기능 완성 시
- 주 1회 "경계 위반(import 규칙) 점검" 리뷰

**마무리 주차**
- Alembic 마이그레이션 정리, 통합 테스트 보강, OpenAPI 문서 점검, 배포(Docker)

---

## 9. 참고 레퍼런스

- FastAPI 공식 문서 — Dependencies, Bigger Applications, Lifespan Events (fastapi.tiangolo.com)
- Clean Architecture (Robert C. Martin) — 의존성 규칙의 원전
- 템플릿: `BrunoTanabe/fastapi-clean-architecture-ddd-template`, `0xTheProDev/fastapi-clean-example` (GitHub) — 구조 참고용
- SQLAlchemy 2.0 async + Pydantic v2 셋업 가이드, Repository 패턴 in Python

---

*이 리포트는 2026년 중반 시점의 FastAPI 0.136.x / Pydantic v2 / SQLAlchemy 2.0 생태계를 기준으로 작성되었습니다. 코드는 패턴 전달용 골격이며, 실제 적용 시 비밀번호 해싱(passlib/bcrypt), 환경변수 분리, 인증(JWT) 등을 추가하세요.*
