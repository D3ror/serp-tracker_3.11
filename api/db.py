from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from api.config import settings

# Async engine for FastAPI/jobs
ASYNC_DATABASE_URL = settings.DATABASE_URL
engine = create_async_engine(ASYNC_DATABASE_URL, future=True, echo=False, pool_size=5, max_overflow=10)
AsyncSessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()

# helper dependency for FastAPI
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
