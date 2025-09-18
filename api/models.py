from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, JSON, create_engine
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime
from api.config import settings
import re

Base = declarative_base()

class Keyword(Base):
    __tablename__ = "keyword"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, unique=True, nullable=False)

    ranks = relationship("Rank", back_populates="keyword")


class Engine(Base):
    __tablename__ = "engine"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)

    ranks = relationship("Rank", back_populates="engine")


class Rank(Base):
    __tablename__ = "rank"

    id = Column(Integer, primary_key=True, index=True)
    keyword_id = Column(Integer, ForeignKey("keyword.id"))
    engine_id = Column(Integer, ForeignKey("engine.id"))
    position = Column(Integer, nullable=False)
    domain = Column(String, nullable=False)  # ✅ required, consistent with app inserts
    fetched_at = Column(DateTime, default=datetime.utcnow)

    keyword = relationship("Keyword", back_populates="ranks")
    engine = relationship("Engine", back_populates="ranks")


class SerpFeature(Base):
    __tablename__ = "serp_feature"

    id = Column(Integer, primary_key=True, index=True)
    keyword_id = Column(Integer, ForeignKey("keyword.id"))
    type = Column(String, nullable=False)
    data = Column(JSON)


class Anomaly(Base):
    __tablename__ = "anomaly"

    id = Column(Integer, primary_key=True, index=True)
    keyword_id = Column(Integer, ForeignKey("keyword.id"))
    metric = Column(String, nullable=False)
    score = Column(Float, nullable=False)
    detected_at = Column(DateTime, default=datetime.utcnow)


# ✅ Helper: Reset DB schema (drop & recreate all tables)
def reset_db():
    sync_url = re.sub(r"\+asyncpg", "", settings.DATABASE_URL)
    engine = create_engine(sync_url, pool_pre_ping=True)

    print("⚠️ Dropping all tables...")
    Base.metadata.drop_all(engine)
    print("✅ Creating fresh tables...")
    Base.metadata.create_all(engine)
    print("🎉 Database schema reset complete.")
