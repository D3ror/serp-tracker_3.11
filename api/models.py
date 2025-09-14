from sqlalchemy import (
    Column, Integer, String, Date, DateTime, ForeignKey, Float, JSON, Text, UniqueConstraint
)
from sqlalchemy.orm import relationship
from datetime import datetime
from api.db import Base

class Engine(Base):
    __tablename__ = "engine"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, index=True)

class SerpProvider(Base):
    __tablename__ = "provider"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, index=True)

class Keyword(Base):
    __tablename__ = "keyword"
    id = Column(Integer, primary_key=True)
    text = Column(String, index=True, nullable=False)
    engine_id = Column(Integer, ForeignKey("engine.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    tags = Column(JSON, nullable=True)  # categories/tags
    vol_7 = Column(Float, nullable=True)
    vol_30 = Column(Float, nullable=True)

    engine = relationship("Engine", lazy="joined")
    ranks = relationship("Rank", back_populates="keyword", cascade="all, delete-orphan")

class Rank(Base):
    __tablename__ = "rank"
    id = Column(Integer, primary_key=True)
    keyword_id = Column(Integer, ForeignKey("keyword.id", ondelete="CASCADE"))
    date = Column(Date, index=True)
    rank = Column(Integer, nullable=True)
    url = Column(String, nullable=True)
    snippet = Column(Text, nullable=True)
    serp_features = Column(JSON, nullable=True)
    provider_id = Column(Integer, ForeignKey("provider.id"), nullable=True)

    keyword = relationship("Keyword", back_populates="ranks")
    provider = relationship("SerpProvider")

    __table_args__ = (UniqueConstraint("keyword_id", "date", name="uq_keyword_date"),)

class Anomaly(Base):
    __tablename__ = "anomaly"
    id = Column(Integer, primary_key=True)
    keyword_id = Column(Integer, ForeignKey("keyword.id"))
    date = Column(Date, index=True)
    score = Column(Float)
    method = Column(String)
    payload = Column(JSON, nullable=True)

    keyword = relationship("Keyword")
