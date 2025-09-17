# api/models.py
from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, JSON
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime

Base = declarative_base()

class Keyword(Base):
    __tablename__ = "keyword"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, unique=True, nullable=False)

    ranks = relationship("Rank", back_populates="keyword", cascade="all, delete-orphan")
    anomalies = relationship("Anomaly", back_populates="keyword", cascade="all, delete-orphan")
    features = relationship("SerpFeature", back_populates="keyword", cascade="all, delete-orphan")


class Engine(Base):
    __tablename__ = "engine"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)

    ranks = relationship("Rank", back_populates="engine", cascade="all, delete-orphan")


class Rank(Base):
    __tablename__ = "rank"

    id = Column(Integer, primary_key=True, index=True)
    keyword_id = Column(Integer, ForeignKey("keyword.id"), nullable=False)
    engine_id = Column(Integer, ForeignKey("engine.id"), nullable=False)

    domain = Column(String, nullable=False)                # which domain ranked
    position = Column(Integer, nullable=False)             # SERP position (1 = top)
    fetched_at = Column(DateTime, default=datetime.utcnow) # timestamp of fetch

    keyword = relationship("Keyword", back_populates="ranks")
    engine = relationship("Engine", back_populates="ranks")


class SerpFeature(Base):
    __tablename__ = "serp_feature"

    id = Column(Integer, primary_key=True, index=True)
    keyword_id = Column(Integer, ForeignKey("keyword.id"), nullable=False)
    type = Column(String, nullable=False)  # e.g. "featured_snippet", "local_pack"
    data = Column(JSON, nullable=True)     # flexible metadata blob

    keyword = relationship("Keyword", back_populates="features")


class Anomaly(Base):
    __tablename__ = "anomaly"

    id = Column(Integer, primary_key=True, index=True)
    keyword_id = Column(Integer, ForeignKey("keyword.id"), nullable=False)
    metric = Column(String, nullable=False)              # what was measured
    score = Column(Float, nullable=False)                # z-score or volatility index
    detected_at = Column(DateTime, default=datetime.utcnow)

    keyword = relationship("Keyword", back_populates="anomalies")
