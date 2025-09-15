# api/models.py
from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, JSON
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime

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
