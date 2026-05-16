"""
OncoVision AI — Database Layer
===============================
SQLite persistence for diagnostic reports using SQLAlchemy.
Each biopsy analysis is stored with its image, AI prediction,
biomarkers, confidence, risk label, and full case analysis.
"""

import os
import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import create_engine, Column, String, Float, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

# ---------------------------------------------------------------------------
# Database configuration
# ---------------------------------------------------------------------------

DB_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DB_DIR, exist_ok=True)

DATABASE_URL = f"sqlite:///{os.path.join(DB_DIR, 'oncovision.db')}"
UPLOAD_DIR = os.path.join(DB_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ---------------------------------------------------------------------------
# ORM Model
# ---------------------------------------------------------------------------

class DiagnosisRecord(Base):
    """Persistent record of a single biopsy analysis."""

    __tablename__ = "diagnoses"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Original upload
    filename = Column(String, nullable=False)
    image_path = Column(String, nullable=False)  # Local path to saved image

    # AI prediction
    prediction = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    risk_label = Column(String, nullable=False)

    # Biological indicators (stored as JSON string)
    biological_indicators = Column(Text, nullable=False)

    # Full case analysis (stored as JSON string)
    case_analysis = Column(Text, nullable=False)

    def to_dict(self) -> dict:
        """Serialize record to a JSON-safe dictionary."""
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "filename": self.filename,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "risk_label": self.risk_label,
            "biological_indicators": json.loads(self.biological_indicators),
            "case_analysis": json.loads(self.case_analysis),
        }


# ---------------------------------------------------------------------------
# Create tables on import
# ---------------------------------------------------------------------------
Base.metadata.create_all(bind=engine)
