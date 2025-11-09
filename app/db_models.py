from sqlalchemy import String, Float, DateTime, func, Integer
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    text: Mapped[str] = mapped_column(String)
    pred_label: Mapped[str] = mapped_column(String)
    prob: Mapped[float] = mapped_column(Float)
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(primary_key=True)
    text: Mapped[str] = mapped_column(String)
    pred_label: Mapped[str] = mapped_column(String)
    true_label: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())
