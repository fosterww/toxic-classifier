from pydantic import BaseModel, Field, ConfigDict


class HealthOut(BaseModel):
    status: str
    model_version: str

    model_config = ConfigDict(protected_namespaces=())


class PredictIn(BaseModel):
    text: str = Field(min_length=1, max_length=5000, description="Raw comment text")


class PredictOut(BaseModel):
    label: str
    prob: float
    low_confidence: bool


class FeedbackIn(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    true_label: int = Field(ge=0, le=1, description="0=clean, 1=toxic")


class FeedbackOut(BaseModel):
    status: str
