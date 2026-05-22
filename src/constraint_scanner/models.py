from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Loose(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class Answer(_Loose):
    id: str
    text: str = ""
    index: int | None = None
    # The public API renames the internal `prob` field to `probability` and
    # replaces poolYes/poolNo with a single `pool: {YES, NO}` dict — see
    # common/src/api/market-types.ts (ApiAnswer / augmentAnswerWithProbability).
    probability: float | None = None
    pool: dict[str, float] | None = None
    # Numeric bucket centre for NUMBER / MULTI_NUMERIC / DATE answers.
    midpoint: float | None = None
    is_other: bool = Field(default=False, alias="isOther")
    resolution: str | None = None


class LiteMarket(_Loose):
    id: str
    question: str
    url: str | None = None

    outcome_type: str = Field(alias="outcomeType")
    mechanism: str | None = None

    probability: float | None = None
    pool: dict[str, float] | None = None
    p: float | None = None
    total_liquidity: float | None = Field(default=None, alias="totalLiquidity")

    value: float | None = None
    min: float | None = None
    max: float | None = None
    is_log_scale: bool | None = Field(default=None, alias="isLogScale")

    volume: float = 0.0
    volume_24_hours: float | None = Field(default=None, alias="volume24Hours")

    is_resolved: bool = Field(default=False, alias="isResolved")
    resolution: str | None = None
    resolution_probability: float | None = Field(default=None, alias="resolutionProbability")

    created_time: int | None = Field(default=None, alias="createdTime")
    close_time: int | None = Field(default=None, alias="closeTime")
    last_updated_time: int | None = Field(default=None, alias="lastUpdatedTime")
    last_bet_time: int | None = Field(default=None, alias="lastBetTime")

    creator_id: str | None = Field(default=None, alias="creatorId")
    creator_username: str | None = Field(default=None, alias="creatorUsername")


class FullMarket(LiteMarket):
    answers: list[Answer] | None = None
    should_answers_sum_to_one: bool | None = Field(default=None, alias="shouldAnswersSumToOne")
    add_answers_mode: str | None = Field(default=None, alias="addAnswersMode")
    unit: str | None = None  # MULTI_NUMERIC unit label, e.g. "USD", "year"
    description: str | dict | None = None
    text_description: str | None = Field(default=None, alias="textDescription")
