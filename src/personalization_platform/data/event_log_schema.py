from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FieldContract:
    name: str
    dtype: str
    description: str
    required: bool = True
    source: str | None = None


@dataclass(frozen=True)
class TableContract:
    name: str
    grain: str
    description: str
    required_fields: tuple[FieldContract, ...]
    optional_fields: tuple[FieldContract, ...] = ()
    future_fields: tuple[FieldContract, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "grain": self.grain,
            "description": self.description,
            "required_fields": [asdict(field) for field in self.required_fields],
            "optional_fields": [asdict(field) for field in self.optional_fields],
            "future_fields": [asdict(field) for field in self.future_fields],
        }


REQUESTS_CONTRACT = TableContract(
    name="requests",
    grain="One row per feed request derived from a user impression group within a session.",
    description=(
        "Captures the request context that produced a set of candidate impressions. "
        "For the first MIND slice, each request is inferred from one impression row in "
        "behaviors.tsv and grouped into a lightweight user session."
    ),
    required_fields=(
        FieldContract(
            name="request_id",
            dtype="string",
            description="Stable request identifier for joining request-level features and impressions.",
            source="derived from split, impression_id, and row position",
        ),
        FieldContract(
            name="user_id",
            dtype="string",
            description="User identifier from MIND behaviors.",
            source="behaviors.tsv user_id",
        ),
        FieldContract(
            name="session_id",
            dtype="string",
            description="Inferred session identifier used to group nearby requests for a user.",
            source="derived from user_id and timestamp sessionization rule",
        ),
        FieldContract(
            name="request_ts",
            dtype="timestamp",
            description="Timestamp when the feed request occurred.",
            source="behaviors.tsv time",
        ),
        FieldContract(
            name="split",
            dtype="string",
            description="Dataset split used to build the request, such as train or valid.",
            source="pipeline config",
        ),
        FieldContract(
            name="candidate_count",
            dtype="int",
            description="Number of items shown in the impression set for the request.",
            source="count of parsed impressions",
        ),
    ),
    optional_fields=(
        FieldContract(
            name="history_length",
            dtype="int",
            description="Number of clicked history items available before the request.",
            required=False,
            source="count of behaviors.tsv history entries",
        ),
        FieldContract(
            name="request_index_in_session",
            dtype="int",
            description="Position of the request within the inferred session.",
            required=False,
            source="derived from per-session ordering",
        ),
    ),
    future_fields=(
        FieldContract(
            name="device_type",
            dtype="string",
            description="Semi-synthetic device label for downstream ranking and experimentation.",
            required=False,
        ),
        FieldContract(
            name="experiment_unit_id",
            dtype="string",
            description="Stable bucketing unit that may later differ from user_id.",
            required=False,
        ),
    ),
)


IMPRESSIONS_CONTRACT = TableContract(
    name="impressions",
    grain="One row per candidate item shown within a request.",
    description=(
        "Captures candidate exposure and label state for ranking training and retrieval evaluation. "
        "Each row comes from splitting the MIND impressions string into item-level records."
    ),
    required_fields=(
        FieldContract(
            name="impression_id",
            dtype="string",
            description="Stable identifier for the request-item exposure record.",
            source="derived from request_id and item_id",
        ),
        FieldContract(
            name="request_id",
            dtype="string",
            description="Request identifier joining back to the requests table.",
            source="derived request_id",
        ),
        FieldContract(
            name="user_id",
            dtype="string",
            description="Repeated user identifier for convenient downstream aggregation.",
            source="behaviors.tsv user_id",
        ),
        FieldContract(
            name="item_id",
            dtype="string",
            description="Candidate content item identifier.",
            source="parsed from behaviors.tsv impressions",
        ),
        FieldContract(
            name="position",
            dtype="int",
            description="Display order of the item within the request candidate set.",
            source="position in parsed impressions list",
        ),
        FieldContract(
            name="clicked",
            dtype="int",
            description="Binary label indicating whether the impression was clicked.",
            source="parsed from behaviors.tsv impressions label",
        ),
    ),
    optional_fields=(
        FieldContract(
            name="topic",
            dtype="string",
            description="Primary item topic carried through from the item-state table for debugging.",
            required=False,
            source="news.tsv category or vertical",
        ),
        FieldContract(
            name="candidate_source",
            dtype="string",
            description="Source label for retrieval once multi-source candidates exist.",
            required=False,
            source="future retrieval pipeline",
        ),
    ),
    future_fields=(
        FieldContract(
            name="dwell_seconds",
            dtype="float",
            description="Semi-synthetic or observed engagement depth label.",
            required=False,
        ),
        FieldContract(
            name="served_rank_score",
            dtype="float",
            description="Model score used for replay-style diagnostics.",
            required=False,
        ),
    ),
)


USER_STATE_CONTRACT = TableContract(
    name="user_state",
    grain="One row per request capturing the user state available before ranking that request.",
    description=(
        "Provides a snapshot of light user history before the request. This stays intentionally narrow "
        "for the smoke slice so later ranking code can join stable columns without requiring a full "
        "feature store design."
    ),
    required_fields=(
        FieldContract(
            name="request_id",
            dtype="string",
            description="Request identifier for point-in-time joins.",
            source="derived request_id",
        ),
        FieldContract(
            name="user_id",
            dtype="string",
            description="User identifier for aggregation and experiment assignment.",
            source="behaviors.tsv user_id",
        ),
        FieldContract(
            name="history_item_ids",
            dtype="array[string]",
            description="Ordered list of prior clicked items visible at request time.",
            source="behaviors.tsv history",
        ),
        FieldContract(
            name="history_click_count",
            dtype="int",
            description="Number of prior clicked items in the visible history.",
            source="count of history_item_ids",
        ),
        FieldContract(
            name="is_cold_start",
            dtype="bool",
            description="Whether the request has no prior click history in the source data.",
            source="history_click_count == 0",
        ),
    ),
    optional_fields=(
        FieldContract(
            name="recent_topic_counts",
            dtype="map[string,int]",
            description="Aggregated topic counts over visible history when item metadata is available.",
            required=False,
            source="derived from history joined to news.tsv metadata",
        ),
    ),
    future_fields=(
        FieldContract(
            name="fatigue_state",
            dtype="map[string,float]",
            description="Topic or creator fatigue state for reranking constraints.",
            required=False,
        ),
        FieldContract(
            name="subscription_tier",
            dtype="string",
            description="Semi-synthetic user segment for experimentation and guardrails.",
            required=False,
        ),
    ),
)


ITEM_STATE_CONTRACT = TableContract(
    name="item_state",
    grain="One row per content item referenced by the event-log slice.",
    description=(
        "Captures item metadata required for retrieval, ranking, and reporting. The first version should "
        "stay close to raw MIND news metadata while making room for explicit operational fields later."
    ),
    required_fields=(
        FieldContract(
            name="item_id",
            dtype="string",
            description="Stable content item identifier.",
            source="news.tsv news_id",
        ),
        FieldContract(
            name="topic",
            dtype="string",
            description="Primary editorial category used by retrieval and diversity constraints.",
            source="news.tsv category",
        ),
        FieldContract(
            name="subcategory",
            dtype="string",
            description="Secondary editorial category or subtopic.",
            source="news.tsv subcategory",
        ),
        FieldContract(
            name="title",
            dtype="string",
            description="Human-readable title for debugging and reporting.",
            source="news.tsv title",
        ),
        FieldContract(
            name="publisher",
            dtype="string",
            description="Publisher or source field used as the initial creator proxy.",
            source="Derived from article URL hostname when a dedicated source field is absent.",
        ),
        FieldContract(
            name="creator_id",
            dtype="string",
            description="Normalized creator identifier used for creator-spread constraints and reporting.",
            source="Derived from publisher plus subcategory when no direct creator field exists.",
        ),
        FieldContract(
            name="published_ts",
            dtype="timestamp",
            description="Published timestamp when available, otherwise null in the first slice.",
            source="news.tsv or null if absent in fixture/raw input",
        ),
    ),
    optional_fields=(
        FieldContract(
            name="abstract",
            dtype="string",
            description="Short content summary for exploratory reporting.",
            required=False,
            source="news.tsv abstract",
        ),
        FieldContract(
            name="entity_ids",
            dtype="array[string]",
            description="Resolved entity identifiers when entity metadata is present.",
            required=False,
            source="news.tsv title_entities and abstract_entities",
        ),
    ),
    future_fields=(
        FieldContract(
            name="freshness_hours",
            dtype="float",
            description="Derived item age used in freshness-aware ranking and reranking.",
            required=False,
        ),
    ),
)


SCHEMA_ASSUMPTIONS = (
    "MIND behaviors rows are treated as request-level records; request_id is inferred because the raw dataset does not provide a stable request key.",
    "Session boundaries are inferred from per-user timestamp gaps and may be simplified in the smoke implementation.",
    "User state is point-in-time and limited to information visible before the request, primarily the provided click history.",
    "Item metadata comes from MIND news records, while publisher, creator, freshness, and experiment fields may need to be derived when the raw source omits them.",
    "Impression labels are binary clicks only in the first slice; richer engagement targets are future extensions.",
)


MIND_MAPPING_NOTES = {
    "behaviors.tsv.impression_id": "Used as part of request_id derivation and request lineage.",
    "behaviors.tsv.user_id": "Maps directly to requests.user_id, impressions.user_id, and user_state.user_id.",
    "behaviors.tsv.time": "Maps to requests.request_ts and informs inferred session_id.",
    "behaviors.tsv.history": "Maps to user_state.history_item_ids and history_click_count.",
    "behaviors.tsv.impressions": "Explodes into impressions.item_id, impressions.position, and impressions.clicked.",
    "news.tsv.news_id": "Maps directly to item_state.item_id.",
    "news.tsv.category": "Maps to item_state.topic.",
    "news.tsv.subcategory": "Maps to item_state.subcategory.",
    "news.tsv.title": "Maps to item_state.title.",
    "news.tsv.abstract": "Maps to item_state.abstract when present.",
}


def build_event_log_schema_contract() -> dict[str, object]:
    tables = [
        REQUESTS_CONTRACT,
        IMPRESSIONS_CONTRACT,
        USER_STATE_CONTRACT,
        ITEM_STATE_CONTRACT,
    ]
    return {
        "contract_version": "v1",
        "design_goal": (
            "Define the minimum stable event-log surface for the first MIND-derived smoke pipeline "
            "without overcommitting to later ranking or experimentation features."
        ),
        "tables": [table.to_dict() for table in tables],
        "assumptions": list(SCHEMA_ASSUMPTIONS),
        "mind_mapping_notes": MIND_MAPPING_NOTES,
    }
