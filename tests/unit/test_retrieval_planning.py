import pytest

from local_lke.errors import RetrievalError
from local_lke.models import (
    MetadataCondition,
    MetadataFilterPlan,
    MetadataOperator,
    QueryRoute,
    RetrievalStrategy,
    RewriteStrategy,
    StructuredQueryPlan,
)
from local_lke.providers import FakeChatProvider
from local_lke.retrieval.planning import MetadataPlanParser, build_query_plan, parse_model_plan


def test_multi_part_plan_is_bounded_and_inspectable() -> None:
    plan = build_query_plan(
        "What is Atlas? And also how is Atlas retained?; Who owns Atlas?",
        strategy=RetrievalStrategy.HYBRID,
        rewrite=RewriteStrategy.STEP_BACK,
        max_subqueries=3,
    )

    assert plan.route is QueryRoute.MULTI_PART
    assert len(plan.subqueries) == 3
    assert plan.rewritten_query is not None
    assert "not evidence" in plan.rationale


def test_metadata_plan_rejects_unallowlisted_fields_and_bad_types() -> None:
    with pytest.raises(ValueError):
        MetadataFilterPlan.model_validate(
            {"conditions": [{"field": "raw_sql", "operator": "eq", "value": "x"}]}
        )
    with pytest.raises(ValueError):
        MetadataCondition(field="page_number", operator=MetadataOperator.GT, value="ten")


def test_model_output_must_be_json_matching_the_plan_schema() -> None:
    parsed = parse_model_plan(
        '{"projections":["region"],"filters":[],"group_by":[],"aggregations":[],"order_by":[],"limit":5}',
        StructuredQueryPlan,
    )
    assert parsed.projections == ["region"]

    with pytest.raises(RetrievalError, match="allowlisted query plan"):
        parse_model_plan("SELECT * FROM users", StructuredQueryPlan)
    with pytest.raises(RetrievalError, match="allowlisted query plan"):
        parse_model_plan('{"raw_sql":"DROP TABLE users"}', StructuredQueryPlan)


def test_model_metadata_output_is_reduced_to_a_pydantic_plan() -> None:
    parser = MetadataPlanParser(
        FakeChatProvider(
            '{"conditions":[{"field":"filename","operator":"contains",'
            '"value":"runbook"}],"allow_unfiltered_fallback":false}'
        )
    )

    plan = parser.plan("Use only the runbook")

    assert plan.conditions[0].field == "filename"
    assert plan.conditions[0].operator is MetadataOperator.CONTAINS
