import chromadb
import pytest
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.retrieve import retrieve

_EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"


@pytest.fixture(scope="module")
def housing_collection():
    ef = SentenceTransformerEmbeddingFunction(model_name=_EMBEDDING_MODEL)
    client = chromadb.EphemeralClient()
    col = client.get_or_create_collection(name="test_housing", embedding_function=ef)
    col.upsert(
        ids=["c0", "c1", "c2", "c3", "c4", "c5"],
        documents=[
            "## Security Deposit\nLandlords may charge no more than 1.5× monthly rent as a security deposit under Michigan law.",
            "## Bus Pass\nWMU students ride all KMetro bus routes for free with their Bronco Card, including routes 3, 16, and 21.",
            "## Henry Hall Rates\nHenry Hall double room costs $6,395.50 per semester with the Bronco Gold Plus meal plan included.",
            "## Lease Cancellation Fees\nCancelling a WMU apartment contract after April 1 costs 100% of one full month's rent.",
            "## Utility Costs\nKalamazoo winter heating bills average $80–$120 per month for a typical one-bedroom apartment.",
            "## Neighborhood Safety\nThe Vine neighborhood near downtown Kalamazoo has higher crime rates than other student rental zones near WMU.",
        ],
        metadatas=[
            {"source_file": "source7_michigan_tenant_law.md", "header": "## Security Deposit", "char_count": 103},
            {"source_file": "source6_winter_commuting.md", "header": "## Bus Pass", "char_count": 102},
            {"source_file": "source1_wmu_residence_halls.md", "header": "## Henry Hall Rates", "char_count": 99},
            {"source_file": "source2_wmu_apartment_policies.md", "header": "## Lease Cancellation Fees", "char_count": 95},
            {"source_file": "source8_utility_costs.md", "header": "## Utility Costs", "char_count": 89},
            {"source_file": "source5_neighborhood_geography.md", "header": "## Neighborhood Safety", "char_count": 102},
        ],
    )
    return col


def test_retrieve_returns_k_results(housing_collection) -> None:
    results = retrieve("How much does on-campus housing cost?", k=3, collection=housing_collection)
    assert len(results) == 3


def test_retrieve_result_has_required_keys(housing_collection) -> None:
    results = retrieve("security deposit rules", k=1, collection=housing_collection)
    assert len(results) == 1
    r = results[0]
    assert "text" in r
    assert "source_file" in r
    assert "header" in r
    assert "distance" in r


def test_retrieve_distance_is_float(housing_collection) -> None:
    results = retrieve("tenant rights", k=2, collection=housing_collection)
    for r in results:
        assert isinstance(r["distance"], float)


def test_retrieve_default_k_is_5(housing_collection) -> None:
    results = retrieve("housing options at WMU", collection=housing_collection)
    assert len(results) == 5


def test_retrieve_returns_relevant_chunk_for_bus_query(housing_collection) -> None:
    results = retrieve("Is the KMetro bus free for WMU students?", k=3, collection=housing_collection)
    top_source_files = [r["source_file"] for r in results]
    assert "source6_winter_commuting.md" in top_source_files


def test_retrieve_k_exceeding_collection_size_returns_all(housing_collection) -> None:
    results = retrieve("housing", k=20, collection=housing_collection)
    assert len(results) == 6
