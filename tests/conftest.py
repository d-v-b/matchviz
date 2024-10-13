import pytest

BIGSTITCHER_XMLS = [
    "s3://aind-open-data/exaSPIM_708373_2024-04-02_19-49-38_alignment_2024-05-07_18-15-25/bigstitcher.xml"
    ]

@pytest.fixture(scope="session")
def bigstitcher_xml(request: pytest.FixtureRequest) -> str:
    return BIGSTITCHER_XMLS[request.param]