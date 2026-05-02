from inference_pipeline import PaperMetadata, doi_syntax_plausible, explain_candidate_ranking


def test_doi_syntax():
    assert doi_syntax_plausible("10.1000/182") is True
    assert doi_syntax_plausible("10.12/short") is False


def test_explain_percentages():
    m = PaperMetadata(
        title="Machine Learning Basics",
        authors=["Anna Smith", "Li Wei"],
        year="2020",
        doi="10.1000/182",
        search_score=0.42,
    )
    text = explain_candidate_ranking(
        "Machine Learning Basics 2020 Smith",
        "10.1000/182",
        m,
    )
    assert "объяснение" in text.lower()
    assert "≈" in text or "%" in text
