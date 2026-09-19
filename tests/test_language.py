from dataclasses import replace
import pandas as pd
from marathon_absa.config import SETTINGS
from marathon_absa.language import HybridLanguageService


class FakeOpenLID:
    def predict(self,text,k=2):
        lower=text.lower()
        if "cuaca" in lower or "sangat" in lower: labels=["__label__zsm_Latn","__label__ind_Latn"]
        else: labels=["__label__eng_Latn","__label__deu_Latn"]
        scores=[0.94,0.03]
        return labels[:k],scores[:k]


def service():
    obj=HybridLanguageService(replace(SETTINGS,openlid_filename="missing-test-model.bin")); obj.openlid=FakeOpenLID(); return obj


def test_empty_is_no_text():
    result=service().detect("",""); assert result["language_status"]=="no_text"; assert result["primary_language"]=="no_text"


def test_metadata_only_is_insufficient():
    result=service().detect("🏃 #KLSCM2024",""); assert result["language_status"]=="insufficient_text"


def test_english_primary_language():
    result=service().detect("The route was beautiful","The route was beautiful"); assert result["primary_language"]=="English"; assert result["language_method"]=="openlid_constrained"


def test_mixed_language_components():
    result=service().detect("The route was beautiful. Cuaca sangat panas.","The route was beautiful. Cuaca sangat panas.")
    names={x["language"] for x in result["detected_languages"]}; assert {"English","Malay"}.issubset(names); assert result["is_mixed_language"]


def test_apply_preserves_compatibility_alias():
    frame=pd.DataFrame([{"normalized_text":"The route was beautiful","linguistic_text":"The route was beautiful"}]); result=service().apply(frame); assert result.loc[0,"detected_language"]==result.loc[0,"primary_language"]



def test_numeric_tail_does_not_create_empty_language_span():
    detector = service()
    text = "The marathon route and volunteers were absolutely wonderful today 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17"
    result = detector.detect(text, text)
    assert result["primary_language"] == "English"

def test_non_linguistic_span_does_not_create_mixed_language():
    detector = service()
    result = detector._compose(
        "Kids Dash done! Great weather this morning",
        "English",
        0.82,
        [("English", "eng_Latn", 0.8), ("zxx", "zxx_Zxxx", 0.1)],
        [[("zxx", "zxx_Zxxx", 0.62)], [("English", "eng_Latn", 0.40)]],
    )
    assert result["primary_language"] == "English"
    assert not result["is_mixed_language"]
    assert [item["language"] for item in result["detected_languages"]] == ["English"]

def test_malay_shortforms_support_malay_detection():
    result = service().detect("sy dh daftar utk larian ni", "sy dh daftar utk larian ni")
    assert result["primary_language"] == "Malay"
    assert all(item["language"] != "Indonesian" for item in result["detected_languages"])


def test_raw_openlid_code_is_never_exposed():
    from marathon_absa.language import _label
    name, _ = _label("__label__hne_Deva")
    assert name != "hne"