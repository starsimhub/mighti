"""Tests for default GBD disability weights and multimorbidity rules (Tasks A/C)."""

import numpy as np

from mighti.analyzers.disability_weights import (
    CORE_CEA_CONDITIONS,
    DEFAULT_GBD_DISABILITY_WEIGHTS,
    DEFAULT_MULTIMORBIDITY_RULE,
    HIV_STAGE_DISABILITY_WEIGHTS,
    SENSITIVITY_MULTIMORBIDITY_RULE,
    adjust_total_yld_for_multimorbidity,
    canonical_dw_key,
    classify_hiv_stage,
    combine_disability_weights,
    get_default_disability_weights,
    hiv_stage_disability_weight,
    multimorbidity_scale,
    resolve_disability_weights,
    resolve_disease_module,
)


def test_core_conditions_present():
    assert set(CORE_CEA_CONDITIONS) == {
        "hiv",
        "type2diabetes",
        "hypertension",
        "cardiovasculardiseases",
        "chronickidneydisease",
    }
    for key, weight in DEFAULT_GBD_DISABILITY_WEIGHTS.items():
        assert 0.0 < weight < 1.0


def test_aliases_canonicalize():
    assert canonical_dw_key("t2d") == "type2diabetes"
    assert canonical_dw_key("Type2Diabetes") == "type2diabetes"
    assert canonical_dw_key("HTN") == "hypertension"
    assert canonical_dw_key("CKD") == "chronickidneydisease"
    assert canonical_dw_key("CVD") == "cardiovasculardiseases"
    assert canonical_dw_key("HIV") == "hiv"


def test_resolve_none_uses_defaults():
    dws = resolve_disability_weights(None)
    assert dws == DEFAULT_GBD_DISABILITY_WEIGHTS
    assert dws is not DEFAULT_GBD_DISABILITY_WEIGHTS  # copy


def test_resolve_explicit_dict_replaces_defaults():
    dws = resolve_disability_weights({"t2d": 0.10, "hiv": 0.20})
    assert dws == {"type2diabetes": 0.10, "hiv": 0.20}


def test_resolve_empty_dict_means_no_yld():
    assert resolve_disability_weights({}) == {}


def test_get_default_subset():
    subset = get_default_disability_weights(["HIV", "Type2Diabetes"])
    assert set(subset) == {"hiv", "type2diabetes"}


class _FakeDiseases:
    def __init__(self, **mods):
        for k, v in mods.items():
            setattr(self, k, v)

    def values(self):
        return [v for k, v in self.__dict__.items() if not k.startswith("_")]

    def get(self, key, default=None):
        return getattr(self, key, default)


class _FakeDisease:
    def __init__(self, name, disease_name=None):
        self.name = name
        self.disease_name = disease_name or name
        self.duration = [1.0]


def test_resolve_disease_module_alias():
    diseases = _FakeDiseases(type2diabetes=_FakeDisease("type2diabetes", "Type2Diabetes"))
    mod, key = resolve_disease_module(diseases, "t2d")
    assert mod is not None
    assert key == "type2diabetes"


def test_microcosting_analyzer_defaults_without_manual_dict():
    from mighti.analyzers.analyzer_cost import MicrocostingAnalyzer

    an = MicrocostingAnalyzer(unit_costs={"art": 50.0})
    assert "hiv" in an.disability_weights
    assert "type2diabetes" in an.disability_weights
    assert "hypertension" in an.disability_weights
    assert "cardiovasculardiseases" in an.disability_weights
    assert "chronickidneydisease" in an.disability_weights
    assert an.multimorbidity_rule == DEFAULT_MULTIMORBIDITY_RULE
    assert an.hiv_yld_mode == "average"


def test_classify_hiv_stage_priority():
    infected = np.array([True, True, True, True, False])
    on_art = np.array([True, False, False, False, False])
    cd4 = np.array([600.0, 150.0, 280.0, 500.0, np.nan])
    falling = np.array([False, False, True, False, False])
    stages = classify_hiv_stage(
        infected=infected, on_art=on_art, cd4=cd4, falling=falling
    )
    assert list(stages) == ["art", "aids", "symptomatic", "early", ""]
    dws = hiv_stage_disability_weight(stages)
    assert dws[0] == HIV_STAGE_DISABILITY_WEIGHTS["art"]
    assert dws[1] == HIV_STAGE_DISABILITY_WEIGHTS["aids"]
    assert dws[4] == 0.0


def test_microcosting_hiv_yld_mode_stage_flag():
    from mighti.analyzers.analyzer_cost import MicrocostingAnalyzer

    an = MicrocostingAnalyzer(hiv_yld_mode="stage")
    assert an.hiv_yld_mode == "stage"


def test_combine_rules_hiv_t2d():
    ws = [0.20, 0.10]
    assert abs(combine_disability_weights(ws, "additive") - 0.30) < 1e-12
    assert abs(combine_disability_weights(ws, "multiplicative") - (1 - 0.8 * 0.9)) < 1e-12
    assert combine_disability_weights(ws, "maximum") == 0.20
    assert DEFAULT_MULTIMORBIDITY_RULE == "multiplicative"
    assert SENSITIVITY_MULTIMORBIDITY_RULE == "additive"


def test_multimorbidity_scale_and_adjust():
    ws = [0.20, 0.10]
    multi = combine_disability_weights(ws, "multiplicative")
    assert abs(multimorbidity_scale(ws, "multiplicative") - multi / 0.30) < 1e-12
    assert multimorbidity_scale([0.2], "multiplicative") == 1.0

    yld = {
        "hiv": np.array([2.0, 0.0, 1.0]),
        "type2diabetes": np.array([1.0, 3.0, 0.5]),
    }
    dws = {"hiv": 0.20, "type2diabetes": 0.10}
    additive = adjust_total_yld_for_multimorbidity(yld, dws, rule="additive")
    assert np.allclose(additive, [3.0, 3.0, 1.5])

    multi_tot = adjust_total_yld_for_multimorbidity(yld, dws, rule="multiplicative")
    # agent 0 and 2 have both conditions → scaled; agent 1 only T2D → unchanged
    assert multi_tot[1] == 3.0
    assert multi_tot[0] < 3.0
    assert multi_tot[2] < 1.5
