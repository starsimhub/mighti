"""
API contract tests for MIGHTI's declared stable public surface.

These tests are intentionally lightweight and should only assert names that are
part of the supported API documentation.
"""

import importlib

import mighti as mi


def test_stable_namespaces_exist():
    """Declared stable namespaces should be importable from top-level `mighti`."""
    for namespace in (
        "diseases",
        "analyzers",
        "interactions",
        "interventions",
        "sdoh",
        "initialization",
    ):
        assert hasattr(mi, namespace), f"Missing stable namespace: mi.{namespace}"


def test_representative_stable_symbols_exist():
    """
    Each stable namespace should expose representative symbols used in examples/docs.
    """
    assert hasattr(mi.diseases, "Type2Diabetes")
    assert hasattr(mi.analyzers, "PrevalenceAnalyzer_HIV")
    assert hasattr(mi.interactions, "NCDHIVConnector")
    assert hasattr(mi.interventions, "ARTwithCASM")
    assert hasattr(mi.sdoh, "NeighbourhoodSituation")


def test_initialization_prevalence_helpers_exist():
    """Initialization helpers should remain available via initialization/prevalence."""
    prev_mod = importlib.import_module("mighti.initialization.prevalence")
    assert hasattr(prev_mod, "initialize_prevalence_data")
    assert hasattr(prev_mod, "age_sex_dependent_prevalence")


def test_plotting_not_exported_top_level_but_available_in_analysis():
    """
    Plotting helpers should not be top-level API, but should remain importable from
    `mighti.analysis.plotting` per public API policy.
    """
    assert not hasattr(mi, "plot_mean_prevalence")
    plotting_mod = importlib.import_module("mighti.analysis.plotting")
    assert hasattr(plotting_mod, "plot_mean_prevalence")
