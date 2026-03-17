Supported API
=============

This page defines the public API surface intended for stable external use.
Code outside this surface may change without notice between releases.

Stable namespaces
-----------------

The following namespaces are supported for user-facing imports and workflows:

- ``mighti.diseases``
- ``mighti.analyzers``
- ``mighti.interactions``
- ``mighti.interventions``
- ``mighti.sdoh``
- ``mighti.initialization``

Experimental/internal
---------------------

The following areas are currently internal or under active development:

- calibration internals (e.g., implementation details under ``mighti.calibration``)
- root-level scripts used as runnable examples (e.g., ``mighti_main.py``)

Guidance
--------

- Prefer namespaced imports, for example ``import mighti as mi`` then
  ``mi.diseases.Type2Diabetes``.
- Avoid depending on private implementation details from internal modules.
- When in doubt, treat only the stable namespaces above as API contracts.
