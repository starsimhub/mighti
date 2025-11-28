# What is CD4 and How is it Used in the HIV Module?

## CD4 Overview

**CD4** (Cluster of Differentiation 4) is a type of T-cell (white blood cell) that plays a crucial role in the immune system. In HIV modeling, CD4 count is used as a key biomarker to track HIV disease progression and immune system health.

## CD4 Count Ranges

- **Normal range**: 500-1200 cells/mm³
- **HIV progression stages**:
  - **>500 cells/mm³**: Early stage, relatively healthy immune system
  - **200-500 cells/mm³**: Moderate immune suppression
  - **<200 cells/mm³**: Severe immune suppression (AIDS-defining)

## How CD4 is Used in the HIV Module

1. **Mortality Calculation** (`make_p_hiv_death`):
   - CD4 count is used to calculate HIV-related mortality probabilities
   - Lower CD4 counts = higher mortality risk
   - The HIV module uses CD4 bins (ranges) to look up mortality probabilities
   - This is why we had to patch `make_p_hiv_death` to handle out-of-bounds indices

2. **Disease Progression**:
   - CD4 count naturally declines over time in untreated HIV
   - ART (antiretroviral therapy) helps maintain or restore CD4 counts
   - When people stop ART, CD4 counts decline again (via `post_art_decline` function)

3. **CD4 Decline After Stopping ART** (`post_art_decline`):
   - When someone stops ART, their CD4 count declines
   - The rate of decline depends on how long they were on ART and how long they've been off
   - This is calculated using `ti_art` (time started ART) and `ti_stop_art` (time stopped ART)
   - This is why we had to patch `post_art_decline` to handle negative durations

## Why We Validate CD4 Values

In our adherence module, we validate and fix CD4 values to prevent errors because:

1. **Invalid values cause crashes**: The HIV module's `step_state` method validates CD4 values and raises `ValueError: Invalid entry for CD4` if values are:
   - NaN (not a number)
   - Negative
   - Zero
   - >= 2000 (unrealistically high)

2. **CD4 values can become invalid** when:
   - People are dropped from ART and re-added quickly
   - Timing values (`ti_art`, `ti_stop_art`) are invalid
   - The `post_art_decline` function returns invalid values

3. **Our fix**: We validate CD4 values before the HIV module processes them, setting invalid values to a reasonable default (500.0 cells/mm³) to prevent crashes.

## CD4 and ART Adherence

While CD4 is not directly used in our adherence module, it's indirectly affected:
- **Low adherence** → People drop out of ART → CD4 declines → Higher mortality
- This creates a feedback loop where adherence problems lead to worse health outcomes

