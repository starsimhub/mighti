# Improvements for ART Adherence Simulation

## 1. Fix "No Interaction" Scenario Logic

**Problem**: In "No interaction" scenarios, adherence is 1.0 for everyone (no AdherenceEngine), so dropout probability should be 0.0. However, we're still seeing some dropout because:
- People might have been added to `_ever_dropped` earlier (during initialization or early timesteps)
- The `_ever_dropped` set persists and excludes people even when dropout probability is now 0.0

**Solution**: Add a check in `ARTAdherenceDisruptor.step()` to detect "No interaction" scenarios (where all adherence values are 1.0) and clear `_ever_dropped` more aggressively.

## 2. Performance Optimizations

**Problem**: Simulation is slow with 1 million people.

**Solutions**:
- Reduce debug output frequency (currently every 5 timesteps, could be every 12 or 24)
- Optimize dropout probability calculation (use vectorized operations more efficiently)
- Cache frequently accessed state arrays
- Only validate CD4 values for people actually scheduled to drop (not all people)

## 3. Increase Statistical Power

**Problem**: Only ~115-119 AUD diagnosed HIV+ individuals, leading to high variance.

**Solutions**:
- Run multiple replicates (e.g., 5-10 runs) and average results
- Increase population size (if computationally feasible)
- Use stratified sampling to ensure sufficient AUD+HIV+ individuals

## 4. Better "No Interaction" Detection

**Problem**: Currently, "No interaction" scenarios are detected implicitly by checking if adherence is 1.0. This is fragile.

**Solution**: Add an explicit parameter to `ARTAdherenceDisruptor` to indicate whether it's in a "No interaction" scenario, and skip dropout logic entirely in those cases.

## 5. Code Improvements

### 5.1 Reduce Debug Output Frequency
- Change debug output from every 5 timesteps to every 12 timesteps (monthly)
- Only print detailed debug info when dropout actually occurs

### 5.2 Optimize CD4 Validation
- Only validate CD4 for people scheduled to drop (not all people)
- Cache CD4 array access

### 5.3 Better `_ever_dropped` Management
- Clear `_ever_dropped` more aggressively when dropout probability is 0.0
- Add a method to reset `_ever_dropped` for "No interaction" scenarios

## 6. Statistical Analysis Improvements

**Problem**: Small sample sizes lead to high variance in coverage estimates.

**Solutions**:
- Add confidence intervals to coverage estimates
- Report standard errors across replicates
- Use bootstrap resampling for small groups

## 7. Validation Improvements

**Problem**: Hard to verify that "No interaction" scenarios truly have no interaction.

**Solution**: Add validation checks that:
- Verify adherence is 1.0 for all people in "No interaction" scenarios
- Verify dropout probability is 0.0 for all people in "No interaction" scenarios
- Warn if dropout occurs in "No interaction" scenarios

