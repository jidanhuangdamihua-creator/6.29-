# D4 Category Fix - Next Steps

## ✅ Completed

- [x] Created fix branch: `fix/d4-first-category-granularity`
- [x] Modified domain_filter: second_category=20 → first_category=15
- [x] Applied to BOTH with and without scenarios
- [x] Verified modifications
- [x] Committed changes (48d4b2dc)

## 📋 Immediate Next Steps

### 1. Regenerate Solidified Data (Required)

Both WITH and WITHOUT scenarios need fresh source pool data:

**Why?** 
- Domain filter changed → different candidate pools
- WITH: 719 → 1254+ candidates (major change)
- Cannot use old solidified data with new config

**Commands** (to be determined based on your data generation pipeline):
```bash
# Find the data generation script
# Likely in scripts/ or data_processing/
# Run for both scenarios
```

### 2. Re-run D4 Experiments

```bash
# WITHOUT scenario
python scripts/run_d4_experiment.py --info-sharing without

# WITH scenario  
python scripts/run_d4_experiment.py --info-sharing with
```

**Expected**:
- WITHOUT: K≥3 satisfied, experiment completes
- WITH: New results (different candidates selected)

### 3. Verify Success

Check:
- [ ] No K<3 errors in logs
- [ ] Results written to outputs/
- [ ] RMSE/sMAPE metrics look reasonable
- [ ] Compare WITH vs WITHOUT results

## 📝 Paper Documentation

### Methods Section

Add explanation:
> "Initial candidate pool using second_category granularity yielded insufficient candidates (K<3) for target stores. Based on raw data verification, we adjusted to first_category granularity, providing 5 valid candidates while maintaining category semantic coherence."

### Limitations Section

Include (as previously drafted):
> "Category IDs in D4 are anonymized, preventing verification of semantic appropriateness. While first_category=15 ensures adequate candidate pool size, actual product relationships remain unverified pending category label availability."

## ⚠️  Important Reminders

1. **Both scenarios need re-run** - not just WITHOUT
2. **This is about feasibility** - not performance optimization
3. **Document the rationale** - based on K≥3 requirement, not results
4. **Keep verification audit trail** - scripts and outputs saved

## 🔄 If Issues Arise

### Issue: "Still getting K<3 errors"

Check:
- Did you regenerate solidified data with new config?
- Are you using the new config files?
- Is the domain_filter actually being applied?

### Issue: "WITH scenario results drastically different"

This is **expected** - candidate pool changed significantly (719→1254+).
Different candidates → different KNN selections → different results.
Document this as part of the fix impact.

### Issue: "How to regenerate solidified data?"

Look for:
- `scripts/generate_solidified_data.py`
- `data_processing/solidify_knn.py`
- Or similar data preparation scripts

If unclear, check existing parquet files' metadata or git history.

---

**Current branch**: `fix/d4-first-category-granularity`
**Status**: Config fixed ✅, awaiting data regeneration and re-run
