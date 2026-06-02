# Quick Setup: Add This Code to patent_analysis.ipynb

## After Step 3 (Results table), Add This New Cell:

```python
# Export the 3 CSV reports with platform/subsystem breakdown
from analyzer import export_csv_reports

print("📊 Exporting CSV reports...")
export_csv_reports(results, OUTPUT_DIR)

print("\n✓ CSV files created:")
print(f"   • 01_platforms_distribution.csv")
print(f"   • 02_subsystems_distribution.csv")  
print(f"   • 03_platform_subsystem_matrix.csv")
```

## That's It!

When you run the existing `patent_analysis.ipynb` notebook:

1. **Step 1-2**: Works as before (loads data, runs analysis)
2. **Step 3**: Results table with old categories  
3. **NEW CELL**: Exports 3 CSV files ← ADD THIS
4. **Step 4+**: Works as before (visualization, inspection, comparison)

## The 3 CSV Files Will Contain:

**01_platforms_distribution.csv**
```
Fixed-Wing Aircraft, 245623, 18.2%
Rotary-Wing / Helicopter, 156789, 11.6%
UAV / Drone, 892456, 66.1%
... (5 platforms total)
```

**02_subsystems_distribution.csv**
```
Propulsion System, 654321
Avionics & Control Systems, 543210
Power & Energy Systems, 498765
... (13 subsystems total)
```

**03_platform_subsystem_matrix.csv**
```
Platform, Fuselage, Propulsion, Landing Gear, ...
Fixed-Wing, 245623, 234567, 156789, ...
Helicopter, 156789, 145678, 123456, ...
... (5 platforms × 13 subsystems matrix)
```

## Ready?

Open your `patent_analysis.ipynb` and add that one cell. Done! 🚀
