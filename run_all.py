import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ["q1_dual_primal.py", "q2_saddle_point.py", "q3_weak_duality.py"]

for script in SCRIPTS:
    print("=" * 60)
    print(f"Running {script}")
    print("=" * 60)
    runpy.run_path(str(HERE / script), run_name="__main__")
    print()

print("Done. All figures are in the figures/ folder.")
