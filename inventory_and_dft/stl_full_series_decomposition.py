
import json
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DASHBOARD_FILE = SCRIPT_DIR.parent / "final_dashboard" / "thesis_dashboard_Final.html"
OUTDIR = SCRIPT_DIR / "stl_full_series_decomposition"
OUTDIR.mkdir(exist_ok=True)

TARGET_ORDER = ["Sales_Amount_Actual", "Cost_Amount_Actual", "Margin"]


def load_decomposition() -> dict:
    html = DASHBOARD_FILE.read_text(encoding="utf-8")
    marker = '<script id="patterns-data" type="application/json">'
    start = html.find(marker)
    if start == -1:
        raise RuntimeError(f"Could not find embedded patterns-data block in {DASHBOARD_FILE}")
    start = html.find(">", start) + 1
    end = html.find("</script>", start)
    return json.loads(html[start:end])["decomposition"]


def strength(component: np.ndarray, resid: np.ndarray) -> float:
    denom = np.var(component + resid)
    if denom == 0:
        return 0.0
    return max(0.0, 1.0 - np.var(resid) / denom)


def main():
    decomposition = load_decomposition()
    summary_rows = []

    for target in TARGET_ORDER:
        d = decomposition[target]
        trend = np.asarray(d["trend"], dtype=float)
        seasonal = np.asarray(d["seasonal"], dtype=float)
        resid = np.asarray(d["residual"], dtype=float)

        trend_strength = strength(trend, resid)
        seasonal_strength = strength(seasonal, resid)

        out = pd.DataFrame({
            "Posting_Date": d["dates"],
            "observed": d["observed"],
            "trend": trend,
            "seasonal": seasonal,
            "resid": resid,
        })
        out.to_csv(OUTDIR / f"stl_{target}.csv", index=False)

        summary_rows.append({
            "target": target,
            "trend_start": trend[0],
            "trend_end": trend[-1],
            "trend_strength": round(trend_strength, 3),
            "seasonal_strength": round(seasonal_strength, 3),
            "resid_std": round(float(np.std(resid)), 1),
        })
        print(f"{target}: trend_strength={trend_strength:.3f} seasonal_strength={seasonal_strength:.3f} "
              f"resid_std={np.std(resid):.1f}")

    pd.DataFrame(summary_rows).to_csv(OUTDIR / "stl_summary.csv", index=False)
    print(f"\nSaved per-day decomposition + summary to {OUTDIR}")


if __name__ == "__main__":
    main()
