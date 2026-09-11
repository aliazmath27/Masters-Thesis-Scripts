# Thesis Analysis Scripts

Supporting scripts for the master's thesis *"OData Integration: A Data Science Approach"* (DEMECAN ERP data, Chapter 4 exploratory analysis and Chapter 5 forecasting).

Clustering/inventory/DFT scripts expect `ILE_Modified.xlsx` (the enriched Item Ledger Entries export) in the same directory unless noted otherwise, and write their output to a folder named after the script. The agent-reasoning scripts under `forecasting/agent_reasoning/` additionally require a `GOOGLE_API_KEY` in a local `.env` file (never committed — see Credentials below).

Install dependencies with `pip install -r requirements.txt` before running anything.

## Structure

```
requirements.txt               Python dependencies for every script in this repo
data_preparation/             Feature-engineering pipeline that built ILE_Modified.xlsx (Ch3 Table 3.2)
2d_clustering/                 2-variable KMeans clustering (category/THC-group/product-family level)
3d_clustering/                 3-variable KMeans clustering
inventory_and_dft/             Inventory turnover/flow, DFT demand-cycle detection, STL decomposition
clustering_supplementary/      Customer-level and SKU-level clustering
forecasting/                   Deep-learning forecasting pipeline (3 generations), MASE, lookback search
forecasting/agent_reasoning/   LLM agent-reasoning forecasting (plain prompting + IC-DP)
final_dashboard/                The results dashboard (thesis_dashboard_Final.html)
```

### Data preparation

Chained in this order on the raw OData export (`ItemLedgerEntriesCustom_all_rows.xlsx`, 58 columns) to produce `ILE_Modified.xlsx` (71 columns — see `Data_Dictionary_Enrichment_Log.md`):

| Script | Adds |
|---|---|
| `product_family_addition.py` | `product_family` (9-family taxonomy: Enua, Truu, Core, Craft, Oktonia, Calama, Pure, Drop, Others) |
| `thc_group_addition.py` | `thc_auto`, `thc_manual`, `thc_final`, `thc_product_group` |
| `unit_price_unit_cost_addition.py` | `unit_quantity`, `total_physical_quantity`, `unit_price`, `unit_cost` |
| `item_category_code_fix.py` | Fills blank `itemCategoryCode` values with "Others" |
| `filter_2023_onwards.py` | Restricts to the 2023-01-01+ analysis window |

### 2D clustering

| Script | Analysis |
|---|---|
| `clustering_all_categories.py` | Quantity vs Margin |
| `category_sales_inventory_clustering.py` | Sales vs Inventory Movement |
| `inventory_cv_clustering.py` | Inventory vs CV |
| `inventory_velocity_clustering.py` | Inventory vs Velocity |
| `margin_sales_clustering.py` | Sales vs Margin |
| `velocity_unit_margin_clustering_k3.py` | Velocity vs Unit Margin |
| `velocity_cv_clustering.py` | Velocity vs CV |
| `unit_price_cost_clustering.py` | Unit Price vs Unit Cost |
| `Unit_Margin_Transactions_clustering.py` | Margin per Unit vs Transactions |

### 3D clustering

| Script | Analysis |
|---|---|
| `clustering_3d_sales_margin_transactions.py` | Sales + Margin per Unit + Transactions |
| `clustering_3d_velocity_cv_margin.py` | Velocity + CV + Margin per Unit |
| `clustering_3d_unit_price_cost_transactions.py` | Unit Price + Unit Cost + Transactions |

### Inventory & DFT

| Script | Analysis |
|---|---|
| `inventory_flow_analysis.py` | Inventory flow (on hand, sold, bought) |
| `inventory_dft_cycles_windowed.py` | DFT demand cycle detection |
| `thc_group_decomposition.py` | STL seasonal/trend decomposition, per THC group |
| `stl_full_series_decomposition.py`* | Trend/seasonal-strength figures behind Ch4 §4.11 (reads the STL decomposition already embedded in `thesis_dashboard_Final.html`) |

\* Authored for this repo — no standalone script for this existed on disk. First attempt independently re-ran STL on the raw daily series, but produced different numbers (trend strength ~0.46-0.51, seasonal ~0.30-0.38) than the cited figures — the exact STL hyperparameters (seasonal/trend smoother window lengths, robustness, etc.) used to build the dashboard weren't independently documented anywhere, so an independent re-fit couldn't be trusted to match. Fixed by having the script read the `trend`/`seasonal`/`residual` arrays that `thesis_dashboard_Final.html` already embeds (the same data its "Seasonal decomposition" tab charts) and compute the strength formula directly from those, in Python instead of JS. This reproduces the cited numbers exactly (trend strength 0.256/0.256/0.259, seasonal strength 0.116/0.039/0.04 for Cost/Margin/Sales) since it's using the literal source data rather than a re-derivation. Confirmed by running it — verified, not a re-guess. No `ILE_Modified.xlsx` or `statsmodels` needed for this script anymore; it only reads `final_dashboard/thesis_dashboard_Final.html` (one directory up).

### Supplementary clustering

| Script | Analysis |
|---|---|
| `customer_segmentation.py` | Customer-level clustering (sales/margin/quantity/transactions) |
| `item_clustering_sku.py` | SKU-level clustering |
| `product_group_clustering.py` | Item-description clustering by category |

### Forecasting

Three generations of the same DL pipeline, kept side by side since the thesis discusses the progression as a finding (not just the final version):

| Script | Analysis |
|---|---|
| `dl_forecasting_tuned.py` | Gen 1 — sequential (OFAT) hyperparameter search |
| `dl_forecasting_tuned_2.py` | Gen 2 — per-target models (Sales/Cost/Margin tuned separately) |
| `dl_forecasting_tuned_3.py` | Gen 3 — full-grid hyperparameter search (main pipeline, computes validation MASE inline) |
| `dl_compute_mase.py` | Final MASE across all models/lookbacks (DL tuned, DL baseline, both agent variants) |
| `dl_lookback_refinement.py` | Dense lookback refinement on top of Gen 2's coarse grid (same hyperparameters, no new tuning — finds a better lookback between coarse-grid candidates) |
| `best_window_analysis.py` | Cross-model ranking — single best (model, lookback) combo per horizon/target across all 5 methods (3 DL + 2 agent variants) |
| `verify_exact_canonical_dl_check.py` | Independent re-check of 11 (horizon, lookback) combinations Gen 3's dense sweep structurally can't reach (its fixed 2-day step size only lands on lookback values sharing its starting parity) — reuses Gen 3's exact architecture/preprocessing and each cell's already-found winning hyperparameters, no new search. Kept as a separate script deliberately: its evidentiary value comes from being run independently of Gen 3 (fresh process, seed reset per cell) — folding it into `dl_forecasting_tuned_3.py` would blur that separation and the two scripts answer different questions (Gen 3 searches for the best config; this one only verifies specific already-known configs at lookbacks Gen 3 never tested) |

Usage note for `verify_exact_canonical_dl_check.py`: uses relative paths (`ILE_Modified.xlsx`, `DL_fine_tuned_3/winning_configs_v3.csv`) rather than the `BASE` convention the other scripts use — run it directly from your Thesis working folder, or edit `RAW_FILE`/`WINNING_CONFIG_PATH` at the top. Set the `MAX_CELLS` environment variable (e.g. `MAX_CELLS=3`) to test on a few cells before running all 99. Results land in a new `dl_check/` folder and can be resumed if interrupted.

### Agent reasoning (`forecasting/agent_reasoning/`)

| Script | Analysis |
|---|---|
| `backtest_reasoning_forecast_full.py` | Plain-prompting Gemini agent, horizon x lookback backtest |
| `ICDP_backtest_reasoning_forecast_full.py` | In-context direct-prompting (IC-DP) variant |

## Credentials

The agent-reasoning scripts read `GOOGLE_API_KEY` from a local `.env` file via `python-dotenv` — no key is hardcoded anywhere in this repo. Create your own `.env` (git-ignored) alongside these scripts with:
```
GOOGLE_API_KEY=your-key-here
```

## Results dashboard (`final_dashboard/`)

`thesis_dashboard_Final.html` — the interactive results viewer (Best Model, All Models, Lookback Optimum, Hyperparameter Tuning, Final Day, and Data Patterns tabs). Fully self-contained (embedded data, no build step) — open the file directly in a browser.

## Note

Scripts have hardcoded local paths (e.g. `C:\Users\aliaz\IU\Thesis`) as `BASE`/`INPUT_FILE` constants — update these before running elsewhere.
