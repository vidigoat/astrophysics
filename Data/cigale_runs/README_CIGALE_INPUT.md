# CIGALE input (mock catalog)

- **Format:** CSV with columns: `id`, `redshift`, and **band fluxes in mJy** (plus `_err` for uncertainties).
- **Filter names** must match CIGALE exactly. Run `pcigale-filters list` to see names (e.g. `GALEX.GALEX.FUV`, `sdss.SDSS.u`). If `mock_catalog.csv` uses a filter name CIGALE doesn’t know, replace it with one from the list.
- **Config:** Set `data_file = mock_catalog.csv` (or full path) in `pcigale.ini`, then run `pcigale genconf`, `pcigale check`, `pcigale run` from this directory.
- After a successful run, tell me: **(1)** the path to the **output results file** (e.g. `out/results.fits`) and **(2)** the **column names** in that file for age, A_V, metallicity (and SFH if used).
