# Project Review — Issues, Mistakes & Corrections

This file documents every issue found and corrected during the development
of the Heart Disease Prediction project.

---

## 1. Wrong Dataset Filename

**Issue:** Dataset was downloaded and saved as `heart_cleveland.csv`.
**README expects:** `data/heart.csv`
**Correction:** Renamed `heart_cleveland.csv` → `heart.csv`.
All source file references updated accordingly.

---

## 2. Wrong Requirements Filename

**Issue:** Dependency file was named `requirement.txt` (missing 's').
**README expects:** `requirements.txt`
**Correction:** Renamed `requirement.txt` → `requirements.txt`.

---

## 3. Model Saved to Wrong Directory

**Issue:** Trained model files (`best_model.pkl`, `scaler.pkl`, `model_meta.json`)
were being saved to a root-level `model/` directory.
**README expects:** `src/model/`
**Correction:** Updated `MODEL_DIR` path in `data_pipeline.py`, `train.py`,
and `predict.py` to use `src/model/`. Root-level `model/` directory removed.

---

## 4. Missing pyproject.toml

**Issue:** `pyproject.toml` was listed in the README project structure but
did not exist in the project.
**Correction:** Created `pyproject.toml` with all project dependencies,
Python version constraint (`>=3.14`), and uv dev dependencies.

---

## 5. Unrelated Files in Project Root

**Issue:** The following files were present but had no relation to the project:
- `index.js` — JavaScript file from a different project
- `package.json` — Node.js config from a different project
- `Heart-disease-script.ipynb` — Empty duplicate notebook
- `build_notebook.py` — Temporary script used to generate the notebook
- `fix_cell.py` — Temporary script used to fix a notebook bug
- `model/` (root level) — Empty leftover directory

**Correction:** All removed.

---

## 6. matplotlib boxplot API Breaking Change

**Issue:** Cell in `exploration.ipynb` used `labels=` parameter in `boxplot()`.
In matplotlib 3.11, this was renamed to `tick_labels=`.
**Error:** `TypeError: Axes.boxplot() got an unexpected keyword argument 'labels'`
**Correction:** Updated to `tick_labels=` in the boxplot cell and switched to
manual `set_xticks()` + `set_xticklabels()` approach for full compatibility.

---

## 7. matplotlib pie() Wrongly Updated

**Issue:** A regex replace intended to fix the boxplot `labels=` parameter
also incorrectly changed `labels=` inside `pie()` to `tick_labels=`.
**Error:** `TypeError: Axes.pie() got an unexpected keyword argument 'tick_labels'`
**Correction:** Fixed `pie()` to use the correct `labels=` parameter.

---

## 8. Notebook Written with UTF-8 BOM Encoding

**Issue:** The initial `exploration.ipynb` was written using PowerShell
`Set-Content` which added a UTF-8 BOM (`\xEF\xBB\xBF`) to the file.
**Error:** `json.decoder.JSONDecodeError: Unexpected UTF-8 BOM`
**Correction:** Rewrote the file using
`System.IO.File::WriteAllText(..., UTF8Encoding(false))` to write
UTF-8 without BOM, and later used Python `nbformat.write()` directly.

---

## 9. sklearn Feature Name Warning in predict.py

**Issue:** `predict.py` was passing a raw numpy array to the model,
but the model was trained on a named DataFrame.
**Warning:** `UserWarning: X does not have valid feature names,
but RandomForestClassifier was fitted with feature names`
**Correction:** Updated `predict.py` to always wrap input in a
`pd.DataFrame` with named columns before calling `model.predict()`.

---

## 10. chr() Hack in Notebook Save Best Model Cell

**Issue:** The notebook builder script used `chr(39)` sequences to avoid
quote conflicts inside Python f-strings embedded in PowerShell strings.
**Error:** `KeyError: "'accuracy'"` — the chr() sequence generated the
key `'accuracy'` (with literal single quotes) instead of `accuracy`.
**Correction:** Moved notebook building to a separate `.py` file written
with `File::WriteAllBytes`, then fixed the cell via a Python patch script.

---

## 11. Missing .gitignore

**Issue:** No `.gitignore` existed, meaning `__pycache__/`, `.venv/`,
`.pkl` model binaries, and `.idea/` IDE files would all be committed.
**Correction:** Created `.gitignore` excluding:
- `__pycache__/`, `*.pyc`
- `.venv/`, `venv/`
- `.ipynb_checkpoints/`
- `src/model/*.pkl` (large binary files)
- `.idea/`, `.vscode/`
- OS files (`Thumbs.db`, `.DS_Store`)

---

## 12. README Project Structure Used Wrong Folder Names

**Issue:** README showed `data/heart.csv` but initial code used
`data/heart_cleveland.csv`. README showed `src/model/` but code
wrote to root `model/`. README listed `requirements.txt` but file
was `requirement.txt`.
**Correction:** All three mismatches corrected (see items 1, 2, 3 above).

---

## Summary of All Corrections

| # | Issue | Severity | Fixed |
|---|---|---|---|
| 1 | Wrong dataset filename | Medium | Yes |
| 2 | Wrong requirements filename | Low | Yes |
| 3 | Model saved to wrong directory | Medium | Yes |
| 4 | Missing pyproject.toml | Low | Yes |
| 5 | Unrelated files in project | Low | Yes |
| 6 | matplotlib boxplot API change | High | Yes |
| 7 | matplotlib pie() wrongly updated | High | Yes |
| 8 | UTF-8 BOM encoding in notebook | High | Yes |
| 9 | sklearn feature name warning | Medium | Yes |
| 10 | chr() hack caused KeyError | High | Yes |
| 11 | Missing .gitignore | Medium | Yes |
| 12 | README vs code path mismatches | Medium | Yes |

All issues have been resolved. The project is fully functional and
consistent with the README specification.