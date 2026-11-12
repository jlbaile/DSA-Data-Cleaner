# Student Perceptions of AI-Generated Dramas: Survey Text Analysis (PySpark)

A reproducible PySpark pipeline for analyzing a Google Forms survey on how students perceive AI-generated dramas as an educational learning medium. It validates consent, cleans and anonymizes the responses, flags duplicates for human review, and runs descriptive text analysis. It also prepares a manual-coding package for the next phase.

Course context: Data Scalability and Analytics (BSIT).

> **Status:** Phases 1-3 are implemented in `ai_drama_analysis_phase1_3.py`. Phases 4-7 (manual coding agreement, validated sentiment, optional clustering, final report) are planned.

---

## Contents

1. [What the script does](#what-the-script-does)
2. [Requirements](#requirements)
3. [Quick start (VS Code, step by step)](#quick-start-vs-code-step-by-step)
4. [Using your own survey file](#using-your-own-survey-file)
5. [Output folder guide](#output-folder-guide)
6. [Reviewing duplicates](#reviewing-duplicates)
7. [Manual coding workflow](#manual-coding-workflow)
8. [Configuration reference](#configuration-reference)
9. [Privacy and GitHub safety](#privacy-and-github-safety)
10. [Method notes and limitations](#method-notes-and-limitations)
11. [Troubleshooting](#troubleshooting)
12. [Roadmap](#roadmap)
13. [Authors](#authors)

---

## What the script does

| Phase | Steps |
|---|---|
| **1. Load, inspect, validate** | Checks the file and the six question columns; reports missing values and timestamp failures; classifies consent; documents eligibility. |
| **2. Clean and transform** | Excludes test and empty submissions; creates stable anonymous IDs (`R0001`...); hashes emails (only to find duplicates); redacts emails, phone numbers and links from the text; flags duplicates (never auto-deletes); converts answers to one-row-per-answer (long) format. |
| **3. Descriptive analysis** | Response completeness, answer length and quality flags, top words (with how many respondents used each), bigrams and trigrams, TF-IDF terms per question, five charts. |
| **Phase 4 prep** | A codebook with nine themes and a manual-coding sheet. |

Design decisions worth knowing:

- Responses collected **before the consent question was added** to the form (blank consent value) are included. This is a group decision, and they are labeled in a `consent_basis` column.
- Negation words (`not`, `no`, `hindi`, `wala`...) are kept in phrase tables so "not accurate" does not collapse into "accurate".
- Text normalization is Unicode-aware, so it works for English, Filipino and Taglish.
- Blank checks strip carriage returns, because Google Forms CSVs use Windows line endings.

## Requirements

| Tool | Version |
|---|---|
| Python | 3.9 or newer (3.10+ recommended) |
| Java (JDK) | 17 or 21 for PySpark 4.x (PySpark 3.5 also works with 8/11/17) |
| Python packages | `pyspark`, `pandas`, `matplotlib` (see `requirements.txt`) |
| Editor | Visual Studio Code with the **Python** extension (any editor works) |

Tested with Python 3.12 and PySpark 4.3. It needs no Hadoop installation.

## Quick start (VS Code, step by step)

### 1. Install the prerequisites (once)

1. **Python:** download it from <https://www.python.org/downloads/>. On Windows, tick **Add python.exe to PATH** in the installer.
2. **Java 17:** install a JDK 17 (for example Temurin from <https://adoptium.net>). Then check it in a terminal:
   ```bash
   java -version
   ```
3. **VS Code:** download it from <https://code.visualstudio.com>, then install the **Python** extension (Extensions icon, search "Python", Install).

### 2. Get the code

Either clone it:

```bash
git clone <your-repository-url>
cd <repository-folder>
```

or click **Code, then Download ZIP** on GitHub, extract it, and use **File, Open Folder** in VS Code.

### 3. Create an isolated environment

Open the VS Code terminal (**Ctrl + `**) in the project folder:

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

> If PowerShell says scripts are disabled, run this once, then activate again:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

Then press **Ctrl+Shift+P**, choose **Python: Select Interpreter**, and pick the one inside `.venv`.

### 4. Install the dependencies

```bash
pip install -r requirements.txt
```

### 5. Add your data

Create a folder `data/` and place the exported Google Forms CSV inside, for example:

```
data/survey_responses.csv
```

The `data/` folder is ignored by Git (see [Privacy](#privacy-and-github-safety)), so the raw file is never uploaded.

### 6. Run it

```bash
python ai_drama_analysis_phase1_3.py "data/survey_responses.csv" "output"
```

The first argument is the CSV and the second is the output folder. Both are optional: without arguments the script uses the `AI_DRAMA_CSV` environment variable, or the defaults at the top of the script.

Prefer the Run button? Add `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Run AI Drama analysis",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/ai_drama_analysis_phase1_3.py",
      "args": ["data/survey_responses.csv", "output"],
      "console": "integratedTerminal"
    }
  ]
}
```

Then press **F5**.

### 7. Check the result

The terminal prints each step, including how many rows were excluded and why. Open `output/reports/run_log.txt` for the same log, and `output/figures/` for the charts.

## Using your own survey file

The script looks for **exactly one** column whose header contains each of these (case-insensitive):

| Question ID | Header must contain |
|---|---|
| Q1 Benefits | `benefits` |
| Q2 Engagement | `interest and engagement` |
| Q3 Problems | `problems` |
| Q4 Accuracy | `accuracy` |
| Q5 Suggestions | `suggestions` |
| Q6 Overall opinion | `overall` |
| Consent | `agree to participate` |
| Email | `email` |
| Timestamp | `timestamp` |

If a header does not match, the script stops with a clear `PROBLEM:` message instead of producing wrong results. To reuse the pipeline for a different survey, edit the `QUESTIONS` list in section 0 of the script.

## Output folder guide

```
output/
├── data_quality/     inspection_columns.csv, column_dictionary.csv, validation_summary.csv,
│                     exclusion_log.csv, duplicate_review.csv, pii_flags.csv
├── processed/        long_responses.csv   (one row per respondent x question)
│                     cleaned_responses.csv (one row per respondent)
├── manual_coding/    codebook.csv, manually_coded_responses_TEMPLATE.csv
├── tables/           a1_response_completeness.csv ... a6_tfidf_terms_per_question.csv
├── figures/          fig1 ... fig5 (.png)
├── reports/          run_log.txt, run_settings.json, interpretation_draft_phase3.txt
└── protected/        id_map_PROTECTED.csv, responses_original_PROTECTED.csv   (NEVER share)
```

Key files:

- `validation_summary.csv` and `exclusion_log.csv`: a record of how many rows came in and why any were removed.
- `processed/`: the anonymized datasets for submission or analysis.
- `protected/`: the link between anonymous IDs and source rows, plus the exact original answers. Keep this private.

## Reviewing duplicates

The script **flags** possible duplicates (same email submitted more than once, or identical answers across three or more questions). It never deletes them on its own.

1. Open `output/data_quality/duplicate_review.csv`.
2. In `review_decision`, type `keep` or `exclude` for each row, and add a reason.
3. Save the file as `output/manual_coding/duplicate_decisions.csv`.
4. Run the script again. Rows marked `exclude` are removed and logged.

## Manual coding workflow

1. Open `manual_coding/codebook.csv` and agree on a definition and one example per theme.
2. Two coders each copy `manually_coded_responses_TEMPLATE.csv` and code **independently** (put `1` in each theme column that applies; fill `coder_id`).
3. Use `uncodable` for answers that cannot be coded.
4. Compare the two files and compute agreement (Phase 4, planned).

With a larger dataset, set `CODING_SAMPLE_PER_QUESTION` (for example `100`) to code a reproducible random sample instead of every answer.

## Configuration reference

All settings are at the top of the script (section 0).

| Setting | Default | Meaning |
|---|---|---|
| `INCLUDE_LEGACY_NO_CONSENT_FIELD` | `True` | Include responses submitted before the consent question existed. |
| `AGREE_PREFIXES` | `("yes",)` | A consent value starting with one of these counts as agreement. |
| `TEST_PATTERNS` | `["example data for testing"]` | Rows containing any of these phrases are treated as test data. |
| `TIMESTAMP_FORMAT` | `M/d/yyyy H:mm:ss` | Google Forms timestamp format. |
| `EMAIL_SALT` | placeholder | Secret used when hashing emails. **Change it before sharing.** |
| `MIN_NGRAM_COUNT` | `2` | Minimum frequency for phrases to appear. |
| `TOP_N_WORDS`, `TOP_N_PER_QUESTION` | `20`, `8` | Size of the word tables. |
| `INCLUDE_TRIGRAMS` | `True` | Also produce three-word phrases. |
| `CODING_SAMPLE_PER_QUESTION` | `None` | `None` means code all answers; an integer means sample that many per question. |
| `RANDOM_SEED` | `42` | Makes sampling reproducible. |
| `ELIGIBILITY_NOTE` | study wording | Eligibility text recorded in the run log. |

## Privacy and GitHub safety

The raw survey export contains **email addresses**. Before your first commit:

1. Use the provided `.gitignore`. It excludes `data/`, `output/`, `.venv/` and any `*.csv`.
2. Never commit `protected/` files, the raw CSV, or `duplicate_decisions.csv`.
3. Change `EMAIL_SALT` and do not publish the real value.
4. Skim the answers for names or schools before quoting any respondent. The script redacts only emails, phone numbers and links.
5. If a sensitive file was already committed, deleting it later does not remove it from Git history. Rewrite the history or recreate the repository.

If you want to show the pipeline publicly, add a small **synthetic** CSV with the same columns (for example `sample_data/synthetic_responses.csv`) and keep the real data private.

**Ethics note:** the form states that the survey is open to students aged 18 and above. Respondents under 18 took part only with a guardian's permission, collected through a letter outside the form. The dataset records neither age nor guardian permission, so keep the signed letters on file separately and describe this in your methodology.

## Method notes and limitations

- Word and phrase counts describe **vocabulary**, not opinion. Confirm any pattern against the actual answers and the manual codebook before reporting it as a finding.
- TF-IDF treats each question as one document (six documents), so scores show terms distinctive to a question compared with the others.
- With small samples, report counts rather than percentages for rare themes.
- Keyword-based flags are only a pre-screen; the manual codebook is the source of truth for themes.

## Troubleshooting

| Problem | Fix |
|---|---|
| `JAVA_GATEWAY_EXITED` or "Java not found" | Install JDK 17 and make sure `java -version` works. If needed, set `JAVA_HOME` to the JDK folder. |
| `CSV not found` | Check the path and quotes. Use forward slashes or quote paths that contain spaces. |
| `PROBLEM: expected exactly 1 column matching...` | A question header was edited. See [Using your own survey file](#using-your-own-survey-file). |
| Fewer respondents than expected | Read `exclusion_log.csv`. It lists every exclusion and its reason. |
| `Python worker failed to connect back` (Windows) | Run from the activated `.venv` so Spark uses the same Python. The script sets this automatically. |
| `FutureWarning: PySpark does not yet fully support pandas >= 3.0` | Harmless, but `requirements.txt` pins `pandas<3` to avoid it. |
| Hadoop or `winutils` warnings on Windows | Safe to ignore; the script does not write through Hadoop. |
| PowerShell blocks `Activate.ps1` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then try again. |

## Roadmap

- [x] Phase 1: load, inspect, validate consent and eligibility
- [x] Phase 2: cleaning, anonymization, duplicate review, long format
- [x] Phase 3: descriptive text analysis and charts
- [ ] Phase 4: manual coding agreement (Cohen's kappa) and theme prevalence
- [ ] Phase 5: optional clustering with clear limitations
- [ ] Phase 6: validated multilingual sentiment
- [ ] Phase 7: final report and individual contribution table

## Authors

Group project for Data Scalability and Analytics (BSIT).

| Member | Contribution |
|---|---|
| _name_ | _contribution_ |
| _name_ | _contribution_ |

