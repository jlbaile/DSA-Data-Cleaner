"""
Student Perceptions of AI-Generated Dramas - Analysis script (Phases 1-3)

Phase 1: load, inspect, validate consent / eligibility, audit data quality
Phase 2: clean, anonymize, flag (not delete) duplicates, stable respondent IDs
Phase 3: descriptive text analysis (completeness, length, words, n-grams, TF-IDF)

Usage:
    python ai_drama_analysis_phase1_3.py [path_to_csv] [output_folder]
If no arguments are given, the paths in section 0 (or the AI_DRAMA_CSV environment variable) are used.

Raw CSV is never modified. Everything is written under the output folder.
"""
import json
import os
import platform
import re
import sys
from datetime import datetime
from functools import reduce

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pyspark
from pyspark.ml.feature import NGram, CountVectorizer, IDF
from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F

# ----------------------------------------------------------------------------
# 0. CONFIGURATION  (the decisions a reviewer may ask about live here)
# ----------------------------------------------------------------------------
DEFAULT_CSV = (r"C:\Users\Johnk\Downloads\Student Perceptions of AI-Generated "
               r"Dramas as an Educational Learning Medium(Responses) - Form Responses 1.csv")
DEFAULT_OUT = r"C:\Users\Johnk\Downloads\AI_Drama_Output"
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("AI_DRAMA_CSV", DEFAULT_CSV)
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT

# Responses submitted BEFORE the consent question was added to the form have a blank consent value.
# Group decision: these are valid and are analysed together with the newer data.
INCLUDE_LEGACY_NO_CONSENT_FIELD = True
AGREE_PREFIXES = ("yes",)                       # consent value must start with one of these (case-insensitive)
TEST_PATTERNS = ["example data for testing"]    # rows containing any of these are test submissions
TIMESTAMP_FORMAT = "M/d/yyyy H:mm:ss"
EMAIL_SALT = "change-this-secret-before-sharing"  # used only to hash emails for duplicate flagging
MIN_NGRAM_COUNT = 2                              # phrases must appear at least this many times
TOP_N_WORDS = 20
TOP_N_PER_QUESTION = 8
INCLUDE_TRIGRAMS = True
CODING_SAMPLE_PER_QUESTION = None                # None = all answers go in the manual-coding sheet; or e.g. 100
RANDOM_SEED = 42

ELIGIBILITY_NOTE = ("The form has no age/eligibility field. Eligibility is treated as self-declared through the "
                    "form's instructions. Describe the exact eligibility wording in the methodology.")

# (question_id, short key, keyword that identifies the column header, category, chart label)
QUESTIONS = [
    ("Q1", "benefits",    "benefits",                "Benefits",        "Benefits"),
    ("Q2", "engagement",  "interest and engagement", "Engagement",      "Interest & engagement"),
    ("Q3", "problems",    "problems",                "Concerns",        "Problems & limitations"),
    ("Q4", "accuracy",    "accuracy",                "Accuracy",        "Accuracy concerns"),
    ("Q5", "suggestions", "suggestions",             "Suggestions",     "Suggestions"),
    ("Q6", "opinion",     "overall",                 "Overall opinion", "Overall opinion"),
]
QN = [q[1] for q in QUESTIONS]
QID = {q[1]: q[0] for q in QUESTIONS}
QCAT = {q[1]: q[3] for q in QUESTIONS}
QLABEL_BY_ID = {q[0]: q[4] for q in QUESTIONS}

# Negation words are NEVER removed from n-grams (they can reverse meaning). They are excluded only from the
# single-word frequency table, where "not"/"hindi" would otherwise dominate.
NEGATORS = {"not", "no", "never", "hindi", "walang", "wala", "dont", "doesnt", "isnt", "arent", "wont", "cant",
            "didnt", "cannot", "without"}
STOP = set("""
the and for are but you all any can had her was one our out has have been that this with they their them then than
there these those what when where which who whom why how will would could should from into about also its just more most
some such very too only own same other because while being does did doing were his him she your yours over under again
once here both each few between through during before after above below off youre theyre thats lot may might much many
make makes made use used using get gets way really even still something anything often sometimes maybe think feel like well
ang ng sa mga na ay at ko ako mo siya sila kami tayo yung yun ito iyon niya nila namin natin para pero kasi dahil lang
naman din rin pa po ba kung baka mas may mayroon nang sana nito dito doon minsan yan yon pag kapag
sya nya ung pwede talaga kaya
""".split()) - NEGATORS
DOMAIN_STOP = {"ai", "generated", "dramas", "drama", "students", "student", "learning", "lesson", "lessons",
               "education", "educational"}
STOP_ALL = sorted(STOP | DOMAIN_STOP)
NEG_L = sorted(NEGATORS)
PLACEHOLDERS = ["na", "n a", "none", "wala", "nothing", "no", "idk", "wala po", "none po", "wala naman"]

PII_PATTERNS = {
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "phone": r"(\+?63|0)9\d{2}[\s-]?\d{3}[\s-]?\d{4}",
    "url":   r"(https?://\S+|www\.\S+)",
}

# Folders
DIRS = {k: os.path.join(OUT_DIR, k) for k in
        ("data_quality", "processed", "protected", "manual_coding", "tables", "figures", "reports")}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

LOG = []


def log(msg=""):
    print(msg)
    LOG.append(str(msg))


os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
spark = (SparkSession.builder.appName("AIDramaTextAnalysis").master("local[*]")
         .config("spark.sql.shuffle.partitions", "4")
         .config("spark.sql.ansi.enabled", "false")
         .config("spark.ui.showConsoleProgress", "false")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")


def to_pd(sdf: DataFrame) -> pd.DataFrame:
    """Small Spark DataFrame -> pandas (avoids pandas-version issues with toPandas)."""
    return pd.DataFrame([r.asDict() for r in sdf.collect()], columns=sdf.columns)


def strip_ws(c):
    """Trim ALL whitespace, including \\r, tabs and non-breaking spaces (Spark's trim() only removes spaces).
    Google Forms CSVs use Windows line endings, so the last column can end with a stray carriage return."""
    return F.regexp_replace(c, r"^[\s\u00a0]+|[\s\u00a0]+$", "")


def save(pdf: pd.DataFrame, folder: str, name: str) -> pd.DataFrame:
    pdf.to_csv(os.path.join(DIRS[folder], f"{name}.csv"), index=False, encoding="utf-8-sig")
    return pdf


# ----------------------------------------------------------------------------
# 1. LOAD AND INSPECT
# ----------------------------------------------------------------------------
log("=== PHASE 1: LOAD, INSPECT, VALIDATE ===")
if not os.path.isfile(CSV_PATH):
    sys.exit(f"CSV not found: {CSV_PATH}\nPass the path as the first argument or set AI_DRAMA_CSV.")
log(f"Input file: {os.path.basename(CSV_PATH)}")

raw = (spark.read.option("header", True).option("multiLine", True)
       .option("quote", '"').option("escape", '"').option("encoding", "UTF-8")
       .csv(CSV_PATH))
n_received = raw.count()
log(f"Original rows: {n_received} | columns: {len(raw.columns)}")

problems = []


def one_col(pattern, label):
    m = [c for c in raw.columns if re.search(pattern, c.lower())]
    if len(m) != 1:
        problems.append(f"{label}: expected exactly 1 column matching '{pattern}', found {len(m)}")
        return None
    return m[0]


ts_col = one_col(r"timestamp", "Timestamp")
consent_col = one_col(r"agree to participate", "Consent")
email_col = one_col(r"email", "Email")
colmap = {key: one_col(re.escape(kw), f"Question {qid} ({key})") for qid, key, kw, _, _ in QUESTIONS}
age_like = [c for c in raw.columns if re.search(r"\bage\b|\bedad\b|birth|year level|grade", c.lower())]

# Inspection report (column names, types, missing values)
miss = raw.agg(*[F.sum(F.when(F.col(f"`{c}`").isNull() | (strip_ws(F.col(f"`{c}`")) == ""), 1).otherwise(0)).alias(str(i))
                 for i, c in enumerate(raw.columns)]).collect()[0]
inspect_pdf = pd.DataFrame({
    "column_index": range(len(raw.columns)),
    "column_header": [re.sub(r"\s+", " ", c).strip()[:120] for c in raw.columns],
    "dtype": [t for _, t in raw.dtypes],
    "missing_or_blank": [int(miss[str(i)] or 0) for i in range(len(raw.columns))],
})
save(inspect_pdf, "data_quality", "inspection_columns")

if problems:
    for p in problems:
        log("PROBLEM: " + p)
    log("Stopping: the form columns do not match what the script expects. Check whether a question was edited.")
    with open(os.path.join(DIRS["data_quality"], "inspection_problems.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(problems))
    spark.stop()
    sys.exit(1)
log("All six question columns, the consent column, the email column and the timestamp column were found.")

pd.DataFrame({"question_id": [q[0] for q in QUESTIONS], "short_key": QN,
              "original_header": [re.sub(r"\s+", " ", colmap[k]).strip() for k in QN]}) \
    .pipe(save, "data_quality", "column_dictionary")

# ----------------------------------------------------------------------------
# 2. VALIDATE CONSENT / ELIGIBILITY (before any column is dropped)
# ----------------------------------------------------------------------------
sel = ([F.col(f"`{ts_col}`").alias("timestamp_raw"),
        F.col(f"`{consent_col}`").alias("consent_raw"),
        F.col(f"`{email_col}`").alias("email_raw")]
       + [F.col(f"`{colmap[k]}`").alias(f"{k}__orig") for k in QN])
df = raw.select(*sel).coalesce(1)  # one partition keeps the file's row order (multiLine files are not split)

# Stable anonymous ID from the source-row position (NOT from the timestamp, which can tie)
df = df.withColumn("src_row", F.row_number().over(Window.orderBy(F.monotonically_increasing_id())))
df = df.withColumn("respondent_id", F.format_string("R%04d", F.col("src_row")))
df = df.withColumn("submitted_at", F.to_timestamp("timestamp_raw", TIMESTAMP_FORMAT))
n_ts_fail = df.filter(F.col("timestamp_raw").isNotNull() & F.col("submitted_at").isNull()).count()
log(f"Timestamp parsing failures: {n_ts_fail}")
df = df.withColumn("email_hash", F.when(strip_ws(F.coalesce("email_raw", F.lit(""))) != "",
                                        F.sha2(F.concat(F.lit(EMAIL_SALT), F.lower(strip_ws(F.col("email_raw")))), 256)))

consent_norm = F.lower(strip_ws(F.coalesce("consent_raw", F.lit(""))))
agreed = reduce(lambda a, b: a | b, [consent_norm.startswith(p) for p in AGREE_PREFIXES])
first_consent_ts = df.filter(agreed).agg(F.min("submitted_at")).collect()[0][0]
log(f"Earliest response with an explicit consent value: {first_consent_ts}")
legacy_cond = (F.col("submitted_at") < F.lit(first_consent_ts)) if first_consent_ts is not None else F.lit(False)
df = df.withColumn("consent_status",
                   F.when(agreed, "agreed")
                    .when(consent_norm != "", "declined_or_unclear_value")
                    .when(legacy_cond, "legacy_no_consent_field")
                    .otherwise("blank_after_consent_added"))
df = df.withColumn("consent_basis",
                   F.when(F.col("consent_status") == "agreed", "form consent item: agreed")
                    .when(F.col("consent_status") == "legacy_no_consent_field",
                          "collected before consent item existed (group decision: include)"))

cs = to_pd(df.groupBy("consent_status").count())
cs_map = dict(zip(cs["consent_status"], cs["count"]))
included_statuses = ["agreed"] + (["legacy_no_consent_field"] if INCLUDE_LEGACY_NO_CONSENT_FIELD else [])
n_consent_ok = int(sum(cs_map.get(s, 0) for s in included_statuses))

if age_like:
    log("WARNING: possible age/eligibility column(s) found: " + str(age_like) +
        " - define an eligibility rule in the script before relying on it.")
    elig_status = "age-like column present: rule not defined yet"
else:
    elig_status = "self-declared (no age field in form)"
log("Eligibility: " + ELIGIBILITY_NOTE)

exclusions = []  # (stage, reason, n_removed)


def exclude(frame, cond, stage, reason):
    cond = F.coalesce(cond, F.lit(False))
    n = frame.filter(cond).count()
    exclusions.append((stage, reason, n))
    log(f"  excluded {n:>3} | {stage}: {reason}")
    return frame.filter(~cond)


log("Exclusions (applied in this order):")
df = exclude(df, ~F.col("consent_status").isin(included_statuses), "consent",
             "no valid consent (declined, unclear, or blank after the consent item was added)")

# ----------------------------------------------------------------------------
# 3. CLEAN, REDACT, TEST/EMPTY ROWS
# ----------------------------------------------------------------------------
WS = r"[\s\u00a0]+"


def redact(c):
    for tag, rx in PII_PATTERNS.items():
        c = F.regexp_replace(c, rx, f"[{tag.upper()}]")
    return c


for k in QN:
    o = f"{k}__orig"
    df = df.withColumn(o, F.when(strip_ws(F.coalesce(F.col(o), F.lit(""))) == "", None).otherwise(F.col(o)))
    df = df.withColumn(k, F.when(F.col(o).isNotNull(), redact(F.regexp_replace(strip_ws(F.col(o)), WS, " "))))

is_test = reduce(lambda a, b: a | b,
                 [F.lower(F.coalesce(F.col(k), F.lit(""))).contains(p) for k in QN for p in TEST_PATTERNS])
df = exclude(df, is_test, "cleaning", "test submission (matches TEST_PATTERNS)")
df = df.withColumn("n_answered", sum(F.col(k).isNotNull().cast("int") for k in QN))
df = exclude(df, F.col("n_answered") == 0, "cleaning", "consented but answered none of the six questions")

# Duplicate REVIEW (flag only, never auto-delete)
df = df.withColumn("answer_key", F.concat_ws("||", *[F.lower(F.coalesce(F.col(k), F.lit(""))) for k in QN]))
by_email = (df.filter(F.col("email_hash").isNotNull()).groupBy("email_hash")
            .agg(F.collect_list("respondent_id").alias("ids"), F.count("*").alias("n")).filter("n > 1")
            .select(F.lit("same email submitted more than once").alias("reason"), "ids"))
by_answers = (df.filter(F.col("n_answered") >= 3).groupBy("answer_key")
              .agg(F.collect_list("respondent_id").alias("ids"), F.count("*").alias("n")).filter("n > 1")
              .select(F.lit("identical answers (3+ questions)").alias("reason"), "ids"))
dup_groups = to_pd(by_email.unionByName(by_answers))
dup_rows = []
for gi, r in enumerate(dup_groups.itertuples(index=False), start=1):
    for rid in sorted(r.ids):
        dup_rows.append({"dup_group": f"D{gi:02d}", "respondent_id": rid, "reason": r.reason,
                         "review_decision": "", "exclusion_reason": ""})
dup_pdf = pd.DataFrame(dup_rows, columns=["dup_group", "respondent_id", "reason", "review_decision",
                                          "exclusion_reason"])
decisions_path = os.path.join(DIRS["manual_coding"], "duplicate_decisions.csv")
if os.path.isfile(decisions_path):
    dec = pd.read_csv(decisions_path, dtype=str).fillna("")
    drop_ids = dec.loc[dec["review_decision"].str.strip().str.lower() == "exclude", "respondent_id"].tolist()
    df = exclude(df, F.col("respondent_id").isin(drop_ids), "duplicate review",
                 "excluded by manual decision in duplicate_decisions.csv")
    log(f"Duplicate decisions file applied ({len(dec)} rows).")
else:
    log(f"Duplicate review: {len(dup_pdf)} record(s) flagged in {dup_pdf['dup_group'].nunique()} group(s); "
        "none excluded. Fill review_decision (keep/exclude) in data_quality/duplicate_review.csv, "
        "save it as manual_coding/duplicate_decisions.csv, and re-run.")
save(dup_pdf, "data_quality", "duplicate_review")

df = df.cache()
total_n = df.count()
log(f"Final respondents proceeding to analysis: {total_n}")

# Validation / exclusion summaries
summary_rows = [
    ("Records received", n_received),
    ("Consent: agreed on the form", int(cs_map.get("agreed", 0))),
    ("Consent: blank, submitted before the consent item existed", int(cs_map.get("legacy_no_consent_field", 0))),
    ("Consent: declined or unclear value", int(cs_map.get("declined_or_unclear_value", 0))),
    ("Consent: blank after the consent item was added", int(cs_map.get("blank_after_consent_added", 0))),
    ("Legacy (pre-consent-item) records included?", "yes" if INCLUDE_LEGACY_NO_CONSENT_FIELD else "no"),
    ("Records passing consent rule", n_consent_ok),
    ("Eligibility procedure", elig_status),
    ("Final respondents", total_n),
]
save(pd.DataFrame(summary_rows, columns=["item", "value"]), "data_quality", "validation_summary")
save(pd.DataFrame(exclusions, columns=["stage", "reason", "records_excluded"]), "data_quality", "exclusion_log")

# Protected ID map (never share): links anonymous IDs to source rows
save(to_pd(df.select("respondent_id", "src_row", "submitted_at", "email_hash").orderBy("src_row")),
     "protected", "id_map_PROTECTED")

# ----------------------------------------------------------------------------
# 4. TRANSFORM TO LONG FORMAT AND NORMALISE TEXT
# ----------------------------------------------------------------------------
log("\n=== PHASE 2: TRANSFORM ===")
long = reduce(DataFrame.unionByName, [
    df.select("respondent_id", "consent_basis", F.lit(QID[k]).alias("question_id"),
              F.lit(k).alias("question_key"), F.lit(QCAT[k]).alias("question_category"),
              F.col(f"{k}__orig").alias("response_original"), F.col(k).alias("response_clean"))
    for k in QN])
long = long.withColumn("answered", F.col("response_clean").isNotNull())

# Unicode-aware normalisation: keep letters of any language; drop digits/punctuation; keep negation words
long = (long
        .withColumn("response_norm",
                    F.trim(F.regexp_replace(
                        F.regexp_replace(F.lower(F.coalesce("response_clean", F.lit(""))), "['\u2019\u2018`]", ""),
                        r"[^\p{L}\s]", " ")))
        .withColumn("response_norm", F.trim(F.regexp_replace("response_norm", WS, " ")))
        .withColumn("tokens_all", F.filter(F.split("response_norm", " +"), lambda w: F.length(w) > 0))
        .withColumn("word_count", F.when(F.col("answered"), F.size("tokens_all")).otherwise(0))
        .withColumn("tokens", F.filter("tokens_all", lambda w: (F.length(w) > 2) & (~w.isin(STOP_ALL))
                                       & (~w.isin(NEG_L))))
        .withColumn("quality_flag",
                    F.when(~F.col("answered"), "empty")
                     .when(F.col("response_norm").isin(PLACEHOLDERS), "placeholder")
                     .when(F.col("word_count") < 3, "very_short")
                     .otherwise("ok")))
# PII scan on the original text (counts only; redaction already applied to response_clean)
pii_conds = {t: F.col("response_original").rlike(rx) for t, rx in PII_PATTERNS.items()}
pii = reduce(DataFrame.unionByName, [
    long.filter(c).select("respondent_id", "question_id", F.lit(t).alias("pii_type")) for t, c in pii_conds.items()])
save(to_pd(pii.orderBy("respondent_id", "question_id")), "data_quality", "pii_flags")
log(f"Possible personal identifiers redacted in {pii.count()} answer(s) (see data_quality/pii_flags.csv).")
log("Reminder: regexes catch emails, phone numbers and links only - skim the answers for names/schools before "
    "quoting anyone.")
long = long.cache()

max_per_resp = long.groupBy("respondent_id").count().agg(F.max("count")).collect()[0][0]
assert max_per_resp <= 6, "A respondent has more than six question records"
log(f"Check passed: at most {max_per_resp} question records per respondent.")

long_out = long.select("respondent_id", "question_id", "question_category", "response_clean", "response_norm",
                       "word_count", "answered", "quality_flag", "consent_basis")
save(to_pd(long_out.orderBy("respondent_id", "question_id")), "processed", "long_responses")
save(to_pd(long.select("respondent_id", "question_id", "response_original")
           .orderBy("respondent_id", "question_id")), "protected", "responses_original_PROTECTED")
wide = df.select("respondent_id", "consent_basis", "n_answered", *[F.col(k).alias(f"{QID[k]}_{k}") for k in QN])
save(to_pd(wide.orderBy("respondent_id")), "processed", "cleaned_responses")
log("Saved processed/long_responses.csv, processed/cleaned_responses.csv (anonymised) and "
    "protected/ files (do NOT submit or share).")

la = long.filter(F.col("answered"))

# ----------------------------------------------------------------------------
# 5. DESCRIPTIVE TEXT ANALYSIS
# ----------------------------------------------------------------------------
log("\n=== PHASE 3: DESCRIPTIVE ANALYSIS ===")

# A1 completeness
a1 = (long.groupBy("question_id")
      .agg(F.sum(F.col("answered").cast("int")).alias("answered"))
      .withColumn("unanswered", F.lit(total_n) - F.col("answered"))
      .withColumn("percent_answered", F.round(F.col("answered") / F.lit(total_n) * 100, 1))
      .orderBy("question_id"))
a1_pdf = save(to_pd(a1), "tables", "a1_response_completeness")
log("[A1] Response completeness"); print(a1_pdf.to_string(index=False))

# A2 response length + quality flags
a2 = (la.groupBy("question_id")
      .agg(F.round(F.avg("word_count"), 1).alias("avg_words"), F.min("word_count").alias("min_words"),
           F.expr("percentile_approx(word_count, 0.5)").alias("median_words"), F.max("word_count").alias("max_words"))
      .orderBy("question_id"))
a2_pdf = save(to_pd(a2), "tables", "a2_response_length")
a2q = (la.groupBy("question_id").pivot("quality_flag", ["ok", "very_short", "placeholder"]).count()
       .na.fill(0).orderBy("question_id"))
save(to_pd(a2q), "tables", "a2b_answer_quality_flags")
log("[A2] Response length (words)"); print(a2_pdf.to_string(index=False))

# A3 top words overall (count + document frequency = distinct respondents)
words = la.select("respondent_id", "question_id", F.explode("tokens").alias("word"))
a3 = (words.groupBy("word").agg(F.count("*").alias("count"), F.countDistinct("respondent_id").alias("respondents"))
      .orderBy(F.desc("count"), "word").limit(TOP_N_WORDS))
a3_pdf = save(to_pd(a3), "tables", "a3_top_words")
log("[A3] Top words overall"); print(a3_pdf.head(10).to_string(index=False))

# A4 top words per question
w = Window.partitionBy("question_id").orderBy(F.desc("count"), "word")
a4 = (words.groupBy("question_id", "word")
      .agg(F.count("*").alias("count"), F.countDistinct("respondent_id").alias("respondents"))
      .withColumn("rank", F.row_number().over(w)).filter(F.col("rank") <= TOP_N_PER_QUESTION)
      .orderBy("question_id", "rank"))
a4_pdf = save(to_pd(a4), "tables", "a4_top_words_per_question")


# A5 bigrams / trigrams (negators kept as the first word so "not accurate" survives)
def is_content(x):
    return (F.length(x) > 2) & (~x.isin(STOP_ALL)) & (~x.isin(NEG_L))


def lead_ok(x):
    return is_content(x) | x.isin(NEG_L)


def ngram_table(n):
    base = la.filter(F.size("tokens_all") >= n)
    g = NGram(n=n, inputCol="tokens_all", outputCol="grams").transform(base)

    def keep(p):
        parts = F.split(p, " ")
        cond = lead_ok(parts[0]) & is_content(parts[1])
        return cond & is_content(parts[2]) if n == 3 else cond
    g = g.withColumn("grams", F.filter("grams", keep))
    ex = g.select("respondent_id", F.explode("grams").alias("phrase"))
    return (ex.groupBy("phrase").agg(F.count("*").alias("count"), F.countDistinct("respondent_id").alias("respondents"))
            .filter(F.col("count") >= MIN_NGRAM_COUNT).orderBy(F.desc("count"), "phrase").limit(15))


a5_pdf = save(to_pd(ngram_table(2)), "tables", "a5_top_bigrams")
log("[A5] Top bigrams"); print(a5_pdf.head(10).to_string(index=False))
if INCLUDE_TRIGRAMS:
    save(to_pd(ngram_table(3)), "tables", "a5b_top_trigrams")

# A6 TF-IDF: each question is one document, so scores show terms distinctive to a question vs the other five
docs = la.groupBy("question_id").agg(F.flatten(F.collect_list("tokens")).alias("terms"))
cv_model = CountVectorizer(inputCol="terms", outputCol="tf", vocabSize=5000, minDF=1.0).fit(docs)
tf = cv_model.transform(docs)
scored = IDF(inputCol="tf", outputCol="tfidf").fit(tf).transform(tf)
vocab = cv_model.vocabulary
tfidf_rows = []
for r in scored.collect():
    for idx, tfv, score in zip(r["tfidf"].indices, [r["tf"][int(i)] for i in r["tfidf"].indices], r["tfidf"].values):
        tfidf_rows.append({"question_id": r["question_id"], "term": vocab[int(idx)], "count": int(tfv),
                           "tfidf": round(float(score), 3)})
a6_pdf = (pd.DataFrame(tfidf_rows, columns=["question_id", "term", "count", "tfidf"])
          .sort_values(["question_id", "tfidf", "count"], ascending=[True, False, False])
          .groupby("question_id").head(TOP_N_PER_QUESTION).reset_index(drop=True))
save(a6_pdf, "tables", "a6_tfidf_terms_per_question")
log("[A6] TF-IDF terms per question saved (6 documents only: read as 'distinctive to this question').")

# ----------------------------------------------------------------------------
# 6. VISUALISATIONS
# ----------------------------------------------------------------------------
log("\n=== VISUALISATIONS ===")
labels = [QLABEL_BY_ID[q] for q in a1_pdf["question_id"]]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(labels, a1_pdf["answered"], color="#16a34a", label="Answered")
ax.bar(labels, a1_pdf["unanswered"], bottom=a1_pdf["answered"], color="#d1d5db", label="Unanswered")
for i, (a, u) in enumerate(zip(a1_pdf["answered"], a1_pdf["unanswered"])):
    ax.text(i, a / 2, str(a), ha="center", va="center", color="white", fontweight="bold")
ax.set_ylabel("Respondents")
ax.set_title(f"Response Completeness per Question (N = {total_n})")
ax.legend()
plt.xticks(rotation=20, ha="right"); plt.tight_layout()
plt.savefig(os.path.join(DIRS["figures"], "fig1_response_completeness.png"), dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(10, 5))
xs = range(len(a2_pdf))
ax.bar([x - 0.2 for x in xs], a2_pdf["avg_words"], width=0.4, color="#8b5cf6", label="Mean")
ax.bar([x + 0.2 for x in xs], a2_pdf["median_words"], width=0.4, color="#c4b5fd", label="Median")
ax.set_xticks(list(xs)); ax.set_xticklabels([QLABEL_BY_ID[q] for q in a2_pdf["question_id"]], rotation=20, ha="right")
ax.set_ylabel("Words per answer"); ax.set_title("Answer Length per Question"); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(DIRS["figures"], "fig2_answer_length.png"), dpi=200); plt.close()


def small_multiples(frame, value_col, label_col, title, fname, color):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (qid, _, _, _, qlabel) in zip(axes.flat, QUESTIONS):
        sub = frame[frame["question_id"] == qid].sort_values(value_col)
        if sub.empty:
            ax.axis("off"); continue
        ax.barh(sub[label_col], sub[value_col], color=color)
        ax.set_title(qlabel, fontsize=11)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(); plt.savefig(os.path.join(DIRS["figures"], fname), dpi=200); plt.close()


small_multiples(a4_pdf, "count", "word", "Most Frequent Words per Question", "fig3_top_words_per_question.png", "#3b82f6")
small_multiples(a6_pdf, "tfidf", "term", "Terms Most Distinctive to Each Question (TF-IDF)",
                "fig4_tfidf_per_question.png", "#f97316")

if not a5_pdf.empty:
    p = a5_pdf.head(10).sort_values("count")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(p["phrase"], p["count"], color="#0ea5e9")
    ax.set_xlabel("Frequency"); ax.set_title("Most Frequent Two-Word Phrases")
    plt.tight_layout(); plt.savefig(os.path.join(DIRS["figures"], "fig5_top_bigrams.png"), dpi=200); plt.close()
log("Figures saved to " + DIRS["figures"])

# ----------------------------------------------------------------------------
# 7. MANUAL-CODING PACKAGE (used in Phase 4)
# ----------------------------------------------------------------------------
CODEBOOK = [
    ("T01", "Understanding lessons", "Dramas make lessons easier to understand or follow.",
     "Mentions clearer, simpler or easier-to-follow explanations.",
     "Only says it is 'interesting' or 'fun' with no link to understanding."),
    ("T02", "Memory and retention", "Student mentions remembering or recalling lesson content.",
     "Mentions remember, recall, retain, memorable, madaling maalala.",
     "General praise with no mention of memory."),
    ("T03", "Interest and engagement", "Attention, enjoyment, curiosity or motivation.",
     "Mentions engaging, fun, interesting, not boring, gets attention.",
     "Complaints that dramas distract from the lesson (use T07)."),
    ("T04", "Accessibility and convenience", "Ease of access, flexible or self-paced learning.",
     "Mentions available anytime, easy to make or watch, helps teachers save time.",
     "Ease of understanding (use T01)."),
    ("T05", "Accuracy and reliability", "Doubt about correctness or trustworthiness of the content.",
     "Mentions errors, wrong or outdated facts, misinformation, need to verify.",
     "Complaints about visuals or voices only (use T06)."),
    ("T06", "AI limitations", "Unnatural visuals, voices, emotions, or other technical flaws.",
     "Mentions unclear or unrealistic visuals, glitches, flat emotions.",
     "Factual mistakes (use T05)."),
    ("T07", "Distraction and overreliance", "Worry about distraction or depending too much on AI.",
     "Mentions losing focus, laziness, replacing teachers or reading.",
     "Wanting teachers to review content (use T08)."),
    ("T08", "Teacher guidance", "Students want teachers to review, guide or supervise use.",
     "Mentions teacher checking, fact-checking with the teacher, supervision.",
     "Mentions teachers only as people who benefit (use T04)."),
    ("T09", "Content improvement", "Recommendations for better scripts, visuals, explanations or examples.",
     "Suggests shorter, relatable, local or more realistic content.",
     "Pure criticism with no improvement suggested."),
]
save(pd.DataFrame(CODEBOOK, columns=["theme_id", "theme", "definition", "apply_when", "do_not_apply_when"])
     .assign(example_response="(fill in after reading real answers)"), "manual_coding", "codebook")

code_src = to_pd(la.filter(F.col("quality_flag") != "placeholder")
                 .select("respondent_id", "question_id", "question_category", "response_clean"))
if CODING_SAMPLE_PER_QUESTION:
    code_src = (code_src.groupby("question_id", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), CODING_SAMPLE_PER_QUESTION), random_state=RANDOM_SEED)))
code_src = code_src.sort_values(["question_id", "respondent_id"]).reset_index(drop=True)
for t, *_ in CODEBOOK:
    code_src[t] = ""
code_src["uncodable"] = ""
code_src["coder_id"] = ""
code_src["notes"] = ""
code_path = os.path.join(DIRS["manual_coding"], "manually_coded_responses_TEMPLATE.csv")
code_src.to_csv(code_path, index=False, encoding="utf-8-sig")
log(f"Manual-coding template: {len(code_src)} answers (put 1 in each theme column that applies; "
    "a second coder fills a copy independently).")

# ----------------------------------------------------------------------------
# 8. DRAFT INTERPRETATION + RUN LOG
# ----------------------------------------------------------------------------
top_words = ", ".join(a3_pdf["word"].head(8))
lowest = a1_pdf.sort_values("percent_answered").iloc[0]
longest = a2_pdf.sort_values("avg_words", ascending=False).iloc[0]
interp = [
    f"Final respondents: {total_n} (received {n_received}).",
    f"Most frequent content words: {top_words}.",
    f"Question answered least often: {QLABEL_BY_ID[lowest['question_id']]} ({lowest['percent_answered']}%).",
    f"Longest answers on average: {QLABEL_BY_ID[longest['question_id']]} ({longest['avg_words']} words).",
    "Word and phrase counts describe vocabulary, not opinion. Confirm any pattern against the actual answers and "
    "the manual codebook before reporting it as a finding.",
]
with open(os.path.join(DIRS["reports"], "interpretation_draft_phase3.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(interp))
settings = {"run_at": datetime.now().isoformat(timespec="seconds"), "python": platform.python_version(),
            "pyspark": pyspark.__version__, "pandas": pd.__version__,
            "include_legacy_no_consent_field": INCLUDE_LEGACY_NO_CONSENT_FIELD, "agree_prefixes": AGREE_PREFIXES,
            "test_patterns": TEST_PATTERNS, "min_ngram_count": MIN_NGRAM_COUNT, "random_seed": RANDOM_SEED,
            "eligibility_note": ELIGIBILITY_NOTE, "records_received": n_received, "final_respondents": total_n}
with open(os.path.join(DIRS["reports"], "run_settings.json"), "w", encoding="utf-8") as f:
    json.dump(settings, f, indent=2, default=str)
with open(os.path.join(DIRS["reports"], "run_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(LOG))
log("\n".join(interp))

spark.stop()
print("\nDone. Open:", OUT_DIR)