
import os
import re
import sys
import textwrap
from functools import reduce

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.ml.feature import NGram
from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F

# ----------------------------------------------------------------------------
# 0. CONFIGURATION
# ----------------------------------------------------------------------------
CSV_PATH = (r"C:\Users\Johnk\Downloads\Student Perceptions of AI-Generated "
            r"Dramas as an Educational Learning Medium(Responses) - "
            r"Form Responses 1.csv")
OUT_DIR = r"C:\Users\Johnk\Downloads\AI_Drama_Output"
TABLE_DIR = os.path.join(OUT_DIR, "tables")
FIG_DIR = os.path.join(OUT_DIR, "figures")
for d in (OUT_DIR, TABLE_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)
print("Output folder:", OUT_DIR)

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

# short name -> keyword found in the original question header
QUESTION_KEYS = {
    "benefits": "benefits",
    "engagement": "interest and engagement",
    "problems": "problems",
    "accuracy": "accuracy",
    "suggestions": "suggestions",
    "opinion": "overall",
}
QN = list(QUESTION_KEYS)
QLABEL = {"benefits": "Benefits", "engagement": "Interest & engagement",
          "problems": "Problems & limitations", "accuracy": "Accuracy concerns",
          "suggestions": "Suggestions", "opinion": "Overall opinion"}

# Words ignored in word-frequency analysis (edit freely)
STOP = set("""
the and for are but not you all any can had her was one our out has have been that this with they their them then than
there these those what when where which who whom why how will would could should from into about also its just more most
some such very too only own same other because while being does did doing were his him she your yours over under again
once here both each few between through during before after above below off dont doesnt cant wont isnt arent youre theyre
thats lot may might much many make makes made use used using get gets way really even still something anything often
sometimes maybe think feel like well
ang ng sa mga na ay at ko ako mo siya sila kami tayo yung yun ito iyon niya nila namin natin para pero kasi dahil lang
naman din rin pa po ba kung baka mas may mayroon wala hindi nang sana nito dito doon minsan yan yon pag kapag
ai generated dramas drama students student learning lesson lessons education educational
""".split())

# Simple sentiment lexicons (English + Filipino). Extend them with words from your own data.
POS = ["good", "great", "helpful", "engaging", "engage", "interesting", "interest", "interactive", "fun", "effective",
       "creative", "easy", "easier", "understand", "understanding", "useful", "beneficial", "enjoy", "enjoyable",
       "relatable", "entertaining", "innovative", "motivating", "motivation", "improve", "better", "accessible",
       "excellent", "support", "memorable", "benefit", "benefits", "positive", "love", "best", "effectively",
       "maganda", "madali", "nakakatulong", "tulong", "interesado", "okay", "ayos", "epektibo", "masaya", "malinaw",
       "nakakaengganyo"]
NEG = ["bad", "harmful", "wrong", "error", "errors", "inaccurate", "fake", "misinformation", "unrealistic", "unclear",
       "confusing", "limited", "problem", "problems", "concern", "concerns", "distract", "distracting", "distraction",
       "unreliable", "outdated", "mistake", "mistakes", "lack", "lacking", "boring", "replace", "dependent",
       "overreliance", "bias", "biased", "risk", "risks", "incorrect", "false", "negative", "difficult", "confuse",
       "mali", "kulang", "peke", "problema", "nakakalito", "luma", "delikado"]
TAGALOG_MARKERS = ["ang", "ng", "sa", "mga", "yung", "yun", "ko", "ako", "mas", "kasi", "baka", "siya", "naman",
                   "lang", "pero", "din", "rin", "ito", "nito", "minsan", "madali", "hindi", "dahil", "kung", "pag",
                   "wala", "ay", "na", "po", "ba"]

# Theme coding: theme -> (question it belongs to, regex searched in the lower-cased answer)
THEMES = {
    "Engagement & interest":      ("benefits",    r"engag|interest|\bfun\b|entertain|enjoy|motivat|attention|engganyo"),
    "Easier understanding":       ("benefits",    r"understand|easier|\beasy\b|clear|simplif|visual|madali|maintindihan|malinaw"),
    "Relatable real-life context": ("benefits",   r"relat|real life|real-life|scenario|situation|context|example"),
    "Creativity & innovation":    ("benefits",    r"creativ|innovat|modern|new way|imagin"),
    "Accuracy & misinformation":  ("accuracy",    r"accura|error|wrong|\bmali\b|fake|misinform|outdated|reliab|incorrect|false|mistake"),
    "Visual / realism issues":    ("problems",    r"unclear|unrealistic|visual|glitch|quality|distort|\bmali\b|kulang"),
    "Over-reliance / less human": ("problems",    r"relian|depend|replace|teacher|human|lazy|critical thinking"),
    "Bias / moral values":        ("problems",    r"bias|moral|\bvalues?\b|ethic|cultur|stereotyp"),
    "Distraction":                ("problems",    r"distract|focus|attention"),
    "Fact-checking / teacher guidance": ("suggestions", r"fact|verif|teacher|guid|supervis|source|review|reference"),
    "Simple & relatable content": ("suggestions", r"simple|relatable|short|concise|cultur|local|filipino"),
}


def flag_name(theme):
    return "theme_" + re.sub(r"\W+", "_", theme.lower()).strip("_")


spark = (SparkSession.builder.appName("AIDramaTextAnalysis").master("local[*]")
         .config("spark.sql.shuffle.partitions", "4")
         .config("spark.sql.ansi.enabled", "false")
         .config("spark.ui.showConsoleProgress", "false")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")


def to_pd(sdf: DataFrame) -> pd.DataFrame:
    """Small Spark DataFrame -> pandas (avoids pandas-version issues with toPandas)."""
    return pd.DataFrame([r.asDict() for r in sdf.collect()], columns=sdf.columns)


def save(sdf: DataFrame, name: str) -> pd.DataFrame:
    pdf = to_pd(sdf)
    pdf.to_csv(os.path.join(TABLE_DIR, f"{name}.csv"), index=False, encoding="utf-8-sig")
    return pdf


# ----------------------------------------------------------------------------
# 1. LOAD AND INSPECT
# ----------------------------------------------------------------------------
print("\n=== 1. LOAD AND INSPECT ===")
raw = (spark.read.option("header", True).option("multiLine", True)
       .option("quote", '"').option("escape", '"').option("encoding", "UTF-8")
       .csv(CSV_PATH))
print("Rows:", raw.count(), "| Columns:", len(raw.columns))
raw.printSchema()

# ----------------------------------------------------------------------------
# 2. CLEAN AND TRANSFORM
# ----------------------------------------------------------------------------
print("\n=== 2. CLEAN AND TRANSFORM ===")
ts_col = next(c for c in raw.columns if "timestamp" in c.lower())
colmap = {}
for short, key in QUESTION_KEYS.items():
    colmap[short] = next(c for c in raw.columns if key in c.lower())
pd.DataFrame({"column": list(colmap), "question": list(colmap.values())}).to_csv(
    os.path.join(OUT_DIR, "column_dictionary.csv"), index=False, encoding="utf-8-sig")

# Select only needed columns (the Email column is deliberately dropped: personal data)
df = raw.select(F.col(f"`{ts_col}`").alias("timestamp"),
                *[F.col(f"`{c}`").alias(s) for s, c in colmap.items()])
print("Dropped columns (personal data / unused):", [c for c in raw.columns if c not in [ts_col, *colmap.values()]])

# Trim, collapse whitespace, blank -> null
for q in QN:
    df = df.withColumn(q, F.when(F.trim(F.col(q)) == "", None)
                             .otherwise(F.regexp_replace(F.trim(F.col(q)), r"\s+", " ")))
df = df.withColumn("submitted_at", F.to_timestamp("timestamp", "M/d/yyyy H:mm:ss")).drop("timestamp")

# Remove test/example rows
is_test = reduce(lambda a, b: a | b, [F.lower(F.coalesce(F.col(q), F.lit(""))).contains("example data for testing")
                                      for q in QN])
n_before = df.count()
df = df.filter(~is_test)
print(f"Removed {n_before - df.count()} test row(s)")

# Remove duplicates / empty rows, assign anonymous IDs
df = df.dropDuplicates(QN).dropna(how="all", subset=QN)
df = df.withColumn("rn", F.row_number().over(Window.orderBy("submitted_at"))) \
       .withColumn("respondent_id", F.format_string("R%02d", F.col("rn"))).drop("rn")
total_n = df.count()
print("Valid respondents:", total_n)

# Theme flags (wide format)
for theme, (q, rx) in THEMES.items():
    df = df.withColumn(flag_name(theme), F.coalesce(F.lower(F.col(q)).rlike(rx), F.lit(False)))
df = df.cache()

# Long format: one row per respondent x question
long = reduce(DataFrame.unionByName,
              [df.select("respondent_id", F.lit(q).alias("question"), F.col(q).alias("answer")) for q in QN]) \
    .filter(F.col("answer").isNotNull())

long = (long
        .withColumn("word_count", F.size(F.split(F.trim("answer"), " +")))
        .withColumn("clean", F.regexp_replace(F.lower("answer"), "['\u2019]", ""))
        .withColumn("clean", F.trim(F.regexp_replace("clean", "[^a-z ]", " ")))
        .withColumn("tokens_all", F.filter(F.split("clean", " +"), lambda w: F.length(w) > 0))
        .withColumn("tokens", F.filter("tokens_all", lambda w: (F.length(w) > 2) & (~w.isin(list(STOP)))))
        .withColumn("pos_n", F.size(F.filter("tokens_all", lambda w: w.isin(POS))))
        .withColumn("neg_n", F.size(F.filter("tokens_all", lambda w: w.isin(NEG))))
        .withColumn("sentiment_score", F.col("pos_n") - F.col("neg_n"))
        .withColumn("sentiment_label", F.when(F.col("sentiment_score") > 0, "Positive")
                                        .when(F.col("sentiment_score") < 0, "Negative")
                                        .otherwise("Neutral"))
        .withColumn("tl_ratio", F.size(F.filter("tokens_all", lambda w: w.isin(TAGALOG_MARKERS)))
                    / F.size("tokens_all"))
        .withColumn("language", F.when(F.col("tl_ratio") >= 0.25, "Filipino")
                                 .when(F.col("tl_ratio") >= 0.05, "Taglish")
                                 .otherwise("English"))).cache()

# Respondent-level summary
resp = long.groupBy("respondent_id").agg(
    F.count("*").alias("answers_given"),
    F.sum("word_count").alias("total_words"),
    F.sum("sentiment_score").alias("overall_sentiment_score"))
resp = resp.withColumn("overall_sentiment", F.when(F.col("overall_sentiment_score") > 0, "Positive")
                                             .when(F.col("overall_sentiment_score") < 0, "Negative")
                                             .otherwise("Neutral"))

# Export processed datasets
long_out = long.select("respondent_id", "question", "answer", "clean", "word_count", "pos_n", "neg_n",
                       "sentiment_score", "sentiment_label", F.round("tl_ratio", 3).alias("tagalog_ratio"),
                       "language")
to_pd(long_out.orderBy("respondent_id", "question")).to_csv(
    os.path.join(OUT_DIR, "processed_dataset_long.csv"), index=False, encoding="utf-8-sig")
wide_out = df.join(resp, "respondent_id", "left").orderBy("respondent_id")
to_pd(wide_out).to_csv(os.path.join(OUT_DIR, "processed_dataset_wide.csv"), index=False, encoding="utf-8-sig")
print("Processed datasets saved.")
long.select("respondent_id", "question", "word_count", "sentiment_score", "sentiment_label", "language").show(8)

# ----------------------------------------------------------------------------
# 3. ANALYSES
# ----------------------------------------------------------------------------
print("\n=== 3. ANALYSES ===")

# Analysis 1: Response length and completeness per question
print("\n[Analysis 1] Response length per question")
a1 = (long.groupBy("question")
      .agg(F.count("*").alias("answers"),
           F.round(F.avg("word_count"), 1).alias("avg_words"),
           F.min("word_count").alias("min_words"),
           F.expr("percentile_approx(word_count, 0.5)").alias("median_words"),
           F.max("word_count").alias("max_words")))
a1.show(truncate=False)
a1_pdf = save(a1, "a1_response_length")

# Analysis 2: Most frequent words overall
print("\n[Analysis 2] Top words overall")
words = long.select("question", F.explode("tokens").alias("word"))
a2 = words.groupBy("word").count().orderBy(F.desc("count"), "word").limit(20)
a2.show(20, truncate=False)
a2_pdf = save(a2, "a2_top_words")

# Analysis 3: Top 5 words per question
print("\n[Analysis 3] Top words per question")
w = Window.partitionBy("question").orderBy(F.desc("count"), "word")
a3 = (words.groupBy("question", "word").count()
      .withColumn("rank", F.row_number().over(w)).filter("rank <= 5")
      .orderBy("question", "rank"))
a3.show(40, truncate=False)
save(a3, "a3_top_words_per_question")

# Analysis 4: Lexicon-based sentiment per question
print("\n[Analysis 4] Sentiment per question")
a4 = (long.groupBy("question").pivot("sentiment_label", ["Positive", "Neutral", "Negative"]).count()
      .na.fill(0))
a4 = a4.join(long.groupBy("question").agg(F.round(F.avg("sentiment_score"), 2).alias("avg_score")), "question")
a4.show(truncate=False)
a4_pdf = save(a4, "a4_sentiment_per_question")
print("Overall respondent sentiment:")
a4b = resp.groupBy("overall_sentiment").count().orderBy(F.desc("count"))
a4b.show()
save(a4b, "a4b_overall_respondent_sentiment")

# Analysis 5: Theme prevalence (percent of respondents mentioning each theme)
print("\n[Analysis 5] Theme prevalence")
a5 = reduce(DataFrame.unionByName, [
    df.agg(F.lit(t).alias("theme"), F.lit(q).alias("question"),
           F.sum(F.col(flag_name(t)).cast("int")).alias("respondents"),
           F.count(F.col(q)).alias("answered"))
    for t, (q, _) in THEMES.items()])
a5 = a5.withColumn("percent", F.round(F.col("respondents") / F.col("answered") * 100, 1)).orderBy(F.desc("percent"))
a5.show(truncate=False)
a5_pdf = save(a5, "a5_theme_prevalence")

# Analysis 6: Most frequent two-word phrases (bigrams)
print("\n[Analysis 6] Top bigrams")
bg = NGram(n=2, inputCol="tokens", outputCol="bigrams").transform(long.filter(F.size("tokens") >= 2))
a6 = (bg.select(F.explode("bigrams").alias("bigram")).groupBy("bigram").count()
      .filter("count >= 2").orderBy(F.desc("count"), "bigram").limit(15))
a6.show(truncate=False)
save(a6, "a6_top_bigrams")

# Analysis 7: Language mix (English / Taglish / Filipino)
print("\n[Analysis 7] Language of responses")
a7 = (long.groupBy("language").count()
      .withColumn("percent", F.round(F.col("count") / long.count() * 100, 1)).orderBy(F.desc("count")))
a7.show()
save(a7, "a7_language_mix")

# Analysis 8: Does longer writing go with more positive/negative sentiment?
corr = long.stat.corr("word_count", "sentiment_score")
corr_resp = resp.stat.corr("total_words", "overall_sentiment_score")
print(f"\n[Analysis 8] Correlation word_count vs sentiment (per answer): {corr:.3f}")
print(f"             Correlation total_words vs overall sentiment (per respondent): {corr_resp:.3f}")
pd.DataFrame({"measure": ["answer-level", "respondent-level"], "pearson_r": [round(corr, 3), round(corr_resp, 3)]}) \
    .to_csv(os.path.join(TABLE_DIR, "a8_length_sentiment_corr.csv"), index=False)

# ----------------------------------------------------------------------------
# 4. VISUALIZATIONS
# ----------------------------------------------------------------------------
print("\n=== 4. VISUALIZATIONS ===")

# Figure 1: top words
p = a2_pdf.sort_values("count")
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(p["word"], p["count"], color="#3b82f6")
ax.set_xlabel("Frequency")
ax.set_title("Top 20 Words in Student Responses")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig1_top_words.png"), dpi=200)
plt.close()

# Figure 2: theme prevalence
p = a5_pdf.sort_values("percent")
colors = {"benefits": "#16a34a", "engagement": "#0ea5e9", "problems": "#dc2626",
          "accuracy": "#f97316", "suggestions": "#8b5cf6", "opinion": "#64748b"}
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(p["theme"], p["percent"], color=[colors[q] for q in p["question"]])
ax.set_xlim(0, 100)
ax.set_xlabel("% of respondents mentioning theme (in the related question)")
ax.set_title("Theme Prevalence in Open-Ended Answers")
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[q]) for q in dict.fromkeys(p["question"])]
ax.legend(handles, [QLABEL[q] for q in dict.fromkeys(p["question"])], title="Question", loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig2_theme_prevalence.png"), dpi=200)
plt.close()

# Figure 3: sentiment per question
p = a4_pdf.set_index("question").reindex(QN)
fig, ax = plt.subplots(figsize=(10, 5))
bottom = pd.Series(0, index=p.index)
for lab, col in [("Positive", "#16a34a"), ("Neutral", "#9ca3af"), ("Negative", "#dc2626")]:
    ax.bar([QLABEL[q] for q in p.index], p[lab], bottom=bottom.values, color=col, label=lab)
    bottom += p[lab]
ax.set_ylabel("Number of responses")
ax.set_title("Lexicon-Based Sentiment per Question")
ax.legend()
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig3_sentiment_per_question.png"), dpi=200)
plt.close()

# Figure 4: average response length
p = a1_pdf.set_index("question").reindex(QN)
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar([QLABEL[q] for q in p.index], p["avg_words"], color="#8b5cf6")
ax.set_ylabel("Average words per response")
ax.set_title("Average Response Length per Question")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig4_response_length.png"), dpi=200)
plt.close()

# Optional Figure 5: word cloud (only if the wordcloud package is installed)
try:
    from wordcloud import WordCloud
    freq = dict(zip(a2_pdf["word"], a2_pdf["count"]))
    wc = WordCloud(width=1000, height=500, background_color="white").generate_from_frequencies(freq)
    wc.to_file(os.path.join(FIG_DIR, "fig5_wordcloud.png"))
except ImportError:
    print("(wordcloud not installed - skipping optional word cloud)")
print("Figures saved to", FIG_DIR)

# ----------------------------------------------------------------------------
# 5. DRAFT INTERPRETATION  (edit and expand in your own words)
# ----------------------------------------------------------------------------
lines = [f"Valid respondents (after removing test data): {total_n}.",
         "Most frequent words: " + ", ".join(a2_pdf.sort_values("count", ascending=False)["word"].head(8)) + "."]
tot_sent = a4_pdf[["Positive", "Neutral", "Negative"]].sum()
lines.append(f"Across all answers: {tot_sent['Positive']} positive, {tot_sent['Neutral']} neutral, "
             f"{tot_sent['Negative']} negative (lexicon-based, so check manually).")
for _, r in a5_pdf.head(3).iterrows():
    lines.append(f"Top theme: {r['theme']} - {r['percent']}% of respondents ({r['respondents']}/{r['answered']}).")
longest = a1_pdf.sort_values("avg_words", ascending=False).iloc[0]
lines.append(f"Longest answers were for '{QLABEL[longest['question']]}' (avg {longest['avg_words']} words).")
lines.append("Caution: small sample and keyword-based coding; read the raw answers to confirm each theme.")
with open(os.path.join(OUT_DIR, "interpretation_draft.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n".join(lines))

spark.stop()
print("\nDone. Open:", OUT_DIR)