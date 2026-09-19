from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .dashboard_data import DASHBOARD_ROOT, load_dashboard_data

ROOT = Path("data/processed/absa_v1/final_outputs/absa_v1_thesis_outputs_v1")
PRIMARY_ASPECTS = ["race_performance", "crowd_community_atmosphere", "physical_experience"]
TEMPORAL_TABLE_ASPECTS = PRIMARY_ASPECTS + ["emotional_experience", "training_preparation_pacing", "route_course"]
COMPOSITION_ASPECTS = ["race_performance", "crowd_community_atmosphere", "physical_experience",
                       "emotional_experience", "training_preparation_pacing", "route_course", "photography_media"]
YEARS = [2019, 2023, 2024, 2025]
COLORS = {"positive": "#087f8c", "negative": "#c54f4f", "mixed": "#c3922e", "neutral": "#78838a"}
SERIES = ["#087f8c", "#c75d3d", "#536f8d", "#8d6a9f"]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "seguisb.ttf" if bold else "segoeui.ttf"
    path = Path("C:/Windows/Fonts") / name
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def _canvas(title: str, subtitle: str, width: int = 1800, height: int = 1100) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), "white"); draw = ImageDraw.Draw(image)
    draw.text((110, 65), title, font=_font(42, True), fill="#14262b")
    draw.text((110, 125), subtitle, font=_font(23), fill="#56686d")
    return image, draw


def _save(image: Image.Image, path: Path, pdf: Path | None = None) -> None:
    image.save(path, "PNG", dpi=(200, 200), optimize=True)
    if pdf is not None: image.save(pdf, "PDF", resolution=200.0)


def _line_figure(year_aspect: pd.DataFrame, labels: dict[str, str], path: Path, pdf: Path) -> None:
    image, draw = _canvas("Principal aspect prevalence across observed KLSCM editions",
        "Document-level prevalence with frozen 95% confidence intervals")
    left, top, right, bottom = 180, 220, 1660, 900
    maximum = .55
    for index in range(0, 6):
        value = index / 10; y = bottom - (value / maximum) * (bottom-top)
        draw.line((left, y, right, y), fill="#dce6e7", width=2)
        draw.text((95, y-14), f"{value:.0%}", font=_font(21), fill="#56686d")
    x_map = {year: left + i*(right-left)/(len(YEARS)-1) for i, year in enumerate(YEARS)}
    for year, x in x_map.items():
        draw.text((x-27, bottom+25), str(year), font=_font(22, True), fill="#14262b")
    draw.text((left, bottom+75), "Event edition", font=_font(22), fill="#56686d")
    draw.text((25, top+250), "Share of analyzed posts", font=_font(21), fill="#56686d")
    shown = year_aspect[year_aspect.aspect.isin(PRIMARY_ASPECTS)]
    for series_index, aspect in enumerate(PRIMARY_ASPECTS):
        group = shown[shown.aspect.eq(aspect)].set_index("year").loc[YEARS]
        points = []
        for year, row in group.iterrows():
            x = x_map[year]; y = bottom - (row.document_prevalence/maximum)*(bottom-top)
            low = bottom - (row.document_prevalence_ci_lower/maximum)*(bottom-top)
            high = bottom - (row.document_prevalence_ci_upper/maximum)*(bottom-top)
            draw.line((x, high, x, low), fill=SERIES[series_index], width=3)
            draw.line((x-8, high, x+8, high), fill=SERIES[series_index], width=3)
            draw.line((x-8, low, x+8, low), fill=SERIES[series_index], width=3)
            points.append((x, y))
        draw.line(points, fill=SERIES[series_index], width=6, joint="curve")
        for x, y in points: draw.ellipse((x-8, y-8, x+8, y+8), fill="white", outline=SERIES[series_index], width=5)
        ly = 955 + series_index*38
        draw.line((1080, ly+12, 1135, ly+12), fill=SERIES[series_index], width=6)
        draw.text((1150, ly), labels[aspect], font=_font(21), fill="#14262b")
    draw.text((110, 1025), "Note. Error bars are frozen Wilson 95% CIs; all values reflect model-estimated labels.", font=_font(19), fill="#56686d")
    _save(image, path, pdf)


def _training_figure(year_aspect: pd.DataFrame, path: Path) -> None:
    image, draw = _canvas("Training, preparation and pacing sentiment by edition",
        "Independent document-level sentiment presence among aspect-bearing posts")
    left, top, right, bottom = 180, 220, 1660, 870; maximum = .75
    for index in range(0, 4):
        value = index*.20; y = bottom-(value/maximum)*(bottom-top)
        draw.line((left,y,right,y),fill="#dce6e7",width=2); draw.text((95,y-14),f"{value:.0%}",font=_font(21),fill="#56686d")
    x_map = {year:left+i*(right-left)/3 for i,year in enumerate(YEARS)}
    data = year_aspect[year_aspect.aspect.eq("training_preparation_pacing")].set_index("year").loc[YEARS]
    for column, label, color in [("positive_document_share_within_aspect_year","Positive present",COLORS["positive"]),
                                  ("negative_document_share_within_aspect_year","Negative present",COLORS["negative"])]:
        points=[]
        for year,row in data.iterrows():
            x=x_map[year]; value=row[column]; y=bottom-(value/maximum)*(bottom-top); points.append((x,y))
            draw.text((x-30,y-42),f"{value:.1%}",font=_font(19,True),fill=color)
        draw.line(points,fill=color,width=7,joint="curve")
        for x,y in points: draw.ellipse((x-9,y-9,x+9,y+9),fill="white",outline=color,width=5)
    for year,x in x_map.items(): draw.text((x-27,bottom+25),str(year),font=_font(22,True),fill="#14262b")
    draw.line((1110,935,1160,935),fill=COLORS["positive"],width=7); draw.text((1175,922),"Positive present",font=_font(21),fill="#14262b")
    draw.line((1110,975,1160,975),fill=COLORS["negative"],width=7); draw.text((1175,962),"Negative present",font=_font(21),fill="#14262b")
    draw.text((110,1020),"Note. Positive and negative values may overlap because a document can contain multiple evaluative propositions for the same aspect.",font=_font(18),fill="#56686d")
    _save(image,path)


def _composition_figure(aspect_sentiment: pd.DataFrame, labels: dict[str,str], path: Path) -> None:
    image, draw = _canvas("Sentiment composition within selected KLSCM aspects",
        "Percentage of model-estimated mentions within each aspect")
    left, top, right = 520, 235, 1640; bar_h=72; gap=38
    pivot=aspect_sentiment[aspect_sentiment.aspect.isin(COMPOSITION_ASPECTS)].pivot(index="aspect",columns="sentiment",values="share_within_aspect")
    for i,aspect in enumerate(COMPOSITION_ASPECTS):
        y=top+i*(bar_h+gap); draw.text((90,y+20),labels[aspect],font=_font(21,True),fill="#14262b")
        x=left
        for sentiment in ["positive","negative","mixed","neutral"]:
            share=float(pivot.loc[aspect,sentiment]); width=(right-left)*share
            draw.rectangle((x,y,x+width,y+bar_h),fill=COLORS[sentiment])
            if width>78: draw.text((x+9,y+22),f"{share:.1%}",font=_font(17,True),fill="white")
            x+=width
    for i,sentiment in enumerate(["positive","negative","mixed","neutral"]):
        x=700+i*230; draw.rectangle((x,1010,x+24,1034),fill=COLORS[sentiment]); draw.text((x+35,1008),sentiment.title(),font=_font(19),fill="#14262b")
    draw.text((110,1065),"Note. Rows use aspect-specific mention denominators and are not document prevalence estimates.",font=_font(18),fill="#56686d")
    _save(image,path)


def _overall_figure(overall: pd.DataFrame, path: Path) -> None:
    image, draw = _canvas("Overall model-estimated sentiment distribution",
        "Share of 15,486 model-estimated aspect mentions")
    left, top, right, bottom=310,245,1600,860; maximum=.70
    for index,row in enumerate(overall.sort_values("mention_share",ascending=False).itertuples()):
        y=top+index*145; width=(right-left)*(row.mention_share/maximum)
        draw.text((95,y+25),row.sentiment.title(),font=_font(24,True),fill="#14262b")
        draw.rectangle((left,y,right,y+72),fill="#edf2f2"); draw.rectangle((left,y,left+width,y+72),fill=COLORS[row.sentiment])
        draw.text((left+width+18,y+19),f"{row.mention_count:,} ({row.mention_share:.1%})",font=_font(22,True),fill="#14262b")
    draw.text((110,955),"Denominator: model-estimated aspect mentions, not posts, people or runners.",font=_font(20),fill="#56686d")
    _save(image,path)


def _topic_figure(topic_long: pd.DataFrame, path: Path) -> None:
    selections = {
        "Race Performance, Endurance and Personal Achievement":"race_performance",
        "Runner Community, Encouragement and Support":"crowd_community_atmosphere",
        "Race Photography and Photo Sharing":"photography_media",
        "Running Clinics and Race Preparation":"training_preparation_pacing",
    }
    image,draw=_canvas("Descriptive topic–aspect alignment","Selected frozen topic and aspect relationships; no inferential testing")
    left,top,right=700,245,1600; maximum=1.0
    for i,(topic,aspect) in enumerate(selections.items()):
        row=topic_long[(topic_long.topic_label.eq(topic)) & (topic_long.aspect.eq(aspect))].iloc[0]
        y=top+i*155; width=(right-left)*(row.document_prevalence/maximum)
        wrapped=topic if len(topic)<54 else topic[:51]+"…"
        draw.text((80,y),wrapped,font=_font(21,True),fill="#14262b"); draw.text((80,y+38),row.aspect_display_label,font=_font(18),fill="#56686d")
        draw.rectangle((left,y,right,y+72),fill="#edf2f2"); draw.rectangle((left,y,left+width,y+72),fill="#087f8c")
        draw.text((left+width+15,y+20),f"{row.document_prevalence:.1%}",font=_font(21,True),fill="#14262b")
    draw.text((110,930),"Document prevalence within each frozen consolidated topic",font=_font(21),fill="#56686d")
    draw.text((110,1010),"Note. Topic–aspect relationships are descriptive only.",font=_font(19),fill="#56686d")
    _save(image,path)


def _md_table(frame: pd.DataFrame) -> str:
    columns=list(frame.columns); lines=["| "+" | ".join(columns)+" |","|"+"|".join(["---"]*len(columns))+"|"]
    for row in frame.itertuples(index=False,name=None): lines.append("| "+" | ".join("" if pd.isna(v) else str(v) for v in row)+" |")
    return "\n".join(lines)


def create_thesis_outputs(root: Path = ROOT, created_at: str | None = None) -> dict[str,Any]:
    if root.exists(): raise FileExistsError(f"Thesis output package already exists: {root}")
    created_at=created_at or datetime.now(timezone.utc).isoformat(); data=load_dashboard_data(); labels=dict(data.metadata["aspect_display_labels"])
    source_paths=sorted(path for path in DASHBOARD_ROOT.iterdir() if path.is_file()); hashes={str(path):_sha(path) for path in source_paths}
    root.mkdir(parents=True); (root/"dashboard_screenshots").mkdir()

    corpus_rows=[("Analyzed substantive posts",data.corpus["total_documents"],"documents"),
        ("Posts with >=1 model-estimated aspect",data.corpus["mention_bearing_documents"],"documents"),
        ("Zero-mention posts",data.corpus["zero_mention_documents"],"documents"),
        ("Model-estimated aspect mentions",data.corpus["total_mentions"],"mentions"),
        ("Mean mentions per document",round(data.corpus["mean_mentions_per_document"],3),"mentions/document"),
        ("Median mentions per document",int(data.corpus["median_mentions_per_document"]),"mentions/document"),
        ("Multi-aspect documents",data.corpus["multi_aspect_documents"],"documents")]
    corpus=pd.DataFrame(corpus_rows,columns=["metric","value","unit"])
    for row in data.years.itertuples(): corpus.loc[len(corpus)]=[f"Analyzed substantive posts ({int(row.year)})",int(row.documents),"documents"]
    corpus.to_csv(root/"table_01_corpus_summary.csv",index=False)
    (root/"table_01_corpus_summary.md").write_text("# Table 1. Corpus summary\n\n"+_md_table(corpus)+"\n",encoding="utf-8")

    overall=data.sentiment_overall[["sentiment","mention_count","mention_share"]].copy(); overall["denominator_description"]="share of model-estimated aspect mentions"
    overall.to_csv(root/"table_02_overall_sentiment.csv",index=False)
    aspect=data.aspects.rename(columns={"affected_document_count":"affected_documents","year_association_adjusted_p":"adjusted_p"})
    aspect[["aspect","display_label","affected_documents","document_prevalence_all","mention_count","mention_share","support_class","inferential_eligible","adjusted_p","cramers_v","effect_category","recommended_for_emphasis"]].to_csv(root/"table_03_aspect_prevalence.csv",index=False)
    sentiment=data.aspect_sentiment.pivot(index=["aspect","display_label","support_class"],columns="sentiment",values="share_within_aspect").reset_index().rename(columns={s:f"{s}_share" for s in ["positive","negative","mixed","neutral"]})
    sentiment=sentiment.merge(data.aspects[["aspect","mention_count","affected_document_count"]].rename(columns={"mention_count":"total_mentions","affected_document_count":"affected_documents"}),on="aspect")
    sentiment.to_csv(root/"table_04_aspect_sentiment.csv",index=False)

    temporal=data.year_aspect[data.year_aspect.aspect.isin(TEMPORAL_TABLE_ASPECTS)].pivot(index="aspect",columns="year",values="document_prevalence").reset_index()
    temporal.columns=["aspect"]+[f"{int(c)}_prevalence" for c in temporal.columns[1:]]
    temp_meta=data.aspects[["aspect","display_label","cramers_v","year_association_adjusted_p","effect_category","recommended_for_emphasis"]].rename(columns={"year_association_adjusted_p":"adjusted_p"})
    temporal=temp_meta.merge(temporal,on="aspect"); temporal.to_csv(root/"table_05_year_aspect_prevalence.csv",index=False)
    topic=data.topics[["topic_id","topic_label","document_count","mention_count","top_aspect_1","top_aspect_1_document_prevalence","top_aspect_2","top_aspect_2_document_prevalence","top_aspect_3","top_aspect_3_document_prevalence","positive_share","negative_share"]].copy(); topic["analysis_type"]="descriptive_only"; topic.to_csv(root/"table_06_topic_aspect_summary.csv",index=False)

    eligible=data.aspects[data.aspects.inferential_eligible].copy(); year_wide=data.year_aspect[data.year_aspect.aspect.isin(eligible.aspect)].pivot(index="aspect",columns="year",values="document_prevalence").reset_index(); year_wide.columns=["aspect"]+[f"{int(c)}_prevalence" for c in year_wide.columns[1:]]
    rows=[]
    for item in eligible.itertuples():
        pairs=data.pairwise[(data.pairwise.aspect.eq(item.aspect)) & data.pairwise.outcome.eq("aspect_prevalence")]
        if len(pairs):
            largest=pairs.iloc[pairs.percentage_point_difference.abs().argmax()]
            difference=largest.percentage_point_difference/100
            note=f"Largest frozen contrast: {int(largest.year_a)}–{int(largest.year_b)}; Holm-adjusted {'significant' if largest.significant_holm else 'not significant'}."
        else: difference=pd.NA; note="No frozen pairwise results (omnibus did not qualify)."
        rows.append({"aspect":item.aspect,"adjusted_p":item.year_association_adjusted_p,"cramers_v":item.cramers_v,
                     "effect_category":item.effect_category,"FDR_significant":item.year_association_fdr_significant,
                     "recommended_for_emphasis":item.recommended_for_emphasis,"largest_pairwise_difference":difference,"pairwise_note":note})
    inferential=year_wide.merge(pd.DataFrame(rows),on="aspect"); inferential.to_csv(root/"table_07_inferential_summary.csv",index=False)

    _line_figure(data.year_aspect,labels,root/"figure_01_primary_aspect_prevalence.png",root/"figure_01_primary_aspect_prevalence.pdf")
    _training_figure(data.year_aspect,root/"figure_02_training_sentiment_by_year.png")
    _composition_figure(data.aspect_sentiment,labels,root/"figure_03_aspect_sentiment_composition.png")
    _overall_figure(overall,root/"figure_04_overall_sentiment.png")
    _topic_figure(data.topic_aspect,root/"figure_05_descriptive_topic_aspect_alignment.png")

    findings=["# Thesis key findings (ABSA V1)","", "These statements summarize frozen model-estimated results and are not causal claims.",""]
    for _, row in data.findings.iterrows():
        if row["finding_type"]=="aspect_prevalence":
            findings.append(f"- **{row['headline']}** Prevalence changed from {row['2019_value']:.1%} in 2019 to {row['2025_value']:.1%} in 2025. The frozen edition association remained detectable after FDR correction (adjusted p = {row['adjusted_p']:.3g}) and had a {row['effect_category']} effect (Cramer's V = {row['effect_size']:.3f}). {row['short_interpretation']}")
        else:
            findings.append(f"- **{row['headline']}** Positive document-level presence was {row['2019_value']:.1%}, {row['2023_value']:.1%}, {row['2024_value']:.1%}, and {row['2025_value']:.1%} in 2019, 2023, 2024, and 2025, respectively. {row['short_interpretation']} {row['caution_note']}")
    findings += ["","Secondary results should not be elevated merely because an adjusted p-value is below .05. Photography/media, weather sentiment, sparse aspects, language comparisons and topic-wide patterns retain the frozen cautions."]
    (root/"THESIS_KEY_FINDINGS_V1.md").write_text("\n".join(findings),encoding="utf-8")

    limitations=f"""# ABSA limitations source (V1)\n\n1. Aspect and sentiment outputs are model-estimated labels, not manually verified labels for all {data.corpus['total_documents']:,} production documents.\n2. The selected system achieved development aspect precision of 0.513 and recall of 0.790.\n3. The predeclared precision target of 0.55 was not met.\n4. Human gold comprised 80 documents and was reused during controlled V1/V2/V3 prompt refinement, so it is development rather than independent confirmatory evidence.\n5. Production predictions were not manually verified corpus-wide.\n6. Statistical tests quantify sampling associations in frozen labels and do not incorporate ABSA classifier uncertainty.\n7. The Instagram corpus is not representative of all KLSCM participants or public opinion.\n8. Language assignments include uncertain and low-quality categories; language comparisons are descriptive and are not cultural inference.\n9. Topic-level analyses are descriptive; no broad topic-wide inferential testing was performed.\n10. Model extraction density increased across editions while zero-mention prevalence declined, which may reflect corpus composition and/or model behavior.\n11. Low- and very-low-support aspects require cautious interpretation.\n12. Observed temporal associations are non-causal and must not be interpreted as proof that KLSCM or its participants changed.\n"""
    (root/"THESIS_ABSA_LIMITATIONS_V1.md").write_text(limitations,encoding="utf-8")
    method="""# ABSA method summary (V1)\n\nThe study constructed a substantive KLSCM social-media corpus through frozen relevance filtering, followed by BERTopic discovery and researcher-finalized consolidation into 32 substantive topics. A frozen ontology defined 20 aspect families and four evaluative sentiment labels. An 80-document single-researcher human audit provided development gold. Three controlled OpenAI prompt versions were assessed using unchanged gold, ontology, schema and matching rules; `absa_v1_instructions_3_aspect_ownership` was selected for production with its known precision limitation.\n\nProduction Batch inference covered 7,704 documents and yielded 15,486 model-estimated mentions, including exact evidence and document metadata linkage. Offline descriptive analysis summarized aspect prevalence and sentiment at document and mention levels. Inferential analysis used documents as the independent unit, binary document×aspect and aspect-sentiment indicators, Pearson chi-square omnibus tests, Cramer's V, BH-FDR within predeclared analysis families and Holm correction for qualifying pairwise year comparisons. Frozen descriptive and inferential outputs were transformed into a static dashboard data mart and presentation-only Streamlit dashboard. No production model output was manually corrected.\n"""
    (root/"THESIS_ABSA_METHOD_SUMMARY_V1.md").write_text(method,encoding="utf-8")
    captions="""# Thesis figure captions (V1)\n\n**Figure 1.** Model-estimated document prevalence of the three principal KLSCM aspect families across the 2019, 2023, 2024 and 2025 editions. Error bars represent frozen Wilson 95% confidence intervals. Temporal associations were evaluated at document level; each aspect had an FDR-adjusted association with a small effect. Results reflect model-estimated ABSA labels rather than manually verified corpus-wide labels.\n\n**Figure 2.** Positive and negative document-level sentiment presence among posts containing training, preparation and pacing mentions. The indicators are independent and may overlap when a post contains multiple evaluative propositions.\n\n**Figure 3.** Sentiment composition of model-estimated mentions within seven selected supported aspect families. Percentages use the total number of mentions for each aspect as denominator.\n\n**Figure 4.** Overall sentiment distribution across 15,486 model-estimated aspect mentions. Percentages do not represent posts, individuals or the broader runner population.\n\n**Figure 5.** Descriptive alignment between four frozen consolidated topics and their conceptually corresponding aspect families. No topic-wide inferential testing was performed.\n"""
    (root/"THESIS_FIGURE_CAPTIONS_V1.md").write_text(captions,encoding="utf-8")
    notes="""# Thesis table notes (V1)\n\n- **Document prevalence:** affected documents divided by all documents in the stated corpus or year denominator.\n- **Mention share:** model-estimated mentions in a category divided by the relevant total mention denominator. Mention rows are nested within documents.\n- **Support class:** frozen affected-document thresholds: very low 0–29, low 30–99, moderate 100–499 and high 500 or more.\n- **Cramer's V:** frozen omnibus association effect size; <.10 trivial, .10–<.20 small, .20–<.30 moderate and ≥.30 strong.\n- **FDR:** Benjamini–Hochberg false-discovery-rate adjustment applied within frozen inferential families.\n- **Model-estimated sentiment:** structured ABSA output from the selected V3 system; it is not manually verified ground truth for the full production corpus.\n- **Pairwise comparisons:** shown only where produced by the frozen post-omnibus Holm-corrected procedure. No comparisons were added for this package.\n"""
    (root/"THESIS_TABLE_NOTES_V1.md").write_text(notes,encoding="utf-8")
    (root/"dashboard_screenshots"/"SCREENSHOTS_NOT_GENERATED.md").write_text("Dashboard screenshots were not generated because no workspace-exportable browser screenshot package is installed. The existing in-app browser supported visual QA but not direct deterministic export into this research directory; no new browser infrastructure was installed.\n",encoding="utf-8")

    try: commit=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True,check=False).stdout.strip() or None
    except OSError: commit=None
    outputs=[path for path in root.rglob("*") if path.is_file()]
    manifest={"protocol":"absa_v1_thesis_outputs_v1","created_at":created_at,"code_version":"marathon_absa.absa_final_outputs:v1","git_commit":commit,
        "production_system":"absa_v1_instructions_3_aspect_ownership","model_precision":.513,"model_recall":.790,
        "source_artifacts":{str(path):{"sha256":hashes[str(path)]} for path in source_paths},
        "generated_tables":[str(path) for path in outputs if path.name.startswith("table_")],
        "generated_figures":[str(path) for path in outputs if path.name.startswith("figure_")],
        "generated_markdown":[str(path) for path in outputs if path.suffix==".md"],
        "execution":{"new_statistical_hypothesis_tests":0,"new_model_inference":0,"new_manual_review":0,"new_topic_modeling":0,"api_calls":0},
        "python_version":platform.python_version(),"output_hashes":{str(path.relative_to(root)):_sha(path) for path in outputs}}
    (root/"final_output_manifest_v1.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    if {str(path):_sha(path) for path in source_paths}!=hashes: raise RuntimeError("Frozen dashboard source changed")
    return {"output_root":str(root),"tables":8,"figures":6,"markdown":7,"screenshots":0,"api_calls":0,"new_statistical_tests":0}
