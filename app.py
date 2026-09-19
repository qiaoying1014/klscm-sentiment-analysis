from __future__ import annotations
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from marathon_absa.config import SETTINGS
from marathon_absa.storage import read_table

st.set_page_config(page_title="KLSCM Multilingual ABSA",page_icon="🏃",layout="wide")
st.title("KLSCM Multilingual Topic Discovery & ABSA")
st.caption("Social-media and blog analysis · GIS and interview triangulation are deferred")

st.info("This is the source-preparation dashboard. Finalized Instagram and long-form review findings are presented in absa_dashboard.py, under Cross-Source Analysis for the 25 parent reviews. Counts below describe preparation records, not the finalized analysis population.")

@st.cache_data(show_spinner=False)
def load_optional(name):
    path=SETTINGS.output_dir/f"{name}.parquet"; return read_table(path) if path.exists() else pd.DataFrame()


def format_detected_languages(components):
    if not isinstance(components,list): return ""
    values=[]
    for component in components:
        if not isinstance(component,dict): continue
        language=str(component.get("language","")).strip()
        if not language: continue
        coverage=component.get("coverage")
        values.append(f"{language} ({coverage:.1%})" if isinstance(coverage,(int,float)) else language)
    return " · ".join(values)

documents=load_optional("documents")
if documents.empty: st.warning("No processed data found. Run `python -m marathon_absa.cli prepare` first."); st.stop()
for col,default in [("primary_language",documents.get("detected_language","undetermined")),("is_mixed_language",False),("language_method","legacy"),("language_adjudication_status","not_required")]:
    if col not in documents: documents[col]=default
documents["event_year"]=documents.event_year.replace("","Unknown").fillna("Unknown").astype(str)
relevance=load_optional("relevance"); topics=load_optional("topic_assignments"); topic_info=load_optional("topic_info"); mentions=load_optional("aspect_mentions"); units=load_optional("units")

with st.sidebar:
    st.header("Filters"); sources=st.multiselect("Source",sorted(documents.source.unique()),default=sorted(documents.source.unique())); years=sorted(documents.event_year.unique()); selected_years=st.multiselect("Event year",years,default=years)
    languages=sorted(documents.primary_language.fillna("undetermined").unique()); selected_languages=st.multiselect("Primary language",languages,default=languages)
    mixed_filter=st.selectbox("Mixed language",["All","Mixed only","Single only"]); methods=sorted(documents.language_method.fillna("legacy").unique()); selected_methods=st.multiselect("Detection method",methods,default=methods)
filtered=documents[documents.source.isin(sources)&documents.event_year.isin(selected_years)&documents.primary_language.fillna("undetermined").isin(selected_languages)&documents.language_method.fillna("legacy").isin(selected_methods)]
if mixed_filter=="Mixed only": filtered=filtered[filtered.is_mixed_language]
elif mixed_filter=="Single only": filtered=filtered[~filtered.is_mixed_language]
doc_ids=set(filtered.document_id)

tabs=st.tabs(["Overview","Language & relevance","Topics","Aspect sentiment","Hashtags & emoji","Evidence","Validation"])
with tabs[0]:
    ready=filtered.processing_status.eq("ready").sum(); filtered_units=units[units.document_id.isin(doc_ids)] if not units.empty else pd.DataFrame()
    c1,c2,c3,c4=st.columns(4); c1.metric("Source records",f"{len(filtered):,}"); c2.metric("Ready documents",f"{ready:,}"); c3.metric("Analysis chunks",f"{len(filtered_units):,}"); c4.metric("Duplicates",f"{filtered.processing_status.eq('duplicate').sum():,}")
    counts=filtered.groupby(["event_year","source"]).size().reset_index(name="documents"); st.plotly_chart(px.bar(counts,x="event_year",y="documents",color="source",barmode="group",title="Documents by event year"),width="stretch")
    blog=filtered[filtered.source=="blog"]
    if not blog.empty:
        blog_docs=blog.groupby("event_year").size().reset_index(name="documents"); blog_chunks=filtered_units[filtered_units.source=="blog"].groupby("event_year").size().reset_index(name="chunks") if not filtered_units.empty else pd.DataFrame()
        left,right=st.columns(2); left.plotly_chart(px.bar(blog_docs,x="event_year",y="documents",title="Blog source records by year",text_auto=True),width="stretch")
        if not blog_chunks.empty: right.plotly_chart(px.bar(blog_chunks,x="event_year",y="chunks",title="Blog analysis chunks by year",text_auto=True),width="stretch")
with tabs[1]:
    excluded_status={"empty","duplicate"}; excluded_language={"no_text","insufficient_text"}
    valid=filtered[~filtered.processing_status.isin(excluded_status)&~filtered.language_status.isin(excluded_language)]
    st.caption("Language distribution excludes empty, duplicate, no-text, and insufficient-text records.")
    lang=valid.groupby(["primary_language","source"]).size().reset_index(name="documents"); st.plotly_chart(px.bar(lang,x="primary_language",y="documents",color="source",barmode="group",title="Final primary-language distribution"),width="stretch")
    blog_lang=valid[valid.source=="blog"].groupby("primary_language").size().reset_index(name="documents")
    if not blog_lang.empty: st.plotly_chart(px.bar(blog_lang,x="primary_language",y="documents",title="Blog languages",text_auto=True),width="stretch")
    quality=filtered[filtered.processing_status.isin(excluded_status)|filtered.language_status.isin(excluded_language)|filtered.language_adjudication_status.eq("pending")]
    st.subheader("Data quality and language review queue"); st.dataframe(quality[["document_id","source","event_year","processing_status","primary_language","language_status","language_method","language_review_reason","language_adjudication_status","is_mixed_language"]],width="stretch",hide_index=True)
    mixed=valid[valid.is_mixed_language].copy(); mixed["detected_languages"]=mixed.detected_languages.map(format_detected_languages); st.subheader("Mixed-language records"); st.dataframe(mixed[["document_id","source","original_text","primary_language","detected_languages","language_method"]],width="stretch",hide_index=True)
    if not relevance.empty:
        rel=relevance[relevance.document_id.isin(doc_ids)].copy()
        st.plotly_chart(px.bar(rel.relevance.value_counts().rename_axis("relevance").reset_index(name="documents"),x="relevance",y="documents",color="relevance",title="Relevance funnel"),width="stretch")
        pending=int(rel.review_status.eq("pending").sum()) if "review_status" in rel else 0
        included=int(rel.include_in_topics.fillna(False).astype(bool).sum()) if "include_in_topics" in rel else int(rel.relevance.eq("relevant").sum())
        a,b=st.columns(2); a.metric("Pending relevance review",f"{pending:,}"); b.metric("Included in topic discovery",f"{included:,}")
        if "reason_code" in rel:
            reasons=rel.reason_code.fillna("unknown").value_counts().rename_axis("reason_code").reset_index(name="documents")
            st.plotly_chart(px.bar(reasons,x="reason_code",y="documents",color="reason_code",title="Relevance decision reasons"),width="stretch")
        rel_context=rel.merge(documents[["document_id","event_year","primary_language"]],on="document_id",how="left")
        by_year=rel_context.groupby(["event_year","relevance"]).size().reset_index(name="documents")
        by_language=rel_context.groupby(["primary_language","relevance"]).size().reset_index(name="documents")
        left,right=st.columns(2)
        left.plotly_chart(px.bar(by_year,x="event_year",y="documents",color="relevance",barmode="group",title="Relevance decisions by year"),width="stretch")
        right.plotly_chart(px.bar(by_language,x="primary_language",y="documents",color="relevance",barmode="stack",title="Relevance decisions by language"),width="stretch")
with tabs[2]:
    if topics.empty: st.info("Run topic discovery after reviewing relevance outputs.")
    else:
        shown=topics[topics.document_id.isin(doc_ids)]; counts=shown.groupby(["topic_id","source"]).size().reset_index(name="units"); st.plotly_chart(px.bar(counts,x="topic_id",y="units",color="source",barmode="group"),width="stretch"); st.dataframe(topic_info,width="stretch",hide_index=True)
with tabs[3]:
    if mentions.empty: st.info("Run ABSA after validating the topic-derived taxonomy.")
    else:
        shown=mentions[mentions.document_id.isin(doc_ids)]; counts=shown.groupby(["aspect","sentiment"]).size().reset_index(name="mentions"); st.plotly_chart(px.bar(counts,x="aspect",y="mentions",color="sentiment",barmode="group"),width="stretch")
with tabs[4]:
    tags=[]; icons=[]
    for row in filtered.itertuples(index=False):
        tags.extend({"feature":x,"source":row.source} for x in row.hashtags); icons.extend({"emoji":a,"alias":b,"source":row.source} for a,b in zip(row.emojis,row.emoji_aliases))
    if tags: st.plotly_chart(px.bar(pd.DataFrame(tags).value_counts(["feature","source"]).reset_index(name="count").sort_values("count",ascending=False).head(40),x="feature",y="count",color="source"),width="stretch")
    if icons: st.dataframe(pd.DataFrame(icons).value_counts(["emoji","alias","source"]).reset_index(name="count").sort_values("count",ascending=False).head(40),width="stretch",hide_index=True)
with tabs[5]:
    if mentions.empty: st.info("Evidence becomes available after ABSA.")
    else: st.dataframe(mentions[mentions.document_id.isin(doc_ids)],width="stretch",hide_index=True)
with tabs[6]:
    validation=load_optional("validation_sample"); st.dataframe(validation,width="stretch",hide_index=True) if not validation.empty else st.info("Create a validation sample with the CLI.")
    manifest=SETTINGS.output_dir/"run_manifest.jsonl"
    if manifest.exists(): st.dataframe(pd.DataFrame([json.loads(x) for x in manifest.read_text(encoding="utf-8").splitlines() if x]),width="stretch",hide_index=True)

