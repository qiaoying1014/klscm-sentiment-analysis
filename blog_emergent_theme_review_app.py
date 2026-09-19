"""Researcher-only, resumable emergent consolidation review."""
import pandas as pd
import streamlit as st
from marathon_absa.blog_emergent_themes import (
    ROOT, AUDIT, DECISION_FILE, DECISIONS, read, sha, load_source, save_decision, public_text,
)

st.set_page_config(page_title='Blog-emergent consolidation', layout='wide')
st.title('Blog-emergent theme consolidation')
st.caption('Support uses unique parent reviews / 25. Original labels came from delegated AI-assisted review; they are not independent human validation. Similarity is advisory only.')
if not (ROOT / (AUDIT + '.csv')).exists():
    st.error('Prepare the blog-emergent audit first.')
    st.stop()
audit = read(ROOT / (AUDIT + '.csv'))
decisions = read(ROOT / DECISION_FILE)
current_version = sha(ROOT / DECISION_FILE)
if st.button('Reload saved decisions'):
    st.session_state.clear()
    st.rerun()
source = load_source()
st.write(f"Resolved: {sum(decisions.researcher_decision.ne('UNCLEAR'))} / {len(decisions)}")
labels = audit.set_index('existing_theme_id').researcher_label_original.to_dict()
identity = st.selectbox('Existing emergent concept', audit.existing_theme_id.tolist(), format_func=lambda x: public_text(labels[x]) + ' — ' + x)
revision_key = 'displayed_revision_' + identity
if revision_key not in st.session_state:
    st.session_state[revision_key] = current_version
version = st.session_state[revision_key]
row = audit.set_index('existing_theme_id').loc[identity]
st.info(public_text(row.notes))
st.write('Aspect:', row.aspect)
st.write(f'Support: {row.parent_review_count}/25 parent reviews; {row.mention_count} mentions')
if int(row.parent_review_count) < 2:
    st.info('Observed in one sampled blog review; requires cautious interpretation.')
units = pd.read_csv('data/processed/units.csv', keep_default_na=False, usecols=['unit_id','document_id','text'])

def show_evidence(theme):
    for mention in source[source.theme_id.eq(theme)].itertuples():
        with st.expander(f'Parent {mention.review_id} · chunk {mention.chunk_id} · mention {mention.blog_mention_id}'):
            st.write(public_text(mention.evidence_text))
            st.caption('English gloss: ' + public_text(mention.english_gloss))
            st.caption('Original decision note: ' + public_text(mention.researcher_note))
            st.write('Review context (inference chunk):')
            for text in units.loc[units.unit_id.eq(mention.chunk_id), 'text']:
                st.write(public_text(text))
show_evidence(identity)
peers = audit[(audit.aspect == row.aspect) & (audit.existing_theme_id != identity)]
st.subheader('Same-aspect candidates')
st.caption('All same-aspect peers are available. Lexical label similarity is a secondary navigation aid, not semantic equivalence or a merge threshold.')
pairs = read(ROOT / 'blog_emergent_theme_candidates_v1.csv')
for peer in peers.itertuples():
    pair = pairs[((pairs.source_id == identity) & (pairs.candidate_id == peer.existing_theme_id)) | ((pairs.candidate_id == identity) & (pairs.source_id == peer.existing_theme_id))]
    with st.expander(public_text(peer.researcher_label_original) + f' · {peer.parent_review_count}/25'):
        st.write(peer.existing_theme_id)
        if len(pair):
            st.caption(f'Lexical similarity: {float(pair.iloc[0].label_similarity):.3f}')
        show_evidence(peer.existing_theme_id)
saved = decisions.set_index('existing_theme_id').loc[identity]
with st.form('decision_' + identity):
    options = ['UNCLEAR','KEEP_SEPARATE','MERGE_WITH_EXISTING_EMERGENT','RENAME_ONLY','EXCLUDE_AS_NON_THEME']
    decision = st.selectbox('Researcher decision', options, index=options.index(saved.researcher_decision))
    targets = [''] + peers.existing_theme_id.tolist()
    target = st.selectbox('Merge target (used only for merge)', targets, index=targets.index(saved.target_existing_theme_id) if saved.target_existing_theme_id in targets else 0, format_func=lambda x: public_text(labels.get(x, 'No target')))
    label = st.text_input('Confirmed final label', value=public_text(saved.final_label))
    reason = st.text_area('Reason (required for merge or exclusion)', value=public_text(saved.reason))
    submitted = st.form_submit_button('Save decision')
if submitted:
    try:
        save_decision(ROOT, identity, decision, target, label, reason, version)
        st.session_state.pop(revision_key, None)
        st.success('Decision saved with backup. Reloading saved state.')
        st.rerun()
    except (ValueError, FileExistsError) as error:
        st.error(str(error))
st.caption('UNCLEAR blocks finalization. Merges must share an aspect, retain meaningful distinctions, and confirm the target’s final label. No frozen Instagram theme is edited here.')
