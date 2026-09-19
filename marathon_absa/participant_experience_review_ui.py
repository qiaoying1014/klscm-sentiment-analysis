"""Local human review interface. No API, inference, or finalization actions."""
import copy
from pathlib import Path

import streamlit as st

from .participant_experience_dashboard_data import ROOT, FIELDS
from .participant_experience_review import (
    CLAIM_STATES, DECISIONS, claim_reviews, load_review, resolve_evidence, save_review,
)


def render_review(root: Path = ROOT):
    st.title('Participant experience: researcher review')
    st.info('Validation means structurally grounded candidate output. It does not mean the claim has been substantively approved by a researcher.')
    st.caption('AI candidate → Researcher revision → Researcher-approved synthesis → Separate finalized/released synthesis')
    try:
        data = load_review(root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f'Review integrity check failed: {exc}')
        return
    if (Path(root) / 'finalized.json').exists() or (Path(root) / 'finalized_manifest.json').exists():
        st.warning('This package is finalized or partially finalized. Review editing is closed.')
        return
    if st.session_state.pop('review_saved', False):
        st.success('Researcher decision saved. No synthesis was finalized.')
    counts = {d: sum(r['decision'] == d for r in data['reviews']) for d in DECISIONS}
    st.write(' · '.join(f'{d}: {n}' for d, n in counts.items()))
    st.caption('Navigation discards unsaved edits. Save the current theme before changing filters or theme.')
    reviewer = st.text_input('Reviewer identity', key='researcher_identity')
    status = st.selectbox('Decision filter', ['All', *DECISIONS])
    aspect = st.selectbox('Aspect filter', ['All', *data['package']['aspects'].values()])
    search = st.text_input('Find theme by label or ID')
    rows = [r for r in data['reviews'] if (status == 'All' or r['decision'] == status)
            and (aspect == 'All' or data['themes'][r['theme_id']]['aspect_label'] == aspect)
            and search.lower() in (r['theme_id'] + data['themes'][r['theme_id']]['theme_label']).lower()]
    if not rows:
        st.info('No themes match these filters.')
        return
    identity = st.selectbox('Theme', [r['theme_id'] for r in rows],
        format_func=lambda i: f"{data['themes'][i]['aspect_label']} / {data['themes'][i]['theme_label']} ({i})")
    row = next(r for r in rows if r['theme_id'] == identity)
    theme = data['themes'][identity]
    st.subheader(theme['theme_label'])
    st.write(theme['evidence_descriptor'])
    st.warning(theme['scope_note'])
    st.write(f"Instagram: {theme['instagram_support_documents']} supporting documents / {data['package']['source_units']['instagram']['total']} analyzed documents. Blog: {theme['blog_support_reviews']} parent reviews / {data['package']['source_units']['blog']['total']} analyzed reviews.")
    if theme['blog_support_reviews'] == 1:
        st.warning('Singleton blog support: multiple mentions from one review are not independent repetition.')
    with st.expander('Source-specific sentiment, unavailable excerpts and generation provenance'):
        st.json({'sentiment_evidence': theme['sentiment_evidence'],
                 'unavailable_evidence_ids': theme.get('unavailable_evidence_ids', []),
                 'request': data['request'], 'candidate_hash': row['candidate_hash'],
                 'generated_at': data['candidates']['generated_at'],
                 'generation_retry_mechanical_provenance': data['candidates'].get('retry_provenance', {}).get('themes', {}).get(identity, {})})
    st.caption('Evidence is shown exactly as stored in the frozen, privacy-redacted package. English glosses are separate from the original excerpt.')
    edited = copy.deepcopy(row['insight'])
    states = claim_reviews(row)
    # Token-derived widget keys prevent carrying edits onto a newer disk revision.
    key = identity + data['token']
    for field in FIELDS:
        st.subheader(field.replace('_', ' ').capitalize())
        left, right = st.columns(2)
        original = data['originals'][identity]['claims'][field]
        current = edited['claims'][field]
        with left:
            st.caption('Immutable AI candidate')
            st.text(original['text'])
            st.caption('Original citations: ' + ', '.join(original['evidence_ids']))
            current['text'] = st.text_area('Researcher working text', value=current['text'], key=key + field + 'text')
            current['evidence_ids'] = st.multiselect('Cited evidence for working text',
                [e['evidence_id'] for e in theme['representative_evidence']],
                default=current['evidence_ids'], key=key + field + 'ids')
            states[field]['state'] = st.selectbox('Claim judgment', CLAIM_STATES,
                index=CLAIM_STATES.index(states[field]['state']), key=key + field + 'state')
            states[field]['note'] = st.text_area('Claim review note', value=states[field]['note'], key=key + field + 'note')
        with right:
            st.caption('Exact cited evidence for researcher working text')
            if not current['evidence_ids']:
                st.write('No cited evidence. Only the prescribed insufficient-evidence directional summary can be saved without citations.')
            for evidence in resolve_evidence(theme, current['evidence_ids']):
                st.markdown(f"**{evidence['source'].upper()} · {evidence['sentiment']}**")
                st.caption(evidence['evidence_id'] + ' · parent ' + evidence['parent_id'])
                st.text(evidence['text'])
                if evidence.get('english_gloss'):
                    st.caption('Existing English gloss (not original evidence)')
                    st.text(evidence['english_gloss'])
            removed = [i for i in original['evidence_ids'] if i not in current['evidence_ids']]
            if removed:
                with st.expander('Original candidate citations removed from working text'):
                    for evidence in resolve_evidence(theme, removed):
                        st.caption(evidence['source'] + ' · ' + evidence['evidence_id'])
                        st.text(evidence['text'])
    with st.expander('All available theme evidence'):
        for evidence in theme['representative_evidence']:
            st.caption(evidence['source'] + ' · ' + evidence['evidence_id'])
            st.text(evidence['text'])
    decision = st.selectbox('Theme decision', DECISIONS, index=DECISIONS.index(row['decision']), key=key + 'decision')
    reason = st.text_area('Theme decision / draft reason', value=row['reason'], key=key + 'reason')
    st.caption('APPROVE requires ACCEPT or REVISE for all seven claims. REVISE and REJECT require claim notes. Every save requires your identity and a reason. Rejection blocks release; it does not delete the theme.')
    if st.button('Save researcher review', type='primary'):
        try:
            save_review(root, data['token'], identity, edited, states, decision, reviewer, reason)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            st.error(str(exc))
        else:
            st.session_state['review_saved'] = True
            st.rerun()
    with st.expander('Saved researcher identity and revision history'):
        st.json({k: row.get(k) for k in ('reviewer', 'reviewed_at', 'reason', 'review_history')})
