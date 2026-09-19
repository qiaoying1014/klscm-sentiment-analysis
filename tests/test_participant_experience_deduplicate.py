import copy

import pytest

from marathon_absa import participant_experience_deduplicate as repair
from marathon_absa import participant_experience_dashboard_data as grounding


def example():
    theme = {'theme_id': 'test', 'blog_support_reviews': 0,
             'representative_evidence': [{'evidence_id': 'exact-id', 'sentiment': 'mixed'}]}
    row = {'theme_id': 'test', 'claims': {f: {'text': 'Test observation.', 'evidence_ids': ['exact-id']}
                                       for f in grounding.FIELDS}}
    row['claims']['organizer_implication']['text'] = 'Organizers may consider reviewing the observation.'
    return row, theme


def test_exact_dedup_only_preserves_raw_text_and_order():
    row, theme = example()
    row['claims']['organizer_insight']['evidence_ids'] *= 2
    before = copy.deepcopy(row)
    fixed, changes = repair.deduplicate(row, theme)
    assert row == before
    assert changes == [{'field': 'organizer_insight', 'before': ['exact-id', 'exact-id'],
                        'after': ['exact-id'], 'removed_count': 1}]
    assert all(fixed['claims'][f]['text'] == row['claims'][f]['text'] for f in grounding.FIELDS)
    grounding.validate_insight(fixed, theme)


@pytest.mark.parametrize('ids', [['exact-i'], ['unknown', 'unknown'], []])
def test_unknown_truncated_and_missing_evidence_never_fixed(ids):
    row, theme = example()
    row['claims']['organizer_insight']['evidence_ids'] = ids
    before = copy.deepcopy(row)
    with pytest.raises(ValueError): repair.deduplicate(row, theme)
    assert row == before


def test_noop_valid_claim_unchanged():
    row, theme = example()
    fixed, changes = repair.deduplicate(row, theme)
    assert fixed == row and changes == []


def test_real_offline_build_all_99_preserves_98_and_records_one_correction():
    candidates, reviews, audit = repair.build(grounding.ROOT)
    assert len(candidates['insights']) == len(reviews) == 99
    assert len({r['theme_id'] for r in reviews}) == 99
    assert audit['unchanged_candidates'] == 98
    assert len(audit['corrections']) == 1
    assert audit['corrections'][0]['theme_id'] == 'physical_experience__rt04'
    assert audit['corrections'][0]['changes'][0]['removed_count'] == 1
    for identity, record in candidates['retry_provenance']['themes'].items():
        if identity != 'physical_experience__rt04':
            assert record['raw_insight_hash'] == record['insight_hash']
    assert all(r['decision'] == 'PENDING' for r in reviews)


def test_existing_review_blocks_writes(tmp_path):
    (tmp_path / 'review.json').write_text('preserve', encoding='utf-8')
    with pytest.raises(FileExistsError): repair.merge(tmp_path)
    assert (tmp_path / 'review.json').read_text() == 'preserve'
