import json
import socket
from pathlib import Path

import pandas as pd
import pytest

from marathon_absa import blog_emergent_themes as e


@pytest.fixture
def sample():
    source = pd.DataFrame([
        dict(theme_id='blog_emergent__a__1',aspect='a',review_id='r0',chunk_id='c1',blog_mention_id='m1',evidence_text='First evidence',english_gloss='First gloss',emergent_blog_theme_label='Toilet queues'),
        dict(theme_id='blog_emergent__a__1',aspect='a',review_id='r0',chunk_id='c2',blog_mention_id='m2',evidence_text='Repeated review evidence',english_gloss='Second gloss',emergent_blog_theme_label='Toilet queues'),
        dict(theme_id='blog_emergent__a__2',aspect='a',review_id='r1',chunk_id='c3',blog_mention_id='m3',evidence_text='Another queue',english_gloss='Queue',emergent_blog_theme_label='Waiting for toilets'),
    ])
    decisions = pd.DataFrame([
        dict(existing_theme_id='blog_emergent__a__1',researcher_decision='KEEP_SEPARATE',target_existing_theme_id='',final_label='Toilet queues',reason=''),
        dict(existing_theme_id='blog_emergent__a__2',researcher_decision='KEEP_SEPARATE',target_existing_theme_id='',final_label='Waiting for toilets',reason=''),
    ])
    return source, decisions, ['r'+str(n) for n in range(25)]


def test_support_lineage_singletons_and_denominator(sample):
    frame, lineage, summary = e.build_taxonomy(*sample)
    assert frame.support_reviews.tolist() == [1,1]
    assert frame.support_mentions.tolist() == [2,1]
    assert frame.review_prevalence.tolist() == [.04,.04]
    assert frame.total_included_reviews.eq(25).all()
    assert frame.singleton_flag.all() and frame.low_support_flag.all()
    assert len(lineage) == 3
    assert lineage.evidence_text.tolist() == sample[0].evidence_text.tolist()
    assert lineage.blog_mention_id.tolist() == sample[0].blog_mention_id.tolist()
    assert set(lineage.theme_id) == set(sample[0].theme_id)
    assert summary['api_calls'] == 0


def merge(sample):
    source, decisions, parents = sample
    decisions.loc[1,['researcher_decision','target_existing_theme_id','final_label','reason']] = ['MERGE_WITH_EXISTING_EMERGENT','blog_emergent__a__1','Toilet queues','Evidence expresses the same waiting problem; no distinct facility is combined.']
    return source, decisions, parents


def test_merge_unique_reviews_and_deterministic_ids(sample):
    source, decisions, parents = merge(sample)
    frame, lineage, summary = e.build_taxonomy(source, decisions, parents)
    assert frame.iloc[0].support_reviews == 2
    assert frame.iloc[0].support_mentions == 3
    assert not frame.iloc[0].low_support_flag and not frame.iloc[0].singleton_flag
    other, _, _ = e.build_taxonomy(source.iloc[::-1], decisions.iloc[::-1], parents[::-1])
    pd.testing.assert_frame_equal(frame, other)
    assert json.loads(frame.iloc[0].source_existing_theme_ids) == sorted(set(source.theme_id))
    assert summary['merged_groups'] == 1


def test_merge_same_parent_once(sample):
    source, decisions, parents = merge(sample)
    source['review_id'] = 'r0'
    frame, _, _ = e.build_taxonomy(source,decisions,parents)
    assert frame.iloc[0].support_reviews == 1 and frame.iloc[0].support_mentions == 3


def test_cross_aspect_rejected(sample):
    source, decisions, parents = merge(sample)
    source.loc[2,'aspect']='b'
    with pytest.raises(ValueError,match='same-aspect'):
        e.build_taxonomy(source,decisions,parents)


@pytest.mark.parametrize('mutation,message', [
    ('unclear','Unresolved'),('missing_id','accounted'),('duplicate_id','accounted'),
    ('empty_label','Readable'),('hash_label','Readable'),('missing_label','Readable'),
    ('denominator','exactly 25'),('parent','Unknown'),('duplicate_mention','Duplicate'),
    ('evidence','missing evidence'),('exclude_reason','reason'),('rename_keep','RENAME_ONLY'),
])
def test_fail_closed(sample,mutation,message):
    s,d,p=sample
    if mutation=='unclear':d.loc[0,'researcher_decision']='UNCLEAR'
    if mutation=='missing_id':d=d.iloc[:1]
    if mutation=='duplicate_id':d=pd.concat([d,d.iloc[:1]])
    if mutation=='empty_label':d.loc[0,'final_label']=' '
    if mutation=='hash_label':d.loc[0,'final_label']='blog_emergent__abc'
    if mutation=='missing_label':d.loc[0,'final_label']=e.MISSING
    if mutation=='denominator':p=p[:-1]
    if mutation=='parent':s.loc[0,'review_id']='unknown'
    if mutation=='duplicate_mention':s.loc[1,'blog_mention_id']='m1'
    if mutation=='evidence':s.loc[0,'evidence_text']=''
    if mutation=='exclude_reason':d.loc[0,'researcher_decision']='EXCLUDE_AS_NON_THEME'
    if mutation=='rename_keep':d.loc[0,'final_label']='New readable label'
    with pytest.raises(ValueError,match=message):e.build_taxonomy(s,d,p)


def test_exclusion_preserves_evidence(sample):
    s,d,p=sample
    d.loc[0,['researcher_decision','reason']]=['EXCLUDE_AS_NON_THEME','Not an evaluative concept']
    frame,lineage,summary=e.build_taxonomy(s,d,p)
    assert len(frame)==1 and len(lineage)==3
    assert lineage.iloc[0].emergent_theme_id==''
    assert lineage.iloc[0].reason=='Not an evaluative concept'
    assert summary['excluded_non_theme_observations']==1


def test_rename_preserves_identity(sample):
    s,d,p=sample
    d.loc[0,['researcher_decision','final_label']]=['RENAME_ONLY','Waiting time for toilets']
    frame,_,_=e.build_taxonomy(s,d,p)
    assert frame.iloc[0].emergent_theme_id==s.iloc[0].theme_id


def test_cycles_and_unconfirmed_merge_rejected(sample):
    s,d,p=merge(sample)
    d.loc[0,'researcher_decision']='MERGE_WITH_EXISTING_EMERGENT'
    d.loc[0,'target_existing_theme_id']=d.loc[1,'existing_theme_id']
    with pytest.raises(ValueError,match='chains/cycles'):e.build_taxonomy(s,d,p)


def test_atomic_backup_and_resumable_conflict(tmp_path):
    path=tmp_path/e.DECISION_FILE
    d=pd.DataFrame([dict(existing_theme_id='a',researcher_decision='UNCLEAR',target_existing_theme_id='',final_label='Readable label',reason='')])
    e.safe_write(path,d.to_csv(index=False))
    version=e.sha(path)
    e.save_decision(tmp_path,'a','KEEP_SEPARATE','','Readable label','',version)
    assert e.read(path).iloc[0].researcher_decision=='KEEP_SEPARATE'
    assert len(list((tmp_path/'emergent_backups').glob('*')))==1
    with pytest.raises(ValueError,match='another session'):
        e.save_decision(tmp_path,'a','UNCLEAR','','Readable label','',version)
    assert not (tmp_path/'blog_emergent_review.lock').exists()


def test_real_integrity_and_no_network(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Network forbidden')
    monkeypatch.setattr(socket.socket,'connect',forbidden)
    result=e.verify_integrity()
    assert result['all_passed'] and all(result['checks'].values())
    assert result['prompt_sha256']=='196606110ecb2b78d56a3220c1bd8957e8e42f2f259c65d1094c037fa21f30d0'
    source=e.load_source()
    assert len(source)==67 and source.theme_id.nunique()==35
    assert source.aspect.nunique()==17 and source.review_id.nunique()==18
    real_read = e.read
    def pending_read(path):
        frame = real_read(path)
        if Path(path).name == e.DECISION_FILE:
            frame['researcher_decision'] = 'UNCLEAR'
        return frame
    monkeypatch.setattr(e, 'read', pending_read)
    before = {str(p): e.sha(p) for p in e.ROOT.glob('blog_emergent_theme*') if p.is_file()}
    with pytest.raises(ValueError,match='Unresolved'):
        e.finalize()
    assert before == {str(p): e.sha(p) for p in e.ROOT.glob('blog_emergent_theme*') if p.is_file()}
    assert e.verify_integrity()['all_passed']


def test_changed_frozen_hash_blocks(tmp_path):
    file=tmp_path/'frozen.txt';file.write_text('before')
    (tmp_path/e.BASELINE).write_text(json.dumps({str(file):e.sha(file)}))
    file.write_text('changed')
    with pytest.raises(ValueError,match='hashes differ'):e.verify_integrity(tmp_path)


def test_cross_source_existing_findings_and_pending_no_writes(monkeypatch):
    from marathon_absa.cross_source_analysis import create_cross_source_analysis
    root=Path('data/processed/cross_source_analysis_v1')
    before={str(p):e.sha(p) for p in root.glob('*') if p.is_file()}
    manifest=json.loads((root/'cross_source_manifest_v1.json').read_text())
    assert manifest['instagram_total_documents']==7704 and manifest['blog_total_reviews']==25
    assert manifest['pooled_prevalence'] is False and manifest['inferential_tests']==manifest['api_calls']==0
    themes=pd.read_csv(root/'cross_source_theme_comparison_v1.csv')
    assert len(themes)==64 and themes.present_in_blogs.sum()==47
    def pending(*args): raise FileNotFoundError('Pending finalization')
    monkeypatch.setattr(e, 'finalized_summary', pending)
    with pytest.raises(FileNotFoundError):create_cross_source_analysis()
    assert before=={str(p):e.sha(p) for p in root.glob('*') if p.is_file()}


def test_summary_labels_and_privacy(tmp_path,monkeypatch,sample):
    frame,lineage,summary=e.build_taxonomy(*sample)
    e.safe_write(tmp_path/e.TAXONOMY,frame.to_csv(index=False))
    e.safe_write(tmp_path/e.DECISION_FILE,sample[1].to_csv(index=False))
    e.safe_write(tmp_path/'blog_emergent_theme_lineage_v1.csv',lineage.to_csv(index=False))
    manifest=dict(finalized=True,unresolved_decisions=0,taxonomy_sha256=e.sha(tmp_path/e.TAXONOMY),decisions_sha256=e.sha(tmp_path/e.DECISION_FILE),lineage_sha256=e.sha(tmp_path/'blog_emergent_theme_lineage_v1.csv'))
    e.safe_write(tmp_path/e.MANIFEST,json.dumps(manifest))
    monkeypatch.setattr(e,'verify_integrity',lambda root: {})
    result=e.finalized_summary(tmp_path)
    assert result.theme_label.str.strip().ne('').all()
    assert 'author' not in result and 'url' not in result
    assert 'https://' not in e.public_text('Visit https://example.com now')
    import marathon_absa.blog_analysis as blog
    authors=json.loads(blog.RAW.read_text(encoding='utf-8-sig'))
    for record in authors:
        if record['author']:
            assert record['author'].casefold() not in e.public_text(record['author']).casefold()
    e.safe_write(tmp_path/e.DECISION_FILE,'changed')
    with pytest.raises(ValueError,match='Stale'):e.finalized_summary(tmp_path)


def test_review_app_loads_without_personal_fields():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file('blog_emergent_theme_review_app.py').run(timeout=30)
    assert not app.exception
    assert len(app.selectbox)>=3
    app.selectbox[0].select('blog_emergent__facilities__366d8c73').run()
    assert not app.exception

def test_regenerated_cross_source_has_readable_emergent_labels(tmp_path,monkeypatch,sample):
    from marathon_absa.cross_source_analysis import create_cross_source_analysis
    frame,_,_=e.build_taxonomy(*sample)
    frame=frame.rename(columns={'emergent_theme_id':'theme_id','emergent_theme_label':'theme_label'})
    monkeypatch.setattr(e,'finalized_summary',lambda root:frame.copy())
    result=create_cross_source_analysis(tmp_path)
    emergent=pd.read_csv(tmp_path/'blog_emergent_theme_summary_v1.csv')
    assert emergent.theme_label.tolist()==['Toilet queues','Waiting for toilets']
    assert emergent.total_included_reviews.eq(25).all()
    assert result['instagram_total_documents']==7704 and result['blog_total_reviews']==25
    assert result['pooled_prevalence'] is False and result['inferential_tests']==0
    original=pd.read_csv('data/processed/cross_source_analysis_v1/cross_source_theme_comparison_v1.csv')
    regenerated=pd.read_csv(tmp_path/'cross_source_theme_comparison_v1.csv')
    pd.testing.assert_frame_equal(original,regenerated)

def test_complete_finalizer_in_isolated_directory(tmp_path,monkeypatch):
    import shutil
    def forbidden(*args,**kwargs):raise AssertionError('Network forbidden')
    monkeypatch.setattr(socket.socket,'connect',forbidden)
    for name in [e.BASELINE,'blog_emergent_themes_v1.csv','blog_theme_mapping_candidates_v1.csv','blog_review_summary_v1.csv']:
        shutil.copy2(e.ROOT/name,tmp_path/name)
    source=e.load_source(tmp_path)
    concepts=source.groupby('theme_id').emergent_blog_theme_label.first()
    decisions=pd.DataFrame([dict(existing_theme_id=identity,researcher_decision='KEEP_SEPARATE',target_existing_theme_id='',final_label=label,reason='Synthetic test decision only') for identity,label in concepts.items()])
    e.safe_write(tmp_path/e.DECISION_FILE,decisions.to_csv(index=False))
    result=e.finalize(tmp_path)
    assert result['source_emergent_ids']==result['finalized_emergent_themes']==35
    assert result['denominator']==25 and result['singleton_themes']==24
    summary=e.finalized_summary(tmp_path)
    assert len(summary)==35 and summary.theme_label.str.strip().ne('').all()
    lineage=e.read(tmp_path/'blog_emergent_theme_lineage_v1.csv')
    assert len(lineage)==67 and set(lineage.blog_mention_id)==set(source.blog_mention_id)

def test_delegated_review_labels_and_full_lineage():
    decisions=e.read(e.ROOT/e.DECISION_FILE)
    assert decisions.researcher_decision.value_counts().to_dict()=={'KEEP_SEPARATE':28,'RENAME_ONLY':7}
    assert decisions.reason.str.contains('not independent human validation',regex=False).all()
    final=e.finalized_summary()
    assert len(final)==35 and final.singleton_flag.sum()==24
    assert final.total_included_reviews.eq(25).all()
    labels=set(final.theme_label)
    assert 'On-course mist-arch cooling experience' in labels
    assert 'Post-race cooling facilities' not in labels
    assert 'Toilet provision and condition (race venue and accommodation)' in labels
    assert 'Volunteering at a visiting-event booth within the race expo' in labels
    assert 'Reported claim about slow average finishing times' in labels
    lineage=e.read(e.ROOT/'blog_emergent_theme_lineage_v1.csv')
    original=e.load_source()
    assert len(lineage)==67 and set(lineage.theme_id)==set(original.theme_id)
    assert set(lineage.blog_mention_id)==set(original.blog_mention_id)
    assert (lineage.theme_id==lineage.emergent_theme_id).all()
    pairs=e.read(e.ROOT/'blog_emergent_theme_pair_review_v1.csv')
    assert len(pairs)==39 and pairs.decision.eq('KEEP_SEPARATE').all()


def test_published_emergent_mart_matches_reviewed_taxonomy():
    root=Path('data/processed/cross_source_analysis_v1')
    final=e.finalized_summary()
    mart=pd.read_csv(root/'blog_emergent_theme_summary_v1.csv')
    assert set(mart.theme_id)==set(final.theme_id)
    assert mart.theme_label.tolist()==final.theme_label.tolist()
    manifest=json.loads((root/'cross_source_manifest_v1.json').read_text())
    assert any('blog_emergent_theme_taxonomy_v1.csv' in key for key in manifest['source_hashes'])
    for key,value in manifest['source_hashes'].items():assert e.sha(Path(key))==value
    assert not {'author','url'}.intersection(mart.columns)
    for text in mart.representative_evidence:
        assert e.public_text(text)==text
