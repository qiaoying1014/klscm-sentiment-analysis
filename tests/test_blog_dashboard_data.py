import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from marathon_absa.blog_dashboard_data import load_blog_dashboard_data
from marathon_absa.blog_emergent_themes import verify_integrity


def test_longform_counts_and_public_allowlists():
    data=load_blog_dashboard_data()
    assert data.counts==dict(reviews=25,mentions=301,chunks=56,aspects=20,matched=47,reference_themes=64,emergent=35,singletons=24)
    assert data.years.support_reviews.sum()==data.languages.support_reviews.sum()==25
    for frame in [data.aspects,data.sentiment,data.matched,data.emergent,data.years,data.languages]:
        assert not {'author','url','review_id','chunk_id'}.intersection(frame.columns)
    assert data.sentiment.groupby('aspect').support_reviews.sum().to_dict()==data.aspects.set_index('aspect').blog_support_reviews.to_dict()
    assert data.emergent.scope_note.str.strip().ne('').all()
    assert verify_integrity()['all_passed']


def test_stale_emergent_mart_is_not_displayed(monkeypatch):
    original=pd.read_csv
    def changed(path,*args,**kwargs):
        frame=original(path,*args,**kwargs)
        if Path(path).name=='blog_emergent_theme_summary_v1.csv':
            frame.loc[0,'support_reviews']=99
        return frame
    monkeypatch.setattr(pd,'read_csv',changed)
    with pytest.raises(ValueError,match='stale'):
        load_blog_dashboard_data()


def test_longform_page_displays_finalized_counts_and_review_provenance():
    app=AppTest.from_file('absa_dashboard.py',default_timeout=45).run()
    app.sidebar.radio[0].set_value('Cross-Source Analysis').run(timeout=45)
    assert not app.exception
    metrics={item.label:item.value for item in app.metric}
    assert metrics['Long-form parent reviews']=='25'
    assert metrics['Blog aspect mentions']=='301'
    assert metrics['Matched Instagram themes']=='47/64'
    assert metrics['Blog-emergent themes']=='35'
    assert any('finalized' in item.value for item in app.success)
    visible=' '.join(item.value for item in [*app.markdown,*app.caption])
    assert 'not independent human validation' in visible
    assert '56 chunks used only for inference' in visible
    assert '17 have none' in visible
    assert '35 blog-emergent themes' in visible
    assert not any('pending' in item.value.lower() for item in app.info)


def test_blog_filters_keep_parent_denominator_and_show_scope():
    app=AppTest.from_file('absa_dashboard.py',default_timeout=45).run()
    app.sidebar.radio[0].set_value('Cross-Source Analysis').run(timeout=45)
    next(x for x in app.selectbox if x.key=='blog_aspect').set_value('facilities').run(timeout=45)
    assert not app.exception
    selector=next(x for x in app.selectbox if x.key=='blog_evidence')
    selector.set_value('blog_emergent__facilities__366d8c73').run(timeout=45)
    visible=' '.join(item.value for item in [*app.markdown,*app.caption])
    assert 'race-venue support is 2/25' in visible and 'accommodation support 1/25' in visible
    tables=[item.value for item in app.dataframe]
    assert any('Blog support' in table and table['Blog support'].str.contains('/25').all() for table in tables)
    next(x for x in app.radio if x.key=='blog_matches').set_value('No blog support').run(timeout=45)
    assert not app.exception


def test_main_pages_remain_explicitly_instagram_only():
    app=AppTest.from_file('absa_dashboard.py',default_timeout=45).run()
    assert any('frozen Instagram analysis' in item.value for item in app.info)
    app.sidebar.radio[0].set_value('Methodology').run(timeout=45)
    assert not app.exception
    assert any('apply to Instagram only' in item.value for item in app.caption)
