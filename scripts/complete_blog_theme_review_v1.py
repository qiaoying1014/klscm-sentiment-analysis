"""Apply the explicitly inspected, user-delegated AI review of 301 blog mentions.

No similarity threshold or automatic top-candidate acceptance is used. Row numbers
below refer to the original candidate table; evidence and identity are audited.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil

import pandas as pd

ROOT = Path('data/processed/blog_analysis_v1')
PATH = ROOT / 'blog_theme_mapping_candidates_v1.csv'
EDITABLE = ['mapping_decision', 'selected_reviewed_theme_id',
            'emergent_blog_theme_label', 'researcher_note', 'review_status']


def main():
    original = pd.read_csv(PATH, keep_default_na=False)
    taxonomy = pd.read_csv('data/processed/absa_v1/aspect_level_themes_review_v1/reviewed_theme_taxonomy.csv', keep_default_na=False)
    assert len(original) == 301 and original.blog_mention_id.is_unique
    assert hashlib.sha256('\n'.join(original.blog_mention_id).encode()).hexdigest() == '522cf85772565b1415a2f92115264b00cf2646f69c621f7bf102b3bbdb71ae7c', 'Candidate identities/order changed; re-review required'
    decisions = {}

    def put(indices, decision, value, reason):
        for i in indices:
            assert i not in decisions, i
            selected = f'{original.loc[i, "aspect"]}__rt{value:02d}' if decision == 'MATCH_EXISTING' else ''
            decisions[i] = [decision, selected, value if decision == 'EMERGENT_BLOG_THEME' else '',
                            'AI-assisted review by Codex, delegated by user; not independent human validation. ' + reason, 'reviewed']

    def m(indices, theme, reason):
        put(indices, 'MATCH_EXISTING', theme, reason)

    def e(indices, label, reason):
        put(indices, 'EMERGENT_BLOG_THEME', label, reason)

    def x(indices, reason):
        put(indices, 'EXCLUDE_FROM_THEME_COMPARISON', '', reason)

    m([14,118,194,210,221,233,265,284,300], 3, 'Evaluates water-station availability, spacing, capacity or usefulness; mixed supply details retained in the original evidence.')
    m([46,149], 2, 'Evaluates adequacy or quality of on-course hydration/fueling supplies.')
    e([195], 'On-course cooling provisions', 'Mist and water cooling are the central evaluation, distinct from drinking/fueling or water-station spacing.')

    m([4,5,6,7,13,21,27,34,35,37,42,62,67,68,96,100,103,108,109,111,134,141,159,166,168,180,182,186,202,215,247,270,285,290], 1, 'Describes shared race atmosphere, camaraderie, celebration or encouragement; the reviewed community theme includes these forms of support. Japanese row 186 describes enjoyable shared warm-up dancing.')
    e([15,140,188,189,253], 'Crowd density and personal space', 'Concerns crowding, pen density or avoiding crowds rather than interpersonal support; frozen aspect is retained.')
    e([54,75], 'Event popularity and participation demand', 'Evaluates demand/popularity rather than directly describing community encouragement.')
    e([224], 'Strength of the participating field', 'Observes the strength of female participants rather than support or camaraderie; retains original aspect.')
    e([264], 'Religious practices within the race community', 'Appreciates roadside prayer by fellow participants; distinct community practice outside the existing support theme.')
    e([266], 'Runner etiquette and litter disposal', 'Criticizes runners discarding rubbish; evaluates community conduct, not facility provision.')

    e([1], 'Pre-race accommodation and location experience', 'Values the pre-race stay/location as a way to experience Kuala Lumpur; no same-aspect existing theme covers lodging.')
    e([56,153], 'Local food and destination experience', 'Evaluates Kuala Lumpur food as part of the destination experience, not aid-station supplies.')
    m([88], 4, 'Shoe laces coming undone directly evaluate running footwear usability.')
    e([214], 'Charity souvenirs and gifts', 'Evaluates the purchased charity plush as a cute gift, not apparel or the act of fundraising.')
    m([236,237], 5, 'Evaluates KLSCM overall, including popularity or a negative reported ranking; does not independently verify the claim.')
    m([246,249], 1, 'Context explicitly concerns KLSCM Run For A Reason fundraising and its anticipated charitable impact.')
    x([241,242,243,244,245], 'Source context identifies Hamburg as the evaluated event (including the continuation about toilets and temperature); exclude comparator-event evidence from KLSCM theme comparison.')

    m([0,11,18,19,41,52,87,102,129], 6, 'Expresses enjoyment, happiness, meditative rhythm or personal value in running.')
    m([3,250,273], 3, 'Expresses race-related nervousness or fear of being unable to finish; reviewed theme includes anxiety during participation.')
    m([8,63,105], 7, 'Frames participation or running-community reflection as personally meaningful or moving; row 105 is a memorial reflection in its source context.')
    m([28,104,157,276], 2, 'Expresses joy or relief at completing or approaching the finish.')
    e([31,39], 'Emotional response to non-completion', 'Source describes sadness/disappointment and resilience around DNF; distinguish from successful finish-line joy and urges to quit.')
    e([107,121,125,133], 'Emotional response to injury and setbacks', 'Context concerns distress, misfortune or hope following injury/setbacks; not a direct assessment of course mental difficulty.')
    m([110], 1, 'Values participation in a landmark KLSCM edition and its lasting significance.')
    e([120,136,251], 'Pre-race excitement and hopeful anticipation', 'Expresses positive anticipation of participation or potential performance, rather than anxiety or a completed marathon memory.')
    m([126,227,267], 5, 'Explicitly concerns giving up, lost motivation or difficulty continuing.')
    m([176], 4, 'Describes the psychological strain of racing in difficult conditions.')

    e([193,268], 'Course signage and distance-marker information', 'Evaluates accuracy or usefulness of course markers/signage; this aspect has no finalized Instagram themes.')
    e([139,187,216], 'Toilet provision and queues', 'Evaluates toilet availability, usability or waiting; no finalized Instagram facilities theme exists.')
    e([174], 'Post-race refreshments', 'Evaluates refreshments after the finish, within the frozen facilities aspect.')
    e([200], 'Mobile connectivity at the race venue', 'Evaluates poor 5G reception at the venue; no same-aspect reference theme exists.')
    e([204,218,252], 'Race-village space and ground conditions', 'Evaluates event-space suitability, muddy ground or warm-up space.')
    e([222], 'Post-race cooling facilities', 'Evaluates the effectiveness of the cooldown mist arch in the finish area.')
    e([230], 'Baggage-facility location and access', 'Evaluates distance to the bag-drop tent; retains facilities ownership rather than moving to organization.')
    m([51,72,165,278], 1, 'Evaluates finisher medal design, receipt or meaning.')
    m([207,229], 2, 'Evaluates finisher apparel or usable finish-area merchandise; row 229 is the provided refreshment bag.')
    e([164], 'Finisher food quality', 'Evaluates an over-ripe banana in the finisher bag, not a medal or apparel item.')

    m([9,64,254,283,297], 1, 'Evaluates flag-off timing, start arrangements or operational queues, including expo waiting lines.')
    m([16,43,48,53,71,76,77,85,98,99,119,169,173,199,205,206,208,260,281,288,292], 2, 'Evaluates organizer delivery, staffing, logistics or event provisions; retains the frozen aspect.')
    e([115], 'Race cutoff rules and time allowances', 'Evaluates the shorter race cutoff; the reference themes do not explicitly cover eligibility to continue under time limits.')
    x([97], 'Evaluation targets the charity foundation broader work for children, not race operational delivery; do not use it as KLSCM organization evidence or change the frozen aspect.')
    m([44,223,274], 1, 'Values race imagery, photo availability or the photographers creating those memories.')
    m([201,203], 2, 'Concerns posing for race photographs or access/queues for a finish-time photo opportunity.')

    m([20,84,128,146,217,256,257,263,269], 1, 'Evaluates bodily condition, ease of exertion, susceptibility to injury or recovery; aligns with the broad reviewed physical-condition theme.')
    m([23,272], 6, 'Muscle cramps are explicitly described as impairing continued running; other physical details remain in the evidence.')
    m([26,106,226,296], 3, 'Describes physical pain, injury, fatigue or difficulty during the race.')
    m([32,177,191], 5, 'Describes exhaustion, physical breakdown or heat-related fatigue that compromises continued exertion.')
    m([40], 2, 'Presents bodily aching as an expected consequence of endurance participation.')
    m([122,124], 8, 'Identifies localized knee or foot injury; more specific than the general pre-race illness theme.')
    e([138,152,154,162], 'Digestive discomfort and stomach cramps', 'Describes meal-related fullness or gastrointestinal cramps, distinct from the leg-muscle cramp reference theme.')
    e([184], 'Heavy sweating during the race', 'Japanese evidence explicitly describes profuse sweating; it does not alone establish acute medical distress.')

    e([61,282], 'Race-pack collection efficiency', 'Evaluates collection speed and organization; no finalized Instagram race-pack/expo themes exist.')
    e([293], 'Expo vendors and activities', 'Evaluates variety of vendors and enjoyable expo activities; distinct from kit collection.')
    m([24,30,33,38], 8, 'Concerns persistence despite difficulty, falling short or DNF, consistent with disappointing outcomes and determination.')
    m([29,69,277], 12, 'Describes completing the marathon or half marathon.')
    m([70,231], 5, 'Directly evaluates achieved finish time against a time threshold or target.')
    m([90,92,151,179,192,198], 7, 'Evaluates maintaining, losing or changing pace during the race.')
    m([91,93,94,197,225,255,280,286], 3, 'Evaluates personal running ability, relative placing, expectations or overall race performance.')
    m([95,279], 1, 'Praises or congratulates other runners for their achieved race times.')
    m([127,158,163,181], 9, 'Evaluates improvement, lack of progress or a new personal-best milestone.')
    m([234], 10, 'Explicitly evaluates the first full-marathon completion.')
    m([235], 2, 'Expresses happiness/pride in the achieved finishing time without an explicit target comparison.')
    e([238], 'Aggregate field finishing times', 'Reports an event-level average finish-time assessment, not personal performance; preserve as a reported claim, not verified fact.')
    m([78,212], 2, 'Evaluates securing an entry or ballot-free registration access.')
    e([156], 'Race-distance entry choice', 'Expresses satisfaction with choosing the half rather than full distance, not access to a slot or bib transfer.')

    m([10,22,73,89,113,147,148,171,190,219,298], 1, 'Explicitly evaluates hills, climbs, flat sections or elevation difficulty.')
    m([45,112,116,145,258,261,262,287,291], 2, 'Evaluates Kuala Lumpur scenery, city streets and visual route variety; secondary elevation details retained.')
    e([50,142,170,232,299], 'Overall course quality and difficulty', 'General route appraisal or technical difficulty without enough specific hill/scenery evidence for either existing theme.')
    e([65,82], 'Course width and running congestion', 'Evaluates available road space and ability to establish an unobstructed rhythm.')
    e([150,160,161], 'Course distance and marker accuracy', 'Evaluates course length or distance-marker discrepancies, not elevation or scenery; frozen route aspect retained.')
    e([228], 'Finish-line visibility', 'Evaluates seeing the finish and timing clock from a distance.')
    x([66], 'Explicitly evaluates Singapore marathon congestion as a comparator, not the KLSCM course.')
    x([101], 'Source is a quoted life-journey/running metaphor, not an identifiable assessment of the KLSCM route.')
    x([132], 'Generic instruction to respect distance lacks a concrete KLSCM course evaluation sufficient for theme comparison.')

    m([25,36,117,185,240,259,289], 1, 'Evaluates medical help, traffic/route safety or safety provision, including lighting; reported deficiencies remain source claims.')
    m([2,123,130,135], 3, 'Evaluates pre-race routines, preparation adequacy, suitable training programs or scheduling a preparatory race.')
    m([74,131], 7, 'Emphasizes disciplined adherence to training and respect for preparation.')
    m([83,86,271,275], 1, 'Evaluates pacing/restraint, drafting or a strategy for managing the remaining race.')
    m([143], 2, 'Source uses recent running improvement as evidence of readiness to attempt an ambitious race goal.')
    e([213], 'Training load and quality balance', 'Evaluates excessive marathon volume and insufficient quality, rather than too little training.')
    e([49,60,80,81,211], 'Transport accessibility and public-transit convenience', 'Evaluates travel access, connections or waiting for transit; no finalized Instagram transport themes exist.')
    m([55,57,58,294], 2, 'Evaluates favorable exchange rates or affordable transport/accommodation as material participation value.')
    m([248], 1, 'Values charitable running as rewarding and worthwhile; context explicitly connects this value to KLSCM participation.')
    m([17,178,196], 3, 'Evaluates encouragement or practical help from race volunteers.')
    e([79], 'Contributing as an expo volunteer', 'Context places the reviewer helping at the GCM booth within the SCKLM expo; evaluates giving support, not receiving volunteer assistance.')
    m([12,47,114,172,175,183,209,239,295], 1, 'Describes race heat/humidity or preparation for it; reported air-quality claims in row 239 are retained as secondary context.')
    m([59,137,144,155,220], 2, 'Evaluates suitable temperatures, cool/cloudy conditions or favorable weather despite residual humidity; row 220 also reports cleared haze.')
    e([167], 'Weather-related muddy ground', 'Evaluates muddy event conditions; preserve frozen weather aspect without recasting it as heat or favorable weather.')

    assert set(decisions) == set(original.index), sorted(set(original.index) - set(decisions))
    revised = original.copy()
    for i, values in decisions.items():
        if original.loc[i, 'review_status'] == 'reviewed':
            continue  # Preserve any existing human/session decisions.
        revised.loc[i, EDITABLE] = values
    lookup = taxonomy.set_index('reviewed_theme_id').aspect.to_dict()
    for row in revised.itertuples():
        assert row.review_status == 'reviewed'
        if row.mapping_decision == 'MATCH_EXISTING':
            assert lookup.get(row.selected_reviewed_theme_id) == row.aspect
        if row.mapping_decision == 'EMERGENT_BLOG_THEME':
            assert row.emergent_blog_theme_label.strip()
    pd.testing.assert_frame_equal(original.drop(columns=EDITABLE), revised.drop(columns=EDITABLE))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    backup = PATH.with_name(f'{PATH.stem}.backup_ai_review_{stamp}.csv')
    shutil.copy2(PATH, backup)
    temporary = PATH.with_suffix('.review.tmp')
    revised.to_csv(temporary, index=False, encoding='utf-8-sig')
    os.replace(temporary, PATH)
    saved = pd.read_csv(PATH, keep_default_na=False)
    pd.testing.assert_frame_equal(saved, revised)
    audit = revised[['blog_mention_id', 'review_id', 'aspect', 'target', 'evidence_text'] + EDITABLE].copy()
    audit.insert(0, 'app_row_number', audit.index + 1)
    audit.to_csv(ROOT / 'blog_theme_ai_review_audit_v1.csv', index=False, encoding='utf-8-sig')
    report = {
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'reviewer': 'Codex AI assistant', 'authorization': 'User explicitly delegated completion of the review',
        'independent_human_validation': False, 'rows': len(saved),
        'decision_counts': saved.mapping_decision.value_counts().to_dict(),
        'existing_themes_used': saved.loc[saved.mapping_decision.eq('MATCH_EXISTING'), 'selected_reviewed_theme_id'].nunique(),
        'emergent_themes': saved.loc[saved.mapping_decision.eq('EMERGENT_BLOG_THEME'), ['aspect', 'emergent_blog_theme_label']].drop_duplicates().to_dict('records'),
        'backup': str(backup), 'mapping_sha256': hashlib.sha256(PATH.read_bytes()).hexdigest(),
        'immutable_columns_unchanged': True, 'pending': int(saved.review_status.ne('reviewed').sum()),
        'method': 'Explicit evidence-by-evidence interpretation against all 64 reviewed labels and scope notes; source chunks checked for ambiguous referents. Similarity rankings not used as acceptance rules.',
        'limitations': ['AI review is not independent human coding or inter-rater reliability evidence.', 'Frozen ABSA aspect/sentiment labels were not repaired.', 'Blog-emergent labels may have single-review support and are not claims of recurring population-level themes.', 'Comparator and insufficiently grounded KLSCM theme evidence excluded only from theme comparison; upstream aspect marts unchanged.'],
        'api_calls': 0,
    }
    (ROOT / 'blog_theme_ai_review_manifest_v1.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
