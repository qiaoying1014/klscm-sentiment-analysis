"""Explicit delegated AI review of the frozen 35 blog-emergent concepts.

Run from repository root with: python -m scripts.complete_blog_emergent_review_v1
Decisions were assessed from all 67 evidence spans and local source context;
this is not independent human coding or inter-rater validation.
"""
import json
from datetime import datetime, timezone
from itertools import combinations

import pandas as pd
from marathon_absa import blog_emergent_themes as e

# Original label -> (optional corrected label, evidence-based scope/decision reason).
REVIEW = {
'On-course cooling provisions': ('', 'Misting and cold water along the half-marathon route address heat relief. Preserve aid_stations_hydration; the related mist-arch observation has frozen facilities ownership and cannot be merged across aspects.'),
'Religious practices within the race community': ('', 'The reviewer appreciates Muslim participants praying beside the route. This is community practice, not evidence that prayer facilities were supplied; separate from crowd size, competitive strength and etiquette.'),
'Strength of the participating field': ('Perceived strength of female half-marathon runners', 'The sole span specifically describes strong female HM runners and being overtaken. Narrow the label to the observed subgroup; do not infer field-wide performance statistics.'),
'Crowd density and personal space': ('', 'Five observations in four parents concern crowd concentration or space at stations/start pens and avoiding crowds. Distinct from demand/sell-out popularity and from running-road width, which has different frozen aspect ownership.'),
'Event popularity and participation demand': ('', 'Both parents describe registration selling out. Demand is not the same concept as physical crowd density, even when one can contribute to the other.'),
'Runner etiquette and litter disposal': ('', 'The criticism concerns runners not using available bins. It evaluates participant behaviour rather than bin provision, religion, popularity or interpersonal encouragement.'),
'Pre-race accommodation and location experience': ('', 'The hotel-neighbourhood context supports a pre-race stay/location appraisal. Do not merge with local food enjoyment: accommodation setting and eating are distinguishable experiences.'),
'Charity souvenirs and gifts': ('', 'The plush toy was bought with registration for the event charity hospice and valued as a gift. It is an optional charity purchase, not a finisher entitlement or a generic food/destination experience.'),
'Local food and destination experience': ('', 'Both spans praise local food in the race-trip setting. Retain as complementary destination experience, not organizer-supplied refreshments or accommodation quality.'),
'Pre-race excitement and hopeful anticipation': ('', 'Three parents express anticipation or hope before participating. Preserve distinction from disappointment after non-completion and emotions around injury setbacks.'),
'Emotional response to non-completion': ('', 'Both spans explicitly concern DNF runners and combine sadness with resilience. This is observed response to non-completion, not necessarily the authors own DNF, and is distinct from pre-race injury anxiety.'),
'Emotional response to injury and setbacks': ('', 'Three explicit spans concern SCKLM injury/setback distress; the concluding hope statement is weaker contextual continuation within that same parent. Preserve all frozen mention lineage, but the theme has only one parent and no additional independent support from that conclusion.'),
'Course signage and distance-marker information': ('', 'The two parents discuss navigational usefulness or marker/GPS agreement. These are information cues; related distance-length observations retain their different frozen route_course aspect. GPS disagreement is a perception, not a measured certification error.'),
'Toilet provision and queues': ('Toilet provision and condition (race venue and accommodation)', 'Context identifies two parents discussing race-venue portable toilets/queues and one discussing hotel toilets. Correct the label so 3/25 is not misrepresented as race-toilet provision: race-venue support is 2/25 and accommodation support 1/25. Retain inherited facilities ownership and all evidence; this is a mixed-setting toilet observation, not three confirmations of one venue service.'),
'Baggage-facility location and access': ('', 'The sole span concerns distance from finish to bag drop. Distinct from toilets, cooling, connectivity, refreshments and the general event-ground condition.'),
'Post-race cooling facilities': ('On-course mist-arch cooling experience', 'Source context says about 5 km before the ending. Correct the erroneous post-race timing. Preserve facilities aspect even though an aid_stations_hydration cooling concept is related; cross-aspect consolidation is prohibited.'),
'Mobile connectivity at the race venue': ('', 'Poor 5G prevented contacting friends and using app tracking at Dataran Merdeka. It is communication infrastructure, not physical space or refreshment/cooling provision.'),
'Race-village space and ground conditions': ('', 'Retain the existing umbrella for usability of the event ground: one parent describes mud and two describe space availability. These dimensions must remain explicit; 3/25 is not support for mud alone or crowding alone. No additional facility concepts are merged into this umbrella.'),
'Post-race refreshments': ('', 'The source criticizes post-run refreshments in general. It is distinct from toilets, baggage, cooling and connectivity. Finisher food quality belongs to a different frozen aspect and cannot be merged.'),
'Finisher food quality': ('', 'The over-ripe banana is explicitly in the finisher goodie bag. Preserve finisher_items ownership and singleton status; it does not establish general refreshment quality across the sample.'),
'Race cutoff rules and time allowances': ('Perceived race cutoff allowance', 'The pre-race article appraises the stated 7h15 allowance relative to ultra/trail events. Retain as a perceived allowance in planning, not verification of official rules or an experienced cutoff incident.'),
'Digestive discomfort and stomach cramps': ('', 'Four mentions in one parent connect pre-race fullness and later stomach cramps. Retain the single digestive episode family; the suggested food cause remains the authors conjecture. Do not merge with sweating.'),
'Heavy sweating during the race': ('', 'Japanese evidence and English gloss describe profuse sweating in heat. This is distinct from digestive discomfort and remains one sampled review.'),
'Expo vendors and activities': ('', 'The reviewer praises vendor variety and activities. This is expo content, not efficiency of collecting the bib/race pack.'),
'Race-pack collection efficiency': ('', 'Two parents describe quick, orderly bib/pack collection. Clear same-concept support already shares one ID; keep distinct from vendors/activities.'),
'Aggregate field finishing times': ('Reported claim about slow average finishing times', 'The parent repeats a third-party ranking claim. Retain as a reported perception, not measured finishing-time data or a verified population comparison. No external factual verification or inferential claim is made.'),
'Race-distance entry choice': ('', 'The runner expresses relief at choosing the half instead of full marathon while experiencing cramps. This evaluates entry-distance choice; preserve frozen registration_entry ownership.'),
'Course distance and marker accuracy': ('Perceived course length and distance-marker accuracy', 'One parent compares markers and recorded finish distance with a Garmin watch. Clarify perception; these observations do not establish an objectively long course or inaccurate certification. Distinct from width, visibility and general course appraisal.'),
'Finish-line visibility': ('', 'Seeing the line and clock from afar helped the runner push toward the finish. This is visual finish approach, not route length, width or overall difficulty.'),
'Course width and running congestion': ('', 'Both parents describe wide/open roads allowing free running. Keep separate from broad course approval/difficulty; shared route vocabulary is not sufficient for a merge.'),
'Overall course quality and difficulty': ('', 'Retain the inherited umbrella without merging narrower route concepts: three parents provide general approval/acceptability and two describe challenge/difficulty. The combined 5/25 must not be stated as five difficulty reports or five positive evaluations.'),
'Training load and quality balance': ('', 'The context explicitly questions doing too many full marathons without sufficient quality and considers focusing on the half. It is preparation/race-load balance, not the separate frozen entry-choice observation.'),
'Transport accessibility and public-transit convenience': ('', 'Four parents describe reaching KL/the event and convenient transit, including a short free-LRT wait. The vague arrangements span is grounded by deciding when to reach the train station in context and adds no extra parent beyond the explicit transit span.'),
'Contributing as an expo volunteer': ('Volunteering at a visiting-event booth within the race expo', 'Full context places the GCM promotional booth inside the SCKLM expo and describes three days of supporting it. Retain the rewarding/demanding contributor experience; it is not an evaluation of the GCM race or of KLSCM on-course volunteer service.'),
'Weather-related muddy ground': ('', 'The runner avoided the muddy race carnival to protect shoes. Preserve weather_conditions ownership; the related facilities-ground observation cannot be merged across frozen aspects.'),
}


def main():
    integrity = e.verify_integrity()
    source = e.load_source()
    if set(source.emergent_blog_theme_label) != set(REVIEW) or source.theme_id.nunique() != 35:
        raise ValueError('The explicit review does not cover the current source exactly')
    path = e.ROOT / e.DECISION_FILE
    previous = e.read(path)
    previous_hash = e.sha(path)
    prefix = 'AI-assisted review by Codex, explicitly delegated by user on 2026-09-07; not independent human validation. '
    records = []
    for identity, group in source.groupby('theme_id', sort=True):
        original = group.emergent_blog_theme_label.iloc[0]
        corrected, reason = REVIEW[original]
        records.append(dict(existing_theme_id=identity, researcher_decision='RENAME_ONLY' if corrected else 'KEEP_SEPARATE',
            target_existing_theme_id='', final_label=corrected or original, reason=prefix+reason))
    decisions = pd.DataFrame(records)
    existing = previous.set_index('existing_theme_id')
    for row in decisions.itertuples():
        old = existing.loc[row.existing_theme_id]
        if old.researcher_decision != 'UNCLEAR' and any(str(old[col]) != str(getattr(row,col)) for col in ['researcher_decision','target_existing_theme_id','final_label','reason']):
            raise ValueError('Previously resolved decision differs; refusing overwrite')
    parents = e.read(e.ROOT/'blog_review_summary_v1.csv').review_id.tolist()
    e.build_taxonomy(source, decisions, parents)
    for row in decisions.itertuples():
        e.save_decision(e.ROOT,row.existing_theme_id,row.researcher_decision,'',row.final_label,row.reason,e.sha(path))
    evidence = source[['theme_id','aspect','review_id','chunk_id','blog_mention_id','evidence_text','english_gloss']].copy()
    evidence = evidence.merge(decisions,left_on='theme_id',right_on='existing_theme_id',validate='many_to_one')
    e.safe_write(e.ROOT/'blog_emergent_theme_ai_review_evidence_v1.csv',evidence.to_csv(index=False))
    pairs=[]
    concepts=source.groupby('theme_id').first().reset_index()
    for aspect,group in concepts.groupby('aspect'):
        for left,right in combinations(group.itertuples(),2):
            pairs.append(dict(aspect=aspect,source_id=left.theme_id,candidate_id=right.theme_id,
                decision='KEEP_SEPARATE',reason=prefix+'Left scope: '+REVIEW[left.emergent_blog_theme_label][1]+' Right scope: '+REVIEW[right.emergent_blog_theme_label][1]))
    e.safe_write(e.ROOT/'blog_emergent_theme_pair_review_v1.csv',pd.DataFrame(pairs).to_csv(index=False))
    result=dict(stage='delegated_ai_blog_emergent_review_v1',completed_at_utc=datetime.now(timezone.utc).isoformat(),
        reviewer_provenance='Codex AI-assisted review explicitly delegated by user; not independent human coding or inter-rater validation',
        source_ids=35,source_mentions=67,reviewed_same_aspect_pairs=len(pairs),renamed=int(decisions.researcher_decision.eq('RENAME_ONLY').sum()),
        unchanged_labels=int(decisions.researcher_decision.eq('KEEP_SEPARATE').sum()),merges=0,exclusions=0,unresolved=0,
        original_decisions_sha256=previous_hash,decisions_sha256=e.sha(path),frozen_hash_verification=integrity,api_calls=0)
    e.safe_write(e.ROOT/'blog_emergent_theme_ai_review_manifest_v1.json',json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
