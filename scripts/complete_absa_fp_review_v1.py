"""Complete the frozen v1 FP diagnostic researcher fields without touching source fields."""
from pathlib import Path
import hashlib, json
import pandas as pd

ROOT = Path("data/processed/absa_v1/development/absa_v1_false_positive_diagnostic_v1")

FP_CATS = [
"aspect_inference_too_far","context_as_evaluative","ontology_boundary_confusion","grounding_related",
"factual_as_evaluative","over_decomposition","duplicate_redundant_mention","sentiment_spillover",
"context_as_evaluative","context_as_evaluative","ontology_boundary_confusion","aspect_inference_too_far",
"sentiment_spillover","ontology_boundary_confusion","factual_as_evaluative","aspect_inference_too_far",
"aspect_inference_too_far","over_decomposition","ontology_boundary_confusion","factual_as_evaluative",
"sentiment_spillover","aspect_inference_too_far","context_as_evaluative","ontology_boundary_confusion",
"aspect_inference_too_far","ontology_boundary_confusion","duplicate_redundant_mention","ontology_boundary_confusion",
"duplicate_redundant_mention","ontology_boundary_confusion","aspect_inference_too_far","ontology_boundary_confusion",
"ontology_boundary_confusion","factual_as_evaluative","ontology_boundary_confusion","sentiment_spillover",
"context_as_evaluative","duplicate_redundant_mention","context_as_evaluative","ontology_boundary_confusion",
"over_decomposition","aspect_inference_too_far","duplicate_redundant_mention","duplicate_redundant_mention",
"ontology_boundary_confusion"]

FP_NOTES = [
"The photo checkpoint statement narrates finally obtaining pictures but does not evaluate the photography service or image quality. Treating that positive experience as a photography_media evaluation requires an unsupported aspect attribution.",
"The Chinese New Year greeting supplies positive affective context, while KLSCM appears only as a hashtag. The greeting is not an evaluation of the runner's KLSCM emotional experience.",
"The unmet mission is a genuine negative affective experience, but the frozen ontology supports emotional_experience rather than a distinct race_performance judgment. The malformed composite evidence is secondary to this aspect-boundary error.",
"The evidence splices non-contiguous hashtags into a training claim and does not ground a coherent evaluative statement. This malformed grounding materially creates the unsupported mention.",
"The thanks concern support and prayers around a media-recognition post, with KLSCM only contextual. They do not evaluate the race crowd or atmosphere.",
"The palliative-care description explains the charity beneficiary and is not an evaluation of KLSCM medical or safety provision. Background cause context was converted into an event-aspect judgment.",
"The injury-free-finish hashtag states an outcome but does not clearly evaluate bodily experience. Positive physical_experience is inferred beyond the frozen gold's supported evaluations.",
"Calling KLSCM an annual major event is descriptive event framing, not praise of its organization or operations. The model converts contextual stature into an operational evaluation.",
"Not aiming for a personal best is a factual statement of race strategy, not a neutral evaluation of performance. Mere performance relevance is insufficient for an ABSA mention.",
"The memorial and cancer narrative carries positive affect outside the KLSCM experience, which is linked only by a hashtag. This personal background should not become an event emotional evaluation.",
"The encouragement is genuine positive social support, which the frozen ontology assigns to crowd_community_atmosphere. It is not a separate evaluation of the author's emotional experience.",
"Come back stronger is prospective encouragement rather than an evaluation of the completed race performance. The model inferred a distinct positive performance judgment too far.",
"The phrase 'bukan senang' describes the rarity/difficulty of getting the shot within an otherwise strongly praising photography passage. Negative polarity was incorrectly attached to the photography aspect.",
"'Worth it' evaluates the overall painful race experience, not monetary value or cost. The model crossed the frozen boundary from emotional experience to value_cost.",
"Hot and humid conditions are reported descriptively in a news-style account of preparation. The text does not express the author's negative weather evaluation.",
"'Finish strong' is encouragement directed to runners, not evidence that a performance was achieved or positively evaluated. The model inferred an outcome from a wish.",
"'All set' communicates readiness, but it does not clearly evaluate preparation quality or satisfaction. A positive training_preparation_pacing mention goes beyond the frozen reference.",
"The achievement statement already supports positive race_performance; extracting Alhamdulillah as a second emotional aspect decomposes one achievement reaction without distinct aspect support.",
"'Tahniah' praises the finisher achievement, which the frozen ontology assigns to race_performance. It is not a separate crowd/community atmosphere evaluation.",
"The singlet-speed line is reported hearsay and playful speculation, not an evaluation of actual race performance. The model treats performance-related wording as evaluative evidence.",
"General gratitude inherits the caption's positive tone but is not specifically attributable to emotional_experience under the frozen reference. The supported positive evaluation concerns the photography.",
"The emoji-only affect does not identify a KLSCM aspect being evaluated. A positive emotional_experience mention requires more aspect-specific support here.",
"The acknowledgement of overwhelming support is part of an official registration/apparel announcement, not an evaluation of race-day crowd atmosphere. Announcement context was converted into an ABSA mention.",
"Being glad to meet pacers is a genuine positive social encounter, which the frozen ontology assigns to crowd_community_atmosphere rather than a separate emotional_experience mention.",
"The happy-runner hashtag supplies generic affect but does not ground an evaluation of a KLSCM aspect. The event-specific emotional interpretation is too inferential.",
"'Not about pace' explicitly removes pacing as the focus of evaluation; the positive experience belongs to emotional_experience. The model assigns negative sentiment to the wrong ontology family.",
"The document already contains the matched positive route evaluation, and this 'balanced' route mention encodes the same evaluation without a distinct additional target. It is redundant at the frozen mention granularity.",
"The surprise at finishing in Stadium Merdeka is an affective reaction to the overall finish experience, not an evaluation of facility amenities or infrastructure. The aspect family crosses the frozen boundary.",
"The document already has a supported race_performance evaluation; 'keep fighting' restates the same perseverance outcome without a distinct second performance judgment. This is a redundant mention.",
"Gratitude for good health evaluates bodily condition, which the frozen ontology assigns to physical_experience. It is not a separate emotional_experience mention.",
"The wish to finish safely and stylishly is prospective encouragement, not an evaluation of an achieved race result. The positive race_performance interpretation goes beyond the expressed experience.",
"Thankfulness for good-luck wishes is genuinely positive social support, but the frozen ontology represents this document under crowd_community_atmosphere rather than a separate emotional_experience mention.",
"'This sentence motivated me' conveys affect tied to perseverance, but the frozen reference assigns the supported evaluations to performance and teammate/community. The extra emotional family crosses that annotation boundary.",
"The prior PB of 4:46 is a factual historical benchmark used to frame tomorrow's target. It is not a separate positive evaluation of the current KLSCM performance.",
"Feeling lazy in the final 2 km describes motivation/affect, not bodily pain or impairment. The negative physical_experience label crosses the aspect boundary.",
"The performance is explicitly better and forward-looking; 'room to improve' does not make the evaluated result mixed. Negative sentiment from a qualifying phrase spills onto an otherwise positive performance judgment.",
"Runny nose, phlegm, and lost sleep describe pre-race background conditions rather than the frozen race-day physical evaluation. Medical context was promoted into an additional event mention.",
"This mixed performance statement duplicates the document's already matched mixed race_performance evaluation. It does not justify an additional mention at the frozen granularity.",
"'Hope you all enjoy the run' is a photographer's wish to runners, not the author's own emotional evaluation of KLSCM. Communicative context was misread as an experienced aspect judgment.",
"'Moga berjaya' is positive encouragement and community support, not an evaluation of an observed race result. The model crosses from crowd/community support into race_performance.",
"'Fun challenge' combines affect with hot-weather difficulty already represented by emotional_experience and weather_conditions. Adding physical_experience decomposes the same experience without distinct bodily evidence.",
"'Endured the grind' refers broadly to training effort and redemption, without a distinct negative bodily-condition judgment. The physical_experience inference extends beyond the supported aspects.",
"The comeback-stronger line repeats the document's supported perseverance and completion evaluation rather than adding a distinct performance mention. It is redundant at the frozen mention level.",
"The personal-best achievement is already represented by the matched positive race_performance mention. This second prediction encodes the same evaluation without a distinct target.",
"The author's honor at guiding and watching is affective context around praise for the winners' performance. The frozen evaluation supports race_performance, not a separate emotional_experience family."]

TP_WHY = [
"'Berjaya tingkatkan' explicitly expresses successful improvement in speed and time, not merely the numerical result.",
"The celebratory hashtags #bestmoments and #runhappy explicitly frame the KLSCM experience as positively felt.",
"'Memang tak cukup training' is a clear negative judgment that preparation was insufficient.",
"The Chinese wording evaluates the result as ordinary and contrasts missing sub-4 with holding 4:30, creating a mixed performance judgment.",
"'Layan view' expresses enjoyment of the scenery while running, making the route reference evaluative.",
"'Penuh peluh & panas' presents heat as an unpleasant recurring race condition rather than a neutral weather fact.",
"The good-luck wish plus smiling emojis is explicit positive encouragement directed to fellow runners.",
"The leg 'wasn't performing at its best' and could not exert force explicitly evaluates impaired physical condition.",
"'Oops moment' marks the apparel-selection outcome during registration as a problem needing correction.",
"'Anda memang awesome' is direct, emphatic praise of the photographer.",
"The runner states a deliberate take-it-easy strategy linked to limited training and hills; this is an expressed pacing choice, not a bare training fact.",
"The colloquial 'raya' comparison and laughing emoji frame completion as a celebratory, enjoyable experience.",
"'Struggling' is an explicit negative judgment of runners' physical difficulty.",
"'Serammmm' directly expresses pre-race fear, with elongation intensifying the negative affect.",
"DNF is not a neutral result here: saving it as a record of personal capability frames non-completion as a negative self-evaluation."]
TP_NOTES = [
"Preserve Malay achievement verbs such as 'berjaya' when attached to improved performance.",
"Hashtags can carry genuine affect when they directly characterize the event experience, even without prose evidence.",
"Colloquial Malay abbreviation ('mmg x cukup') is an explicit insufficiency judgment that a conservative policy must retain.",
"Preserve mixed Chinese comparative performance language rather than reducing it to factual timing.",
"Indonesian/Malay colloquial enjoyment language can implicitly evaluate scenery and route.",
"A single adjective can be sufficient when narrative framing makes the adverse experience clear.",
"Short multilingual encouragement with emojis remains legitimate community evaluation.",
"This is direct functional/physical impairment and should survive stricter precision filtering.",
"Light euphemism can still encode a concrete complaint when a corrective action follows.",
"Retain colloquial emphatic praise tied to a clear photography target.",
"Neutral sentiment is legitimate for an explicitly chosen strategy; conservative extraction should not require positive or negative polarity.",
"Culturally natural race-as-festival language and emoji jointly express positive affect.",
"Preserve difficulty verbs even when the observation includes other runners rather than only the author.",
"A one-word colloquial affective reaction is sufficient when its race context is unambiguous.",
"Outcome abbreviations can be evaluative when surrounding self-reflection supplies the judgment."]

FN_NOTES = [
"The completion hashtag functions as an implicit positive achievement, not merely participation. The model missed a hashtag-only race_performance signal; this is implicit evaluation in sparse text.",
"The best-of-luck wish explicitly encourages KLSCM runners and invokes energy, endurance, and finishing strong. The model missed explicit community evaluation in an announcement-style caption.",
"'Tahniah semua' with a thumbs-up explicitly praises the full-marathon finishers' achievement. The model captured community atmosphere but missed the Malay performance-directed praise.",
"'Mental koyak. Tak ready' explicitly describes severe mental strain and lack of readiness during the race. The model found related emotional/training aspects but missed the physical_experience boundary used by the frozen gold.",
"The writer explicitly thanks the running group for inviting them to participate, a positive social/community evaluation. The Malay gratitude signal was missed amid a long performance narrative.",
"'Sangat sukar untuk melepaskan diri' explicitly complains about difficulty escaping the last starting pen/pack until km10. The model missed the organization_operations boundary in a dense multilingual narrative.",
"'Totally worth it' explicitly evaluates the painful overall race experience positively. The model assigned the phrase to value_cost and missed the frozen emotional_experience boundary.",
"Uncertain and conflicting finish-gate messages from marshals are an explicit negative operational experience. The model missed the organization aspect while extracting surrounding struggle and panic.",
"Reaching the finish with only 13 minutes before gate closure implicitly marks successful performance, intensified by 'Drama weyy'. The model missed the achievement embedded in timing narrative.",
"'Glad to meet' explicitly evaluates meeting the KLSCM pacers positively. The model captured general emotion but missed the crowd_community_atmosphere boundary.",
"'Ready to run or ruin?' playfully combines readiness with apprehension, yielding mixed pre-race affect. The model mapped it only to performance and missed subtle emotional_experience.",
"'Less than what I did last year' explicitly evaluates the three-month mileage as comparatively insufficient. The model focused on PB/target performance and missed the negative preparation comparison.",
"'Nampak gaya boleh sub 6' expresses hopeful confidence about readiness before the marathon. The model interpreted it as performance but missed the frozen positive emotional/readiness aspect.",
"'Pursue your goal, aim high. All the best' is direct positive encouragement to the sole team representative. The model captured emotion but missed its community-directed function.",
"'Bucket list checked' frames first-marathon completion as a valued achievement, reinforced by the pride hashtag. The model missed implicit positive race_performance in very short text.",
"The e-certificate availability is a neutral finisher-item status under the frozen gold. This is an explicit, difficult boundary case showing that neutral ABSA items must not be eliminated wholesale."]

def digest(df, review_cols):
    cols=[c for c in df.columns if c not in review_cols]
    payload=df[cols].astype(str).to_json(orient="split",force_ascii=False,index=False)
    return hashlib.sha256(payload.encode()).hexdigest()

def update(name, review_cols, values):
    path=ROOT/name
    df=pd.read_csv(path,keep_default_na=False)
    before=digest(df,review_cols); ids=df.review_row_id.tolist()
    assert all(len(column_values) == len(df) for column_values in values.values())
    for col, vals in values.items():
        assert len(vals)==len(df); df[col]=vals
    assert ids==df.review_row_id.tolist() and before==digest(df,review_cols)
    df.to_csv(path,index=False,encoding="utf-8")
    reread=pd.read_csv(path,keep_default_na=False)
    assert ids==reread.review_row_id.tolist() and before==digest(reread,review_cols)
    return {"file":str(path),"rows":len(df),"source_sha256":before}

results=[]
results.append(update("absa_v1_fp_manual_review_sample_v1.csv",["fp_error_category","fp_error_subcategory","researcher_notes","reviewed","confidence"],{
"fp_error_category":FP_CATS,"fp_error_subcategory":[""]*45,"researcher_notes":FP_NOTES,"reviewed":[True]*45,
"confidence":["medium" if i in {1,3,7,10,12,17,22,25,31,32,33,35,37,40,41} else "high" for i in range(1,46)]}))
results.append(update("absa_v1_tp_contrast_sample_v1.csv",["why_legitimate_evaluation","researcher_notes","reviewed"],{
"why_legitimate_evaluation":TP_WHY,"researcher_notes":TP_NOTES,"reviewed":[True]*15}))
results.append(update("absa_v1_fn_contrast_sample_v1.csv",["researcher_notes","reviewed"],{
"researcher_notes":FN_NOTES,"reviewed":[True]*16}))
print(json.dumps(results,indent=2,ensure_ascii=False))
