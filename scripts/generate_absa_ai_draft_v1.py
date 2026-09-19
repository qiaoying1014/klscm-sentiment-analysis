from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pandas as pd

from marathon_absa.absa_v1 import (
    DRAFT_ANNOTATIONS_PATH, DRAFT_MANIFEST_PATH, DRAFT_PROGRESS_PATH, SAMPLE_PATH,
    annotation_columns, stable_mention_id, validate_draft_package,
)


def mention(aspect: str, target: str, sentiment: str, evidence: str,
            expression: str = "explicit", gloss: str = "", note: str = "") -> dict:
    return {"aspect": aspect, "target": target, "sentiment": sentiment, "evidence_text": evidence,
            "expression_type": expression, "english_gloss": gloss, "analysis_notes": note}


SPECS: dict[int, list[dict]] = {
    1: [mention("race_performance", "Malaysia men's podium finishers", "positive",
                "You’ve all earned your place on the podium, and we’re so proud of you!", note="Explicit pride in achievement.")],
    2: [mention("race_performance", "sub-3 personal best", "positive",
                "From past struggles to a sub-3 PB , this was more than a race, it was redemption.", note="Achievement and redemption."),
        mention("crowd_community_atmosphere", "race-day community and atmosphere", "positive",
                "Seeing familiar faces, hearing the cheers 📣, and sharing the road with so many passionate runners made this day unforgettable."),
        mention("emotional_experience", "overall KLSCM experience", "positive",
                "We all passed something bigger than a test, we proved our spirit.", expression="implicit")],
    3: [mention("race_performance", "full-marathon finishers", "positive", "Tahniah semua 👍",
                gloss="Congratulations everyone")],
    4: [mention("photography_media", "race photography and captured moments", "positive",
                "I shoot so many just don't want to miss your priceless moments.")],
    5: [mention("training_preparation_pacing", "KLSCM running clinic coaching", "positive", "give expert advice")],
    6: [mention("emotional_experience", "memories made while running", "positive",
                "Its all about the memories you make.")],
    7: [mention("weather_conditions", "race heat", "negative", "larian penuh peluh & panas",
                gloss="a run full of sweat and heat"),
        mention("route_course", "KM8 hill", "negative", "bukit yg penuh kejam", gloss="a very cruel hill"),
        mention("crowd_community_atmosphere", "roadside supporters and busker", "positive",
                "Tq Supporter & busker di tepi jalan. Signboard “meow meow meow meow” you made my day.")],
    12: [mention("photography_media", "commemorative race photos", "positive",
                 "Terima kasih utk rakaman gambar sebagai kenangan!!! 🥰🥰",
                 gloss="Thank you for capturing photos as memories")],
    13: [mention("race_performance", "marathon result", "mixed",
                 "成绩单中规中矩，没破 4️⃣，但稳住 4️⃣:3️⃣0️⃣。",
                 gloss="The result was average; I did not break four hours but held 4:30."),
         mention("emotional_experience", "finish at Stadium Merdeka", "positive",
                 "冲线那一刻，真的有种在跑国际赛事的感觉 🇲🇾✨",
                 gloss="Crossing the line genuinely felt like running an international event")],
    16: [mention("crowd_community_atmosphere", "meeting KLSCM pacers", "positive",
                 "glad to meet with other KLSCM Seiko pacers for the first time 🌝")],
    17: [mention("training_preparation_pacing", "10 km pacing strategy", "positive",
                 "my strategy was to take it easy so I could keep running, even on the hilly roads."),
         mention("physical_experience", "condition at the finish", "positive",
                 "finish with a strong smile and a body that felt good."),
         mention("crowd_community_atmosphere", "family supporter experience", "positive",
                 "Our little friend was so excited, cheering for his safe Mami, and even got involved!"),
         mention("emotional_experience", "cheering at KLSCM", "positive",
                 "I always feel super excited when cheering on my sisters and friends in the last 500 m.")],
    21: [mention("race_performance", "first full-marathon completion", "positive",
                 "Bucket list checked. First full marathon.", expression="implicit")],
    22: [mention("crowd_community_atmosphere", "encouragement for the full-marathon runner", "positive",
                 "Pursue your goal, aim high. All the best for your Full Marathon.")],
    23: [mention("race_performance", "anticipated race completion", "positive",
                 "Harap Abam dpt habiskan larian dgn jaya.", expression="implicit",
                 gloss="Hope I can finish the run successfully"),
         mention("emotional_experience", "pre-race feeling", "negative", "Serammmm..",
                 gloss="Scary / nervous")],
    25: [mention("registration_entry", "apparel selection during registration", "negative",
                 "some of you had an oops moment with your apparel selection during the registration process.",
                 expression="implicit")],
    26: [mention("physical_experience", "full-marathon exertion", "negative",
                 "A full marathon is no joke. The final 8-10km perhaps will test your determination and endurance."),
         mention("organization_operations", "finish-gate closing information", "negative",
                 "the marshals kept saying the gate closes at 10.35am, not sure either checkpoint or gate before FL."),
         mention("race_performance", "finishing before the gate closed", "positive",
                 "We reached FL at 7:02 with the gate closing in 13mins. Drama weyy...", expression="implicit")],
    29: [mention("emotional_experience", "pre-race readiness", "mixed",
                 "This time I'm ready to run or ruin?", expression="implicit")],
    30: [mention("training_preparation_pacing", "race preparation", "negative",
                 "mmg x cukup training", gloss="really did not have enough training"),
         mention("physical_experience", "ankle condition", "negative", "ankle pun belum fully recover",
                 gloss="the ankle had not fully recovered"),
         mention("race_performance", "sub-one-hour goal", "positive",
                 "Alhamdulillah finished with target achieved.. 🤲☺️")],
    32: [mention("emotional_experience", "KLSCM memories", "positive", "#bestmoments",
                 expression="implicit", note="Evaluative hashtag materially conveys positive experience.")],
    33: [mention("race_performance", "third full-marathon completion", "positive",
                 "Alhamdulillah my 3rd FM", expression="implicit")],
    34: [mention("photography_media", "race photo", "positive", "Yesss! I got photo too! tqtqtq Feikei Ming bro 😘x3000")],
    35: [mention("crowd_community_atmosphere", "encouragement for KLSCM runners", "positive",
                 "Wishing all runners the best of luck — may the energy, endurance and spirit carry us to the finish line strong! 🙌")],
    36: [mention("race_performance", "half-marathon completion", "positive",
                 "就这样跑完了，才发现自己比想象只中的更好🤍",
                 gloss="After finishing, I realized I was better than I had imagined"),
         mention("crowd_community_atmosphere", "teammate's perseverance", "positive",
                 "我敬佩妳的毅力，这一路不容易，但我们做到了！",
                 gloss="I admire your perseverance; it was not easy, but we did it")],
    37: [mention("physical_experience", "mental readiness and strain", "negative",
                 "Mental koyak. Tak ready.", gloss="Mentally broken; not ready"),
         mention("race_performance", "continuing despite wanting to DNF", "positive",
                 "Ikut kan hati km13 nk DNF. \nLawan tetap lawan", gloss="At km13 I wanted to DNF, but kept fighting", expression="implicit")],
    38: [mention("crowd_community_atmosphere", "good-luck encouragement to runners", "positive",
                 "Moga baik2 semua.. Gud luck peeps😊😊", gloss="Hope everyone is well; good luck")],
    40: [mention("crowd_community_atmosphere", "support for runners", "positive",
                 "demi memberi sokongan dan dorongan yang tak berbelah bahagi.", gloss="to give unwavering support and encouragement"),
         mention("emotional_experience", "annual KLSCM occasion", "positive",
                 "Kiranya acara besar yang dinanti-nanti setiap tahun.Raja segala marathon katanya.",
                 gloss="A major annual event people await; said to be the king of marathons")],
    41: [mention("emotional_experience", "2023 KLSCM completion", "positive", "Done for 2023 ✅", expression="implicit")],
    43: [mention("race_performance", "DNF outcome", "negative", "DNF [Did Not Finish]")],
    45: [mention("race_performance", "full-marathon result", "mixed",
                 "Didnt beat my timing last year but definitely will come back stronger  in Gods will 🙏🏻😄"),
         mention("physical_experience", "strength and endurance", "positive",
                 "thank God for the strength and endurance 🙏🏻")],
    46: [mention("emotional_experience", "desired finish-line cheering experience", "negative",
                 "nak rasa vibes kena cheer beramai2 tu sambil gate nak tutup, tapi tu lah tak kesampaian",
                 gloss="wanted the mass cheering near gate closure, but it did not happen"),
         mention("organization_operations", "overall event operation", "positive", "apa pun semua berjalan lancar",
                 gloss="everything went smoothly"),
         mention("physical_experience", "swollen foot and worsening knee", "negative",
                 "kaki bengkak pula, aku tak tahu lah kenapa, lutut aku makin teruk rasa.",
                 gloss="my foot was swollen and my knee felt increasingly worse")],
    49: [mention("race_performance", "Mwangangi's KLSCM victory", "positive",
                 "Kemenangan ini amat bermakna buat Mwangangi", gloss="This victory was very meaningful for Mwangangi"),
         mention("route_course", "2025 course balance", "positive",
                 "Laluan tahun ini seimbang dengan gabungan kawasan rata dan berbukit",
                 gloss="This year's route was balanced between flat and hilly areas"),
         mention("race_performance", "improved pace and time", "positive",
                 "saya berjaya tingkatkan kelajuan serta catatan masa.", gloss="I succeeded in improving my speed and time")],
    50: [mention("route_course", "KLSCM routes and scenery", "positive",
                 "Each route has been carefully planned to showcase the stunning skyline and iconic landmarks of Kuala Lumpur, giving you a scenic tour of our vibrant city as you race to the finish line."),
         mention("organization_operations", "on-course support", "positive",
                 "we’ve made sure that you’ll be well-supported along the way.")],
    51: [mention("training_preparation_pacing", "final-week marathon guide", "positive",
                 "our guide on 6 essential things to do in the final week before your marathon, which many of you have found incredibly helpful before!"),
         mention("training_preparation_pacing", "preparation tips", "positive",
                 "we've got some fantastic tips to help you get mentally and physically prepared for the big day.")],
    52: [mention("race_performance", "KLSCM half-marathon time", "positive",
                 "Finished another 21KM - this time, better! 🏅"),
         mention("crowd_community_atmosphere", "running with friends", "positive",
                 "thank you friends for making running less lonely, and more fun and inspring!")],
    53: [mention("race_performance", "half-marathon result", "negative",
                 "成绩单中规中矩，没破 4️⃣，但稳住 4️⃣:3️⃣0️⃣。" if False else "跑渣敢敢set目标230\n掐表时间249\nOfficials result sub246",
                 gloss="Targeted 2:30, watch time 2:49, official result under 2:46"),
         mention("training_preparation_pacing", "future improvement", "positive", "回家继续努力呗！",
                 gloss="Go home and keep working hard", expression="implicit")],
    54: [mention("emotional_experience", "full-marathon memory", "positive",
                 "Pengalaman dan memori yang bermakna dalam hidup.", gloss="A meaningful experience and memory in life"),
         mention("physical_experience", "inability to keep running", "negative",
                 "Walau tidak mampu untuk terus berlari, masih lagi boleh berjalan.", gloss="Although unable to keep running, I could still walk"),
         mention("race_performance", "full-marathon completion", "positive", "Full Marathon Finisher (42.2km) ✅", expression="implicit")],
    56: [mention("race_performance", "42.195 km personal best", "positive",
                 "I renewed my 42.195km personal best from 02:55:47 - 02:49:07 after one and a half month 💪🏻"),
         mention("route_course", "final U-turn", "negative", "It's really a killer u-turn point 😂"),
         mention("physical_experience", "leg cramps", "negative", "both legs cramp at the last 500m")],
    58: [mention("route_course", "scenery while running", "positive", "Lari santai sambil layan view",
                 gloss="Relaxed running while enjoying the view")],
    59: [mention("race_performance", "first full-marathon completion", "positive",
                 "Proud to have crossed that finish line"),
         mention("race_performance", "race result", "negative", "the result fell below my expectations."),
         mention("physical_experience", "race-day leg condition", "negative",
                 "my left leg wasn’t performing at its best; it just couldn’t exert full force."),
         mention("crowd_community_atmosphere", "people and race-day vibe", "mixed",
                 "The people, the vibe… everything was just overwhelming.", expression="implicit")],
    60: [mention("race_performance", "second full-marathon time", "positive",
                 "dpt timing yg baik utk kali  ke 2 fm dgn catatan masa 4jam 14minit",
                 gloss="got a good time of 4:14 for my second full marathon"),
         mention("physical_experience", "health during the achievement", "positive",
                 "syukur pada Allah diberikan kesihatan yg baik.", gloss="grateful to be given good health"),
         mention("crowd_community_atmosphere", "runner friends", "positive", "hampa memang padu.👍🏻🇲🇾",
                 gloss="you all were excellent")],
    62: [mention("race_performance", "10 km placing", "positive", "10KM finished with No. 10", expression="implicit")],
    63: [mention("race_performance", "full-marathon completion", "positive", "#Donefullmarathon", expression="implicit")],
    64: [mention("weather_conditions", "hot and humid race weather", "negative", "hot and humid weather"),
         mention("emotional_experience", "half-marathon challenge", "positive", "surely was a fun challenge"),
         mention("crowd_community_atmosphere", "cheer-zone support", "positive",
                 "I was super happy to finally see the members from @kyserunkrew and @webethirsty at the cheer zone. 🥵 \nThanks for the energy boost!")],
    65: [mention("race_performance", "third placing against men", "positive", "phew still 3rd placing! 😅😅😅")],
    67: [mention("race_performance", "first full-marathon completion and time", "positive",
                 "Tahniah untuk diri sendiri sebab dapat habiskan Marathon pertama.", gloss="Congratulations to myself for finishing my first marathon"),
         mention("organization_operations", "release from the last starting pen", "negative",
                 "berlari dari pen yang terakhir sangat sukar untuk melepaskan diri dan pada km ke 10 baru dapat keluar dari ‘pack’.",
                 gloss="Starting from the last pen made it hard to get free of the pack until km10"),
         mention("crowd_community_atmosphere", "running group invitation", "positive",
                 "Terima kasih kepada @the_zenith_runners kerana mengajak saya untuk menyertai untuk perlumbaan ini.",
                 gloss="Thanks to The Zenith Runners for inviting me to join the race")],
    68: [mention("training_preparation_pacing", "three-month training mileage", "negative",
                 "less than what i did last year (432km) when i set my PB of 4:46."),
         mention("race_performance", "sub-five-hour target", "positive",
                 "will strive to do my best inshaaAllah 😁", expression="implicit")],
    69: [mention("race_performance", "sub-five-hour goal and Malaysian PB", "mixed",
                 "Didn’t break the sub-5 jinx in Malaysia (yet!) 😩", note="Negative goal miss balanced by separate positive PB mention."),
         mention("race_performance", "Malaysian personal best", "positive",
                 "Still, a Malaysian PB by 37s haha so I'll take that! Progress is progress 💪"),
         mention("weather_conditions", "race heat and humidity", "negative", "weather too hot and humid"),
         mention("physical_experience", "illness, fatigue, and cramps", "negative",
                 "jetlag, runny nose, phlegm, couldn’t sleep last night, weather too hot and humid, and the usual culprit — cramps")],
    71: [mention("training_preparation_pacing", "pre-race sports massage", "positive",
                 "Getting sport massage 2 days before is really recommended 👍")],
    73: [mention("race_performance", "third half-marathon result", "mixed",
                 "Did not get a PB for my 3rd HM.. boohoo! Nevertheless, glad that it’s finally done & dusted.")],
    74: [mention("photography_media", "KLSCM photograph and photographer", "positive",
                 "Gambar terbaik masa KLSCM 2023.", gloss="Best photo from KLSCM 2023"),
         mention("photography_media", "photographer", "positive", "Terbaik photog. Kredit to House2A Photog. Anda mmg awesome.",
                 gloss="Excellent photographer; you are awesome")],
    75: [mention("emotional_experience", "10 km and 21 km race experience", "positive",
                 "It was crazy but i had fun and enjoy the race.")],
    77: [mention("race_performance", "10 km finish", "mixed", "Not fast, but I finished.")],
    78: [mention("race_performance", "half-marathon personal best", "positive",
                 "Did my Personal Best on KLSCM Half Marathon 21km 😬 from 1:42:## to 1:37:41")],
    79: [mention("physical_experience", "post-race pain", "negative", "Can’t describe the pain behind that smile."),
         mention("emotional_experience", "overall race experience", "positive", "But totally worth it 😁")],
}


def main() -> None:
    if DRAFT_ANNOTATIONS_PATH.exists() or DRAFT_PROGRESS_PATH.exists() or DRAFT_MANIFEST_PATH.exists():
        raise FileExistsError("Versioned AI draft package already exists; refusing overwrite")
    sample = pd.read_csv(SAMPLE_PATH)
    rows: list[dict] = []
    progress: list[dict] = []
    for record in sample.itertuples(index=False):
        document_id = str(record.document_id)
        caption = str(record.original_text)
        specs = SPECS.get(int(record.sample_order), [])
        for index, spec in enumerate(specs):
            evidence = spec["evidence_text"]
            if evidence not in caption:
                raise ValueError(f"Evidence mismatch at sample order {record.sample_order}: {evidence!r}")
            start = caption.find(evidence)
            rows.append({"mention_id": stable_mention_id(document_id, index, spec["aspect"], evidence),
                         "document_id": document_id, "aspect": spec["aspect"], "target": spec["target"],
                         "sentiment": spec["sentiment"], "evidence_text": evidence,
                         "evidence_start": start, "evidence_end": start + len(evidence),
                         "expression_type": spec["expression_type"], "language": record.primary_language,
                         "english_gloss": spec["english_gloss"], "contributing_hashtags": "[]",
                         "contributing_emoji": "[]", "emerging_aspect_name": "",
                         "analysis_notes": spec["analysis_notes"]})
        progress.append({"document_id": document_id, "sample_order": int(record.sample_order),
                         "annotation_status": "ai_draft_complete",
                         "ai_draft_no_evaluative_aspect_mention": not bool(specs),
                         "ai_draft_mention_count": len(specs)})
    # IDs include ordered evidence; compute them only after all draft content is fixed.
    for index, row in enumerate(rows):
        within_doc_index = sum(1 for prior in rows[:index] if prior["document_id"] == row["document_id"])
        row["mention_id"] = stable_mention_id(row["document_id"], within_doc_index, row["aspect"], row["evidence_text"])
    DRAFT_ANNOTATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=annotation_columns()).to_csv(DRAFT_ANNOTATIONS_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(progress).to_csv(DRAFT_PROGRESS_PATH, index=False, encoding="utf-8-sig")
    counts = validate_draft_package()
    manifest = {"version": "absa_v1_ai_assisted_draft_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "provisional_ai_draft_pending_researcher_review", "model_role": "Codex provisional annotator",
                "human_gold": False, "sample_sha256": hashlib.sha256(SAMPLE_PATH.read_bytes()).hexdigest(),
                "annotations_sha256": hashlib.sha256(DRAFT_ANNOTATIONS_PATH.read_bytes()).hexdigest(),
                "progress_sha256": hashlib.sha256(DRAFT_PROGRESS_PATH.read_bytes()).hexdigest(), **counts}
    DRAFT_MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
