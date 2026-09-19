from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from openai import OpenAI


class CachedOpenAI:
    def __init__(self, cache_dir: Path, model: str):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.client = OpenAI()

    def _key(self, stage: str, prompt_version: str, text: str, extra: str = "") -> str:
        raw = "|".join([stage, prompt_version, self.model, text, extra])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def structured(self, stage: str, prompt_version: str, instructions: str, text: str, schema: dict, extra: str = "") -> tuple[dict, dict]:
        key = self._key(stage, prompt_version, text, extra)
        path = self.cache_dir / stage / f"{key}.json"
        if path.exists():
            stored = json.loads(path.read_text(encoding="utf-8"))
            return stored["result"], {**stored["metadata"], "cached": True}
        path.parent.mkdir(parents=True, exist_ok=True)
        error = None
        for attempt in range(5):
            started = time.perf_counter()
            try:
                response = self.client.responses.create(
                    model=self.model,
                    instructions=instructions,
                    input=text,
                    text={"format": {"type": "json_schema", "name": stage, "strict": True, "schema": schema}},
                )
                result = json.loads(response.output_text)
                usage = getattr(response, "usage", None)
                metadata = {
                    "model": self.model, "prompt_version": prompt_version,
                    "latency_seconds": round(time.perf_counter() - started, 3),
                    "input_tokens": getattr(usage, "input_tokens", None),
                    "output_tokens": getattr(usage, "output_tokens", None), "cached": False,
                }
                path.write_text(json.dumps({"result": result, "metadata": metadata}, ensure_ascii=False, indent=2), encoding="utf-8")
                return result, metadata
            except Exception as exc:  # SDK exposes several transient subclasses
                error = exc
                if attempt < 4:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"OpenAI {stage} failed after retries: {error}") from error

    def embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


RELEVANCE_INSTRUCTIONS = """Classify an Instagram caption's semantic relevance to the Kuala Lumpur Standard
Chartered Marathon (KLSCM). Relevant is not limited to a start-to-finish event journey. It includes
meaningful KLSCM-linked emotional or physical experience (for example excitement, nervousness, energy,
fatigue, pain, relief, happiness, disappointment, achievement, or pride), as well as event-specific
registration, preparation or training, travel and logistics, participant information, participation,
race-day experience, results, spectator support, and evaluation. Interpret the complete original_text, including meaningful
hashtags, emoji, multilingual wording, colloquial expressions, and Malay shorthand.

A meaningful feeling or bodily-state phrase explicitly anchored to KLSCM can be relevant even when no journey
stage is named. One feeling, multiple feelings, an event theme, or both may qualify.

A KLSCM identifier such as #KLSCM, #KLSCM2019, or an unambiguous event-name variant is event-reference
evidence, but it does not make an otherwise unrelated cue relevant. Classify a caption as relevant only when
the available text explicitly attributes a participation, preparation, completion, achievement, race-day,
travel, support, informational, or evaluative cue to KLSCM. Proximity alone is insufficient when another
named event is the main subject, the cue clearly describes another activity, or KLSCM is merely one item in
a hashtag list. Short captions such as "#KLSCM2023 training", "Gear check #KLSCM2019", or an explicit KLSCM
result remain relevant because their cue is explicitly anchored to KLSCM.

Apply a strict substantive-evidence test. A generic running cue, motivational phrase, distance, date,
celebration, photo credit, apparel, or readiness language does not become KLSCM-specific merely because a
KLSCM tag is present. Relevant requires text that identifies a concrete KLSCM action, status, information,
experience, result, or support relationship; otherwise return irrelevant when another subject is clear, or
ambiguous when unavailable visual context is essential.

A caption containing only event hashtags, handles, URLs, or content whose meaning truly requires an unseen
image remains ambiguous; do not invent image content. Generic running with no KLSCM identifier or other
KLSCM-specific context is irrelevant. Another event as the actual subject, product calls-to-action or pure
commercial promotion, lifestyle or engagement spam, and content with no KLSCM connection are irrelevant.
Official or sponsor content is relevant when it provides KLSCM logistics, participant information,
participation, experience, results, support, or evaluation; pure promotion without substantive event content
is irrelevant. Use ambiguous only when available text and metadata cannot support either relevant or
irrelevant, not merely because the caption is brief.

Evidence must be a short exact span from original_text. It may be a hashtag when that hashtag supplies the
event identity, but include the accompanying semantic cue when one exists. Use exactly one allowed reason_code
matching the principal reason."""

RELEVANCE_ADJUDICATION_INSTRUCTIONS = """Independently adjudicate a borderline KLSCM relevance decision.
Inspect original_text, linguistic_text, hashtags, language metadata, and the first-pass decision; correct an
overconfident first pass when needed. Optimize for high recall without treating every KLSCM hashtag as
substantive content.

A KLSCM identifier is event-reference evidence, not automatic proof that every nearby cue concerns KLSCM.
Return relevant only when the available text explicitly anchors registration, preparation or training,
logistics, participation, race-day activity, completion, results, achievement, experience, spectator support,
information, or evaluation to KLSCM. Short or colloquial captions can qualify, but a merely plausible cue is
not enough when another named event or activity is the actual subject, KLSCM appears only in a broad hashtag
list, or the post is a product call-to-action or pure promotion.

Before returning relevant, verify both conditions: (1) KLSCM is the actual event being described, and (2) the
caption states a concrete event-journey action, status, fact, experience, result, or support relationship.
Generic motivation, ordinary exercise, photo context, apparel, dates, distances, or celebration alone fail this
test even when a KLSCM hashtag is present.

Return ambiguous for a truly hashtag-only record, unavailable-image dependence, or genuinely conflicting or
insufficient evidence. Return irrelevant for another event as the main subject, generic running without
substantive KLSCM context, product calls-to-action or pure promotion, lifestyle/spam, or no event connection.
Never infer unavailable image content. Evidence must be an exact original_text span and may include the KLSCM
identifier plus its explicitly attributable semantic cue."""


RELEVANCE_VERIFICATION_INSTRUCTIONS = """Independently verify a proposed KLSCM relevance decision. Return a binary substantive decision and separate evidence-gate fields. Set klscm_is_actual_subject true only when the caption text makes KLSCM, rather than another event or generic activity, the event being described. Set concrete_event_relation true for a text-supported KLSCM emotional or physical experience (including excitement, nervousness, energy, tiredness, pain, relief, happiness, disappointment, achievement, or pride), or a KLSCM registration, preparation, logistics, information, participation, race experience, result, support, or evaluation relationship. A meaningful feeling explicitly anchored to KLSCM does not need to name a journey stage. Select an exclusion whenever another event, promotion, lifestyle/spam, generic running, unavailable-image dependence, or insufficient text controls the decision. A KLSCM hashtag beside generic motivation, dates, distances, apparel, celebration, photo credit, or ordinary exercise is not sufficient. substantive_relevance may be relevant only when both evidence gates are true and exclusion_trigger is none. Evidence must be an exact caption span."""

def absa_instructions(aspects: list[str]) -> str:
    return f"""Perform multilingual aspect-based sentiment analysis of KLSCM content.
Return zero or more opinion-bearing aspect mentions. One text may contain opposing sentiments for different
aspects. Evidence must be copied exactly from the input. Neutral is factual evaluation without polarity;
return no mention for content that expresses no aspect evaluation. Allowed aspects: {', '.join(aspects)}.
Use other_emerging only if none fits and name the emerging aspect. Treat emoji and hashtags as evidence only
when they contribute meaning. Do not translate before analysis; provide a short English gloss afterwards."""
RELEVANCE_V8_FEW_SHOTS = [
    ("Finished KLSCM in 4:32.", "explicit", "event_achievement"),
    ("We did it! So proud! #KLSCM2024", "supported", "event_emotional_experience"),
    ("My legs are destroyed #KLSCM2024", "supported", "event_physical_experience"),
    ("Feeling blessed.", "none", "no_meaningful_content"),
    ("Never give up. Run harder.", "none", "generic_running"),
    ("First long run for KLSCM 2025 complete.", "explicit", "event_preparation"),
    ("Easy 10K before breakfast.", "none", "generic_running"),
    ("PB at Penang Bridge Marathon!", "none", "other_event"),
    ("KLSCM crowd was better than Penang's.", "explicit", "event_evaluation"),
    ("Queue ambil race kit KLSCM hampir dua jam.", "explicit", "event_logistics"),
    ("Flag-off KLSCM sangat kelam-kabut.", "explicit", "event_organization"),
    ("Final 10K panas gila #KLSCM", "supported", "event_route_weather_facilities"),
    ("Loved the KLSCM route through the city.", "explicit", "event_route_weather_facilities"),
    ("Medic helped my ankle at KLSCM.", "explicit", "event_safety_medical"),
    ("RM120 feels expensive for KLSCM.", "explicit", "event_cost_value"),
    ("Penat tapi best volunteer dekat water station #KLSCM", "supported", "event_social_experience"),
    ("Cheering for my sister! #KLSCM2024", "supported", "event_social_experience"),
    ("SUB 4!!! medal #KLSCM2024", "supported", "event_achievement"),
    ("Really disappointed after KLSCM.", "explicit", "event_emotional_experience"),
    ("Register now! #KLSCM2025", "supported", "promotional_only"),
    ("KLSCM race-kit collection closes at 8pm.", "explicit", "event_information"),
    ("#KLSCM2024", "weak", "no_meaningful_content"),
    ("Sunday vibes #KLSCM2024", "weak", "other_meaningful_event_content"),
    ("This was everything #KLSCM", "weak", "event_emotional_experience"),
    ("So proud! #KLSCM", "supported", "event_emotional_experience"),
    ("crying fire medal #KLSCM2024", "weak", "no_meaningful_content"),
    ("Gementar untuk KLSCM esok.", "explicit", "event_emotional_experience"),
    ("The crowd kept me going #KLSCM", "supported", "event_social_experience"),
    ("Kaki dah koyak but worth every KM #KLSCM", "supported", "event_physical_experience"),
    ("Panas gila weh, final 5K seksa #KLSCM", "supported", "event_physical_experience"),
    (".", "none", "no_meaningful_content"),
    ("See you there! #KLSCM2025", "weak", "other_meaningful_event_content"),
    ("Buy two shoes, get 20% off #KLSCM", "supported", "promotional_only"),
    ("KLSCM shuttle was efficient; baggage was a mess.", "explicit", "event_logistics"),
    ("Running makes me happy #KLSCM", "weak", "generic_running"),
    ("Almost cried crossing the finish line #KLSCM", "supported", "event_emotional_experience"),
    ("Quads still sore after KLSCM.", "explicit", "event_physical_experience"),
    ("Boston done. Next target KLSCM.", "explicit", "event_preparation"),
    ("KLSCM support never replied.", "explicit", "event_organization"),
    ("Look at this #KLSCM2024", "weak", "no_meaningful_content"),
]

RELEVANCE_V8_INSTRUCTIONS = """You assess multilingual Instagram caption relevance for KLSCM topic discovery and later ABSA. Favor high recall for meaningful KLSCM experiences while controlling generic, unrelated, promotional, and empty content. Return only assessment fields in the supplied schema; never decide the human label, routing outcome, operational reason code, topic eligibility, or sentiment eligibility.

Event connection: explicit means KLSCM is directly named in the semantic sentence or clause containing the meaningful content. supported means KLSCM context comes from a hashtag or other permitted textual metadata combined with meaningful semantic content. A caption is not explicit merely because it contains a KLSCM hashtag. weak means a possible link relies on an isolated hashtag, vague wording, contradiction, or unseen context. none means no reasonable KLSCM connection or a clearly different subject.

Meaningful event-linked emotion, physical condition, achievement, social experience, action, opinion, evaluation, organization, logistics, safety, route, weather, facilities, cost, information, or reflection is useful. A detailed journey is not required. Generic feelings and generic running are not relevant. A KLSCM hashtag can establish context only when meaningful semantic content accompanies it; it is not semantic content by itself. Do not infer images, participants, actions, or sentiment not present in text. Assess Malay, English, other languages, slang, emoji, and code switching directly without requiring translation. Use weak and recommend review only for genuinely insufficient cases; do not over-route short but well-supported experiences. Evidence must be exact or closely grounded caption text. Contradiction and image dependence must be explicit.

Content types are defined by the schema names. event_information is substantive participant information, not a sales call. promotional_only is a call to buy, register, or engage without meaningful experience or substantive operational information. Comparisons involving KLSCM are event_evaluation with other_event secondary. The primary type is the dominant analytical content.

Targeted examples (caption => event_connection, primary_content_type):
""" + "\n".join(f"{caption} => {connection}, {content}" for caption, connection, content in RELEVANCE_V8_FEW_SHOTS)

RELEVANCE_V8_VERIFICATION_INSTRUCTIONS = RELEVANCE_V8_INSTRUCTIONS + """

Act as an independent stronger-model verification. Do not defer to the first assessment. Reassess the caption under the same schema and policy. Subtype differences within the included family are acceptable; focus especially on disagreements that change included versus excluded family, explicit/supported versus none, meaningful versus no meaningful content, grounded versus image-dependent interpretation, or final routing implications.
"""
