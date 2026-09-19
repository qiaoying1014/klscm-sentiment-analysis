from __future__ import annotations

ASPECTS = [
    "route_scenery", "weather_climate", "event_organization", "registration_communication",
    "expo_race_kit", "transport_accessibility", "crowd_atmosphere", "aid_stations_refreshments",
    "facilities_amenities", "safety_medical", "cost_value", "personal_race_experience", "other_emerging",
]

LANGUAGE_REVIEW_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "primary_language": {"type": "string"}, "language_iso": {"type": "string"},
        "detected_languages": {"type": "array", "items": {"type": "object", "additionalProperties": False,
            "properties": {"language": {"type": "string"}, "iso_script": {"type": "string"}, "coverage": {"type": "number"}},
            "required": ["language", "iso_script", "coverage"]}},
        "is_mixed_language": {"type": "boolean"}, "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    },
    "required": ["primary_language", "language_iso", "detected_languages", "is_mixed_language", "confidence", "reason"],
}

RELEVANCE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "relevance": {"type": "string", "enum": ["relevant", "ambiguous", "irrelevant"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason_code": {"type": "string", "enum": [
            "event_registration", "event_preparation", "event_logistics", "event_information",
            "event_participation", "event_result_achievement", "event_experience_evaluation",
            "event_spectator_support", "unrelated_event", "generic_running",
            "commercial_promotion", "lifestyle_or_spam", "no_event_connection",
            "insufficient_text", "hashtag_only", "image_dependent", "weak_event_connection",
            "conflicting_evidence",
        ]},
        "evidence": {"type": "string"}, "english_gloss": {"type": "string"},
        "model_observed_language": {"type": "string"}, "code_switching_note": {"type": "string"},
    },
    "required": ["relevance", "confidence", "reason_code", "evidence", "english_gloss", "model_observed_language", "code_switching_note"],
}

RELEVANCE_VERIFICATION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "substantive_relevance": {"type": "string", "enum": ["relevant", "irrelevant"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "klscm_is_actual_subject": {"type": "boolean"},
        "concrete_event_relation": {"type": "boolean"},
        "exclusion_trigger": {"type": "string", "enum": ["none", "another_event", "commercial_promotion", "lifestyle_or_spam", "generic_running", "image_dependent", "insufficient_text"]},
        "reason_code": RELEVANCE_SCHEMA["properties"]["reason_code"],
        "evidence": {"type": "string"},
        "english_gloss": {"type": "string"},
        "model_observed_language": {"type": "string"},
        "code_switching_note": {"type": "string"},
    },
    "required": ["substantive_relevance", "confidence", "klscm_is_actual_subject", "concrete_event_relation", "exclusion_trigger", "reason_code", "evidence", "english_gloss", "model_observed_language", "code_switching_note"],
}

ABSA_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"mentions": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "aspect": {"type": "string"}, "target": {"type": "string"},
            "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative", "mixed"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1}, "evidence": {"type": "string"},
            "english_gloss": {"type": "string"}, "aspect_expression": {"type": "string", "enum": ["explicit", "implicit"]},
            "contributing_hashtags": {"type": "array", "items": {"type": "string"}},
            "contributing_emojis": {"type": "array", "items": {"type": "string"}},
            "emerging_aspect": {"type": "string"}, "model_observed_language": {"type": "string"},
            "code_switching_note": {"type": "string"},
        },
        "required": ["aspect", "target", "sentiment", "confidence", "evidence", "english_gloss", "aspect_expression", "contributing_hashtags", "contributing_emojis", "emerging_aspect", "model_observed_language", "code_switching_note"],
    }}}, "required": ["mentions"],
}


V8_EVENT_CONNECTIONS = ["explicit", "supported", "weak", "none"]
V8_INCLUDED_CONTENT_TYPES = [
    "event_registration", "event_preparation", "event_logistics", "event_participation",
    "event_achievement", "event_emotional_experience", "event_physical_experience",
    "event_social_experience", "event_evaluation", "event_organization",
    "event_route_weather_facilities", "event_safety_medical", "event_cost_value",
    "post_event_reflection", "event_information", "other_meaningful_event_content",
]
V8_EXCLUDED_CONTENT_TYPES = [
    "generic_running", "other_event", "promotional_only", "spam_or_unrelated",
    "no_meaningful_content",
]
V8_CONTENT_TYPES = V8_INCLUDED_CONTENT_TYPES + V8_EXCLUDED_CONTENT_TYPES
RELEVANCE_V8_ASSESSMENT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "event_connection": {"type": "string", "enum": V8_EVENT_CONNECTIONS},
        "primary_content_type": {"type": "string", "enum": V8_CONTENT_TYPES},
        "secondary_content_types": {"type": "array", "items": {"type": "string", "enum": V8_CONTENT_TYPES}},
        "meaningful_content_present": {"type": "boolean"},
        "event_link_evidence": {"type": "string"},
        "analytical_content_evidence": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "contradiction_present": {"type": "boolean"},
        "contradiction_note": {"type": "string"},
        "image_dependent": {"type": "boolean"},
        "requires_review_recommendation": {"type": "boolean"},
        "review_reason": {"type": "string", "enum": ["", "weak_event_connection", "low_confidence", "contradictory_evidence", "image_dependent", "insufficient_text", "event_attribution_uncertain", "content_category_uncertain", "other"]},
        "language_observed": {"type": "string"},
        "code_switching_note": {"type": "string"},
        "short_explanation": {"type": "string"},
    },
    "required": ["event_connection", "primary_content_type", "secondary_content_types", "meaningful_content_present", "event_link_evidence", "analytical_content_evidence", "confidence", "contradiction_present", "contradiction_note", "image_dependent", "requires_review_recommendation", "review_reason", "language_observed", "code_switching_note", "short_explanation"],
}

