# KLSCM relevance v8 annotation guideline

Policy version: `v8_event_experience_binary`

## Question and labels

> Should this caption be included in KLSCM topic discovery based on the available caption text and permitted metadata?

Use only `include` or `exclude`. A temporary difficult case is a review status, not a final class.

- `include`: reasonable KLSCM connection plus meaningful experiential, emotional, physical, logistical, social, evaluative, organizational, informational, or journey-related content.
- `exclude`: no defensible KLSCM connection, or only generic running/emotion, another event, promotion, spam, or meaningless content.

## Decision hierarchy

1. Is there a reasonable textual KLSCM connection?
2. Is meaningful analytical content present?
3. Is the record merely generic running, another event, promotion, spam, or meaningless text?
4. Would the decision require an unseen image or video?
5. Decide `include` or `exclude` from available evidence.

A KLSCM hashtag can support event context when meaningful text accompanies it; it is not meaningful content by itself. Do not infer image content. Do not reject a caption solely because it is short. Include KLSCM-linked feelings and physical states, but exclude generic feelings or workout states without the event link. Exclude promotion-only content; retain substantive participant information for topics. Exclude another marathon unless KLSCM is meaningfully compared or discussed. Assess Malay, English, slang, emoji, and code switching directly.

Optional fields are `v8_human_event_connection`, `v8_human_primary_content_type`, `v8_difficult_case`, and `v8_review_notes`. Do not change a label merely because a model disagrees.

## Adjudication and second review

Preserve the first label. A second reviewer independently labels 50-100 stratified difficult cases without model output, covering short feelings, physical states, hashtag boundaries, promotions, comparisons, image dependence, Malay/code switching, and model-routing disagreements. Report raw agreement and Cohen's kappa. Adjudicate disagreements from the written policy and append the final label, rationale, reviewer, and timestamp without overwriting either original judgment.
