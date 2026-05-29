# Supervision and Limitations

HaluEval provides response-level labels for most task-specific records. The general split includes hallucination spans, which can be used when the span text is locatable in the generated response.

The model therefore produces risk scores, not proof that a claim is false. Graph structure and semantic drift are useful warning signals, but factual verification still requires source evidence or an entailment system.
