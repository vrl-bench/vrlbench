"""Optional native SDK context policies for long interactive evaluations."""


def summarizing_context(*, llm, max_size=18, max_tokens=60000, keep_first=2):
    """Bound the live view while retaining the original SDK event artifacts.

    The actor injects its accounted LLM, so summary requests remain on the same
    provider journal. Select this policy explicitly for both comparison arms.
    """
    from openhands.sdk.context.condenser import LLMSummarizingCondenser

    return LLMSummarizingCondenser(
        llm=llm, max_size=max_size, max_tokens=max_tokens, keep_first=keep_first,
        hard_context_reset_max_retries=1,
    )
