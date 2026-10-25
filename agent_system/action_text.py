"""Shared ALFWorld command boundaries, independent of model tokenization."""
import re


def first_action_span(text):
    """Return the first opening/closing tag's content offsets in original text.

    Match the environment's case-insensitive first-tag policy, including its
    first closing tag (not a later, better-formed block). Missing tags return
    None. Reversed tags yield an empty slice, as in the existing projection.
    Format validity and admissibility remain the environment's responsibility.
    """
    opening = re.search(r"<action>", text, re.IGNORECASE | re.ASCII)
    closing = re.search(r"</action>", text, re.IGNORECASE | re.ASCII)
    if opening is None or closing is None:
        return None
    return opening.end(), closing.start()
