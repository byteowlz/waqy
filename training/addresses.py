"""Natural address spellings shared by collection, synthesis, owner weighting and gates.

A word spec's address_variants are pooled into its existing addressed head; they do not
create additional classifiers. E.g. Yo Hex and Hey Hex both train hey_hex, never bare-only.
"""
import re

ADDRESS_WORDS = {"hey", "he", "hei", "okay", "ok", "oye", "eh", "yo"}
ADDRESS = re.compile(r"^(hey|he|hei|okay|ok|oye|eh|yo)[\s,]+", re.I)
PREFIXES = tuple(word + " " for word in sorted(ADDRESS_WORDS))
FILLER = re.compile(r"^(?:also,\s*äh,|so,\s*uhm,|vale,)\s*", re.I)


def is_addressed(text: str) -> bool:
    return bool(ADDRESS.match(FILLER.sub("", text.strip())))
