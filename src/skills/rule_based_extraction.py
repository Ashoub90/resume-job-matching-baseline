"""rule_based_extraction.py

Explainable, rule-based skill extraction utilities.

This module exposes a single function `extract_skills` which performs conservative,
case-insensitive string matching against a provided skills dictionary. The
skills dictionary is expected to be the same structure as
`data/skills_dictionary.json` (categories -> skill -> [variants]).

Design notes / limitations:
- Matching is done by normalizing both the input text and each variant: lowercasing
	and replacing non-alphanumeric characters with spaces, then searching for
	whole-token or phrase matches.
- Short variants (length < 3, e.g. "ml", "js") are only matched when they
	appear as individual tokens to reduce false positives.
- This is intentionally conservative and explainable: it does NOT perform fuzzy
	matching, spelling correction, embeddings, or any ML-based inference. Typos
	and unseen abbreviations will only match if present in the provided variants.
- The function does not consult any external resources and does not modify the
	skills dictionary; it treats the provided `skills_dict` as the single source
	of truth.

API:
		extract_skills(text: str, skills_dict: dict) -> set[str]

Example usage (not executed here):
		skills = extract_skills(cv_text, skills_dictionary)

"""

from typing import Dict, Iterable, List, Set
import re


def _normalize(text: str) -> str:
		"""Normalize text for conservative matching.

		- Lowercase
		- Replace any non-alphanumeric character with a single space
		- Collapse multiple spaces and strip

		Returns a string surrounded by spaces to simplify boundary checks.
		"""
		if not text:
				return " "
		# Lowercase
		t = text.lower()
		# Replace non-alphanumeric with spaces
		t = re.sub(r"[^a-z0-9]+", " ", t)
		# Collapse spaces
		t = re.sub(r"\s+", " ", t).strip()
		# Pad with spaces to make substring matching with boundaries easier
		return f" {t} " if t else " "


def _variant_normalized(variant: str) -> str:
		"""Normalize a variant phrase the same way as the text normalization.

		Returns the normalized variant without surrounding padding.
		"""
		if not variant:
				return ""
		v = variant.lower()
		v = re.sub(r"[^a-z0-9]+", " ", v)
		v = re.sub(r"\s+", " ", v).strip()
		return v


def extract_skills(text: str, skills_dict: Dict[str, Dict[str, List[str]]]) -> Set[str]:
		"""Extract normalized skill keys from raw text using a rule-based approach.

		Args:
				text: Raw, unstructured text (CV or job description).
				skills_dict: A skills dictionary structured as
						{ category: { skill_name: [variant1, variant2, ...], ... }, ... }

		Returns:
				A set of skill names (the keys from the skills dictionary, e.g. "Python",
				"SQL") that were conservatively matched in the input text.

		Matching rules (summary):
		- For every skill in every category, each provided variant is normalized and
			searched for in the normalized input text.
		- Phrase variants (containing spaces after normalization) are matched when
			the full phrase appears as a contiguous sequence of tokens.
		- Short variants (normalized length < 3) are only matched if they appear
			as an exact token in the token set to reduce accidental matches.

		Limitations:
		- No fuzzy or approximate matching. Misspellings or unseen abbreviations
			won't match unless included in the variants list.
		- May still produce false positives for very short tokens present in text
			(e.g., "r" or "c") despite conservative handling.

		This function is intentionally simple and explainable so it can be used as
		a deterministic baseline for downstream evaluation and debugging.
		"""

		normalized_text = _normalize(text)
		tokens = set(normalized_text.strip().split())

		matched: Set[str] = set()

		# Iterate categories and their skills
		for _category, skills in (skills_dict or {}).items():
				if not isinstance(skills, dict):
						# Skip malformed category entries
						continue
				for skill_name, variants in skills.items():
						# Defensive checks
						if not skill_name or not isinstance(variants, Iterable):
								continue

						skill_found = False

						for variant in variants:
								if not variant or not isinstance(variant, str):
										continue

								vnorm = _variant_normalized(variant)
								if not vnorm:
										continue

								# Short tokens (e.g., "ml", "js") should match only as whole tokens
								if len(vnorm) < 3:
										if vnorm in tokens:
												skill_found = True
												break
										else:
												continue

								# For longer variants / phrases, require a phrase match bounded by
								# token boundaries. We can check by searching for ' space + vnorm + space '
								# in the normalized text which is also padded with spaces.
								pattern = f" {vnorm} "
								if pattern in normalized_text:
										skill_found = True
										break

						if skill_found:
								matched.add(skill_name)

		return matched

