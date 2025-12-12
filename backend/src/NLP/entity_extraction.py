"""
entity_extraction.py

Features:
- Portuguese-first fuzzy matching (heuristic prioritization).
- Unicode normalization (accent-preserving & accent-stripped forms).
- Per-collection thresholds tuned for better precision.
- Tag domain filtering (only search tags for tag intents).
- Cooking-time extraction and list_by_time support.
- Backwards-compatible function signatures for pipeline.py usage.

Dependencies:
- spacy
- rdflib
- rapidfuzz

Drop-in replacement: keep your pipeline.py as-is except for the small changes listed later.
"""
from __future__ import annotations
import re, unicodedata
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import spacy
from spacy.pipeline import EntityRuler

from rdflib import Graph, Namespace, RDF, RDFS
from rapidfuzz import process, fuzz

from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException

# ADD: Argos Translate (offline, fast)
import argostranslate.translate as _argos_translate

import logging, warnings
logging.getLogger("stanza").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Language pt package default expects mwt*")

EX = Namespace("http://example.org/recipes#")

STOPWORDS = {
    "en": {"with","of","the","a","an","and","or","for","to","in","on","at","by",
           "recipe","recipes"},  # added
    "pt": {"com","de","do","da","das","dos","e","ou","para","em","no","na","nos","nas",
           "receita","receitas"},  # added
}
DOMAIN_NOISE = {"recipe","recipes","receita","receitas","ingredientes","ingredients"}

THRESHOLDS = {
    "recipe_name": 70,   
    "ingredient": 85,    
    "tag": 90,           
}

SAFETY_MAX_RESULTS = {
    "ingredient": 10,
    "tag": 10,
    "recipe_name": 10,
    "time": 10,
}


# If candidate contains explicit Portuguese features, boost preference
PT_PREF_BOOST = 1


# 1. COOKING TIME REGEX
# Lenient patterns (removed strict \b boundaries around units)
TIME_PATTERNS = [
    # 0: Combined H + M (e.g., 1h30, 1h 30m)
    re.compile(r"(?P<h>\d{1,2})\s*h\s*(?P<m>\d{1,2})", re.I),
    # 1: Hours only
    re.compile(r"(?P<val>\d{1,2})\s*(?:h|hora|horas|hour|hours)", re.I),
    # 2: Minutes only - Ensure \b is at start to avoid matching inside numbers, but NOT at end to allow "30min"
    re.compile(r"\b(?P<val>\d{1,3})\s*(?:minutos|min|mins|minute|minutes|munutes|minuts)", re.I),
]

# Pattern to detect "less than" / "under" context
RANGE_PREFIX_REGEX = re.compile(r"\b(menos de|less than|under|até|up to|max|maximum)\b", re.I)

def parse_time_constraints(text: str) -> Tuple[Optional[int], Optional[int]]:
    if not text or not isinstance(text, str):
        return None, None
    
    # 1. Normalize numbers first (PT & EN)
    t = normalize_pt_numbers(text.lower())
    print(f"[TIME DEBUG] Parsing: '{t}'")

    # 1. Fast fail
    if not re.search(r"\d", t):
        print("[TIME DEBUG] No digits found.")
        return None, None

    extracted_val = None

    # 2. Extraction Logic
    # Try Combined H:M first (Index 0)
    m_hm = TIME_PATTERNS[0].search(t)
    if m_hm:
        try:
            extracted_val = int(m_hm.group("h")) * 60 + int(m_hm.group("m"))
            print(f"[TIME DEBUG] Matched H+M: {extracted_val}")
        except ValueError:
            pass
    
    # Try Hours only (Index 1)
    if extracted_val is None:
        m_h = TIME_PATTERNS[1].search(t)
        if m_h:
            try:
                extracted_val = int(m_h.group("val")) * 60
                print(f"[TIME DEBUG] Matched Hours: {extracted_val}")
            except ValueError:
                pass

    # Try Minutes only (Index 2)
    if extracted_val is None:
        m_min = TIME_PATTERNS[2].search(t)
        if m_min:
            try:
                extracted_val = int(m_min.group("val"))
                print(f"[TIME DEBUG] Matched Minutes: {extracted_val}")
            except ValueError:
                pass

    if extracted_val is None:
        print("[TIME DEBUG] No time value extracted.")
        return None, None

    # 3. Determine if it is Exact or Range (Max)
    # Check if a range prefix exists in the text
    range_match = RANGE_PREFIX_REGEX.search(t)
    if range_match:
        print(f"[TIME DEBUG] Range detected ('{range_match.group(0)}'). Max: {extracted_val}")
        return None, extracted_val  # It is a max limit (Range)
    else:
        print(f"[TIME DEBUG] Exact time: {extracted_val}")
        return extracted_val, None  # It is exact

def content_tokens(chunks, lang, translate_to: Optional[str] = None, bag_mode: bool = False, original_text: Optional[str] = None):
    """
    Normalize query chunks into canonical token order.
    Steps:
      1. Translate each chunk (if translate_to is specified)
      2. Remove stopwords
      3. Sort tokens alphabetically for canonical order

    When bag_mode=True:
      - Bag all tokens from: original chunks, per-chunk translations, AND full original_text + its translation.
    """
    stop = STOPWORDS.get(lang, set())
    if not chunks and not original_text:
        return []

    if not bag_mode:
        keep = []
        for chunk in chunks or []:
            text = chunk
            # Step 1: translate first
            if translate_to and translate_to != lang:
                text = translate_between(text, lang, translate_to)

            # Step 2: lowercase & tokenize
            words = re.findall(r"\b\w+\b", text.lower())

            # Step 3: remove stopwords
            words = [w for w in words if w not in stop and w not in DOMAIN_NOISE]

            if words:
                # Step 4: canonical token order
                words = sorted(words)
                keep.append(" ".join(words))
        return keep

    # bag mode
    bag = {}
    def add_tokens(txt: str, lang_for_stop: str):
        if not txt:
            return
        local_stop = STOPWORDS.get(lang_for_stop, set())
        for w in re.findall(r"\b\w+\b", txt.lower()):
            if w in local_stop or w in DOMAIN_NOISE:
                continue
            key = strip_accents(w)
            # prefer PT-accented form
            if key not in bag or (has_portuguese_chars(w) and not has_portuguese_chars(bag[key])):
                bag[key] = w

    # 1) Original chunks (no per-chunk translation)
    for c in chunks or []:
        add_tokens(c, lang)

    # 2) Full original_text + single full-text translation
    if original_text:
        add_tokens(original_text, lang)
        if translate_to and translate_to != lang:
            try:
                full_t = translate_between(original_text, lang, translate_to)
                # use stopwords of target language on translated text
                add_tokens(full_t, translate_to)
            except Exception:
                pass

    # Canonical order by accent-stripped key
    return [bag[k] for k in sorted(bag.keys())]

def detect_language(text: str) -> str:
    """
    Detects the language of the input text using langdetect.
    Returns ISO-639-1 codes (e.g., 'pt', 'en', 'es'), or 'unknown'.
    """
    if not text or not isinstance(text, str):
        return "unknown"

    try:
        langs = detect_langs(text)
        best = langs[0]

        # optional threshold: filter out weak predictions
        if best.prob < 0.60:
            return "unknown"

        return best.lang
    except LangDetectException:
        return "unknown"

def cooking_time_to_minutes(text: str) -> Optional[int]:
    if not text or not isinstance(text, str):
        return None
    t = text.lower()
    
    # 1. Try Combined H + M first
    for pat in TIME_PATTERNS[3:]:
        m = pat.search(t)
        if m:
            d = m.groupdict()
            if "h" in d and "m" in d:
                try:
                    return int(d["h"]) * 60 + int(d["m"])
                except ValueError:
                    continue

    # 2. Try Minutes (Pattern 0)
    m = TIME_PATTERNS[0].search(t)
    if m:
        try:
            return int(m.group("val"))
        except ValueError:
            pass

    # 3. Try Hours (Patterns 1 and 2)
    for pat in TIME_PATTERNS[1:3]:
        m = pat.search(t)
        if m:
            try:
                return int(m.group("val")) * 60
            except ValueError:
                continue

    return None



# 2. SPACY PIPELINE
def build_spacy_pipeline(lang_priority: str = "pt"):
    """
    Load a PT or EN spaCy model. Fall back to blank "xx".
    Adds EntityRuler for simple TIME cues.
    """
    models = ["pt_core_news_sm", "en_core_web_sm"] if lang_priority == "pt" \
             else ["en_core_web_sm", "pt_core_news_sm"]

    nlp = None
    for m in models:
        try:
            nlp = spacy.load(m)
            break
        except Exception:
            pass

    if nlp is None:
        nlp = spacy.blank("xx")

    # EntityRuler for TIME keywords
    if "entity_ruler" not in nlp.pipe_names:
        try:
            ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})
        except Exception:
            ruler = EntityRuler(nlp)
            nlp.add_pipe(ruler)
    else:
        ruler = nlp.get_pipe("entity_ruler")

    ruler.add_patterns([
        {"label": "TIME", "pattern": [{"LOWER": {"IN": ["min", "mins", "minutos", "minuto"]}}]},
        {"label": "TIME", "pattern": [{"LOWER": {"IN": ["h", "hora", "horas", "hour", "hours"]}}]},
    ])
    return nlp



# 3. KG INDEX + FUZZY SEARCH (with normalization)
def strip_accents(s: str) -> str:
    """Return accent-stripped lowercased form"""
    if not isinstance(s, str):
        return s
    nk = unicodedata.normalize("NFKD", s)
    return "".join([c for c in nk if not unicodedata.combining(c)]).lower()

def has_portuguese_chars(s: str) -> bool:
    """Heuristic: accents or 'ç' or typical Portuguese words"""
    if not isinstance(s, str):
        return False
    if re.search(r"[áàâãéêíóôõúçÁÀÂÃÉÊÍÓÔÕÚ]", s):
        return True
    # also check common PT words
    if re.search(r"\b(para|com|sem|receita|rápido|fácil|fáceis|cozinhar|açorda|bacalhau)\b", s, re.I):
        return True
    return False

@dataclass
class KGIndex:
    recipes: List[str]
    ingredients: List[str]
    tags: List[str]
    recipe_label_to_subject: Dict[str, Any]
    graph: Graph
    # precomputed normalized maps
    recipes_norm: List[str]
    ingredients_norm: List[str]
    tags_norm: List[str]

    @classmethod
    def from_ttl(cls, ttl_path: str):
        g = Graph()
        g.parse(ttl_path, format="turtle")

        def labels_of(rdf_type) -> List[str]:
            vals = []
            for s in g.subjects(RDF.type, rdf_type):
                lab = g.value(s, RDFS.label)
                if lab is not None:
                    vals.append(str(lab))
            return vals

        recipes = []
        recipe_label_to_subject = {}
        for s in g.subjects(RDF.type, EX.Recipe):
            lab = g.value(s, RDFS.label)
            if lab is not None:
                label_str = str(lab)
                recipes.append(label_str)
                recipe_label_to_subject[label_str] = s

        ingredients = labels_of(EX.Ingredient)
        tags = labels_of(EX.Tag)

        # Precompute normalized lists (accent stripped & lower)
        recipes_norm = [strip_accents(x) for x in recipes]
        ingredients_norm = [strip_accents(x) for x in ingredients]
        tags_norm = [strip_accents(x) for x in tags]

        return cls(
            recipes=recipes,
            ingredients=ingredients,
            tags=tags,
            recipe_label_to_subject=recipe_label_to_subject,
            graph=g,
            recipes_norm=recipes_norm,
            ingredients_norm=ingredients_norm,
            tags_norm=tags_norm
        )

    def _choose_collection(self, collection: str):
        if collection == "recipes":
            return self.recipes, self.recipes_norm
        if collection == "ingredients":
            return self.ingredients, self.ingredients_norm
        if collection == "tags":
            return self.tags, self.tags_norm
        raise ValueError("Unknown collection: " + collection)

    def search(self, collection: str, query: str, limit=5, score_cutoff=50, prefer_pt=False):
        """
        Query: original user candidate string.
        prefer_pt: boolean hint to prefer pt-labeled items when candidate looks portuguese.
        Returns list of (match_label, score_float).
        """
        items, items_norm = self._choose_collection(collection)
        if not query:
            return []

        # Build search pool as (orig_label, norm_label)
        query_norm = strip_accents(query)
        raw_results = process.extract(query_norm, items_norm, scorer=fuzz.WRatio, limit=limit, score_cutoff=score_cutoff)
        # raw_results: list of (matched_norm_label, score, idx)
        results = []
        for matched_norm, score, idx in raw_results:
            orig_label = items[idx]
            adj_score = float(score)
            # boost Portuguese-labelled items if requested and heuristic indicates PT
            if prefer_pt and has_portuguese_chars(orig_label):
                adj_score = min(100.0, adj_score + PT_PREF_BOOST)
            results.append((orig_label, adj_score))
        return results
    
import os, pickle

def load_kg_cached(ttl_path: str, cache_path: str = None):
    if cache_path is None:
        cache_path = ttl_path + ".pkl"

    ttl_mtime = os.path.getmtime(ttl_path)

    # Try to load cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)

            if data.get("_ttl_mtime") == ttl_mtime:
                print("✅ Loaded KG from cache")
                return data["kg"]

        except Exception as e:
            print("⚠️ Cache invalid, rebuilding:", e)

    # Rebuild KG
    print("♻️ Rebuilding KG from TTL...")
    kg = KGIndex.from_ttl(ttl_path)

    # Save cache
    with open(cache_path, "wb") as f:
        pickle.dump({
            "kg": kg,
            "_ttl_mtime": ttl_mtime
        }, f)

    print("✅ KG rebuilt and cached")
    return kg



# 4. NLP CANDIDATES
def extract_candidates(text: str, nlp):
    """
    Returns:
        {
            "candidate_chunks": [str,...],
            "cooking_time": int|None,
            "raw_doc": doc
        }
    """
    doc = nlp(text)
    # gather noun_chunks (avoid duplicates)
    time = cooking_time_to_minutes(text)  # always from original text
    noun_chunks = []
    try:
        for nc in doc.noun_chunks:
            chunk = nc.text.strip()
            if len(chunk) >= 2:
                noun_chunks.append(chunk)
    except Exception:
        # fallback simple token-based grouping for blank models
        noun_chunks = [t.text for t in doc if len(t.text) > 2]

    # Also include quoted strings and direct objects found as PROPN/NOUN tokens
    quoted = re.findall(r'["\']([^"\']{2,})["\']', text)
    noun_chunks.extend([q for q in quoted])

    # Filter generic words
    noun_chunks = [c for c in noun_chunks if c.lower()]

    # Deduplicate preserving order
    seen = set()
    final_chunks = []
    for c in noun_chunks:
        ck = c.lower().strip()
        if ck not in seen:
            seen.add(ck)
            final_chunks.append(c)

    # Use the new parser
    exact_m, max_m = parse_time_constraints(text)

    return {
        "candidate_chunks": final_chunks,
        "cooking_time": exact_m,
        "max_minutes": max_m,
        "doc": doc
    }


# 5. LINKING (intent-aware, PT-first)
def _token_contains(a: str, b: str) -> bool:
    # accent-insensitive containment
    sa = strip_accents(a)
    sb = strip_accents(b)
    return sb in sa or sa in sb

def link_candidates_to_kg(candidates: Dict[str, Any], kg: KGIndex, intent: str) -> Dict[str, Any]:
    # detect from chunks, fallback to unknown text
    candidate_text = candidates.get("doc").text if candidates.get("doc") is not None else " ".join(candidates.get("candidate_chunks", []))
    lang = detect_language(" ".join(candidates.get("candidate_chunks", [])) or candidate_text)
    if lang not in {"pt","en"}:
        lang = "pt" if has_portuguese_chars(candidate_text) else "en"
    other_lang = "en" if lang == "pt" else "pt"

    bag_mode = intent in {"list_by_ingredient","list_by_tag"}
    chunks = content_tokens(
        candidates.get("candidate_chunks", []),
        lang,
        translate_to=other_lang,
        bag_mode=bag_mode,
        original_text=candidate_text,  
    )

    # FIX: Capture both exact and max minutes from candidates
    cooking_time_raw = candidates.get("cooking_time") if intent == "list_by_time" else None
    max_minutes_raw = candidates.get("max_minutes") if intent == "list_by_time" else None

    out = {
        "ingredient": [], 
        "recipe_name": [], 
        "tag": [], 
        "cooking_time": cooking_time_raw,
        "max_minutes": max_minutes_raw  # Pass this through
    }

    text_join = " ".join(chunks).lower() if chunks else ""
    prefer_pt = has_portuguese_chars(text_join)

    if intent == "list_by_time":
        # FIX: Do NOT overwrite cooking_time with empty list.
        # Just return the extracted values so pipeline can use them.
        return out

    if intent == "list_by_ingredient":
        for tok in chunks:
            matches = kg.search("ingredients", tok, limit=5,
                                score_cutoff=THRESHOLDS["ingredient"],
                                prefer_pt=prefer_pt)
            out["ingredient"].extend(matches)

        out["ingredient"] = _cap_scored_list(
            out["ingredient"],
            SAFETY_MAX_RESULTS["ingredient"]
        )
        return out

    if intent == "list_by_tag":
        tag_cutoff = THRESHOLDS["tag"]
        for tok in chunks:
            matches = kg.search("tags", tok, limit=5,
                                score_cutoff=tag_cutoff,
                                prefer_pt=prefer_pt)
            out["tag"].extend(matches)

        out["tag"].extend(_expand_time_tag_synonyms(chunks))

        out["tag"] = _cap_scored_list(
            out["tag"],
            SAFETY_MAX_RESULTS["tag"]
        )
        return out

    # Recipe-oriented intents
    if intent in {"find_recipe", "retrieve_ingredients"}:
        # 1) Build candidate phrases: original ordered chunks + full text
        candidate_phrases: list[str] = []
        # use original noun chunks (preserve order and spacing)
        candidate_phrases.extend(candidates.get("candidate_chunks", []))
        # add full original text (better recall)
        full_text = candidates.get("doc").text if candidates.get("doc") is not None else " ".join(candidates.get("candidate_chunks", []))
        if full_text:
            candidate_phrases.append(full_text)

        # 2) Exact/containment matches over KG recipe labels
        primary: list[tuple[str, float]] = []
        for phrase in sorted(set(candidate_phrases), key=len, reverse=True):
            for label in kg.recipes:
                if _token_contains(label, phrase):
                    primary.append((label, 100.0))
            if primary:
                break

        if primary:
            out["recipe_name"] = primary
            return out

        # 3) Fuzzy fallback using the longest normalized phrase
        longest = max(candidate_phrases, key=len) if candidate_phrases else ""
        if longest:
            matches = kg.search("recipes", longest,
                                limit=10,
                                score_cutoff=max(THRESHOLDS["recipe_name"], 75),
                                prefer_pt=prefer_pt)
            out["recipe_name"].extend(matches)

        out["recipe_name"] = _cap_scored_list(
            out["recipe_name"],
            SAFETY_MAX_RESULTS["recipe_name"]
        )
        return out

    return out


# 6. MASTER FUNCTION (compatible with your pipeline)
# Minimal accent/spelling normalization (fast) for frequent Portuguese misspellings

# Protected culinary terms we prefer not to be mistranslated
_PROTECTED_PT_TERMS = {"açorda", "brás", "braz"}


def _protect_terms(original: str, translated: str) -> str:
    """
    Ensure protected terms remain (or are re-injected) if Argos mangles them.
    Strategy: if a protected term appears in original (accent-insensitive) but
    not in translated (any form), append the original term at end or replace suspicious token.
    """
    out = translated
    orig_tokens = set(re.findall(r"\b\w+\b", original.lower()))
    for term in _PROTECTED_PT_TERMS:
        # term match accent-insensitive
        norm_term = strip_accents(term)
        if any(strip_accents(t) == norm_term for t in orig_tokens):
            # If missing in translation, reinsert
            if norm_term not in strip_accents(out):
                out = out.strip() + f" {term}"
    return out

def translate_between(text: str, src_lang: str, tgt_lang: str) -> str:
    """
    Offline Argos translation with PT accent restoration + protected term preservation.
    """
    if not text or src_lang == tgt_lang:
        return text
    if src_lang not in {"pt", "en"} or tgt_lang not in {"pt", "en"}:
        return text
    prep = text
    translated = _argos_translate.translate(prep, src_lang, tgt_lang)
    if src_lang == "pt":
        translated = _protect_terms(prep, translated)
    return translated


def _merge_scored_lists(a, b):
    """
    Deduplicate by label; keep the highest score. Return sorted descending.
    """
    a = a or []
    b = b or []
    best = {}
    for label, score in a + b:
        s = float(score)
        if label not in best or s > best[label]:
            best[label] = s
    return sorted(best.items(), key=lambda x: x[1], reverse=True)

def _cap_scored_list(items, limit: int):
    """
    Keeps the TOP N results by score.
    Deduplicates by label.
    """
    best = {}
    for label, score in items:
        score = float(score)
        if label not in best or score > best[label]:
            best[label] = score

    return sorted(best.items(), key=lambda x: x[1], reverse=True)[:limit]


#pipeline-compatible function
def extract_and_link(text: str, kg: KGIndex, nlp, intent: str):
    lang = detect_language(text)
    if lang not in {"pt", "en"}:
        lang = "pt" if has_portuguese_chars(text) else "en"
    other = "en" if lang == "pt" else "pt"

    nlp_primary = nlp if getattr(nlp, "lang", None) == lang else build_spacy_pipeline(lang)
    candidates_primary = extract_candidates(text, nlp_primary)

    bag_intents = {"list_by_ingredient","list_by_tag"}
    linked_primary = link_candidates_to_kg(candidates_primary, kg, intent=intent)

    # For bag-mode intents skip secondary pass
    if intent in bag_intents:
        return {
            "ingredient": linked_primary.get("ingredient", []),
            "recipe_name": linked_primary.get("recipe_name", []),
            "tag": linked_primary.get("tag", []),
            "cooking_time": linked_primary.get("cooking_time") if intent == "list_by_time" else None,
            "max_minutes": linked_primary.get("max_minutes") if intent == "list_by_time" else None, # Add this
            "detected_language": lang,
            "original_query": text,
            "translated_query": translate_between(text, lang, other),
        }

    # Dual-pass for other intents
    translated_text = translate_between(text, lang, other)
    nlp_secondary = build_spacy_pipeline(other)
    candidates_secondary = extract_candidates(translated_text, nlp_secondary)
    linked_secondary = link_candidates_to_kg(candidates_secondary, kg, intent=intent)

    merged = {
        "ingredient": _merge_scored_lists(linked_primary.get("ingredient", []), linked_secondary.get("ingredient", [])),
        "recipe_name": _merge_scored_lists(linked_primary.get("recipe_name", []), linked_secondary.get("recipe_name", [])),
        "tag": _merge_scored_lists(linked_primary.get("tag", []), linked_secondary.get("tag", [])),
        # FIX: Merge max_minutes from both passes
        "cooking_time": linked_primary.get("cooking_time") or linked_secondary.get("cooking_time") if intent == "list_by_time" else None,
        "max_minutes": linked_primary.get("max_minutes") or linked_secondary.get("max_minutes") if intent == "list_by_time" else None,
        "detected_language": lang,
        "original_query": text,
        "translated_query": translated_text,
    }
    return merged

# Time / speed related PT expressions mapped to KG tags
_TIME_TAG_SYNONYMS = {
    "pouco tempo": ["30-minutes-or-less", "15-minutes-or-less", "time-to-make"],
    "rapido": ["30-minutes-or-less", "15-minutes-or-less", "time-to-make", "quick"],
    "rápido": ["30-minutes-or-less", "15-minutes-or-less", "time-to-make", "quick"],
    "rápidas": ["30-minutes-or-less", "15-minutes-or-less", "time-to-make", "quick"],
    "rapidas": ["30-minutes-or-less", "15-minutes-or-less", "time-to-make", "quick"],
    "rápida": ["30-minutes-or-less", "15-minutes-or-less", "time-to-make", "quick"],
    "rapida": ["30-minutes-or-less", "15-minutes-or-less", "time-to-make", "quick"],
    "fácil": ["easy"],
    "facil": ["easy"],
    "faceis": ["easy"],
    "fáceis": ["easy"],
}

def _expand_time_tag_synonyms(chunks: List[str]) -> List[tuple[str, float]]:
    out = []
    for c in chunks:
        key = strip_accents(c.lower())
        if key in _TIME_TAG_SYNONYMS:
            for tag in _TIME_TAG_SYNONYMS[key]:
                out.append((tag, 96.0))  # high confidence synthetic match
    return out

def _fallback_tokens_from_text(text: str, lang: str, other_lang: str) -> List[str]:
    stop = STOPWORDS.get(lang, set())
    bag = {}
    def add(txt: str):
        for w in re.findall(r"\b\w+\b", txt.lower()):
            if w in stop or w in DOMAIN_NOISE:
                continue
            key = strip_accents(w)
            # prefer PT-accented form
            if key not in bag or (has_portuguese_chars(w) and not has_portuguese_chars(bag[key])):
                bag[key] = w
    add(text)
    try:
        t = translate_between(text, lang, other_lang)
        add(t)
    except Exception:
        pass
    return [bag[k] for k in sorted(bag.keys())]

def normalize_pt_numbers(text: str) -> str:
    """
    Converts Portuguese AND English number words (e.g., 'quarenta e cinco', 'forty five') into digits ('45').
    Focuses on common cooking time ranges (0-100).
    """
    mapping = {
        # Portuguese
        "um": 1, "uma": 1, "dois": 2, "duas": 2, "três": 3, "tres": 3, "quatro": 4,
        "cinco": 5, "seis": 6, "sete": 7, "oito": 8, "nove": 9, "dez": 10,
        "onze": 11, "doze": 12, "treze": 13, "quatorze": 14, "catorze": 14, "quinze": 15,
        "dezesseis": 16, "dezassete": 17, "dezoito": 18, "dezenove": 19, "vinte": 20,
        "trinta": 30, "quarenta": 40, "cinquenta": 50, "sessenta": 60,
        "setenta": 70, "oitenta": 80, "noventa": 90, "cem": 100,
        
        # English
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
        "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
        "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
        "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100
    }
    
    words = text.lower().split()
    new_words = []
    i = 0
    while i < len(words):
        val = mapping.get(words[i])
        
        if val is not None:
            # Look ahead for PT "e" + number (e.g., "vinte e cinco")
            if i + 2 < len(words) and words[i+1] == "e" and words[i+2] in mapping:
                val += mapping[words[i+2]]
                new_words.append(str(val))
                i += 3 
            # Look ahead for EN compound (e.g., "forty five" or "forty-five")
            elif i + 1 < len(words) and words[i+1] in mapping:
                 # e.g. "forty five" -> 40 + 5
                 # Only combine if first is a multiple of 10 (20,30...) and second is single digit
                 if val >= 20 and val % 10 == 0:
                     val += mapping[words[i+1]]
                     new_words.append(str(val))
                     i += 2
                 else:
                     new_words.append(str(val))
                     i += 1
            else:
                new_words.append(str(val))
                i += 1
        else:
            # Handle hyphenated English words like "forty-five"
            if "-" in words[i]:
                parts = words[i].split("-")
                if len(parts) == 2 and parts[0] in mapping and parts[1] in mapping:
                    v1 = mapping[parts[0]]
                    v2 = mapping[parts[1]]
                    if v1 >= 20 and v1 % 10 == 0:
                        new_words.append(str(v1 + v2))
                        i += 1
                        continue
            
            new_words.append(words[i])
            i += 1
            
    return " ".join(new_words)

def extract_time_info(text: str) -> dict:
    # 1. Normalize Portuguese words to digits
    # "receitas em quarenta e cinco minutos" -> "receitas em 45 minutos"
    text = normalize_pt_numbers(text) 

    match = re.search(r'(\d+)', text)
    if match:
        return {"cooking_time": int(match.group(1)), "max_minutes": None}

    # No explicit time found, fallback to parsing logic
    exact, max_range = parse_time_constraints(text)
    return {"cooking_time": exact, "max_minutes": max_range}

