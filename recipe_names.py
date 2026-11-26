"""Shared recipe and ingredient name normalization."""
import re
import unicodedata


def key(text):
    text = unicodedata.normalize('NFKC', str(text)).casefold()
    return ' '.join(''.join(c if (c.isalnum() or unicodedata.category(c).startswith('M')) else ' '
                           for c in text).split())

def title_key(text):
    text = re.sub(r'\b(?:briyani|biriyani|biriani)\b', 'biryani', key(text))
    return re.sub(r'\b(?:kachi|kachchi|katchi)\b', 'kacchi', text)

def family_key(text):
    canonical = title_key(text)
    if re.search(r'\bkacchi\b.*\bbiryani\b', canonical):
        return 'kacchi biryani'
    # Strip catalog-generated numeric/version suffixes while retaining actual dish descriptors.
    return re.sub(r'\s+(?:v|version|variant|style)?\s*\d+$', '', canonical)
