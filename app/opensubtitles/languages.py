"""Human-readable language names for OpenSubtitles language codes."""

from __future__ import annotations

LANGUAGE_NAMES = {
    "ab": "Abkhazian", "af": "Afrikaans", "sq": "Albanian", "am": "Amharic", "ar": "Arabic",
    "an": "Aragonese", "hy": "Armenian", "as": "Assamese", "at": "Asturian", "az": "Azerbaijani",
    "eu": "Basque", "be": "Belarusian", "bn": "Bengali", "bs": "Bosnian", "br": "Breton",
    "bg": "Bulgarian", "my": "Burmese", "ca": "Catalan", "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)", "ze": "Chinese bilingual", "zh": "Chinese", "hr": "Croatian",
    "cs": "Czech", "da": "Danish", "nl": "Dutch", "en": "English", "eo": "Esperanto",
    "et": "Estonian", "ex": "Extremaduran", "fi": "Finnish", "fr": "French", "gd": "Gaelic",
    "gl": "Galician", "ka": "Georgian", "de": "German", "el": "Greek", "he": "Hebrew",
    "hi": "Hindi", "hu": "Hungarian", "is": "Icelandic", "ig": "Igbo", "id": "Indonesian",
    "ia": "Interlingua", "ga": "Irish", "it": "Italian", "ja": "Japanese", "kn": "Kannada",
    "kk": "Kazakh", "km": "Khmer", "ko": "Korean", "ku": "Kurdish", "lv": "Latvian",
    "lt": "Lithuanian", "lb": "Luxembourgish", "mk": "Macedonian", "ms": "Malay",
    "ml": "Malayalam", "ma": "Manipuri", "mr": "Marathi", "mn": "Mongolian", "me": "Montenegrin",
    "nv": "Navajo", "ne": "Nepali", "se": "Northern Sami", "no": "Norwegian", "oc": "Occitan",
    "or": "Odia", "fa": "Persian", "pl": "Polish", "pt-pt": "Portuguese", "pt": "Portuguese",
    "pt-br": "Portuguese (Brazil)", "pm": "Portuguese (Mozambique)", "ps": "Pashto",
    "ro": "Romanian", "ru": "Russian", "sr": "Serbian", "sd": "Sindhi", "si": "Sinhalese",
    "sk": "Slovak", "sl": "Slovenian", "so": "Somali", "es": "Spanish", "ea": "Spanish (Europe)",
    "sp": "Spanish (Europe)", "es-mx": "Spanish (Latin America)", "sw": "Swahili", "sv": "Swedish",
    "sy": "Syriac", "tl": "Tagalog", "ta": "Tamil", "tt": "Tatar", "te": "Telugu", "th": "Thai",
    "tr": "Turkish", "tk": "Turkmen", "uk": "Ukrainian", "ur": "Urdu", "uz": "Uzbek",
    "vi": "Vietnamese", "cy": "Welsh",
}


def language_name(code: str) -> str:
    c = (code or "").strip().lower()
    return LANGUAGE_NAMES.get(c) or LANGUAGE_NAMES.get(c.split("-")[0]) or c.upper()

