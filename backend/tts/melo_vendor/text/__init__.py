from .symbols import *


_symbol_to_id = {s: i for i, s in enumerate(symbols)}


def cleaned_text_to_sequence(cleaned_text, tones, language, symbol_to_id=None):
    symbol_to_id_map = symbol_to_id if symbol_to_id else _symbol_to_id
    phones = [symbol_to_id_map[symbol] for symbol in cleaned_text]
    tone_start = language_tone_start_map[language]
    tones = [i + tone_start for i in tones]
    lang_id = language_id_map[language]
    lang_ids = [lang_id for _ in phones]
    return phones, tones, lang_ids


def get_bert(norm_text, word2ph, language, device):
    if language != "ZH":
        raise NotImplementedError(f"Unsupported Melo vendor language: {language}")
    from .chinese_bert import get_bert_feature

    return get_bert_feature(norm_text, word2ph, device)
