from spellchecker import SpellChecker

checker = SpellChecker()


def is_word(word: str) -> bool:
    return len(checker.unknown([word])) == 0
