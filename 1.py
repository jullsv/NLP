import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import pymorphy3

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


def extract_word_pairs(file_path):
    morph = pymorphy3.MorphAnalyzer()

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except FileNotFoundError:
        print(f"Ошибка: файл '{file_path}' не найден.")
        return

    valid_combinations = {
        ('NOUN', 'NOUN'),
        ('NOUN', 'ADJF'),
        ('ADJF', 'NOUN'),
        ('ADJF', 'ADJF')
    }

    sentences = sent_tokenize(text, language='russian')

    for sentence in sentences:
        tokens = word_tokenize(sentence, language='russian')

        for i in range(len(tokens) - 1):
            word1 = tokens[i]
            word2 = tokens[i + 1]

            if not word1.replace('-', '').isalpha():
                continue

            if not word2.replace('-', '').isalpha():
                continue

            parse1 = morph.parse(word1.lower())[0]
            parse2 = morph.parse(word2.lower())[0]

            if (parse1.tag.POS, parse2.tag.POS) not in valid_combinations:
                continue

            if parse1.tag.number != parse2.tag.number:
                continue

            if parse1.tag.case != parse2.tag.case:
                continue

            if parse1.tag.number == 'sing':
                if parse1.tag.gender != parse2.tag.gender:
                    continue

            print(parse1.normal_form, parse2.normal_form)


if __name__ == '__main__':
    extract_word_pairs('text.txt')
