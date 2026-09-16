import nltk
from nltk.tokenize import word_tokenize
import pymorphy3

nltk.download('punkt', quiet=True)

def process_text_file(file_path):
    morph = pymorphy3.MorphAnalyzer()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    raw_tokens = word_tokenize(text)
    words = [token for token in raw_tokens if token.isalpha()]
    
    ALLOWED_POS = {'NOUN', 'ADJF', 'ADJS'}
    matched_pairs = []
    
    for i in range(len(words) - 1):
        parse1 = morph.parse(words[i])[0]
        parse2 = morph.parse(words[i+1])[0]
        
        if parse1.tag.POS in ALLOWED_POS and parse2.tag.POS in ALLOWED_POS:
            
            same_case = (parse1.tag.case == parse2.tag.case) and (parse1.tag.case is not None)
            same_number = (parse1.tag.number == parse2.tag.number) and (parse1.tag.number is not None)
            
            if parse1.tag.number == 'sing' and parse2.tag.number == 'sing':
                same_gender = (parse1.tag.gender == parse2.tag.gender) and (parse1.tag.gender is not None)
            else:
                same_gender = True
                
            if same_case and same_number and same_gender:
                lemma1 = parse1.normal_form
                lemma2 = parse2.normal_form
                matched_pairs.append((lemma1, lemma2))
                
    return matched_pairs

if __name__ == "__main__":
    results = process_text_file("text.txt")
    print("Найденные пары лемм:")
    for lemma1, lemma2 in results:
        print(f"{lemma1} {lemma2}")