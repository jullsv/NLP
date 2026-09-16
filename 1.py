import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import pymorphy3

nltk.download('punkt', quiet=True)

def extract_word_pairs(file_path):
    m = pymorphy3.MorphAnalyzer()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Ошибка: Файл '{file_path}' не найден.")
        return

    valid_combos = {
        ('NOUN', 'NOUN'),
        ('NOUN', 'ADJF'),
        ('ADJF', 'NOUN'),
        ('ADJF', 'ADJF')
    }
    
    stop_pos = {'PREP', 'CONJ', 'PRCL', 'INTJ'}

    sentences = sent_tokenize(text)
    
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        words = [t.lower() for t in tokens if t.replace('-', '').isalpha()]
        
        meaningful_words = [
            w for w in words 
            if not any(p.tag.POS in stop_pos for p in m.parse(w))
        ]
        
        for i in range(len(meaningful_words) - 1):
            w1_str = meaningful_words[i]
            w2_str = meaningful_words[i+1]
            
            parses1 = m.parse(w1_str)
            parses2 = m.parse(w2_str)
            
            match_found = False
            for p1 in parses1:
                tag1_str = str(p1.tag)
                if any(pron in tag1_str for pron in ('A-PRO', 'NPRO', 'A-num', 'Apro')):
                    continue
                    
                for p2 in parses2:
                    tag2_str = str(p2.tag)
                    if any(pron in tag2_str for pron in ('A-PRO', 'NPRO', 'A-num', 'Apro')):
                        continue
                    
                    if (p1.tag.POS, p2.tag.POS) not in valid_combos:
                        continue
                    
                    if p1.normal_form == p2.normal_form:
                        continue
                    
                    if not (p1.tag.number and p2.tag.number and p1.tag.number == p2.tag.number):
                        continue
                    if not (p1.tag.case and p2.tag.case and p1.tag.case == p2.tag.case):
                        continue
                    
                    if p1.tag.number == 'sing':
                        if not (p1.tag.gender and p2.tag.gender and p1.tag.gender == p2.tag.gender):
                            continue
                    
                    print(f"{p1.normal_form} {p2.normal_form}")
                    match_found = True
                    break
                
                if match_found:
                    break

if __name__ == '__main__':
    extract_word_pairs('text.txt')
