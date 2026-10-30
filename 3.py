import re
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

MODEL_NAME = "DeepPavlov/rubert-base-cased"
SENTENCES = {
    "меридианам": "Ученики изучали меридианы и учились следовать [MASK].",
    "ментально": "Они изменились физически и [MASK].",
}
TOP_N = 10
FIRST_BEAM = 350
SECOND_BEAM = 25
BATCH_SIZE = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()


def is_russian(token):
    return re.fullmatch(r"[А-Яа-яЁё]+", token) is not None


def get_allowed_tokens():
    first, endings = [], []
    for idx in range(len(tokenizer)):
        token = tokenizer.convert_ids_to_tokens(idx)
        if token.startswith("##"):
            if is_russian(token[2:]):
                endings.append(idx)
        elif is_russian(token):
            first.append(idx)
    return first, endings


FIRST_IDS, ENDING_IDS = get_allowed_tokens()
ENDING_TENSOR = torch.tensor(ENDING_IDS, dtype=torch.long, device=DEVICE)


def prepare(sentence):
    if sentence.count("[MASK]") != 1:
        raise ValueError("В предложении должен быть один пропуск [MASK].")
    text = sentence.replace("[MASK]", f"{tokenizer.mask_token} {tokenizer.mask_token}")
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    positions = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero().flatten().tolist()
    if len(positions) != 2:
        raise ValueError("Не удалось сформировать две маски.")
    return inputs, positions


def search_candidates(sentence):
    inputs, (first_pos, second_pos) = prepare(sentence)

    with torch.inference_mode():
        first_logits = model(**inputs).logits[0, first_pos]
        first_log_probs = torch.log_softmax(first_logits, dim=-1)

    available_first = torch.tensor(FIRST_IDS, dtype=torch.long, device=DEVICE)
    start_scores = first_log_probs[available_first]
    best_start_indices = torch.topk(start_scores, min(FIRST_BEAM, len(FIRST_IDS))).indices
    starters = available_first[best_start_indices].tolist()

    candidates = {}
    for offset in range(0, len(starters), BATCH_SIZE):
        batch = starters[offset:offset + BATCH_SIZE]
        batch_ids = inputs["input_ids"].repeat(len(batch), 1)
        batch_ids[:, first_pos] = torch.tensor(batch, dtype=torch.long, device=DEVICE)
        attention_mask = inputs["attention_mask"].repeat(len(batch), 1)

        with torch.inference_mode():
            logits = model(input_ids=batch_ids, attention_mask=attention_mask).logits[:, second_pos, :]
            log_probs = torch.log_softmax(logits, dim=-1)

        continuation_scores = log_probs[:, ENDING_TENSOR]
        top_indices = torch.topk(continuation_scores, min(SECOND_BEAM, len(ENDING_IDS)), dim=-1).indices

        for j, first_id in enumerate(batch):
            end_ids = [ENDING_IDS[k] for k in top_indices[j].tolist()]
            for end_id in end_ids:
                word = tokenizer.convert_tokens_to_string(
                    tokenizer.convert_ids_to_tokens([first_id, end_id])
                )
                if not is_russian(word):
                    continue
                if tokenizer.encode(word, add_special_tokens=False) != [first_id, end_id]:
                    continue
                score = first_log_probs[first_id].item() + log_probs[j, end_id].item()
                candidates[word] = max(candidates.get(word, float("-inf")), score)

    ranking = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
    return ranking


def main():
    print(f"Модель: {MODEL_NAME}")

    for target, sentence in SENTENCES.items():
        ranking = search_candidates(sentence)
        rank = next((i for i, (word, _) in enumerate(ranking, 1) if word == target), None)

        print(f"\nСлово: {target}")
        print(f"Предложение: {sentence}")
        print("Топ-10 двухтокенных кандидатов (ограниченный поиск):")

        for position, (word, score) in enumerate(ranking[:TOP_N], 1):
            mark = " *" if word == target else ""
            print(f"{position:2}. {word:22} {score:.4f}{mark}")

        if rank is None:
            print("Целевое слово не найдено среди кандидатов.")
        else:
            print(f"Место слова «{target}»: {rank}")
            print(f"В топ-10 кандидатов: {'да' if rank <= TOP_N else 'нет'}")


if __name__ == "__main__":
    main()
