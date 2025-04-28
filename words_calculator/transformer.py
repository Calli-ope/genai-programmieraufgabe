from transformers import BertTokenizer, BertModel
import torch
import numpy as np

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")


def get_embedding(word):
    inputs = tokenizer(
        word,
        padding=True,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=True,
    )
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def calculate(calculation: str):
    try:
        elements = calculation.lower().split()
        if len(elements) < 3 or len(elements) % 2 == 0:
            raise ValueError(
                "Invalid calculation! Format example: 'king - man + woman'."
            )

        positive = [elements[0]]
        negative = []

        for i in range(1, len(elements), 2):
            operator, word = elements[i], elements[i + 1]
            if operator == "+":
                positive.append(word)
            elif operator == "-":
                negative.append(word)
            else:
                raise ValueError(f"Invalid operator: {operator}")

        target_vector = sum(get_embedding(word) for word in positive)
        target_vector -= sum(get_embedding(word) for word in negative)

        with open("words_calculator/common_words.txt", "r") as f:
            word_list = [line.strip() for line in f if line.strip()]

        similarities = []
        for word in word_list:
            emb = get_embedding(word)
            sim = cosine_similarity(target_vector, emb)
            similarities.append((word, sim))

        similarities = sorted(similarities, key=lambda x: -x[1])

        print(f"Result: {similarities[0][0]}\n")

    except Exception as e:
        print(f"Error: {e}")


print("'exit' to terminate")
while True:
    print("Calculation:")
    input1 = input()
    if input1 == "exit":
        break
    calculate(input1)
