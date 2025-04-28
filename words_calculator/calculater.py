import gensim.downloader as api

print("Loading model")
model = api.load("glove-wiki-gigaword-50")
print("Model loaded")


def calculate(calculation: str):
    try:
        elements = calculation.lower().split()
        if len(elements) < 3 or len(elements) % 2 == 0:
            raise ValueError("Invalid calculation!")

        positive = [elements[0]]
        negative = []

        for i in range(1, len(elements), 2):
            operator, word = elements[i], elements[i + 1]
            if operator == "+":
                positive.append(word)
            elif operator == "-":
                negative.append(word)
            else:
                raise ValueError(f"Invalid Operator: {operator}")

        most_similar = model.most_similar(positive=positive, negative=negative)

        print(f"Result: {most_similar[0][0]}\n")

    except Exception as e:
        print(e)


print("'exit' to terminate")
while True:
    print("Calculation:")
    input1 = input()
    if input1 == "exit":
        break
    calculate(input1)

# example calculations:
# king - man + woman
# paris - france + germany
#
