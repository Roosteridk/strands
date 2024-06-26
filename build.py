import json


# Maps caharacter onto number between 0 - 25
# def abt(letter):
#     return ord(letter) - 97


def build_prefix_dict(min_length=4, max_length=18):
    # stores distance from nearest word
    prefixes: dict[str, int] = {}

    with open("NWL2020.txt", "r") as file:  # Note: NWL2020 doesnt contain word freqs
        for line in file.readlines():
            word = line.split(" ")[0].lower()
            if min_length <= len(word) <= max_length:
                # loop through every prefix in the word. i.e banana -> b, ba, ban, bana, banan, banana
                for i in range(len(word)):
                    prefix = word[: i + 1]
                    diff = len(word) - len(prefix)
                    prefixes[prefix] = min(
                        prefixes.get(prefix, diff), len(word) - len(prefix)
                    )
    return prefixes


test = build_prefix_dict()
print(len(test))
with open("prefixes.json", "w") as outfile:
    json.dump(test, outfile)
print([k for k, v in test.items() if v == 0][0])
