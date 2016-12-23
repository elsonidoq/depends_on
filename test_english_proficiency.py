import english_proficiency
from pprint import pprint

sents = [
   "Hi! My name is Pablo. How are you?",
    # :P
    "I guess I should get some well written and bad written examples to show here"
]

for sent in sents:
    print sent
    print "*"*len(sent)
    pprint(english_proficiency.get_features(sent))
