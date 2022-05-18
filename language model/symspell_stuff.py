# import symspell
from symspellpy import SymSpell, Verbosity

# instantiate it
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = "wiki_fa_80k.txt"
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1, encoding='utf-8')

# input sample:
input_term = "اهوار"  # misspelling of "اهواز" It's a city name!

# lookup the dictionary
suggestions = sym_spell.lookup(input_term, Verbosity.ALL, max_edit_distance=2)
# display suggestion term, term frequency, and edit distance
for suggestion in suggestions[:5]:
    print(suggestion)