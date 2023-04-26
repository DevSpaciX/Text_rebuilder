import itertools

import nltk


def extract_noun_phrases(tree_string):
    """
    This function extracts noun phrases consisting of multiple NPs separated by tags such as comma or CC, groups them,
    and returns the noun phrases together with their positions in the main tree.
    """
    tree = nltk.tree.Tree.fromstring(tree_string)
    noun_phrases = []
    for subtree in tree.subtrees(filter=lambda s: s.label() == "NP"):
        i = 0
        while i < len(subtree):
            if subtree[i].label() != "NP":
                i += 1
                continue
            p = i
            noun_group = []
            noun_positions = []
            while p + 2 <= len(subtree) and (subtree[p+1].label() == "CC" or subtree[p+1].label() == ",") \
                    and subtree[p+2].label() == "NP":
                noun_group.append(subtree[p+2])
                p += 2
            if noun_group:
                noun_group.insert(0, subtree[i])
                for noun in noun_group:
                    for pos in tree.treepositions():
                        if tree[pos] == noun:
                            noun_positions.append(pos)
                noun_phrases.append([noun_group, noun_positions])
            i = p + 1
    return noun_phrases


def generate_noun_permutations(tree, noun_phrases, limit=20):
    """
    This function generates variants of permutations of corresponding noun phrases with each other in a number not
    exceeding the given limit.
    """
    tree = nltk.tree.Tree.fromstring(tree)
    permutations = []
    noun_positions = []
    for noun_phrase in noun_phrases:
        noun_positions.extend(noun_phrase[1])
        permutations.append(list(itertools.permutations(noun_phrase[0])))
    permutation_options = itertools.product(*permutations)
    results = []
    for option in permutation_options:
        variant = [noun for noun_group in option for noun in noun_group]
        new_tree = tree.copy(deep=True)
        for noun, pos in zip(variant, noun_positions):
            new_tree[pos] = noun.copy(deep=True)
        if new_tree.pformat(margin=999) != tree.pformat(margin=999):
            results.append({"tree": new_tree.pformat(margin=999)})
        if len(results) == limit:
            return results
    return results