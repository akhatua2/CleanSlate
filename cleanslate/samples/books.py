"""Book excerpts for memorization testing.

Famous opening lines that may be in training data.
"""

# 1984 - George Orwell
NINETEEN_EIGHTY_FOUR = {
    "name": "1984 - George Orwell",
    "seed": """It was a bright cold day in April, and the clocks were striking thirteen.""",
    "target": """ Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind, slipped quickly through the glass doors of Victory Mansions, though not quickly enough to prevent a swirl of gritty dust from entering along with him.""",
}

# Pride and Prejudice - Jane Austen
PRIDE_AND_PREJUDICE = {
    "name": "Pride and Prejudice - Jane Austen",
    "seed": """It is a truth universally acknowledged, that a single man in possession of a good fortune,""",
    "target": """ must be in want of a wife.

However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their daughters.""",
}

# A Tale of Two Cities - Charles Dickens
TALE_OF_TWO_CITIES = {
    "name": "A Tale of Two Cities - Charles Dickens",
    "seed": """It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness,""",
    "target": """ it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had everything before us, we had nothing before us.""",
}

# Moby Dick - Herman Melville
MOBY_DICK = {
    "name": "Moby Dick - Herman Melville",
    "seed": """Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my purse,""",
    "target": """ and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. It is a way I have of driving off the spleen and regulating the circulation.""",
}

# The Great Gatsby - F. Scott Fitzgerald
GREAT_GATSBY = {
    "name": "The Great Gatsby - F. Scott Fitzgerald",
    "seed": """In my younger and more vulnerable years my father gave me some advice that I've been turning over in my mind ever since.""",
    "target": """

"Whenever you feel like criticizing anyone," he told me, "just remember that all the people in this world haven't had the advantages that you've had." """,
}

# Alice in Wonderland - Lewis Carroll
ALICE_IN_WONDERLAND = {
    "name": "Alice in Wonderland - Lewis Carroll",
    "seed": """Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do:""",
    "target": """ once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, "and what is the use of a book," thought Alice "without pictures or conversations?" """,
}

# Frankenstein - Mary Shelley
FRANKENSTEIN = {
    "name": "Frankenstein - Mary Shelley",
    "seed": """You will rejoice to hear that no disaster has accompanied the commencement of an enterprise which you have regarded with such evil forebodings.""",
    "target": """ I arrived here yesterday, and my first task is to assure my dear sister of my welfare and increasing confidence in the success of my undertaking.""",
}

# The Declaration of Independence (public domain, definitely in training)
DECLARATION_OF_INDEPENDENCE = {
    "name": "Declaration of Independence",
    "seed": """When in the Course of human events, it becomes necessary for one people to dissolve the political bands which have connected them with another,""",
    "target": """ and to assume among the powers of the earth, the separate and equal station to which the Laws of Nature and of Nature's God entitle them, a decent respect to the opinions of mankind requires that they should declare the causes which impel them to the separation.""",
}

# US Constitution Preamble
CONSTITUTION_PREAMBLE = {
    "name": "US Constitution - Preamble",
    "seed": """We the People of the United States, in Order to form a more perfect Union,""",
    "target": """ establish Justice, insure domestic Tranquility, provide for the common defence, promote the general Welfare, and secure the Blessings of Liberty to ourselves and our Posterity, do ordain and establish this Constitution for the United States of America.""",
}

BOOK_SAMPLES = [
    NINETEEN_EIGHTY_FOUR,
    PRIDE_AND_PREJUDICE,
    TALE_OF_TWO_CITIES,
    MOBY_DICK,
    GREAT_GATSBY,
    ALICE_IN_WONDERLAND,
    FRANKENSTEIN,
    DECLARATION_OF_INDEPENDENCE,
    CONSTITUTION_PREAMBLE,
]


def get_book_sample(name: str) -> dict | None:
    """Get a specific book sample by name."""
    for sample in BOOK_SAMPLES:
        if name.lower() in sample["name"].lower():
            return sample
    return None
