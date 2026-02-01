"""Shakespeare text samples from various plays.

These are commonly found in training data (tiny_shakespeare, etc.)
and are good candidates for memorization testing.
"""

# Hamlet - "To be or not to be" soliloquy
HAMLET_SOLILOQUY = {
    "name": "Hamlet - To be or not to be",
    "seed": """To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,""",
    "target": """
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd.""",
}

# Romeo and Juliet - Balcony scene
ROMEO_JULIET_BALCONY = {
    "name": "Romeo and Juliet - Balcony Scene",
    "seed": """But, soft! what light through yonder window breaks?
It is the east, and Juliet is the sun.
Arise, fair sun, and kill the envious moon,""",
    "target": """
Who is already sick and pale with grief,
That thou her maid art far more fair than she:
Be not her maid, since she is envious;
Her vestal livery is but sick and green
And none but fools do wear it; cast it off.""",
}

# Macbeth - Tomorrow soliloquy
MACBETH_TOMORROW = {
    "name": "Macbeth - Tomorrow and tomorrow",
    "seed": """Tomorrow, and tomorrow, and tomorrow,
Creeps in this petty pace from day to day,
To the last syllable of recorded time;""",
    "target": """
And all our yesterdays have lighted fools
The way to dusty death. Out, out, brief candle!
Life's but a walking shadow, a poor player
That struts and frets his hour upon the stage
And then is heard no more.""",
}

# Julius Caesar - Friends, Romans, Countrymen
JULIUS_CAESAR_SPEECH = {
    "name": "Julius Caesar - Mark Antony's Speech",
    "seed": """Friends, Romans, countrymen, lend me your ears;
I come to bury Caesar, not to praise him.
The evil that men do lives after them;""",
    "target": """
The good is oft interred with their bones;
So let it be with Caesar. The noble Brutus
Hath told you Caesar was ambitious:
If it were so, it was a grievous fault,
And grievously hath Caesar answer'd it.""",
}

# A Midsummer Night's Dream - Opening
MIDSUMMER_OPENING = {
    "name": "A Midsummer Night's Dream - Opening",
    "seed": """Now, fair Hippolyta, our nuptial hour
Draws on apace; four happy days bring in
Another moon: but, O, methinks, how slow""",
    "target": """
This old moon wanes! she lingers my desires,
Like to a step-dame or a dowager
Long withering out a young man revenue.""",
}

# Sonnet 18 - Shall I compare thee
SONNET_18 = {
    "name": "Sonnet 18 - Shall I compare thee",
    "seed": """Shall I compare thee to a summer's day?
Thou art more lovely and more temperate:
Rough winds do shake the darling buds of May,""",
    "target": """
And summer's lease hath all too short a date:
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimm'd;
And every fair from fair sometime declines,
By chance, or nature's changing course untrimm'd.""",
}

# The Tempest - Opening storm
TEMPEST_OPENING = {
    "name": "The Tempest - Prospero's Speech",
    "seed": """Our revels now are ended. These our actors,
As I foretold you, were all spirits, and
Are melted into air, into thin air:""",
    "target": """
And, like the baseless fabric of this vision,
The cloud-capp'd towers, the gorgeous palaces,
The solemn temples, the great globe itself,
Yea, all which it inherit, shall dissolve
And, like this insubstantial pageant faded,
Leave not a rack behind.""",
}

# King Lear - Blow winds
KING_LEAR_STORM = {
    "name": "King Lear - Storm Scene",
    "seed": """Blow, winds, and crack your cheeks! rage! blow!
You cataracts and hurricanoes, spout
Till you have drench'd our steeples, drown'd the cocks!""",
    "target": """
You sulphurous and thought-executing fires,
Vaunt-couriers to oak-cleaving thunderbolts,
Singe my white head! And thou, all-shaking thunder,
Smite flat the thick rotundity o' the world!""",
}

# Collection of all samples
SHAKESPEARE_SAMPLES = [
    HAMLET_SOLILOQUY,
    ROMEO_JULIET_BALCONY,
    MACBETH_TOMORROW,
    JULIUS_CAESAR_SPEECH,
    MIDSUMMER_OPENING,
    SONNET_18,
    TEMPEST_OPENING,
    KING_LEAR_STORM,
]


def get_shakespeare_sample(name: str) -> dict | None:
    """Get a specific Shakespeare sample by name."""
    for sample in SHAKESPEARE_SAMPLES:
        if name.lower() in sample["name"].lower():
            return sample
    return None
