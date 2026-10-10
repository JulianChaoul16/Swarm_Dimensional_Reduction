"""Harry Potter character names used to identify wolves in the pack."""

NAMES = (
    "Harry Potter", "Hermione Granger", "Ron Weasley", "Albus Dumbledore",
    "Minerva McGonagall", "Severus Snape", "Rubeus Hagrid", "Sirius Black",
    "Remus Lupin", "Luna Lovegood", "Neville Longbottom", "Ginny Weasley",
    "Fred Weasley", "George Weasley", "Arthur Weasley", "Molly Weasley",
    "Bill Weasley", "Charlie Weasley", "Percy Weasley", "Draco Malfoy",
    "Lucius Malfoy", "Narcissa Malfoy", "Dobby", "Kreacher",
    "Nymphadora Tonks", "Alastor Moody", "Kingsley Shacklebolt",
    "Cedric Diggory", "Cho Chang", "Fleur Delacour", "Viktor Krum",
    "Oliver Wood", "Hedwig", "Fawkes", "Dolores Umbridge",
    "Bellatrix Lestrange", "Peter Pettigrew", "Lord Voldemort",
    "Garrick Ollivander", "Sybill Trelawney",
)


def wolf_name(index: int) -> str:
    """Return a distinct name for a zero-based wolf index of any pack size."""
    if index < 0:
        raise ValueError("Wolf index must be nonnegative.")
    cycle, offset = divmod(index, len(NAMES))
    return NAMES[offset] if cycle == 0 else f"{NAMES[offset]} ({cycle + 1})"
