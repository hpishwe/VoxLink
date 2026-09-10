# intent.py

INTENT_PATTERNS = {
    "REQUEST_HELP": [
        ["send", "help"],
        ["need", "help"],
        ["require", "help"],
        ["request", "help"],
        ["send", "rescue"],
        ["need", "rescue"],
        ["emergency"],
        ["assist", "me"],
        ["save", "me"],
    ],

    "SEND_COMMAND": [
        ["stop"],
        ["start"],
        ["return"],
        ["move"],
        ["go"],
        ["open"],
        ["close"],
        ["activate"],
        ["deactivate"],
    ],

    "REPORT_FAILURE": [
        ["has", "failed"],
        ["have", "failed"],
        ["failure"],
        ["malfunction"],
        ["not", "working"],
        ["broken"],
        ["damaged"],
    ],

    "REPORT_MEASUREMENT": [
        ["temperature"],
        ["pressure"],
        ["voltage"],
        ["current"],
        ["battery"],
        ["speed"],
        ["percent"],
        ["percentage"],
        ["degrees"],
        ["kpa"],
        ["volts"],
        ["amps"],
    ],

    "REPORT_STATUS": [
        ["status"],
        ["reached"],
        ["operational"],
        ["working"],
        ["online"],
        ["offline"],
        ["ready"],
    ],

    "REPORT_LOCATION": [
        ["location"],
        ["located"],
    ],

    "WARNING": [
        ["warning"],
        ["danger"],
        ["critical"],
        ["unsafe"],
        ["overheated"],
    ],

    "CONFIRMATION": [
        ["confirmed"],
        ["confirm"],
        ["yes"],
        ["correct"],
    ],

    "CANCELLATION": [
        ["cancel"],
        ["abort"],
    ],
}


def detect_intent(text):

    text = text.lower()

    best_intent = "UNKNOWN"
    best_score = 0

    for intent, patterns in INTENT_PATTERNS.items():

        score = 0

        for pattern in patterns:

            if all(word in text for word in pattern):
                score += len(pattern)

        if score > best_score:
            best_score = score
            best_intent = intent

    return best_intent