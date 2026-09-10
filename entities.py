# entities.py

import re


def extract_entities(text):

    entities = {}

    text_lower = text.lower()

    # -------------------------
    # CONDITION
    # -------------------------

    conditions = [
        "injured",
        "damaged",
        "failed",
        "broken",
        "overheated",
        "low battery",
        "no signal"
    ]

    for condition in conditions:

        if condition in text_lower:

            entities["condition"] = condition.upper()
            break

    # -------------------------
    # LOCATION
    # -------------------------

    location_patterns = [
        r"\bsector\s+\d+\b",
        r"\bwaypoint\s+\d+\b",
        r"\bgate\s+\d+\b",
        r"\bstation\s+[a-zA-Z0-9]+\b"
    ]

    for pattern in location_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            entities["location"] = match.group(0).upper()
            break

    # -------------------------
    # MEASUREMENT
    # -------------------------

    measurement_pattern = (
        r"(\d+(?:\.\d+)?)\s*"
        r"(percent|%|degrees?\s*celsius|°c|kpa|km/h|km|"
        r"volts?|v|amps?|a|hz)"
    )

    match = re.search(
        measurement_pattern,
        text,
        re.IGNORECASE
    )

    if match:

        value = float(match.group(1))

        if value.is_integer():
            value = int(value)

        unit = match.group(2).upper()

        entities["value"] = value
        entities["unit"] = unit

        # Identify measurement parameter

        measurement_parameters = [
            "temperature",
            "pressure",
            "voltage",
            "current",
            "speed",
            "battery"
        ]

        for parameter in measurement_parameters:

            if re.search(
                r"\b" + parameter + r"\b",
                text_lower
            ):

                entities["parameter"] = parameter.upper()
                break

    # -------------------------
    # ACTION
    # -------------------------

    actions = [
        "stop",
        "start",
        "return",
        "move",
        "open",
        "close",
        "check",
        "activate",
        "deactivate"
    ]

    found_actions = []

    for action in actions:

        if re.search(
            r"\b" + action + r"\b",
            text_lower
        ):

            found_actions.append(action.upper())

    if found_actions:
        entities["actions"] = found_actions

    # -------------------------
    # OBJECT
    # -------------------------

    objects = [
        "rover",
        "sensor",
        "valve",
        "battery",
        "motor",
        "equipment",
        "station",
        "communication system",
        "communication"
    ]

    for obj in objects:

        if re.search(
            r"\b" + obj + r"\b",
            text_lower
        ):

            entities["object"] = obj.upper()
            break

    return entities