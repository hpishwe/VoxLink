# packet.py

import json


PRIORITY_MAP = {
    "REQUEST_HELP": "EMERGENCY",
    "WARNING": "HIGH",
    "SEND_COMMAND": "HIGH",
    "REPORT_FAILURE": "HIGH",
    "CANCELLATION": "HIGH",

    "REPORT_STATUS": "NORMAL",
    "REPORT_MEASUREMENT": "NORMAL",
    "REPORT_LOCATION": "NORMAL",
    "CONFIRMATION": "NORMAL",

    "UNKNOWN": "NORMAL"
}


# Short codes for compact transmission
INTENT_CODES = {
    "REQUEST_HELP": "H",
    "SEND_COMMAND": "C",
    "REPORT_STATUS": "S",
    "REPORT_FAILURE": "F",
    "REPORT_MEASUREMENT": "M",
    "REPORT_LOCATION": "L",
    "WARNING": "W",
    "CONFIRMATION": "A",
    "CANCELLATION": "X",
    "UNKNOWN": "U"
}


PRIORITY_CODES = {
    "EMERGENCY": "3",
    "HIGH": "2",
    "NORMAL": "1"
}


CONDITION_CODES = {
    "INJURED": "I",
    "DAMAGED": "D",
    "FAILED": "F",
    "BROKEN": "B",
    "OVERHEATED": "O",
    "LOW BATTERY": "LB",
    "NO SIGNAL": "NS"
}


OBJECT_CODES = {
    "ROVER": "R",
    "SENSOR": "S",
    "VALVE": "V",
    "BATTERY": "B",
    "MOTOR": "M",
    "EQUIPMENT": "E",
    "STATION": "ST",
    "COMMUNICATION SYSTEM": "CS",
    "COMMUNICATION": "C"
}


ACTION_CODES = {
    "STOP": "ST",
    "START": "SA",
    "RETURN": "RT",
    "MOVE": "MV",
    "OPEN": "OP",
    "CLOSE": "CL",
    "CHECK": "CK",
    "ACTIVATE": "AC",
    "DEACTIVATE": "DA"
}


UNIT_CODES = {
    "PERCENT": "%",
    "%": "%",
    "DEGREES CELSIUS": "C",
    "°C": "C",
    "KPA": "K",
    "KM/H": "KH",
    "KM": "KM",
    "V": "V",
    "VOLTS": "V",
    "VOLT": "V",
    "A": "A",
    "AMPS": "A",
    "AMP": "A",
    "HZ": "HZ"
}


def create_packet(text, intent, entities):

    packet = {
        "intent": intent,
        "priority": PRIORITY_MAP.get(
            intent,
            "NORMAL"
        ),
        "entities": entities
    }

    return packet


def serialize_packet(packet):

    return json.dumps(
        packet,
        separators=(",", ":")
    )


def create_compact_packet(packet):

    intent = packet.get("intent", "UNKNOWN")
    priority = packet.get("priority", "NORMAL")
    entities = packet.get("entities", {})

    parts = []

    # -------------------------
    # INTENT
    # -------------------------

    intent_code = INTENT_CODES.get(
        intent,
        "U"
    )

    parts.append(intent_code)

    # -------------------------
    # PRIORITY
    # -------------------------

    priority_code = PRIORITY_CODES.get(
        priority,
        "1"
    )

    parts.append(priority_code)

    # -------------------------
    # CONDITION
    # -------------------------

    condition = entities.get("condition")

    if condition:

        condition_code = CONDITION_CODES.get(
            condition,
            condition
        )

        parts.append("I=" + condition_code)

    # -------------------------
    # LOCATION
    # -------------------------

    location = entities.get("location")

    if location:

        location_value = (
            location
            .replace("SECTOR ", "S")
            .replace("WAYPOINT ", "W")
            .replace("GATE ", "G")
            .replace("STATION ", "T")
            .replace(" ", "")
        )

        parts.append("L=" + location_value)

    # -------------------------
    # OBJECT
    # -------------------------

    obj = entities.get("object")

    if obj:

        object_code = OBJECT_CODES.get(
            obj,
            obj
        )

        parts.append("O=" + object_code)

    # -------------------------
    # ACTIONS
    # -------------------------

    actions = entities.get("actions", [])

    if actions:

        action_codes = []

        for action in actions:

            code = ACTION_CODES.get(
                action,
                action
            )

            action_codes.append(code)

        parts.append(
            "A=" + ",".join(action_codes)
        )

    # -------------------------
    # PARAMETER
    # -------------------------

    parameter = entities.get("parameter")

    if parameter:

        parts.append(
            "P=" + parameter
        )

    # -------------------------
    # VALUE
    # -------------------------

    value = entities.get("value")

    if value is not None:

        parts.append(
            "V=" + str(value)
        )

    # -------------------------
    # UNIT
    # -------------------------

    unit = entities.get("unit")

    if unit:

        unit_code = UNIT_CODES.get(
            unit,
            unit
        )

        parts.append(
            "U=" + unit_code
        )

    return "|".join(parts)