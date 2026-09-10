# decoder.py


def decode_packet(packet):

    intent = packet.get("intent")
    entities = packet.get("entities", {})

    # --------------------------------
    # REQUEST HELP
    # --------------------------------

    if intent == "REQUEST_HELP":

        condition = entities.get("condition")
        location = entities.get("location")

        sentence = "Emergency. Help is required."

        if condition:
            sentence += f" Condition: {condition}."

        if location:
            sentence += f" Location: {location}."

        return sentence

    # --------------------------------
    # SEND COMMAND
    # --------------------------------

    if intent == "SEND_COMMAND":

        actions = entities.get("actions", [])
        obj = entities.get("object")
        location = entities.get("location")

        sentence = "Command."

        if obj:
            sentence += f" Target: {obj}."

        if actions:

            sentence += " Actions: "
            sentence += ", ".join(actions) + "."

        if location:
            sentence += f" Location: {location}."

        return sentence

    # --------------------------------
    # MEASUREMENT
    # --------------------------------

    if intent == "REPORT_MEASUREMENT":

        parameter = entities.get("parameter")
        obj = entities.get("object")
        value = entities.get("value")
        unit = entities.get("unit")

        sentence = "Measurement reported."

        if parameter:

            sentence += f" Parameter: {parameter}."

        elif obj:

            sentence += f" Parameter: {obj}."

        if value is not None:

            sentence += f" Value: {value} {unit}."

        return sentence

    # --------------------------------
    # STATUS
    # --------------------------------

    if intent == "REPORT_STATUS":

        obj = entities.get("object")
        location = entities.get("location")

        sentence = "Status report."

        if obj:
            sentence += f" Object: {obj}."

        if location:
            sentence += f" Location: {location}."

        return sentence

    # --------------------------------
    # FAILURE
    # --------------------------------

    if intent == "REPORT_FAILURE":

        condition = entities.get("condition")
        obj = entities.get("object")

        sentence = "Failure reported."

        if obj:
            sentence += f" Object: {obj}."

        if condition:
            sentence += f" Condition: {condition}."

        return sentence

    # --------------------------------
    # LOCATION
    # --------------------------------

    if intent == "REPORT_LOCATION":

        location = entities.get("location")

        if location:
            return f"Location reported: {location}."

        return "Location reported."

    # --------------------------------
    # WARNING
    # --------------------------------

    if intent == "WARNING":

        return "Warning. Critical condition reported."

    # --------------------------------
    # CONFIRMATION
    # --------------------------------

    if intent == "CONFIRMATION":

        return "Confirmation received."

    # --------------------------------
    # CANCELLATION
    # --------------------------------

    if intent == "CANCELLATION":

        return "Cancellation command received."

    # --------------------------------
    # UNKNOWN
    # --------------------------------

    return "Message received."