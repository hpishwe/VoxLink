# main.py

from intent import detect_intent
from entities import extract_entities
from packet import (
    create_packet,
    serialize_packet,
    create_compact_packet
)
from decoder import decode_packet


def process_message(text):

    print("\n" + "=" * 75)

    print("\nINPUT:")
    print(text)

    # -------------------------
    # INTENT
    # -------------------------

    intent = detect_intent(text)

    # -------------------------
    # ENTITIES
    # -------------------------

    entities = extract_entities(text)

    # -------------------------
    # NORMAL SEMANTIC PACKET
    # -------------------------

    packet = create_packet(
        text,
        intent,
        entities
    )

    semantic_json = serialize_packet(packet)

    # -------------------------
    # COMPACT PACKET
    # -------------------------

    compact_packet = create_compact_packet(
        packet
    )

    # -------------------------
    # DECODE
    # -------------------------

    reconstructed = decode_packet(packet)

    # -------------------------
    # SIZE MEASUREMENT
    # -------------------------

    original_bytes = len(
        text.encode("utf-8")
    )

    json_bytes = len(
        semantic_json.encode("utf-8")
    )

    compact_bytes = len(
        compact_packet.encode("utf-8")
    )

    if original_bytes > 0:

        compression = (
            1 -
            (compact_bytes / original_bytes)
        ) * 100

    else:

        compression = 0

    # -------------------------
    # OUTPUT
    # -------------------------

    print("\nINTENT:")
    print(intent)

    print("\nENTITIES:")
    print(entities)

    print("\nSEMANTIC JSON:")
    print(semantic_json)

    print("\nCOMPACT PACKET:")
    print(compact_packet)

    print("\nRECONSTRUCTED MESSAGE:")
    print(reconstructed)

    print("\n--- TRANSMISSION ANALYSIS ---")

    print(
        f"Original text bytes : {original_bytes}"
    )

    print(
        f"Semantic JSON bytes : {json_bytes}"
    )

    print(
        f"Compact packet bytes: {compact_bytes}"
    )

    print(
        f"Compact reduction   : {compression:.2f}%"
    )


if __name__ == "__main__":

    test_messages = [

        "Please send help immediately, I am injured and I'm near Sector 5.",

        "The rover has reached waypoint 17.",

        "Stop the rover and return to waypoint 8.",

        "Battery is at 38 percent.",

        "The thermal sensor temperature is 72 degrees Celsius.",

        "The communication system has failed.",

        "The rover is operational at waypoint 12.",

        "Warning, the temperature is 95 degrees Celsius.",

        "The rover is located near Gate 3.",

        "Cancel the previous command."

    ]

    for message in test_messages:

        process_message(message)