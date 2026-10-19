from iiil.tasks.autotune import AutotuneTask
from iiil.tasks.empty import EmptyTask
from iiil.tasks.open import OpenTask
from iiil.tasks.open_radio import OpenRadioTask
from iiil.tasks.pick_place import PickPlaceTask
from iiil.tasks.pnp_hard import PickPlaceHardTask
from iiil.tasks.pour import PourTask
from iiil.tasks.radio import RadioTask
from iiil.tasks.slice import SliceTask


TASK_NAME_DICT = {
    "AutotuneTask": "autotune",
    "EmptyTask": "real",
    "OpenTask": "open",
    "OpenRadioTask": "open_radio",
    "PickPlaceHardTask": "pnp_hard",
    "PickPlaceTask": "pnp",
    "PourTask": "pour",
    "RadioTask": "radio",
    "SliceTask": "slice",
}

TASK_INSTRUCTION_DICT = {
    # simulation tasks
    "pnp": "Pick up and place the strawberry onto the plate.",
    "pnp_hard": "Pick up and place the red strawberry, then pick up and place the purple strawberry.",
    "open_radio": "Open the fridge, then pick up and turn on the radio.",
    # real tasks
    "books": "Place the basket onto the table, and put the two books from the shelf into the basket.",
    "popcorn": "Pick up the red popcorn bag from the shelf and place it into the microwave.",
    "bread": "Retrieve the bowl from the drawer, put the two bread in from the plate, and place the bowl on the right side of the countertop.",
    "radio": "Pick up the radio on the table and poke the orange button.",
    "mug": "Pick up the mug from the table and hang it on the rack.",
    "peg": "Pick up the peg from the table and insert it into the hole.",
    }


SUBTASK_NAME_DICT = {
    "pnp": ["pick up strawberry", "place strawberry"],
    "pnp_hard": ["pick up red strawberry", "place red strawberry", "pick up purple strawberry", "place purple strawberry"],
    "open_radio": ["open fridge", "navigate to table", "pick up radio"],
    "books": ["pick up basket", "place basket", "pick up book 1", "pick up book 2", "place books"],
    "popcorn": ["pick up popcorn", "open microwave", "place popcorn", "close microwave"],
    "bread": ["open drawer", "pick up bowl", "close drawer", "put bread 1", "put bread 2", "place bowl"],
    "radio": ["navigate to radio", "pick up radio", "poke button"],
    "mug": ["pick up mug", "hang mug"],
    "peg": ["pick up peg", "insert peg"],
}



__all__ = [
    "AutotuneTask",
    "EmptyTask",
    "OpenTask",
    "OpenRadioTask",
    "PickPlaceHardTask",
    "PickPlaceTask",
    "PourTask",
    "RadioTask",
    "SliceTask",
    "TASK_NAME_DICT",
    "TASK_INSTRUCTION_DICT"
]
