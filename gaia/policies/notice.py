"""Notice policy: a member who is at home and awake switches off a wasted load in another room.

Each minute, for every open waste interval (any load the waste tracker is counting: lights, fans,
AC, TV, exhaust, geyser overrun), the household is checked in order. Each awake member at home
who is not in the load's room notices it with p = persona notice_rate. The first to notice
switches it off (power 0, source wall, intent noticed_waste, trigger notice), and the next
load is considered. Loads already off are skipped without a draw.

Runs last in POLICY_ORDER. Other policies read the device state, so a load switched off here is
turned on again by its own policy at its next qualifying moment.
"""

import numpy as np

from gaia.engine.actuator import Intent
from gaia.engine.context import TickContext
from gaia.schema.events import Source

POWER = "power"


class Notice:
    name = "notice"

    def tick(self, ctx: TickContext, rng: np.random.Generator) -> list[Intent]:
        pairs = ctx.open_waste()
        if not pairs:
            return []
        out: list[Intent] = []
        for load_id, room_id in pairs:
            if not ctx.devices.is_on(load_id):
                continue  # already off (a pending off state, or a switch-off this minute)
            for person in ctx.people:  # household order
                if person.room_id is None or person.asleep or person.room_id == room_id:
                    continue
                if rng.random() < person.persona.notice_rate:
                    out.append(Intent(
                        load_id=load_id, capability=POWER, value=0.0, person_id=person.person_id,
                        activity=person.activity, intent="noticed_waste", trigger="notice",
                        device_action="turn_off", policy=self.name, source=Source.wall,
                    ))
                    break
        return out
