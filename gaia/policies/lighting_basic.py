"""G1 main-light behaviour as a policy. Only each room's main light is ever switched.

Per minute, members are visited in household order. For each one:
- Leaving a room: if the room's light is on and nobody awake remains there, and no later member
  also leaves it awake this minute, the light goes off with a 15% (persona) chance of being
  forgotten. A forgotten light stays on; the waste tracker records it.
- Falling asleep: the bedroom light goes off if no other awake member is in that room. If
  nobody else is awake in the home, each lit room with nobody in it goes off with p=0.5
  (cleanup, by the last member to fall asleep).
- Awake and present in a dark room whose light is off: the light goes on, when the persona
  needs light for the current activity.

Light state is tracked within the tick, so two decisions in one minute see each other.
"""

import numpy as np

from gaia.behaviour.needs import indoor_lux, light_needed
from gaia.engine.actuator import Intent
from gaia.engine.context import PersonNow, TickContext
from gaia.home.manifest import HomeManifest, Load, Room
from gaia.interaction.choose import forgets
from gaia.schema.events import Source

CLEANUP_P = 0.5
_SLEEP = "sleep"


class LightingBasic:
    name = "lighting_basic"

    def __init__(self) -> None:
        self._manifest: HomeManifest | None = None
        self._rooms: dict[str, Room] = {}
        self._mains: dict[str, Load | None] = {}
        self._main_rooms: list[str] = []  # room ids with a main light, in manifest order
        self._pending: dict[str, bool] = {}  # load id -> state after this tick's intents

    def tick(self, ctx: TickContext, rng: np.random.Generator) -> list[Intent]:
        self._bind(ctx.manifest)
        self._pending = {}
        people = ctx.people
        changed = ctx.changed  # no room or sleep change this minute: leave and sleep rules are idle
        intents: list[Intent] = []
        for i, person in enumerate(people):
            if changed:
                if person.prev_room_id is not None and person.prev_room_id != person.room_id:
                    self._leave(ctx, people, i, rng, intents)
                if person.asleep and not person.prev_asleep:
                    self._fall_asleep(ctx, people, i, rng, intents)
            if not person.asleep and person.room_id is not None:
                self._presence(ctx, person, intents)
        return intents

    # --- rules ----------------------------------------------------------------

    def _leave(self, ctx, people, i, rng, out: list[Intent]) -> None:
        person = people[i]
        room = person.prev_room_id
        if room is None or room == person.room_id:
            return
        main = self._mains.get(room)
        if main is None or not self._is_on(ctx, main.load_id):
            return
        if any(q.room_id == room and not q.asleep for j, q in enumerate(people) if j != i):
            return
        if any(q.prev_room_id == room and q.room_id != room and not q.asleep for q in people[i + 1:]):
            return  # a later awake leaver in the same minute decides the exit
        if forgets(person.persona, rng):
            return  # the light stays on; waste tracking notices
        self._switch(main, False, person, person.activity, "energy_saving", "room_exit",
                     "turn_off", None, out)

    def _fall_asleep(self, ctx, people, i, rng, out: list[Intent]) -> None:
        person = people[i]
        if not (person.asleep and not person.prev_asleep):
            return
        bed = person.room_id
        others_awake = [q for j, q in enumerate(people) if j != i and not q.asleep]
        bed_main = self._mains.get(bed) if bed is not None else None
        if (bed_main is not None and self._is_on(ctx, bed_main.load_id)
                and not any(q.room_id == bed for q in others_awake)):
            self._switch(bed_main, False, person, _SLEEP, "illumination", "sleep", "turn_off",
                         None, out)
        if others_awake:
            return
        for room_id in self._main_rooms:
            main = self._mains[room_id]
            if not self._is_on(ctx, main.load_id) or any(q.room_id == room_id for q in people):
                continue
            if rng.random() < CLEANUP_P:
                self._switch(main, False, person, _SLEEP, "cleanup", "last_to_sleep", "turn_off",
                             Source.app, out)

    def _presence(self, ctx, person: PersonNow, out: list[Intent]) -> None:
        room = person.room_id
        if person.asleep or room is None:
            return
        main = self._mains.get(room)
        if main is None or self._is_on(ctx, main.load_id):
            return
        indoor = indoor_lux(ctx.outdoor_lux, self._rooms[room])
        if not light_needed(indoor, person.persona, person.activity):
            return
        trigger = "room_entry" if person.prev_room_id != room else "darkness"
        self._switch(main, True, person, person.activity, "illumination", trigger, "turn_on",
                     None, out)

    # --- helpers --------------------------------------------------------------

    def _switch(self, main: Load, on: bool, person: PersonNow, activity: str, intent: str,
                trigger: str, device_action: str, source: Source | None, out: list[Intent]) -> None:
        out.append(Intent(
            load_id=main.load_id, capability="power", value=1.0 if on else 0.0,
            person_id=person.person_id, activity=activity, intent=intent, trigger=trigger,
            device_action=device_action, policy=self.name, source=source,
        ))
        self._pending[main.load_id] = on

    def _is_on(self, ctx: TickContext, load_id: str) -> bool:
        pending = self._pending.get(load_id)
        return ctx.devices.is_on(load_id) if pending is None else pending

    def _bind(self, manifest: HomeManifest) -> None:
        if manifest is self._manifest:
            return
        self._manifest = manifest
        self._rooms = {room.room_id: room for room in manifest.rooms}
        self._mains = {room.room_id: manifest.main_light(room.room_id) for room in manifest.rooms}
        self._main_rooms = [room.room_id for room in manifest.rooms if self._mains[room.room_id]]
