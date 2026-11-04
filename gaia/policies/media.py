"""Media and AV policy: TV sessions with their set-top box or streamer, music on speakers and
soundbars, pauses during TV sessions, and home theatre evenings. It touches only media.* classes.

TV (the media.tv of the room where an awake person starts the tv activity):
- tv activity start: the TV goes on with the source (set-top box with persona tv_source_pref when
  the room has one, else the streamer, else tv_app), the source device goes on, volume is set to
  the persona's volume_pref, and playback is playing. A session already on in the room is joined.
- Pauses: each minute with an awake occupant in the room draws p = PAUSE_P. A pause resumes after
  PAUSE_MIN to PAUSE_MAX minutes (timer). A forgotten TV is never paused.
- Activity end: the tv activity ends while the person is still in the room and nobody else in the
  room is watching: the TV goes off, and its source goes to standby (power 0).
- Last awake leaver: when the leaver leaves the TV room while the TV is still on, the policy rng
  decides forget (persona forgetfulness) or off (room_exit). A forgotten TV stays on, unattended,
  until a timer (FORGET_OFF_MIN to FORGET_OFF_MAX minutes) switches it off. Starting a new tv
  session takes a forgotten TV over.

Music (cook, dinner, relax, wind_down in a room with a media.speaker or media.soundbar, the first
such load in manifest order):
- Activity start in such a room with no music on: a draw with p = persona music_affinity. The
  source is spotify (0.5), airplay (0.4) or radio (0.1). Power on, source, volume, play.
- Stop (playback stopped, then power off) when nobody awake in the room is doing a music activity
  any more. So a music activity followed by another music activity in the same room continues.

Home theatre (the room holding a media.projector, media.av_receiver and media.screen_motorised):
- Each local weekend or festival day draws once with p = THEATRE_P. The session starts at a
  uniform minute in [20:00, 23:00) and runs THEATRE_LEN_MIN to THEATRE_LEN_MAX minutes. It starts
  only if someone is awake at home at its start minute, with that person as the intent's person.
  Nobody is moved into the room in the occupancy grids, so the theatre loads are not waste
  (waste_rule none). The theatre room's lights are left to the lighting policy.

A TV or speaker that another policy switched off (a notice) loses its session or music at the next
minute: the policy reads the device, so no pause, timer or start acts on a set that is off.

Quiet minutes (no change, no TV session, no timer due) return at once. Within a minute the order is
fixed, so no load gets two intents that the actuator would drop as duplicates: TV and music ends,
then starts, then pauses, then timers, then the theatre. Iteration uses dicts and tuples.
"""

from dataclasses import dataclass

import numpy as np

from gaia.engine.actuator import Intent
from gaia.engine.context import PersonNow, TickContext
from gaia.home.manifest import Load
from gaia.interaction.choose import forgets
from gaia.world.calendar import DayType

PAUSE_P = 1 / 45
PAUSE_MIN, PAUSE_MAX = 1, 5  # a pause resumes after this many minutes, inclusive
FORGET_OFF_MIN, FORGET_OFF_MAX = 30, 180  # a forgotten TV goes off after this many minutes, inclusive
THEATRE_P = 0.3
THEATRE_START_MIN, THEATRE_START_MAX = 20 * 60, 23 * 60  # local minute of day, start drawn from [min, max)
THEATRE_LEN_MIN, THEATRE_LEN_MAX = 120, 180  # inclusive
MUSIC_ACTIVITIES = frozenset({"cook", "dinner", "relax", "wind_down"})
MUSIC_SOURCES: tuple[tuple[str, float], ...] = (("spotify", 0.5), ("airplay", 0.4), ("radio", 0.1))
TV_ACTIVITY = "tv"
ENTERTAINMENT = "entertainment"
ENERGY_SAVING = "energy_saving"
TV_CLASS = "media.tv"
STB_CLASS = "media.stb_dth"
STREAMER_CLASS = "media.streamer"
AUDIO_CLASSES = frozenset({"media.speaker", "media.soundbar"})
THEATRE_CLASSES = ("media.projector", "media.av_receiver", "media.screen_motorised")
THEATRE_SOURCE = "hdmi1"
EPS = 1e-9
_NO_ONE = (None, "none", "timer")  # the actor of a timer event: nobody, the system

# A cause for one command: (person id or None, activity label, trigger).
_Who = tuple[str | None, str, str]


@dataclass(slots=True)
class _TvSession:
    room_id: str
    tv: Load
    source: Load | None  # the set-top box or streamer feeding the TV; None for tv_app
    since: int  # minute the session turned on
    forgotten: bool = False
    resume_at: int | None = None  # minute a pause ends
    paused_by: _Who | None = None  # who paused it, so the resume is attributed to them
    off_at: int | None = None  # minute a forgotten TV goes off


def _action(capability: str, value: float | str) -> str:
    if capability == "power":
        return "turn_on" if value == 1.0 else "turn_off"
    if capability == "source":
        return "select_source"
    if capability == "volume":
        return "set_volume"
    if capability == "playback":
        return {"playing": "play", "paused": "pause", "stopped": "stop"}[value]
    if capability == "position":
        return "open" if value == 1.0 else "close"  # 1 is down, 0 is rolled up
    raise ValueError(f"no device action for {capability!r}")


def _music_source(rng: np.random.Generator) -> str:
    u = rng.random()
    acc = 0.0
    for name, p in MUSIC_SOURCES:
        acc += p
        if u < acc:
            return name
    return MUSIC_SOURCES[-1][0]


class Media:
    name = "media"

    def __init__(self) -> None:
        self._manifest = None
        self._tv: dict[str, Load] = {}  # room id -> TV
        self._stb: dict[str, Load] = {}  # room id -> set-top box
        self._streamer: dict[str, Load] = {}  # room id -> streamer
        self._audio: dict[str, Load] = {}  # room id -> first speaker or soundbar
        self._theatre_room: str | None = None
        self._theatre: dict[str, Load] = {}  # theatre class -> load in the theatre room
        self._sessions: dict[str, _TvSession] = {}  # TV load id -> session
        self._music: dict[str, Load] = {}  # room id -> audio load in a music session
        self._activity: dict[str, str] = {}  # person id -> activity at the last changed minute
        self._theatre_plan: tuple[int, int] | None = None  # (start minute, length)
        self._theatre_end: tuple[int, str] | None = None  # (end minute, person id of the start)
        self._next_timer: int | None = None

    def tick(self, ctx: TickContext, rng: np.random.Generator) -> list[Intent]:
        self._bind(ctx)
        self._reconcile(ctx)
        minute = ctx.minute
        if (not ctx.changed and ctx.local_minute_of_day and not self._sessions
                and (self._next_timer is None or minute < self._next_timer)):
            return []
        out: list[Intent] = []
        if ctx.local_minute_of_day == 0 and self._theatre_room is not None:
            self._plan_theatre(ctx, rng)
        if ctx.changed:
            moves = self._changes(ctx)
            self._tv_ends(ctx, moves, rng, out)
            self._music_ends(ctx, moves, out)
            self._starts(ctx, moves, rng, out)
        self._pauses(ctx, rng, out)
        self._timers(ctx, out)
        self._theatre_step(ctx, out)
        self._next_timer = self._earliest()
        return out

    def _reconcile(self, ctx: TickContext) -> None:
        """Drop sessions and music whose set was switched off outside this policy (a notice). A
        stale session would otherwise pause, resume or time out a TV that is off, and the actuator
        would power it back on."""
        for load_id in [k for k in self._sessions if not ctx.devices.is_on(k)]:
            del self._sessions[load_id]
        for room in [r for r, audio in self._music.items() if not ctx.devices.is_on(audio.load_id)]:
            del self._music[room]

    # --- changed minutes -----------------------------------------------------

    def _changes(self, ctx: TickContext) -> dict[str, str]:
        """Person id -> previous activity, for each person whose activity or room moved."""
        moves: dict[str, str] = {}
        for person in ctx.people:
            prev = self._activity.get(person.person_id, person.activity)
            self._activity[person.person_id] = person.activity
            if prev != person.activity or person.prev_room_id != person.room_id:
                moves[person.person_id] = prev
        return moves

    def _tv_ends(self, ctx: TickContext, moves: dict[str, str], rng, out: list[Intent]) -> None:
        people = ctx.people
        for i, person in enumerate(people):
            if person.person_id not in moves:
                continue
            left = person.prev_room_id is not None and person.prev_room_id != person.room_id
            if left:
                self._tv_left(ctx, people, i, rng, out)
            elif (moves[person.person_id] == TV_ACTIVITY and person.activity != TV_ACTIVITY
                  and person.room_id is not None):
                session = self._session_in(person.room_id)
                if session is not None and not session.forgotten and not self._watching(ctx, person):
                    self._tv_off(ctx, session, (person.person_id, person.activity, "activity_end"), out)

    def _tv_left(self, ctx: TickContext, people: tuple[PersonNow, ...], i: int, rng,
                 out: list[Intent]) -> None:
        person = people[i]
        room = person.prev_room_id
        session = self._session_in(room)
        if session is None or session.forgotten:
            return
        if any(q.room_id == room and not q.asleep for j, q in enumerate(people) if j != i):
            return  # someone awake is still in the room
        if any(q.prev_room_id == room and q.room_id != room and not q.asleep for q in people[i + 1:]):
            return  # a later awake leaver in the same minute decides
        if forgets(person.persona, rng):
            session.forgotten = True
            session.off_at = ctx.minute + int(rng.integers(FORGET_OFF_MIN, FORGET_OFF_MAX + 1))
            return
        self._tv_off(ctx, session, (person.person_id, person.activity, "room_exit"), out)

    def _music_ends(self, ctx: TickContext, moves: dict[str, str], out: list[Intent]) -> None:
        candidates: dict[str, PersonNow] = {}  # room id the person left or was in -> the person
        for person in ctx.people:
            if person.person_id in moves and person.prev_room_id is not None:
                candidates[person.prev_room_id] = person
        for room, person in candidates.items():
            audio = self._music.get(room)
            if audio is None:
                continue
            if any(q.room_id == room and not q.asleep and q.activity in MUSIC_ACTIVITIES
                   for q in ctx.people):
                continue
            del self._music[room]
            trigger = "room_exit" if person.room_id != room else "activity_end"
            who = (person.person_id, person.activity, trigger)
            self._change(ctx, audio, "playback", "stopped", who, out)
            self._change(ctx, audio, "power", 0.0, who, out)

    def _starts(self, ctx: TickContext, moves: dict[str, str], rng, out: list[Intent]) -> None:
        for person in ctx.people:
            if person.person_id not in moves or person.asleep or person.room_id is None:
                continue
            if moves[person.person_id] == person.activity:
                continue
            if person.activity == TV_ACTIVITY:
                self._tv_start(ctx, person, rng, out)
            elif person.activity in MUSIC_ACTIVITIES:
                self._music_start(ctx, person, rng, out)

    # --- TV --------------------------------------------------------------------

    def _tv_start(self, ctx: TickContext, person: PersonNow, rng, out: list[Intent]) -> None:
        tv = self._tv.get(person.room_id)
        if tv is None:
            return
        old = self._sessions.get(tv.load_id)
        if old is not None and not old.forgotten:
            return  # joins the session already on
        source, src = self._pick_source(person, rng)
        who = (person.person_id, person.activity, "activity_start")
        if old is not None and old.source is not None and (src is None or src.load_id != old.source.load_id):
            self._change(ctx, old.source, "power", 0.0, who, out)  # the forgotten source goes off
        self._sessions[tv.load_id] = _TvSession(person.room_id, tv, src, ctx.minute)
        if src is not None:
            self._change(ctx, src, "power", 1.0, who, out)
        self._change(ctx, tv, "power", 1.0, who, out)
        self._change(ctx, tv, "source", source, who, out)
        self._change(ctx, tv, "volume", person.persona.volume_pref, who, out)
        self._change(ctx, tv, "playback", "playing", who, out)

    def _pick_source(self, person: PersonNow, rng) -> tuple[str, Load | None]:
        room = person.room_id
        stb = self._stb.get(room)
        streamer = self._streamer.get(room)
        if stb is not None and (streamer is None or rng.random() < person.persona.tv_source_pref):
            return "stb", stb
        if streamer is not None:
            return "streamer", streamer
        return "tv_app", None

    def _tv_off(self, ctx: TickContext, session: _TvSession, who: _Who, out: list[Intent],
                intent: str = ENTERTAINMENT) -> None:
        """Stop playback, then power the TV off and its source to standby. The next session starts
        from stopped, so its first play is a command."""
        del self._sessions[session.tv.load_id]
        self._change(ctx, session.tv, "playback", "stopped", who, out, intent)
        self._change(ctx, session.tv, "power", 0.0, who, out, intent)
        if session.source is not None:
            self._change(ctx, session.source, "power", 0.0, who, out, intent)  # the box goes to standby

    def _pauses(self, ctx: TickContext, rng, out: list[Intent]) -> None:
        for session in tuple(self._sessions.values()):
            if session.forgotten or session.resume_at is not None or session.since == ctx.minute:
                continue
            watchers = ctx.awake_in(session.room_id)
            if not watchers or rng.random() >= PAUSE_P:
                continue
            head = watchers[0]
            who = (head.person_id, head.activity, "timer")
            session.resume_at = ctx.minute + int(rng.integers(PAUSE_MIN, PAUSE_MAX + 1))
            session.paused_by = who
            self._change(ctx, session.tv, "playback", "paused", who, out)

    def _timers(self, ctx: TickContext, out: list[Intent]) -> None:
        for session in tuple(self._sessions.values()):
            if session.forgotten and session.off_at is not None and ctx.minute >= session.off_at:
                self._tv_off(ctx, session, _NO_ONE, out, ENERGY_SAVING)  # no resume: one playback intent
            elif session.resume_at is not None and ctx.minute >= session.resume_at:
                session.resume_at = None
                self._change(ctx, session.tv, "playback", "playing", session.paused_by or _NO_ONE, out)

    def _session_in(self, room_id: str | None) -> _TvSession | None:
        if room_id is None:
            return None
        tv = self._tv.get(room_id)
        return None if tv is None else self._sessions.get(tv.load_id)

    def _watching(self, ctx: TickContext, person: PersonNow) -> bool:
        return any(q.room_id == person.room_id and q.activity == TV_ACTIVITY and not q.asleep
                   and q.person_id != person.person_id for q in ctx.people)

    # --- music -----------------------------------------------------------------

    def _music_start(self, ctx: TickContext, person: PersonNow, rng, out: list[Intent]) -> None:
        room = person.room_id
        audio = self._audio.get(room)
        if audio is None or room in self._music:
            return
        if rng.random() >= person.persona.music_affinity:
            return
        who = (person.person_id, person.activity, "activity_start")
        self._change(ctx, audio, "power", 1.0, who, out)
        self._change(ctx, audio, "source", _music_source(rng), who, out)
        self._change(ctx, audio, "volume", person.persona.volume_pref, who, out)
        self._change(ctx, audio, "playback", "playing", who, out)
        self._music[room] = audio

    # --- home theatre ----------------------------------------------------------

    def _plan_theatre(self, ctx: TickContext, rng) -> None:
        """Once per local day, on weekend and festival days only."""
        self._theatre_plan = None
        if ctx.day_type == DayType.workday:
            return
        if rng.random() >= THEATRE_P:
            return
        start = ctx.minute + int(rng.integers(THEATRE_START_MIN, THEATRE_START_MAX))
        length = int(rng.integers(THEATRE_LEN_MIN, THEATRE_LEN_MAX + 1))
        self._theatre_plan = (start, length)

    def _theatre_step(self, ctx: TickContext, out: list[Intent]) -> None:
        if self._theatre_end is not None and ctx.minute >= self._theatre_end[0]:
            owner = self._theatre_end[1]
            self._theatre_end = None
            self._theatre_off(ctx, (owner, "none", "schedule"), out)
        if self._theatre_plan is not None and ctx.minute >= self._theatre_plan[0]:
            start, length = self._theatre_plan
            self._theatre_plan = None
            awake = [p for p in ctx.people if p.room_id is not None and not p.asleep]
            if awake:
                head = awake[0]
                self._theatre_end = (start + length, head.person_id)
                self._theatre_on(ctx, head, out)

    def _theatre_on(self, ctx: TickContext, head: PersonNow, out: list[Intent]) -> None:
        who = (head.person_id, head.activity, "schedule")
        rec = self._theatre["media.av_receiver"]
        proj = self._theatre["media.projector"]
        screen = self._theatre["media.screen_motorised"]
        self._change(ctx, rec, "power", 1.0, who, out)
        self._change(ctx, rec, "source", THEATRE_SOURCE, who, out)
        self._change(ctx, rec, "volume", head.persona.volume_pref, who, out)
        self._change(ctx, proj, "power", 1.0, who, out)
        self._change(ctx, proj, "source", THEATRE_SOURCE, who, out)
        self._change(ctx, screen, "position", 1.0, who, out)  # screen down

    def _theatre_off(self, ctx: TickContext, who: _Who, out: list[Intent]) -> None:
        rec = self._theatre["media.av_receiver"]
        proj = self._theatre["media.projector"]
        screen = self._theatre["media.screen_motorised"]
        self._change(ctx, proj, "power", 0.0, who, out)
        self._change(ctx, rec, "power", 0.0, who, out)
        self._change(ctx, screen, "position", 0.0, who, out)  # screen rolled up

    # --- intents ---------------------------------------------------------------

    def _change(self, ctx: TickContext, load: Load, capability: str, value: float | str, who: _Who,
                out: list[Intent], intent: str = ENTERTAINMENT) -> None:
        """Send a command only when the device does not already hold the value."""
        current = ctx.devices.get(load.load_id, capability)
        if capability == "playback" and not ctx.devices.is_on(load.load_id):
            current = "stopped"  # a set that is off holds no playback, whatever it last played
        if isinstance(value, str):
            if current == value:
                return
        elif abs(float(current) - value) <= EPS:
            return
        person_id, activity, trigger = who
        out.append(Intent(
            load_id=load.load_id, capability=capability, value=value, person_id=person_id,
            activity=activity, intent=intent, trigger=trigger,
            device_action=_action(capability, value), policy=self.name, source=None,
        ))

    def _earliest(self) -> int | None:
        times: list[int] = []
        if self._theatre_plan is not None:
            times.append(self._theatre_plan[0])
        if self._theatre_end is not None:
            times.append(self._theatre_end[0])
        for session in self._sessions.values():
            if session.resume_at is not None:
                times.append(session.resume_at)
            if session.off_at is not None:
                times.append(session.off_at)
        return min(times, default=None)

    # --- setup -----------------------------------------------------------------

    def _bind(self, ctx: TickContext) -> None:
        manifest = ctx.manifest
        if manifest is self._manifest:
            return
        self._manifest = manifest
        tv: dict[str, Load] = {}
        stb: dict[str, Load] = {}
        streamer: dict[str, Load] = {}
        audio: dict[str, Load] = {}
        by_room: dict[str, dict[str, Load]] = {}
        for ld in manifest.loads:
            lc = ld.load_class
            if lc == TV_CLASS:
                tv.setdefault(ld.room_id, ld)
            elif lc == STB_CLASS:
                stb.setdefault(ld.room_id, ld)
            elif lc == STREAMER_CLASS:
                streamer.setdefault(ld.room_id, ld)
            elif lc in AUDIO_CLASSES:
                audio.setdefault(ld.room_id, ld)
            elif lc in THEATRE_CLASSES:
                by_room.setdefault(ld.room_id, {}).setdefault(lc, ld)
        self._tv, self._stb, self._streamer, self._audio = tv, stb, streamer, audio
        self._theatre_room = None
        self._theatre = {}
        for room_id, found in by_room.items():
            if all(cls in found for cls in THEATRE_CLASSES):
                self._theatre_room = room_id
                self._theatre = {cls: found[cls] for cls in THEATRE_CLASSES}
                break
        self._activity = {p.person_id: p.activity for p in ctx.people}
