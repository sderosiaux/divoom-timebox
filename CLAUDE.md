# CLAUDE.md

## Stop the agent before touching the panel

A launchd agent may be driving the device right now. It does not refuse a second
connection, it fights it: every connection drops the ACL link first to get the
channel back from macOS audio, so the two processes take turns kicking each
other off and the panel flickers.

```bash
./install-agent.sh stop        # before any manual run
./install-agent.sh start       # after
launchctl print gui/$UID/io.conduktor.divoom | grep pid   # is it running?
tail ~/Library/Logs/io.conduktor.divoom.log
```

## Running things

`.venv/bin/python` — there is no active virtualenv in the shell. `./build.sh`
recompiles the two Swift binaries (`bin/divoom-bridge`, `bin/snapshot`); they are
not versioned.

```bash
.venv/bin/python divoom.py               # interactive shell
.venv/bin/python divoom.py fill red      # one-shot
.venv/bin/python -m pytest tests/ -q
```

The differential tests skip unless `ref/` holds clones of divo and hass-divoom
(the README has the commands). Skipped, not failed — check the count.

## Three device behaviours that will bite

They are documented in the README with measurements. In code:

- Palettes cap at **255** colours, not 256. `proto.frame_body` raises above that;
  call `canvas.reduce_palette` first.
- Animation chunks need **pacing**. Send them with `gap=proto.CHUNK_GAP` or the
  device drops the tail and falls back to the clock, having acked every write.
- The **first image** on a fresh channel is acked and ignored. Send a throwaway
  frame after connecting.

## Never pick a colour by eye

This cost three separate rounds of rework. Tones chosen on a monitor are wrong on
the panel, and not slightly: twice, the tone picked by eye ranked second-to-last
once measured. Contrast on LEDs comes from saturation as much as brightness, and
the panel blooms, so neighbouring tones stop being distinguishable.

Measure instead. Send alternating bands of the two tones, read them back, and
compare:

```python
frame = []
for y in range(16):
    frame += [SKIN if (y // 2) % 2 == 0 else candidate] * 16
```

Sanity-check every measurement: if the reference tone reads identically across
several candidates, the panel coordinates are stale and you are sampling the
background. I published a bogus table once by skipping that check.

## Do not resample artwork down to 16x16

Downscaling a sprite or a logo looks like the obvious move and produces mush: a
24x29 face carries shading the panel cannot resolve, and a geometric mark turns
into a smudge. Two approaches that work:

- Redraw at 16x16, using the source only for proportions (`faces.DoomGuy`).
- Render by **coverage**, not resampling (`effects.Logo`). Supersample, measure
  what fraction of each LED the shape covers, and use that fraction as
  brightness. A hard on/off threshold makes the shape jump a whole LED at a time,
  which reads as stepping rather than motion.

`faces.SpriteFace` is the rejected downscaling approach, kept as `doomsprite` for
comparison. Do not make it the default again.

## Where things go

```
effects.py   any class with .next() -> list[RGB]; register in ABSTRACT
             Logo.MODES for a new logo animation
faces.py     AsciiFace for hand-drawn (16-char rows), SpriteFace for sheets
             register in CHARACTERS
serve.py     segments() decides what the unattended agent cycles through
vision.py    the camera loop
```

## The camera loop

`calibrate` before any measurement, and again whenever the device or camera
moves. It forces full brightness itself, because it compares colour channels and
a dim panel loses to the room's reflection on the glass.

Calibration fails with `0 px followed red→green→blue` when the panel is out of
frame or occluded. Check with a snapshot before debugging anything else.

The camera clips bright LEDs to white, so it cannot judge skin tones or anything
near white. It is reliable for geometry, for lit/unlit, and for comparing two
tones against each other. For aesthetics, ask.
