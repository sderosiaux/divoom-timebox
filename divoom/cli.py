"""Interactive shell for poking at a Timebox Evo."""

import argparse
import sys
import time

from PIL import Image

from . import effects, faces, player, proto, vision
from .events import EventFeed
from .bridge import BridgeError, Link, paired, services
from .canvas import Canvas, load_image, parse_color, reduce_palette, render_text, white_balance

HELP = """
  fill <colour>              paint the whole screen
  px <x> <y> <colour>        set one pixel (0,0 = top-left), then resend
  clear                      black screen
  checker [a] [b] [size]     checkerboard
  gradient                   RGB ramp, exercises a large palette
  palette <n>                cycle n distinct colours (probes bits-per-pixel)
  bisect                     find the palette size where decoding breaks
  border <colour>            outline only, shows if edges are cut off
  img <path>                 png / jpg / gif (animated gifs are sent as animation)
  text <message>             scrolling text
  bright <0-100>             backlight
  light <colour> [0-100]     built-in lamp mode
  clock                      back to the clock face
  score <red> <blue>         scoreboard
  ping                       ask the device which view it is showing
  raw <cmd-hex> [args-hex]   build a framed message by hand, e.g. raw 74 32
  send <hex>                 write bytes verbatim, no framing
  crc narrow|wide            checksum width once the byte sum passes 65535;
                             only shows up on large palettes (try `gradient`)
  show                       preview the local framebuffer in the terminal
  calibrate                  find the panel in the webcam frame (black vs white)
  check                      photograph the panel and diff it against what we sent
  autotest                   send every pattern and verify each one by camera
  play <effect> [seconds]    plasma fire life rain stars bounce rings
                             snake pong breakout tetris invaders pacface ghostface
                             pacman tron
                             hyperspace tunnel
                             streams circuit metro synapses
  arcade [seconds]           cycle the ten arcade / space demos (autoplay)
  networks [seconds]         cycle the four panning network styles
  demo [seconds]             one pass of all fourteen new scenes (default 5s each)
  face [who] [health]        doom · doomsprite · max · shades — driven by a TCP event feed
  pan <image> [seconds]      slow drift + zoom over an image bigger than 16x16
  logo [mode] [path] [sec]   pulse · spin · zoom · ripple · orbit · beat
  screensaver [seconds]      cycle through every effect forever
  fps <n>                    streaming rate (default 20)
  warm <0-1>                 undo the panel's cold white (default 0 = faithful)
  selftest                   same sequence, but you confirm each step by eye
  mediatest                  camera-check the image and animation paths
  effecttest                 camera-check one fixed frame of every effect
  verbose                    toggle hex logging
  help / quit
"""

PATTERNS = [
    ("rouge plein", "fill red"),
    ("vert plein", "fill green"),
    ("bleu plein", "fill blue"),
    ("blanc plein", "fill white"),
    ("damier 2px", "checker white black 2"),
    ("cadre rouge", "clear"),
    ("cadre rouge", "border red"),
    ("dégradé (palette large)", "gradient"),
    ("pixel unique en (0,0)", "clear"),
    ("pixel unique en (0,0)", "px 0 0 white"),
    ("pixel unique en (15,15)", "px 15 15 red"),
]


SETTLE = 0.8  # give the panel time to repaint before the shutter opens


class Shell:
    def __init__(self, link: Link, verbose: bool, camera: int = 0):
        self.link = link
        self.canvas = Canvas()
        self.verbose = verbose
        self.camera = camera
        self.fps = 20.0
        self.colors = 255
        self.seed = 1
        self.warm = 0.0
        self.feed: EventFeed | None = None
        link.verbose = verbose

    def prime(self) -> None:
        """Burn one frame after connecting.

        The Evo acknowledges the first image command sent on a freshly opened
        channel and then ignores it, leaving whatever was on screen before.
        """
        self.push()
        time.sleep(0.4)

    def prepare(self, pixels: list) -> list:
        return reduce_palette(white_balance(pixels, self.warm), self.colors)

    def push(self) -> None:
        self.link.send(*proto.image_messages(self.prepare(self.canvas.pixels)))

    def do(self, line: str) -> bool:
        parts = line.split()
        if not parts:
            return True
        cmd, args = parts[0].lower(), parts[1:]

        match cmd:
            case "quit" | "exit" | "q":
                return False
            case "help" | "?":
                print(HELP)
            case "verbose":
                self.verbose = self.link.verbose = not self.verbose
                print(f"verbose {'on' if self.verbose else 'off'}")
            case "show":
                print(self.canvas.preview())
            case "fill":
                self.canvas.fill(parse_color(args[0]))
                self.push()
            case "clear":
                self.canvas.fill((0, 0, 0))
                self.push()
            case "px":
                x, y = int(args[0]), int(args[1])
                self.canvas.set(x, y, parse_color(args[2]))
                self.push()
            case "checker":
                a = parse_color(args[0]) if args else (255, 255, 255)
                b = parse_color(args[1]) if len(args) > 1 else (0, 0, 0)
                self.canvas.checker(a, b, int(args[2]) if len(args) > 2 else 2)
                self.push()
            case "palette":
                self.canvas.palette_probe(int(args[0]))
                self.push()
            case "bisect":
                self.bisect()
            case "gradient":
                self.canvas.gradient()
                self.push()
            case "border":
                self.canvas.border(parse_color(args[0]))
                self.push()
            case "img":
                frames = load_image(args[0])
                if len(frames) == 1:
                    self.canvas.pixels = frames[0][0]
                    self.push()
                else:
                    self.link.send(*proto.animation_messages(frames), gap=proto.CHUNK_GAP)
                print(f"{len(frames)} frame(s)")
            case "text":
                frames = render_text(" ".join(args), (255, 255, 255))
                self.link.send(*proto.animation_messages(frames), gap=proto.CHUNK_GAP)
                print(f"{len(frames)} frame(s)")
            case "bright":
                self.link.send(proto.brightness(int(args[0])))
            case "light":
                level = int(args[1]) if len(args) > 1 else 100
                self.link.send(proto.light(parse_color(args[0]), level))
            case "clock":
                self.link.send(proto.clock())
            case "score":
                self.link.send(proto.scoreboard(int(args[0]), int(args[1])))
            case "ping":
                self.link.send(proto.get_view())
                time.sleep(0.5)
                for line in self.link.drain():
                    print(f"  {line}")
            case "raw":
                payload = bytes.fromhex(args[1]) if len(args) > 1 else b""
                self.link.send(proto.message(int(args[0], 16), payload))
            case "send":
                self.link.send(bytes.fromhex("".join(args)))
            case "crc":
                proto.WIDE_CRC = args[0].lower() == "wide"
                print(f"crc {'wide (32-bit)' if proto.WIDE_CRC else 'narrow (16-bit)'}")
            case "play":
                self.play(effects.build(args[0], self.seed),
                          float(args[1]) if len(args) > 1 else None)
            case "pan":
                self.play(effects.Pan(args[0], self.seed),
                          float(args[1]) if len(args) > 1 else None)
            case "face":
                self.face(args)
            case "logo":
                self.logo(args)
            case "screensaver":
                self.screensaver(float(args[0]) if args else 45.0)
            case "arcade":
                self.screensaver(float(args[0]) if args else 30.0, effects.ARCADE)
            case "networks":
                self.screensaver(float(args[0]) if args else 30.0, effects.NETWORKS)
            case "demo":
                self.demo(float(args[0]) if args else 5.0)
            case "warm":
                self.warm = max(0.0, min(1.0, float(args[0])))
                print(f"balance {self.warm:.0%} — 0 = fidèle, 1 = neutre à l'œil")
                self.push()
            case "fps":
                self.fps = float(args[0])
                print(f"{self.fps:g} fps")
            case "calibrate":
                self.calibrate()
            case "check":
                self.check("check")
            case "autotest":
                self.autotest()
            case "mediatest":
                self.mediatest()
            case "effecttest":
                self.effecttest()
            case "selftest":
                self.selftest()
            case _:
                print(f"unknown command {cmd!r} — try help")
        return True

    def play(self, effect: effects.Effect, seconds: float | None) -> None:
        print(f"{effect.name} à {self.fps:g} fps — Ctrl-C pour arrêter")
        try:
            result = player.play(self.link, effect, self.fps, seconds, self.colors, self.warm)
        except KeyboardInterrupt:
            print("\ninterrompu")
            return
        print(f"  {result.frames} frames en {result.seconds:.1f}s → {result.fps:.1f} fps")

    def face(self, args: list[str]) -> None:
        character = faces.CHARACTERS.get(args[0] if args else "doom")
        if character is None:
            print(f"personnages : {', '.join(sorted(faces.CHARACTERS))}")
            return
        health = int(args[1]) if len(args) > 1 else 100

        if self.feed is None:
            self.feed = EventFeed()
        print(f"événements sur 127.0.0.1:{self.feed.port} — "
              "hurt <n> · heal <n> · health <n> · evil · revive")
        print("  eg:  printf 'hurt 25\\n' | nc 127.0.0.1 8777")
        self.play(faces.Face(self.seed, character, health, self.fps, self.feed), None)

    LOGO = "assets/logo/conduktor.png"

    def logo(self, args: list[str]) -> None:
        mode, seconds = "pulse", None
        path = str(effects.Path(__file__).resolve().parent.parent / self.LOGO)
        for arg in args:
            if arg in effects.Logo.MODES:
                mode = arg
            elif arg.replace(".", "", 1).isdigit():
                seconds = float(arg)
            else:
                path = arg
        self.play(effects.Logo(path, mode, self.seed, fps=self.fps), seconds)

    def screensaver(self, each: float, registry=None) -> None:
        print(f"rotation toutes les {each:g}s — Ctrl-C pour arrêter")
        seed = self.seed
        try:
            while True:
                for name in (registry if registry is not None else sorted(effects.ABSTRACT)):
                    seed += 1
                    print(f"  {name}")
                    player.play(self.link, effects.build(name, seed), self.fps, each,
                                self.colors, self.warm)
        except KeyboardInterrupt:
            print("\ninterrompu")

    def demo(self, each: float = 5.0) -> None:
        names = list(effects.ARCADE | effects.NETWORKS)
        try:
            for index, name in enumerate(names, 1):
                print(f"  {index}/{len(names)}  {name} — {each:g}s", flush=True)
                player.play(self.link, effects.build(name, self.seed), self.fps, each,
                            self.colors, self.warm)
        except KeyboardInterrupt:
            print("\ninterrompu")

    def mediatest(self) -> None:
        """Verify the still-image and device-animation paths through the camera."""
        panel = vision.Panel.load()

        quadrants = Image.new("RGB", (64, 64))
        for i, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255))):
            quadrants.paste(color, ((i % 2) * 32, (i // 2) * 32, (i % 2) * 32 + 32, (i // 2) * 32 + 32))
        path = vision.CAPTURES / "quadrants.png"
        quadrants.save(path)

        print("\n→ img (64x64 réduit en 16x16)")
        frames = load_image(str(path))
        self.canvas.pixels = frames[0][0]
        self.push()
        time.sleep(SETTLE)
        img_score = self.check("media-img")

        # An animation cannot be judged from one photo, so make each frame last
        # long enough that a handful of shots must land on different ones.
        print("\n→ animation embarquée (4 aplats, 1.5s chacun)")
        palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        self.link.send(*proto.animation_messages([([c] * 256, 1500) for c in palette]),
                       gap=proto.CHUNK_GAP)
        seen = set()
        for shot in range(6):
            # Uneven spacing: a fixed interval aliases against the 6s loop and
            # keeps landing on the same frame.
            time.sleep(0.4 + shot * 0.45)
            grid = vision.classify_grid(vision.read(
                vision.snapshot(f"media-anim-{shot}.jpg", self.camera), panel))
            top = max(set(grid), key=grid.count)
            seen.add(top)
            print(f"  photo {shot + 1}: {top} ({grid.count(top) / len(grid):.0%})")

        print(f"\n  img         {'PASS' if img_score >= 0.9 else 'FAIL'}  {img_score:.0%}")
        ok = len(seen - {"."}) >= 2
        print(f"  animation   {'PASS' if ok else 'FAIL'}  {len(seen - {'.'})} couleurs distinctes vues")

    def effecttest(self) -> None:
        """Freeze each effect on a fixed frame and confirm the panel matches it."""
        panel = vision.Panel.load()
        results = []
        for name in sorted(effects.ABSTRACT):
            effect = effects.build(name, seed=1)
            for _ in range(30):
                pixels = effect.next()
            self.canvas.pixels = reduce_palette(pixels, self.colors)
            self.push()
            self.rebase()
            seen = vision.read(vision.snapshot(f"effect-{name}.jpg", self.camera), panel)
            score = max(vision.agreement(vision.expected_grid(self.canvas.pixels),
                                         vision.classify_grid(seen)),
                        vision.similarity(self.canvas.pixels, seen))
            results.append((name, score))
            print(f"  {name:8} {score:5.0%}")

        print("\nrésumé")
        for name, score in results:
            print(f"  {'PASS' if score >= 0.85 else 'FAIL'}  {score:5.0%}  {name}")

    def calibrate(self) -> None:
        # Calibration compares colour channels, so it needs the panel at full
        # output: left dim by an earlier command, the room's reflection on the
        # glass outshines the LEDs and no pixel passes the dominance test.
        self.link.send(proto.brightness(100))
        time.sleep(0.4)

        shots = []
        for name, color in (("red", (255, 0, 0)), ("green", (0, 255, 0)),
                            ("blue", (0, 0, 255)), ("white", (255, 255, 255))):
            self.canvas.fill(color)
            self.push()
            time.sleep(SETTLE)
            shots.append(vision.snapshot(f"calib-{name}.jpg", self.camera))

        self.canvas.fill((0, 0, 0))
        self.push()
        time.sleep(SETTLE)
        dark = vision.snapshot("calib-dark.jpg", self.camera)

        panel = vision.locate(*shots, dark)
        panel.save()
        left, top, right, bottom = panel.box
        w, h = right - left, bottom - top
        print(f"panneau à {panel.box}, {w}x{h}px → pas {w / 15:.1f}x{h / 15:.1f}px, rayon {panel.radius}px")
        if min(w, h) < 16 * 4:
            print("! très petit ; rapproche la caméra ou l'écran si les lectures sont bruitées")

    def rebase(self) -> None:
        """Re-measure the black level right before a reading.

        The panel is glossy and mirrors the room, so the dark reference taken at
        calibration goes stale as soon as anything moves or the light shifts.
        Two frames seconds apart drift far less than a whole session does.
        """
        panel = vision.Panel.load()
        before = self.canvas.pixels
        self.canvas.fill((0, 0, 0))
        self.push()
        time.sleep(SETTLE)
        panel.black = [list(c) for c in vision.sample(
            vision.snapshot("rebase.jpg", self.camera), panel)]
        panel.save()
        self.canvas.pixels = before
        self.push()
        time.sleep(SETTLE)

    def check(self, label: str) -> float:
        panel = vision.Panel.load()
        shot = vision.snapshot(f"{label}.jpg", self.camera)
        seen = vision.read(shot, panel)
        expected = vision.expected_grid(self.canvas.pixels)
        observed = vision.classify_grid(seen)
        structure = vision.agreement(expected, observed)
        colour = vision.similarity(self.canvas.pixels, seen)

        left, right = vision.render(expected).splitlines(), vision.render(observed).splitlines()
        print("  attendu           vu")
        for a, b in zip(left, right):
            print(f"  {a}  {b}")
        print(f"  structure {structure:.0%} · couleur {colour:.0%}")
        return max(structure, colour)

    def bisect(self) -> None:
        """Find the palette size at which the device stops decoding our images."""
        vision.Panel.load()
        for size in (2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 64, 65, 128, 129, 255, 256):
            bits = max(1, (size - 1).bit_length())
            self.canvas.palette_probe(size)
            self.push()
            time.sleep(SETTLE)
            panel = vision.Panel.load()
            seen = vision.read(vision.snapshot(f"bisect-{size:03d}.jpg", self.camera), panel)
            score = max(vision.agreement(vision.expected_grid(self.canvas.pixels),
                                         vision.classify_grid(seen)),
                        vision.similarity(self.canvas.pixels, seen))
            print(f"  {size:3d} couleurs / {bits} bit(s)  {score:5.0%}  {'ok' if score >= 0.9 else 'CASSE'}")

    def autotest(self) -> None:
        vision.Panel.load()  # fail fast if we never calibrated
        results = []
        for step, (label, command) in enumerate(PATTERNS):
            self.do(command)
            if command.startswith("clear"):
                continue
            print(f"\n→ {label}")
            self.rebase()
            score = self.check(f"auto-{step:02d}")
            results.append((label, score))

        print("\nrésumé")
        for label, score in results:
            print(f"  {'PASS' if score >= 0.90 else 'FAIL'}  {score:5.0%}  {label}")

    def selftest(self) -> None:
        for label, command in PATTERNS:
            print(f"\n→ {label}   ({command})")
            self.do(command)
            if command.startswith(("clear",)):
                continue
            input("   appuie sur Entrée quand tu as regardé l'écran (Ctrl-C pour arrêter) ")
        print("\nséquence terminée")


def pick_device(hint: str | None) -> tuple[str, str]:
    rows = paired()
    if not rows:
        raise SystemExit("aucun appareil appairé")

    if hint:
        matches = [r for r in rows if hint.lower() in r[0].lower() or hint.lower() in r[2].lower()]
    else:
        matches = [r for r in rows if any(k in r[2].lower() for k in ("divoom", "timebox", "evo"))]

    if len(matches) == 1:
        return matches[0][0], matches[0][2]
    if not matches:
        print("appareils appairés :")
        for address, state, name in rows:
            print(f"  {address}  {state:9}  {name}")
        raise SystemExit("aucun Timebox trouvé — appaire-le puis relance, ou passe --mac")
    print("plusieurs candidats :")
    for address, state, name in matches:
        print(f"  {address}  {state:9}  {name}")
    raise SystemExit("précise avec --mac")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="divoom", description="Talk to a Divoom Timebox Evo over RFCOMM")
    ap.add_argument("--mac", help="device address or a substring of its name")
    ap.add_argument("--channel", type=int, default=1, help="RFCOMM channel (default 1)")
    ap.add_argument("--keep-audio", action="store_true",
                    help="do not drop the existing connection first")
    ap.add_argument("-v", "--verbose", action="store_true", help="log every frame in hex")
    ap.add_argument("--camera", type=int, default=0, help="webcam index for calibrate/check/autotest")
    ap.add_argument("command", nargs="*", help="run once and exit instead of opening the shell")
    ap.add_argument("--list", action="store_true", help="list paired devices and exit")
    ap.add_argument("--services", action="store_true", help="list the device's RFCOMM channels and exit")
    opts = ap.parse_args(argv)

    if opts.list:
        for address, state, name in paired():
            print(f"{address}  {state:9}  {name}")
        return 0

    address, name = pick_device(opts.mac)

    if opts.services:
        print(f"{name} ({address})")
        for line in services(address):
            print(f"  {line}")
        return 0

    print(f"connexion à {name} ({address}) sur le canal {opts.channel}…")
    try:
        link = Link(address, opts.channel, opts.keep_audio, opts.verbose)
    except BridgeError as exc:
        print(f"échec : {exc}", file=sys.stderr)
        print("pistes : essaie --services pour voir les canaux annoncés, ou --channel 2/4",
              file=sys.stderr)
        return 1
    print(f"canal ouvert, MTU {link.mtu}. `help` pour les commandes, `selftest` pour tout dérouler.")

    shell = Shell(link, opts.verbose, opts.camera)
    shell.prime()
    try:
        if opts.command:
            shell.do(" ".join(opts.command))
            return 0
        while True:
            try:
                line = input("divoom> ")
            except EOFError:
                break
            try:
                if not shell.do(line):
                    break
            except (BridgeError, vision.VisionError, ValueError, IndexError, OSError) as exc:
                print(f"! {exc}")
    except KeyboardInterrupt:
        pass
    finally:
        link.close()
    return 0
