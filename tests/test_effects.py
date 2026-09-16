"""Autoplay must remain valid and animated across collisions and round changes."""

from types import SimpleNamespace

import numpy as np
import pytest

from divoom import effects, proto, serve
from divoom.canvas import reduce_palette
from divoom.cli import Shell

NEW = effects.ARCADE | effects.NETWORKS


@pytest.mark.parametrize("name", NEW)
def test_reproducible_animation_and_wire_format(name):
    first, second = effects.build(name, 17), effects.build(name, 17)
    seen = set()
    for tick in range(300):
        pixels = first.next()
        assert pixels == second.next()
        assert len(pixels) == proto.PIXELS
        assert all(len(pixel) == 3 and all(type(v) is int and 0 <= v <= 255 for v in pixel)
                   for pixel in pixels)
        seen.add(tuple(pixels))
        if tick % 15 == 0:
            assert proto.image_messages(reduce_palette(pixels))
    assert len(seen) > 10, f"{name} stopped animating"


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_snake_eats_without_overlapping_or_leaving_board(seed):
    snake = effects.Snake(seed)
    longest = 0
    for _ in range(1800):
        snake.next()
        assert len(snake.body) == len(set(snake.body))
        assert set(snake.body) <= snake.cells
        assert snake.food not in snake.body
        assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
                   for a, b in zip(snake.body, snake.body[1:]))
        longest = max(longest, len(snake.body))
    assert longest >= 12


def test_tetris_clears_a_line_and_drops_the_remaining_blocks():
    game = effects.Tetris()
    game.board[14, :9] = 1
    game.piece = [(0, y) for y in range(4)]
    game.x, game.y, game.color = 9, 11, 2
    game._lock()
    assert game.lines == 1
    assert game.clearing == [14]
    for _ in range(8):
        game.next()
    assert not game.clearing
    assert np.count_nonzero(game.board) == 3
    assert list(game.board[12:, 9]) == [2, 2, 2]


def test_tetris_autoplay_actually_completes_lines():
    game = effects.Tetris(9)
    for _ in range(1800):
        game.next()
    assert game.lines >= 10


def test_pong_scores_and_serves_after_a_miss():
    game = effects.Pong()
    game.pause = 0
    game.x, game.y, game.dx, game.dy = 15, 3, 0.4, 0.1
    game.paddles[1] = 14
    game.next()
    assert game.scores == [1, 0]
    assert game.pause > 0
    assert game.x == 7.5


def test_pong_does_not_catch_a_ball_behind_the_paddle():
    game = effects.Pong()
    game.pause = 0
    game.x, game.y, game.dx, game.dy = 14.2, 8, 0.3, 0.1
    game.paddles[1] = game.targets[1] = 8
    game.next()
    assert game.dx > 0
    assert game.x > 14.2


def test_pong_bounces_once_when_crossing_a_paddle():
    game = effects.Pong()
    game.pause = 0
    game.x, game.y, game.dx, game.dy = 13.8, 8, 0.3, 0.1
    game.paddles[1] = game.targets[1] = 8
    game.next()
    assert game.dx < 0
    assert game.x < 14
    game.next()
    assert game.dx < 0


def test_breakout_removes_the_brick_it_hits():
    game = effects.Breakout()
    game.pause = 0
    game.x, game.y, game.dx, game.dy = 1, 7.6, 0, -0.3
    game.next()
    assert (0, 3) not in game.bricks
    assert len(game.bricks) == 15
    assert game.dy > 0


def test_breakout_restarts_after_a_clear():
    game = effects.Breakout()
    game.bricks.clear()
    game.pause = 1
    game.next()
    assert len(game.bricks) == 16


def test_breakout_preserves_progress_when_a_ball_is_lost():
    game = effects.Breakout()
    game.bricks.remove((0, 0))
    game.pause = 0
    game.x, game.y, game.dx, game.dy = 0, 15.1, 0, 0.3
    game.paddle = 13
    game.next()
    assert len(game.bricks) == 15
    assert (0, 0) not in game.bricks
    assert game.pause > 0


def test_tetris_moves_and_rotates_through_valid_poses():
    game = effects.Tetris(1)
    moved = rotated = False
    for _ in range(500):
        previous_x, previous_y, previous_piece = game.x, game.y, game.piece[:]
        previous_board = game.board.copy()
        game.next()
        if game.piece:
            assert game._fits(game.piece, game.x, game.y)
        if game.piece and np.array_equal(previous_board, game.board) and game.y >= previous_y:
            moved |= game.x != previous_x
            rotated |= sorted(game.piece) != sorted(previous_piece)
    assert moved and rotated


def test_tron_has_four_riders_and_trails_expire():
    game = effects.Tron(1)
    assert len(game.heads) == 4
    sentinel = (0, 0)
    game.trails[sentinel] = 0
    game.ages[sentinel] = 1
    game.next()
    game.next()
    assert sentinel not in game.trails
    assert game.ages.keys() == game.trails.keys()


def test_invaders_starts_a_new_wave_after_a_clear():
    game = effects.Invaders()
    game.aliens.clear()
    game.next()
    assert game.pause > 0
    for _ in range(game.pause):
        game.next()
    assert len(game.aliens) == 9
    assert not game.shots and not game.bombs


def test_invader_shot_hits_the_sprite_edge():
    game = effects.Invaders()
    game.tick = 1
    game.shots = [(4, 3)]
    game.next()
    assert (0, 0) not in game.aliens
    assert len(game.aliens) == 8
    assert not game.shots


def test_invaders_keeps_wave_progress_after_a_ship_hit():
    game = effects.Invaders()
    game.shield = 0
    game.aliens.pop()
    remaining = game.aliens[:]
    game.bombs = [(game.ship, 14)]
    game.next()
    assert game.pause > 0
    for _ in range(game.pause):
        game.next()
    assert game.aliens == remaining
    assert not game.bombs
    assert game.shield > 0


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_invaders_wave_does_not_finish_in_the_first_eight_seconds(seed):
    game = effects.Invaders(seed)
    for _ in range(160):
        game.next()
        assert game.aliens
        assert not game.restart
        assert all(0 <= x + game.offset <= 16 - game.WIDTH for x, y in game.aliens)


@pytest.mark.parametrize("name", ["pacman", "tron", "invaders", "breakout", "pong"])
def test_long_run_crosses_rounds_without_stalling(name):
    game = effects.build(name, 31)
    recent = set()
    for tick in range(3000):
        pixels = game.next()
        if tick >= 2800:
            recent.add(tuple(pixels))
        if name == "pacman":
            assert game.hero in game.cells
            assert set(game.ghosts) <= game.cells
        elif name == "tron":
            assert len(game.trails) <= proto.PIXELS
            assert all(0 <= x < 16 and 0 <= y < 16 for x, y in game.trails)
        elif name == "invaders":
            assert len(game.shots) < 16 and len(game.bombs) < 16
    assert len(recent) > 20


@pytest.mark.parametrize("collection", ["arcade", "networks", "screensaver"])
def test_service_rotation_contains_every_registered_effect(collection):
    registry = {"arcade": effects.ARCADE, "networks": effects.NETWORKS,
                "screensaver": effects.ABSTRACT}[collection]
    stream = serve.segments(SimpleNamespace(what=collection, each=5))
    names = []
    for _ in registry:
        effect, duration = next(stream)
        names.append(effect.name)
        assert duration == 5
    assert set(names) == set(registry)
    assert next(stream)[0].name == names[0]


def test_single_effect_keeps_state_across_heartbeat_spans():
    stream = serve.segments(SimpleNamespace(what="play", mode="snake"))
    first, duration = next(stream)
    for _ in range(30):
        first.next()
    second, _ = next(stream)
    assert first is second
    assert second.tick == 30
    assert duration is None


@pytest.mark.parametrize("command, registry", [("arcade", effects.ARCADE), ("networks", effects.NETWORKS)])
def test_shell_dispatches_collection(monkeypatch, command, registry):
    shell = Shell(SimpleNamespace(verbose=False), False)
    calls = []
    monkeypatch.setattr(shell, "screensaver", lambda *args: calls.append(args))
    shell.do(f"{command} 5")
    assert calls == [(5.0, registry)]


def test_demo_plays_every_new_scene_once(monkeypatch):
    from divoom import player

    shell = Shell(SimpleNamespace(verbose=False), False)
    calls = []
    monkeypatch.setattr(player, "play", lambda link, effect, fps, seconds, *args:
                        calls.append((effect.name, seconds)))
    shell.do("demo 5")
    assert calls == [(name, 5.0) for name in NEW]


def test_mix_advances_effects_after_each_logo(monkeypatch):
    modes = effects.Logo.MODES
    monkeypatch.setattr(effects, "Logo", lambda *args, **kwargs: SimpleNamespace(name="logo"))
    effects.Logo.MODES = modes
    stream = serve.segments(SimpleNamespace(what="mix", path=None, fps=20, each=5))
    names = []
    for _ in effects.ABSTRACT:
        assert next(stream)[0].name == "logo"
        names.append(next(stream)[0].name)
    assert set(names) == set(effects.ABSTRACT)
