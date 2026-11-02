import numpy as np
from unittest.mock import patch

from shadow_ai.soliton_security import SecurityGuard, line_clear
from test_interactive_tactics import game


def stage(game, count=5):
    game.env.grid[:] = 0
    game.lead_alive = False
    game.aux_guards = [SecurityGuard(f'G{index + 1}',
                        np.array([5. + index * 5., 10.], dtype=np.float32),
                        0., ((5., 10.), (7., 12.))) for index in range(count)]
    game.swat_units = []
    game.env.episode_has_intel[:] = 1
    game.env.known_intel_present[:] = 1
    game.env.player_has_intel[:] = 0
    game.env._intel = [np.array(point, dtype=np.float32)
                       for point in [(10., 10.), (30., 10.), (50., 10.)]]
    return game.aux_guards


def test_sentries_spread_between_sites_and_keep_objects_clear(game):
    guards = stage(game)
    posts = [game._security_assignment(guard) for guard in guards]
    assert {post['key'] for post in posts} == {'INTEL:0', 'INTEL:1', 'INTEL:2'}
    for index, post in enumerate(posts):
        assert post['state'] == 'PROTECT INTEL'
        assert np.linalg.norm(post['goal'] - post['anchor']) >= .9
        assert line_clear(game.env.grid, post['goal'], post['anchor'])
        assert all(np.linalg.norm(post['goal'] - other['goal']) >= 1.1
                   for other in posts[:index])
    # Caller update order and changing physical positions do not swap watches.
    for guard, post in reversed(list(zip(guards, posts))):
        guard.pos += np.array([1., 2.], dtype=np.float32)
        again = game._security_assignment(guard)
        assert again['key'] == post['key']
        assert np.array_equal(again['goal'], post['goal'])


def test_perimeter_watch_patrols_and_remains_separated(game):
    guards = stage(game, count=8)
    game.env.episode_has_intel[:] = [1, 0, 0]
    before = [game._security_assignment(guard) for guard in guards]
    assert all(post is not None for post in before)
    game.env.steps += int(8.1 / game.game_dt)
    after = [game._security_assignment(guard) for guard in guards]
    for old, new in zip(before, after):
        assert old['key'] == new['key']
        assert np.linalg.norm(old['goal'] - new['goal']) > 1.
    for index, post in enumerate(after):
        assert all(np.linalg.norm(post['goal'] - other['goal']) >= 1.1
                   for other in after[:index])


def test_exfil_is_only_watched_after_every_theft_is_known(game):
    guards = stage(game)
    game.env.player_has_intel[:] = 1
    assert all(game._security_assignment(guard)['state'] == 'PROTECT INTEL'
               for guard in guards)
    game.env.known_intel_present[:] = [0, 1, 0]
    assert all(game._security_assignment(guard)['key'] == 'INTEL:1' for guard in guards)
    game.env.known_intel_present[:] = 0
    posts = [game._security_assignment(guard) for guard in guards]
    assert all(post['key'] == 'EXFIL' and post['state'] == 'SECURE EXFIL' for post in posts)
    assert len({tuple(post['goal']) for post in posts}) == len(guards)


def test_medical_watch_uses_known_wound_and_only_available_stations(game):
    guards = stage(game)
    game.player_vitals.health = .2
    assert not any(key.startswith('MEDKIT:') for key, _, _ in game._security_sites())
    game.medical_watch.hit(guards[0].guard_id)
    stations = [site for site in game._security_sites() if site[0].startswith('MEDKIT:')]
    assert stations
    assert any(game._security_assignment(guard)['state'] == 'WATCH MEDKIT' for guard in guards)
    key = stations[0][0].split(':', 1)[1]
    game.medical_watch.consume(key)
    assert all(site[0] != f'MEDKIT:{key}' for site in game._security_sites())
    assert all(game._security_assignment(guard)['key'] != f'MEDKIT:{key}' for guard in guards)


def test_swat_only_fills_uncovered_intel_with_one_member_per_site(game):
    stage(game, count=1)
    swat = [SecurityGuard(f'S{index}', np.array([20., 11. + index], dtype=np.float32),
                          0., ((20., 11.),)) for index in range(3)]
    game.swat_units = [('ASSAULT', guard) for guard in swat]
    regular_post = game._security_assignment(game.aux_guards[0])
    posts = [game._swat_security_assignment(guard) for guard in swat]
    assigned = [post for post in posts if post is not None]
    assert len(assigned) == 2
    assert len({post['key'] for post in assigned}) == 2
    assert all(post['key'] != regular_post['key'] for post in assigned)
    # New regular coverage releases SWAT support instead of doubling the watch.
    for index in range(2):
        game.aux_guards.append(SecurityGuard(f'G{index + 2}', np.array([20., 10.], dtype=np.float32),
                                             0., ((20., 10.),)))
    assert all(game._swat_security_assignment(guard) is None for guard in swat)


def test_dead_or_treating_sentries_do_not_claim_security_posts(game):
    guards = stage(game, count=3)
    guards[0].unconscious = True
    guards[1].state = 'TREATMENT'
    assert game._security_assignment(guards[0]) is None
    assert game._security_assignment(guards[1]) is None
    assert game._security_assignment(guards[2]) is not None


def test_post_selection_keeps_walls_and_does_not_fall_back_to_pile_on_object(game):
    guards = stage(game, count=5)
    game.env.episode_has_intel[:] = [1, 0, 0]
    game.env.grid[:] = 1
    game.env.grid[10, 5:14] = 0
    for guard in guards:
        guard.pos = np.array([8., 10.], dtype=np.float32)
    posts = [game._security_assignment(guard) for guard in guards]
    valid = [post for post in posts if post is not None]
    assert valid
    for index, post in enumerate(valid):
        assert line_clear(game.env.grid, post['goal'], post['anchor'])
        assert np.linalg.norm(post['goal'] - post['anchor']) >= .9
        assert all(np.linalg.norm(post['goal'] - other['goal']) >= 1.1 for other in valid[:index])


def test_swat_support_is_finite_and_then_releases_post(game):
    stage(game, count=0)
    guard = SecurityGuard('S0', np.array([8., 10.], dtype=np.float32), 0., ((8., 10.),))
    game.swat_units = [('ASSAULT', guard)]
    game._finish_swat_security_search(guard)
    assert guard.state == 'SUPPORT INTEL'
    assert guard._security_support_remaining == 12.
    position = guard.pos.copy()
    for _ in range(int(12. / game.game_dt) + 2):
        game._step_swat_security_support(guard)
    assert guard.state == 'WITHDRAW'
    assert np.array_equal(position, guard.pos)
    assert guard.guard_id not in game._swat_security_posts
    game._finish_swat_security_search(guard)
    assert guard.state == 'WITHDRAW'


def test_returning_regular_releases_swat_support_before_timeout(game):
    stage(game, count=0)
    game.env.episode_has_intel[:] = [1, 0, 0]
    guard = SecurityGuard('S0', np.array([8., 10.], dtype=np.float32), 0., ((8., 10.),))
    game.swat_units = [('ASSAULT', guard)]
    game._finish_swat_security_search(guard)
    assert guard.state == 'SUPPORT INTEL'
    game.aux_guards = [SecurityGuard('G1', np.array([7., 10.], dtype=np.float32), 0., ((7., 10.),))]
    assert game._step_swat_security_support(guard)
    assert guard.state == 'WITHDRAW'
    assert guard.guard_id not in game._swat_security_posts


def test_reset_clears_roster_allocations_and_swat_claims(game):
    stage(game, count=1)
    game._security_assignment(game.aux_guards[0])
    guard = SecurityGuard('S0', np.array([8., 10.], dtype=np.float32), 0., ((8., 10.),))
    game.swat_units = [('ASSAULT', guard)]
    assert game._swat_security_assignment(guard) is not None
    assert game._security_posts and game._swat_security_posts
    game._reset_security_distribution()
    assert game._security_roster is None
    assert not game._security_posts and not game._swat_security_posts
    assert game._security_geometry is None and game._security_phase is None
    assert not game._security_layouts and not game._security_perimeters


def test_withdrawing_swat_does_not_reserve_support_for_another_member(game):
    stage(game, count=0)
    game.env.episode_has_intel[:] = [1, 0, 0]
    first = SecurityGuard('S0', np.array([8., 10.], dtype=np.float32), 0., ((8., 10.),))
    second = SecurityGuard('S1', np.array([7., 10.], dtype=np.float32), 0., ((7., 10.),))
    game.swat_units = [('ASSAULT', first), ('ASSAULT', second)]
    assert game._swat_security_assignment(first) is not None
    assert game._swat_security_assignment(second) is None
    first.state = 'WITHDRAW'
    assert game._swat_security_assignment(second) is not None
    assert first.guard_id not in game._swat_security_posts


def test_perimeter_geometry_is_reused_for_one_hundred_frames(game):
    guards = stage(game, count=7)
    game.env.episode_has_intel[:] = [1, 0, 0]
    original = [game._security_assignment(guard) for guard in guards]
    with patch('shadow_ai.security_distribution.line_clear', wraps=line_clear) as clear, \
            patch.object(game.navigator, 'path_length', wraps=game.navigator.path_length) as path:
        for _ in range(100):
            game.env.steps += 1
            posts = [game._security_assignment(guard) for guard in guards]
            assert all(np.array_equal(before['goal'], after['goal'])
                       for before, after in zip(original, posts))
        assert clear.call_count == 0
        assert path.call_count == 0
    original[0]['goal'][:] = 0
    original[0]['anchor'][:] = 0
    again = game._security_assignment(guards[0])
    assert np.any(again['goal']) and np.any(again['anchor'])


def test_wall_change_invalidates_cached_post_during_same_tick(game):
    guard = stage(game, count=1)[0]
    game.env.episode_has_intel[:] = [1, 0, 0]
    before = game._security_assignment(guard)
    steps = game.env.steps
    x, y = game.env._cell(before['goal'])
    game.env.grid[y, x] = 1
    after = game._security_assignment(guard)
    assert game.env.steps == steps
    assert after is not None
    assert not np.array_equal(after['goal'], before['goal'])
    assert line_clear(game.env.grid, after['anchor'], after['goal'])


def test_unavailable_perimeter_slot_is_cached_until_geometry_changes(game):
    guard = stage(game, count=1)[0]
    game.env.episode_has_intel[:] = [1, 0, 0]
    guard.pos = np.array([10., 10.], dtype=np.float32)
    game.env.grid[:] = 1
    game.env.grid[10, 10] = 0
    assert game._security_assignment(guard) is None
    with patch.object(game.navigator, 'goal', wraps=game.navigator.goal) as goal:
        for _ in range(20):
            assert game._security_assignment(guard) is None
        assert goal.call_count == 0
    game.env.grid[10, 11:14] = 0
    assert game._security_assignment(guard) is not None
