"""Train and evaluate optional tactical policies against the playable mechanics.

The existing guard/SWAT movement models remain intact. The new policies learn
discretionary medical, cover and coordination decisions from a shared schema.
Only a candidate that passes held-out checks is published to active.json.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from shadow_ai.tactical_training_env import TacticalTrainingEnv, SCENARIOS, STANDALONE_SCENARIOS


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = None
PHYSICAL_SCENARIOS = {'wounded_station', 'wounded_medic', 'wounded_personal_kit',
    'cover_medic', 'exposed_wounded', 'empty_weapon_contact', 'exhausted_contact',
    'camera_theft', 'search_report', 'all_intel_missing'}


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(temporary, path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(model, mode, episodes=16, seed=900_000):
    """Separate action selection from actual simulated effects on fixed seeds."""
    env = TacticalTrainingEnv(mode=mode, seed=seed, training=False, horizon=24, physics_frames=12)
    results = {}
    try:
        cases = [('campaign', scenario) for scenario in SCENARIOS]
        if mode == 'swat':
            cases += [('standalone', scenario) for scenario in STANDALONE_SCENARIOS]
        for scenario_index, (backend, scenario) in enumerate(cases):
            correct = decisions = completed = 0
            applicable_episodes = applicable_completed = 0
            rewards, progress = [], []
            for episode in range(episodes):
                observation, info = env.reset(seed=seed + scenario_index * 1000 + episode,
                                              options={'scenario': scenario, 'backend': backend})
                episode_reward = 0.
                physical_success = False
                applicable = bool(info.get('physical_applicable', False))
                for _ in range(256):
                    action = 0 if model is None else int(model.predict(observation, deterministic=True)[0])
                    expected = info.get('expected_actions', [info['expected_action']])
                    correct += int(action in expected)
                    decisions += 1
                    observation, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    progress.append(float(info.get('physical_progress', 0.)))
                    physical_success |= bool(info.get('physical_success', info.get('success', False)))
                    if terminated or truncated:
                        break
                completed += int(physical_success)
                applicable_episodes += int(applicable)
                applicable_completed += int(applicable and physical_success)
                rewards.append(episode_reward)
            case_name = scenario if backend == 'campaign' else f'{backend}:{scenario}'
            results[case_name] = dict(episodes=episodes, decisions=decisions,
                accuracy=correct / max(1, decisions), physical_success_rate=completed / episodes,
                applicable_episodes=applicable_episodes, applicable_completed=applicable_completed,
                mean_reward=float(np.mean(rewards)), mean_progress=float(np.mean(progress)))
    finally:
        env.close()
    return dict(scenarios=results,
        accuracy=float(np.mean([row['accuracy'] for row in results.values()])),
        minimum_accuracy=min(row['accuracy'] for row in results.values()),
        physical_success_rate=sum(row['applicable_completed'] for row in results.values()) /
                              max(1, sum(row['applicable_episodes'] for row in results.values())),
        minimum_physical_effect_rate=min(row['physical_success_rate'] for name, row in results.items()
                                        if name.split(':')[-1] in PHYSICAL_SCENARIOS),
        mean_reward=float(np.mean([row['mean_reward'] for row in results.values()])))


class TrainingProgress(BaseCallback):
    def __init__(self, path, mode, run_start):
        super().__init__()
        self.path, self.mode, self.run_start = path, mode, run_start
        self.last = 0

    def _on_step(self):
        if self.num_timesteps - self.last >= 2048:
            self.last = self.num_timesteps
            payload = dict(status='TRAINING', mode=self.mode, steps=self.num_timesteps,
                           elapsed_seconds=round(time.monotonic() - self.run_start, 1))
            write_json(self.path, payload)
            print(json.dumps(payload), flush=True)
        return True


def gate(candidate, baseline, accuracy, minimum):
    return (candidate['accuracy'] >= accuracy and candidate['minimum_accuracy'] >= minimum
            and candidate['minimum_physical_effect_rate'] >= .75
            and candidate['mean_reward'] >= baseline['mean_reward']
            and candidate['physical_success_rate'] >= baseline['physical_success_rate'])


def render_report(report, output):
    lines = ['# Addestramento tattico', '', f"Stato: {report['activation']}", '',
             '| Modello | Decisioni corrette | Minimo per scenario | Effetti nei casi applicabili | Passi PPO |',
             '|---|---:|---:|---:|---:|']
    for mode, row in report['modes'].items():
        metrics = row['candidate']
        lines.append(f"| {mode} | {metrics['accuracy']:.2%} | {metrics['minimum_accuracy']:.2%} | "
                     f"{metrics['physical_success_rate']:.1%} | {row['steps']} |")
    lines += ['', 'Valutazione su semi separati dal training. Le percentuali misurano le scelte',
              'previste dalla dottrina del curriculum; non sono una stima della vittoria contro un giocatore umano.',
              'Gli effetti fisici misurano richieste/cure effettive e movimento verso il bersaglio tattico.',
              'Il riferimento CONTINUE misura l’assenza di un intervento contestuale aggiuntivo, non partite con i vecchi modelli.',
              'I modelli originali di movimento e percezione restano disponibili.']
    (Path(output) / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def rehearse_doctrine(model, env, samples, epochs, seed):
    """Initialize tactical choices from observable doctrine, then refine via PPO.

    Labels depend exclusively on the same 32 features supplied during play.
    Short sampled trajectories execute real actions; held-out seeds are unused.
    """
    if samples <= 0:
        return dict(samples=0, epochs=0)
    observations, actions = [], []
    observation, info = env.reset(seed=seed)
    for index in range(samples):
        observations.append(observation.copy())
        action = int(info['expected_action'])
        actions.append(action)
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated or (index + 1) % 8 == 0:
            observation, info = env.reset()
    inputs = torch.as_tensor(np.asarray(observations), dtype=torch.float32, device=model.device)
    targets = torch.as_tensor(actions, dtype=torch.long, device=model.device)
    counts = torch.bincount(targets, minlength=6).float().clamp_min(1)
    weights = len(actions) / (6 * counts)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    model.policy.set_training_mode(True)
    for group in model.policy.optimizer.param_groups:
        group['lr'] = 5e-4
    for _ in range(epochs):
        for batch in torch.randperm(samples, generator=generator).split(128):
            _, log_probability, _ = model.policy.evaluate_actions(inputs[batch], targets[batch])
            loss = -(weights[targets[batch]] * log_probability).mean()
            model.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), .5)
            model.policy.optimizer.step()
    return dict(samples=samples, epochs=epochs, class_counts=counts.int().tolist())


def main():
    global OUTPUT_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=32768, help='PPO steps per training round and mode')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--eval-episodes', type=int, default=24, help='Held-out episodes per scenario and mode')
    parser.add_argument('--seed', type=int, default=20261029)
    parser.add_argument('--eval-seed', type=int, default=1900000)
    parser.add_argument('--accuracy', type=float, default=.92)
    parser.add_argument('--minimum-accuracy', type=float, default=.80)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--no-activate', action='store_true')
    parser.add_argument('--resume', type=Path, help='Continue role checkpoints from an earlier training directory')
    parser.add_argument('--modes', nargs='+', choices=('guard', 'swat'), default=['guard', 'swat'])
    parser.add_argument('--physics-frames', type=int, default=3, help='30 Hz frames per training decision; evaluation uses 12')
    parser.add_argument('--doctrine-samples', type=int, default=8192)
    parser.add_argument('--doctrine-epochs', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--entropy', type=float, default=.005)
    args = parser.parse_args()
    if args.steps <= 0 or args.rounds <= 0 or args.eval_episodes <= 0 or args.physics_frames <= 0:
        parser.error('steps, rounds and eval-episodes must be positive')

    # Training uses the same executors without recursively invoking active models.
    os.environ['SHADOW_CONTEXT_MODELS'] = 'off'
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    started = time.monotonic()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = (args.out or ROOT / '.runtime' / 'training' / stamp).resolve()
    OUTPUT_PATH = output
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / 'report.json'
    report = dict(version=1, started_utc=datetime.now(timezone.utc).isoformat(),
        seed=args.seed, device='cpu', torch_version=torch.__version__,
        schema=dict(observations=32, actions=6), training_steps_per_round=args.steps,
        training_physics_frames=args.physics_frames, evaluation_physics_frames=12,
        simulated_episode_seconds=9.6, pretraining='Observable tactical doctrine followed by PPO on physical effects',
        baseline_kind='CONTINUE contextual action; original neural movement models are unchanged',
        physical_metric='Episodes with an observed medical request, treatment, or measured physical movement toward the selected goal',
        original_models={str(path): file_hash(ROOT / path) for path in (
            Path('models_v7_1_guard/guard_v7_1_final.zip'), Path('models_v7_1_swat/swat_v7_1_final.zip'))},
        evaluation_episodes_per_scenario=args.eval_episodes,
        validation_seed=700_000, held_out_seed=args.eval_seed,
        activation='PENDING', modes={})
    report['resumed_from'] = str(args.resume.resolve()) if args.resume else None
    if args.resume and (args.resume / 'report.json').is_file():
        previous_report = json.loads((args.resume / 'report.json').read_text(encoding='utf-8'))
        if previous_report.get('schema') == report['schema']:
            for mode, row in previous_report.get('modes', {}).items():
                checkpoint_path = Path(row['checkpoint'])
                if (mode not in args.modes and row.get('passed') and checkpoint_path.is_file()
                        and file_hash(checkpoint_path) == row.get('sha256')):
                    row = dict(row)
                    candidate = evaluate(PPO.load(checkpoint_path, device='cpu'), mode,
                                         episodes=args.eval_episodes, seed=args.eval_seed)
                    baseline = evaluate(None, mode, episodes=args.eval_episodes, seed=args.eval_seed)
                    row.update(candidate=candidate, baseline=baseline, evaluation_seed=args.eval_seed,
                               passed=gate(candidate, baseline, args.accuracy, args.minimum_accuracy))
                    report['modes'][mode] = row
    write_json(report_path, report)
    for mode in args.modes:
        mode_index = ('guard', 'swat').index(mode)
        print(f'TRAINING {mode.upper()}: {output}', flush=True)
        train_env = Monitor(TacticalTrainingEnv(mode=mode, seed=args.seed + mode_index,
            training=True, physics_frames=args.physics_frames,
            horizon=max(1, int(np.ceil(288 / args.physics_frames)))))
        previous = args.resume / f'{mode}_context.zip' if args.resume else None
        if previous is not None and previous.is_file():
            model = PPO.load(previous, env=train_env, device='cpu')
        else:
            model = PPO('MlpPolicy', train_env, n_steps=256, batch_size=128, n_epochs=6,
                learning_rate=5e-4, gamma=.95, gae_lambda=.90, ent_coef=.015,
                policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
                seed=args.seed + mode_index, device='cpu', verbose=0)
        initial_parameters = {key: value.clone() for key, value in model.policy.state_dict().items()}
        model.set_random_seed(args.seed + mode_index)
        doctrine = rehearse_doctrine(model, train_env, args.doctrine_samples,
                                    args.doctrine_epochs, args.seed + mode_index + 101)
        model.learning_rate = args.learning_rate
        model.lr_schedule = lambda _: args.learning_rate
        model.ent_coef = args.entropy
        print(json.dumps(dict(mode=mode, doctrine=doctrine)), flush=True)
        baseline_validation = evaluate(None, mode, episodes=8, seed=700_000)
        validations, best_score = [], (False, float('-inf'), float('-inf'))
        checkpoint = output / f'{mode}_context.zip'
        for round_index in range(args.rounds):
            model.learn(total_timesteps=args.steps, reset_num_timesteps=False,
                        callback=TrainingProgress(output / 'progress.json', mode, started))
            validation = evaluate(model, mode, episodes=8, seed=700_000)
            validations.append(dict(round=round_index + 1, steps=model.num_timesteps, metrics=validation))
            score = (gate(validation, baseline_validation, args.accuracy, args.minimum_accuracy),
                     validation['minimum_accuracy'], validation['accuracy'])
            if score > best_score:
                best_score = score
                model.save(checkpoint)
            print(json.dumps(dict(mode=mode, round=round_index + 1, validation=validation)), flush=True)
            if gate(validation, baseline_validation, args.accuracy, args.minimum_accuracy):
                model.save(checkpoint)
                break
        best_model = PPO.load(checkpoint, device='cpu')
        baseline = evaluate(None, mode, episodes=args.eval_episodes, seed=args.eval_seed)
        candidate = evaluate(best_model, mode, episodes=args.eval_episodes, seed=args.eval_seed)
        changed = any(not torch.equal(initial_parameters[key], value)
                      for key, value in best_model.policy.state_dict().items())
        passed = changed and gate(candidate, baseline, args.accuracy, args.minimum_accuracy)
        report['modes'][mode] = dict(checkpoint=str(checkpoint), sha256=file_hash(checkpoint),
            steps=model.num_timesteps, weights_changed=changed,
            training_seed=args.seed + mode_index, evaluation_seed=args.eval_seed,
            checkpoint_steps=best_model.num_timesteps, doctrine=doctrine,
            baseline=baseline, candidate=candidate, passed=passed, validations=validations)
        train_env.close()
        write_json(report_path, report)
        print(json.dumps(dict(mode=mode, held_out=candidate, passed=passed)), flush=True)

    accepted = all(row['passed'] for row in report['modes'].values())
    report['activation'] = 'APPROVED' if accepted else 'REJECTED'
    report['elapsed_seconds'] = round(time.monotonic() - started, 2)
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(report_path, report)
    if accepted and not args.no_activate and set(report['modes']) == {'guard', 'swat'}:
        manifest_path = ROOT / '.runtime' / 'training' / 'active.json'
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if manifest_path.exists():
            (output / 'previous_manifest.json').write_bytes(manifest_path.read_bytes())
        write_json(manifest_path, dict(version=1,
            **{mode: row['checkpoint'] for mode, row in report['modes'].items()},
            report=str(report_path)))
        report['activation'] = 'ACTIVE'
        write_json(report_path, report)
    write_json(output / 'progress.json', dict(status=report['activation'], report=str(report_path),
                                             elapsed_seconds=report['elapsed_seconds']))
    render_report(report, output)
    print(f"REPORT {report_path} ({report['activation']})", flush=True)
    return 0 if accepted else 2


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        if OUTPUT_PATH is not None:
            report_file = OUTPUT_PATH / 'report.json'
            payload = json.loads(report_file.read_text(encoding='utf-8')) if report_file.exists() else {}
            payload.update(activation='FAILED', error=f'{type(exc).__name__}: {exc}',
                           finished_utc=datetime.now(timezone.utc).isoformat())
            write_json(report_file, payload)
            write_json(OUTPUT_PATH / 'progress.json', dict(status='FAILED', error=payload['error'],
                                                         report=str(report_file)))
        raise
