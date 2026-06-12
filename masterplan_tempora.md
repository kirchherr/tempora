# TEMPORA Masterplan für Codex

**Projektname:** TEMPORA  
**Langname:** Temporale Emergenz und mathematische Beweisbarkeit selbstorganisierender neuronaler Systeme  
**Ziel dieses Dokuments:** Dieses `masterplan.md` ist die operative Arbeitsgrundlage für Codex. Es übersetzt das Forschungskonzept in konkrete, testbare Entwicklungsaufgaben, Repo-Struktur, Codex-Prompts, Akzeptanzkriterien und Meilensteine.

---

## 0. Leitprinzip

TEMPORA darf nicht als großes, unprüfbares KI-Versprechen starten. Der erste Release ist ein **beweisnahes Forschungs-MVP**:

> Ein reproduzierbares Python-Framework, das synthetische dynamische Systeme erzeugt, ein kontraktives kontinuierliches neuronales System trainiert, Selbstorganisation durch kontrollierte Plastizität simuliert, Stabilitätszertifikate berechnet und topologische Ähnlichkeit zwischen Eingabe- und Latentraum misst.

Der erste wissenschaftliche Claim lautet nicht:

> „TEMPORA beweist allgemeine temporale Semantik.“

Sondern:

> „Unter expliziten Annahmen können kontraktive kontinuierliche neuronale Systeme stabile latente Repräsentationen für kontrollierte temporale Dynamiken bilden; topologische Ähnlichkeit wird empirisch über persistente Homologie gemessen und gegen Baselines getestet.“

---

## 1. Sofortiger Start für Codex

### 1.1 Erste Codex-Anweisung

Kopiere diesen Prompt in Codex, nachdem `masterplan.md` im Repo liegt:

```text
Read masterplan.md completely before making changes.

You are working on TEMPORA, a research codebase for proof-guided temporal self-organization in continuous-time neural systems.

Your first task is to initialize the repository only.

Create or update:
- AGENTS.md
- README.md
- pyproject.toml
- src/tempora/__init__.py
- tests/test_smoke.py
- docs/mvp_spec.md
- docs/theory/assumptions.md
- .github/workflows/ci.yml

Use a src/ Python package layout.
Use pytest, ruff, and mypy.
Do not implement the full model yet.
Keep all mathematical claims conservative.
Run the test/lint/typecheck commands if possible and report results.
```

### 1.2 Zweite Codex-Anweisung nach dem ersten PR

```text
Continue with Phase 2 from masterplan.md: implement synthetic data generators.

Before coding:
- Inspect the current repository.
- Check AGENTS.md.
- Identify existing conventions.

Then implement only the synthetic data module and its tests.
Do not implement models in this task.
```

---

## 2. Empfohlene Codex-Arbeitsweise

Codex soll wie ein technischer Forschungspartner arbeiten, nicht wie ein einmaliger Codegenerator.

### 2.1 Dauerhafte Projektregeln

- `AGENTS.md` enthält dauerhafte Projektanweisungen für Codex.
- `masterplan.md` enthält den vollständigen Entwicklungsplan.
- Jede Codex-Aufgabe muss klein, testbar und reviewbar sein.
- Keine großen Datensätze ins Repo committen.
- Jeder neue Code braucht Tests.
- Jede neue mathematische Behauptung braucht Annahmen und Grenzen.
- Jede Experimentfunktion muss reproduzierbar sein.

### 2.2 Arbeitsmodus pro Task

Für jede Aufgabe soll Codex diesen Ablauf einhalten:

1. `masterplan.md` und `AGENTS.md` lesen.
2. Bestehende Dateien inspizieren.
3. Kurz planen, welche Dateien geändert werden.
4. Kleine Implementierung durchführen.
5. Tests ergänzen.
6. `pytest`, `ruff`, `mypy` ausführen, falls möglich.
7. Ergebnis zusammenfassen:
   - geänderte Dateien,
   - erfüllte Akzeptanzkriterien,
   - offene Punkte,
   - Testergebnisse.

### 2.3 Standardbefehle

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy src
```

Falls das Projekt `uv` nutzt:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

---

## 3. Zielbild für TEMPORA v0.1

TEMPORA v0.1 ist fertig, wenn folgende Fähigkeiten vorhanden sind:

```text
[ ] Reproduzierbares Python-Repo mit CI.
[ ] Synthetische Dynamikdaten: Kreis, Torus, Lorenz, Rössler.
[ ] Zeitverzerrung, Rauschen und Maskierung als Augmentierungen.
[ ] Contractive CTRNN / Neural-ODE-artiges Modell.
[ ] Projektion rekurrenter Gewichte in einen kontraktiven Parameterraum.
[ ] Kontraktionsmarge als numerisches Zertifikat.
[ ] Oja-/Hebb-artige Plastizität mit Homöostase.
[ ] Plastizitätsupdates verletzen die Kontraktionsbedingung nicht.
[ ] Persistente Homologie für Eingabe- und Latent-Trajektorien.
[ ] Bottleneck- oder Wasserstein-Distanz zwischen Persistenzdiagrammen.
[ ] Lyapunov-Schätzung für latente Trajektorien.
[ ] Invarianztest unter Zeitstreckung/-stauchung.
[ ] Mindestens zwei Baselines: GRU und unconstrained Neural ODE.
[ ] End-to-End-Benchmark mit JSON-Metriken und Markdown-Report.
[ ] Theorie-Dokumentation mit konservativen Theorem-Skizzen.
```

---

## 4. Repository-Struktur

Codex soll auf folgende Zielstruktur hinarbeiten:

```text
tempora/
  AGENTS.md
  README.md
  masterplan.md
  pyproject.toml
  .gitignore
  .pre-commit-config.yaml
  .github/
    workflows/
      ci.yml

  configs/
    synth_circle.yaml
    synth_torus.yaml
    synth_lorenz.yaml
    synth_rossler.yaml
    contractive_ctrnn.yaml
    benchmark_smoke.yaml

  src/
    tempora/
      __init__.py

      data/
        __init__.py
        types.py
        circle.py
        torus.py
        lorenz.py
        rossler.py
        time_warp.py
        augmentations.py
        dataloaders.py

      models/
        __init__.py
        contractive_ctrnn.py
        neural_ode.py
        plasticity.py
        encoders.py
        projections.py

      training/
        __init__.py
        losses.py
        trainer.py
        callbacks.py
        checkpoints.py

      metrics/
        __init__.py
        jacobian.py
        contraction.py
        lyapunov.py
        tda.py
        invariance.py
        reconstruction.py

      baselines/
        __init__.py
        gru.py
        vanilla_neural_ode.py
        reservoir.py

      experiments/
        __init__.py
        run_synthetic.py
        evaluate_topology.py
        evaluate_stability.py
        compare_baselines.py

      proof/
        __init__.py
        assumptions.py
        certificates.py
        theorem_checks.py

      viz/
        __init__.py
        plots.py
        phase_portraits.py
        persistence_diagrams.py

  scripts/
    train_synth.py
    eval_synth.py
    make_report.py

  tests/
    test_smoke.py
    test_circle.py
    test_torus.py
    test_lorenz.py
    test_rossler.py
    test_time_warp.py
    test_contractive_ctrnn.py
    test_projection.py
    test_plasticity.py
    test_tda_metrics.py
    test_lyapunov.py
    test_end_to_end_synth.py

  docs/
    mvp_spec.md
    theory/
      assumptions.md
      theorem_01_contraction.md
      theorem_02_learning_stability.md
      theorem_03_tda_preservation.md
    experiments/
      synthetic_protocol.md
      benchmark_spec.md
      results_template.md

  notebooks/
    01_lorenz_latent_demo.ipynb
    02_persistence_demo.ipynb

  outputs/
    .gitkeep
```

---

## 5. AGENTS.md Vorlage

Codex soll diese Datei im Repo anlegen oder anpassen.

```md
# TEMPORA Agent Instructions

## Project goal

TEMPORA is a research codebase for proof-guided temporal self-organization in continuous-time neural systems.

The first milestone is not a general AGI system and not a broad proof of temporal semantics. The first milestone is a reproducible MVP that:

1. Generates synthetic dynamical-system datasets.
2. Trains a contractive CTRNN / Neural-ODE-like model.
3. Applies local plasticity with projection into a contractive parameter set.
4. Computes stability certificates.
5. Computes persistent homology metrics between input and latent trajectories.
6. Compares against simple baselines.

## Engineering rules

- Prefer small, reviewable pull requests.
- Do not add large datasets to the repository.
- All public functions need type hints.
- Use deterministic seeds in tests.
- Keep experiments reproducible from CLI commands.
- Every feature must include tests.
- Avoid silent numerical failures; check for NaN and Inf.
- Use docstrings for mathematical functions.
- Keep theory claims conservative and explicitly state assumptions.
- Do not overclaim what the experiments prove.

## Standard commands

Run these commands unless the repository later standardizes different commands:

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy src
```

## Definition of done

A task is done only when:

1. Tests pass or failures are clearly explained.
2. New behavior is covered by tests.
3. README or docs are updated if user-facing behavior changed.
4. Numerical assumptions are documented.
5. Any experiment produces reproducible artifacts under `outputs/`.

## Research constraints

Do not claim that TEMPORA proves general temporal semantic preservation.
For now, claims are limited to:

- contraction certificates under explicit assumptions,
- empirical topology preservation via persistent homology metrics,
- reproducible synthetic benchmarks,
- comparison against simple baselines.
```

---

## 6. Mathematischer MVP-Kern

### 6.1 Modellform

Das erste TEMPORA-Modell soll bewusst einfach sein:

\[
\dot z(t) = -D z(t) + W \tanh(z(t)) + B \phi(u(t)) + b
\]

Dabei:

- \(z(t)\): latenter Zustand,
- \(u(t)\): Eingabezeitreihe,
- \(D\): positive diagonale Dämpfung,
- \(W\): rekurrente Matrix,
- \(B\): Eingabeprojektion,
- \(\phi\): optionaler Encoder,
- \(b\): Bias.

### 6.2 Kontraktionsbedingung

Für den MVP reicht eine einfache hinreichende Bedingung:

\[
d_{\min}(D) > L_\sigma \lVert W \rVert_2 + \kappa
\]

Dabei ist:

- \(L_\sigma\): Lipschitz-Konstante der Aktivierung,
- \(\lVert W \rVert_2\): Spektralnorm von \(W\),
- \(\kappa > 0\): Sicherheitsmarge.

Codex soll dafür implementieren:

```python
contraction_margin = min(D) - L_sigma * spectral_norm(W)
```

Das Modell gilt numerisch als kontraktiv, wenn:

```text
contraction_margin > margin
```

### 6.3 Projektion

Nach jedem Update von \(W\) muss Codex die Matrix in den zulässigen Bereich zurückprojizieren:

```text
if L_sigma * ||W||_2 >= min(D) - margin:
    W <- W * ((min(D) - margin) / (L_sigma * ||W||_2 + eps))
```

### 6.4 Plastizität

Version 1 nutzt eine Oja-/Hebb-artige Regel:

\[
\Delta W = \eta (yy^T - \alpha W) - \lambda W
\]

Danach immer:

\[
W \leftarrow \Pi_{\mathcal C}(W)
\]

Das heißt: Selbstorganisation darf Stabilität nicht zerstören.

---

## 7. Wissenschaftliche Claims für Version 0.1

### Claim A: Kontraktive latente Dynamik

Unter der Bedingung

\[
d_{\min}(D) > L_\sigma \lVert W \rVert_2
\]

ist die latente Dynamik im MVP-Modell kontraktiv in \(z\), sofern die übrigen Annahmen dokumentiert sind.

### Claim B: Stabilität unter plastischem Lernen mit Projektion

Wenn jeder plastische Update-Schritt durch eine Projektion in den kontraktiven Parameterraum abgeschlossen wird, bleibt die rekurrente Dynamik innerhalb der zertifizierten Stabilitätsklasse.

### Claim C: Empirische topologische Nähe

Wenn Eingabe- und Latent-Trajektorien auf kompakten Punktwolken verglichen werden, kann ihre topologische Ähnlichkeit über Persistenzdiagramme und Bottleneck-/Wasserstein-Distanzen gemessen werden.

Nicht behaupten:

```text
TEMPORA beweist allgemeine Semantik.
TEMPORA beweist Homöomorphie für beliebige Daten.
TEMPORA beweist robuste Repräsentation aller realen Videos.
TEMPORA ist AGI oder ein allgemeiner Beweiser temporaler Intelligenz.
```

---

## 8. Phasenplan

## Phase 0: Repo-Initialisierung

### Ziel

Ein sauberes Python-Forschungsrepo, das Codex zuverlässig bearbeiten kann.

### Codex-Prompt

```text
Initialize the TEMPORA Python research repository.

Requirements:
- Read masterplan.md first.
- Use a src/ layout.
- Add pyproject.toml.
- Configure pytest, ruff, and mypy.
- Add GitHub Actions CI.
- Add AGENTS.md using the template from masterplan.md.
- Add README with project scope, setup, and commands.
- Add docs/mvp_spec.md and docs/theory/assumptions.md.
- Add one smoke test.

Do not implement models yet.
Keep the PR small and reviewable.
Run tests and report results.
```

### Zu erstellende Dateien

```text
AGENTS.md
README.md
pyproject.toml
src/tempora/__init__.py
tests/test_smoke.py
docs/mvp_spec.md
docs/theory/assumptions.md
.github/workflows/ci.yml
```

### Akzeptanzkriterien

```text
[ ] Paket ist installierbar.
[ ] pytest läuft.
[ ] ruff läuft.
[ ] mypy läuft oder ist sinnvoll konfiguriert.
[ ] README enthält Projektziel und Nicht-Ziele.
[ ] AGENTS.md enthält dauerhafte Codex-Regeln.
```

---

## Phase 1: Synthetische Daten

### Ziel

Reproduzierbare Ground-Truth-Zeitreihen für kontrollierte Experimente.

### Codex-Prompt

```text
Implement synthetic dynamical-system data generators for TEMPORA.

Requirements:
- Read masterplan.md and AGENTS.md first.
- Implement circle trajectory generator.
- Implement torus trajectory generator.
- Implement Lorenz system generator.
- Implement Rossler system generator.
- Implement time-warp augmentation.
- Implement Gaussian noise augmentation.
- Implement missing-segment masking.
- Use typed dataclasses with times, observations, clean_states, and metadata.
- Add deterministic tests for shapes, seeds, finite values, and metadata.
- Add a small plotting script or helper, but do not commit generated images.

Do not implement models in this task.
Run pytest and ruff.
```

### Zu erstellende Dateien

```text
src/tempora/data/types.py
src/tempora/data/circle.py
src/tempora/data/torus.py
src/tempora/data/lorenz.py
src/tempora/data/rossler.py
src/tempora/data/time_warp.py
src/tempora/data/augmentations.py
src/tempora/data/dataloaders.py
tests/test_circle.py
tests/test_torus.py
tests/test_lorenz.py
tests/test_rossler.py
tests/test_time_warp.py
```

### Datenobjekt

Codex soll eine Dataclass ähnlich dieser verwenden:

```python
from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass(frozen=True)
class TemporalDataset:
    times: np.ndarray
    observations: np.ndarray
    clean_states: np.ndarray
    metadata: dict[str, Any]
```

### Akzeptanzkriterien

```text
[ ] Gleicher Seed erzeugt gleiche Daten.
[ ] Alle Arrays haben dokumentierte Shapes.
[ ] Keine NaN-/Inf-Werte.
[ ] Zeitverzerrung verändert Sampling/Dauer, aber nicht die Trajektorienklasse.
[ ] Rauschen und Maskierung sind separat testbar.
```

---

## Phase 2: Contractive CTRNN

### Ziel

Das erste mathematisch kontrollierte TEMPORA-Modell.

### Codex-Prompt

```text
Implement the contractive CTRNN model.

Equation:
dz/dt = -D z + W tanh(z) + B phi(u(t)) + b

Requirements:
- PyTorch module.
- D positive via softplus parameterization or explicit positive parameter handling.
- W projection enforcing L_sigma * ||W||_2 < min(D) - margin.
- contraction_margin(model) -> float or tensor.
- Jacobian symmetric-part eigenvalue estimator on sampled latent states.
- ODE integration with torchdiffeq if available.
- Provide a simple fixed-step RK4 fallback if torchdiffeq is unavailable.
- Add tests for projection, finite outputs, gradient flow, and contraction margin.
- Add docstrings explaining the sufficient contraction condition.

Do not implement plasticity yet.
Run pytest, ruff, and mypy.
```

### Zu erstellende Dateien

```text
src/tempora/models/contractive_ctrnn.py
src/tempora/models/projections.py
src/tempora/metrics/contraction.py
src/tempora/metrics/jacobian.py
tests/test_contractive_ctrnn.py
tests/test_projection.py
```

### Akzeptanzkriterien

```text
[ ] Projektion reduziert Spektralnorm, wenn nötig.
[ ] Nach Projektion ist contraction_margin positiv.
[ ] Forward-Pass liefert finite Werte.
[ ] Gradienten können berechnet werden.
[ ] Tests decken Verletzung vor Projektion und Korrektur nach Projektion ab.
```

---

## Phase 3: Plastizität und Selbstorganisation

### Ziel

Lokales Lernen einbauen, ohne die Stabilitätsgarantie zu zerstören.

### Codex-Prompt

```text
Add Oja-style local plasticity to the contractive CTRNN.

Requirements:
- Implement plasticity update for recurrent weights W.
- Add homeostatic decay.
- After every update, project W back into the contractive set.
- Log contraction margin before and after update.
- Add predictive loss for next-step latent or observation prediction.
- Add a minimal training loop on circle trajectories.
- Add tests showing repeated plasticity updates preserve contraction constraints.

Keep theory claims conservative.
Run pytest, ruff, and mypy.
```

### Zu erstellende Dateien

```text
src/tempora/models/plasticity.py
src/tempora/training/losses.py
src/tempora/training/trainer.py
src/tempora/training/callbacks.py
tests/test_plasticity.py
tests/test_end_to_end_synth.py
```

### Akzeptanzkriterien

```text
[ ] Wiederholte Plastizitätsupdates verletzen Kontraktionsbedingung nicht.
[ ] Gewichtsnormen bleiben beschränkt.
[ ] Trainingsverlust sinkt auf einem kleinen Kreis-Dataset.
[ ] Metriken werden als JSON speicherbar.
```

---

## Phase 4: Topologische Datenanalyse

### Ziel

Topologische Ähnlichkeit messbar machen.

### Codex-Prompt

```text
Implement topological data analysis metrics for TEMPORA.

Requirements:
- Use ripser.py or GUDHI for persistence diagrams if installed.
- Support H0 and H1.
- Compute bottleneck or Wasserstein distance where available.
- Add clear fallback errors if optional TDA dependencies are missing.
- Test on circle data: H1 should contain one dominant loop.
- Add visualization helper for persistence diagrams.
- Update docs explaining what these metrics do and do not prove.

Do not claim that the metric proves semantic equivalence.
Run tests.
```

### Zu erstellende Dateien

```text
src/tempora/metrics/tda.py
src/tempora/viz/persistence_diagrams.py
src/tempora/experiments/evaluate_topology.py
tests/test_tda_metrics.py
docs/theory/theorem_03_tda_preservation.md
```

### Akzeptanzkriterien

```text
[ ] Kreis-Dataset zeigt eine dominante H1-Struktur.
[ ] Rauschen verändert TDA-Metriken erwartungsgemäß.
[ ] Input- und Latent-Punktwolken können verglichen werden.
[ ] Fehlende optionale Dependencies erzeugen klare Fehlermeldungen.
```

---

## Phase 5: Stabilität und Invarianz

### Ziel

TEMPORA falsifizierbar machen.

### Codex-Prompt

```text
Implement stability and invariance evaluation for TEMPORA.

Requirements:
- Estimate largest Lyapunov exponent from latent trajectories.
- Compute representation distance under time warping.
- Compute representation distance under Gaussian noise.
- Compute representation distance under missing-segment masking.
- Add CLI script that runs evaluation on circle, torus, Lorenz, and Rossler datasets.
- Save metrics to outputs/<run_id>/metrics.json.
- Save figures to outputs/<run_id>/figures/.
- Add tests for metric shapes, finite values, and deterministic seeds.

Run pytest and ruff.
```

### Zu erstellende Dateien

```text
src/tempora/metrics/lyapunov.py
src/tempora/metrics/invariance.py
src/tempora/metrics/reconstruction.py
src/tempora/experiments/evaluate_stability.py
scripts/eval_synth.py
tests/test_lyapunov.py
```

### Akzeptanzkriterien

```text
[ ] Lyapunov-Schätzung liefert finite Werte.
[ ] Zeitverzerrungs-Invarianz ist numerisch messbar.
[ ] Rauschrobustheit ist numerisch messbar.
[ ] Evaluation schreibt reproduzierbare JSON-Artefakte.
```

---

## Phase 6: Baselines

### Ziel

TEMPORA wissenschaftlich vergleichbar machen.

### Codex-Prompt

```text
Implement baseline models for TEMPORA.

Requirements:
- GRU baseline.
- Unconstrained Neural ODE baseline.
- Simple reservoir computing baseline if feasible.
- Same train/eval interface as the TEMPORA model.
- No contraction projection in baselines.
- Add tests for interface compatibility.
- Update compare_baselines.py.

Do not tune for maximum performance yet.
Focus on fair, reproducible comparison.
```

### Zu erstellende Dateien

```text
src/tempora/baselines/gru.py
src/tempora/baselines/vanilla_neural_ode.py
src/tempora/baselines/reservoir.py
src/tempora/experiments/compare_baselines.py
```

### Akzeptanzkriterien

```text
[ ] Alle Baselines nutzen dieselbe Dataset-Schnittstelle.
[ ] Alle Baselines liefern dieselben Metrikfelder.
[ ] Baseline-Ergebnisse sind reproduzierbar.
[ ] Vergleichsbericht zeigt TEMPORA vs. Baselines.
```

---

## Phase 7: End-to-End-Benchmark

### Ziel

Ein vollständiger reproduzierbarer Benchmarklauf.

### Codex-Prompt

```text
Create an end-to-end synthetic benchmark for TEMPORA.

Requirements:
- Train TEMPORA on circle, torus, Lorenz, and Rossler datasets.
- Evaluate prediction MSE, contraction margin, Lyapunov estimate, TDA distance, and time-warp invariance.
- Compare against GRU and unconstrained Neural ODE baselines.
- Save all outputs to outputs/<run_id>/.
- Generate a Markdown report.
- Include seed, config, git commit hash, dependency versions, and runtime metadata.
- Add a small CI-safe smoke benchmark.

Keep the default benchmark small enough for CI.
```

### Zu erstellende Dateien

```text
src/tempora/experiments/run_synthetic.py
src/tempora/experiments/compare_baselines.py
scripts/train_synth.py
scripts/eval_synth.py
scripts/make_report.py
configs/benchmark_smoke.yaml
docs/experiments/benchmark_spec.md
docs/experiments/results_template.md
```

### Akzeptanzkriterien

```text
[ ] Ein CLI-Befehl startet den Smoke-Benchmark.
[ ] Metriken werden in outputs/<run_id>/metrics.json gespeichert.
[ ] Figuren werden in outputs/<run_id>/figures/ gespeichert.
[ ] Ein Markdown-Report wird erzeugt.
[ ] Bericht trennt Claims, Evidenz und offene Punkte.
```

---

## Phase 8: Theorie-Dokumentation

### Ziel

Mathematische Aussagen sauber und konservativ dokumentieren.

### Codex-Prompt

```text
Create the theory documentation for TEMPORA v0.1.

Requirements:
- Document all assumptions.
- Write theorem_01_contraction.md.
- Write theorem_02_learning_stability.md.
- Write theorem_03_tda_preservation.md.
- Each theorem must include:
  - statement,
  - assumptions,
  - proof sketch,
  - implementation correspondence,
  - limitations,
  - tests or metrics that relate to the claim.
- Do not overclaim general semantic preservation.
```

### Zu erstellende Dateien

```text
docs/theory/assumptions.md
docs/theory/theorem_01_contraction.md
docs/theory/theorem_02_learning_stability.md
docs/theory/theorem_03_tda_preservation.md
```

### Akzeptanzkriterien

```text
[ ] Jeder Claim hat Annahmen.
[ ] Jeder Claim hat Limitationen.
[ ] Jeder Claim ist mit Code/Metriken verknüpft.
[ ] Keine allgemeine Homöomorphie- oder Semantikbehauptung.
```

---

## Phase 9: Release TEMPORA v0.1

### Ziel

Ein erster veröffentlichbarer Forschungsrelease.

### Codex-Prompt

```text
Prepare TEMPORA v0.1 release.

Requirements:
- Check README for setup, usage, examples, and limitations.
- Ensure all tests pass.
- Ensure docs are linked.
- Ensure benchmark smoke command works.
- Ensure outputs/ is ignored except .gitkeep.
- Add CITATION.cff if appropriate.
- Add LICENSE placeholder if missing and ask the human to choose license if not specified.
- Add CHANGELOG.md.
- Create a release checklist in docs/release_v0_1.md.

Do not invent benchmark results.
Only report results that were actually generated.
```

### Akzeptanzkriterien

```text
[ ] README ist nutzbar.
[ ] CI ist grün.
[ ] Smoke-Benchmark läuft.
[ ] Release-Checkliste existiert.
[ ] Keine erfundenen Ergebnisse.
```

---

## 9. Master-Prompts für einzelne Codex-Sessions

### 9.1 Repo explorieren

```text
Inspect this repository and summarize:
1. Current structure.
2. Existing dependencies.
3. Existing tests.
4. Gaps relative to masterplan.md.
5. Recommended next smallest task.

Do not modify files.
```

### 9.2 Nächste kleine Aufgabe wählen

```text
Read masterplan.md and AGENTS.md.
Identify the next incomplete phase.
Propose the smallest useful implementation task.
Then wait for confirmation before editing files.
```

### 9.3 Implementierung durchführen

```text
Read masterplan.md and AGENTS.md.
Implement only the task described below.

Task:
<INSERT TASK HERE>

Constraints:
- Keep the diff small.
- Add tests.
- Update docs if behavior changes.
- Run relevant checks.
- Report changed files and test results.
```

### 9.4 Numerische Stabilität prüfen

```text
Review the current implementation for numerical stability issues.

Focus on:
- NaN/Inf risks,
- exploding norms,
- unstable ODE integration,
- incorrect spectral norm estimation,
- missing eps constants,
- shape errors,
- nondeterministic tests.

Do not make changes first.
Return findings with file/line references and proposed fixes.
```

### 9.5 Mathematische Claims prüfen

```text
Review docs and code comments for overclaimed mathematical statements.

Flag any claim that suggests:
- general temporal semantic preservation,
- guaranteed homomorphism/homeomorphism for arbitrary data,
- proof of real-world video understanding,
- stability without explicit assumptions.

Suggest conservative rewrites.
```

### 9.6 Pull Request Review

```text
Review this PR for:
1. numerical stability issues,
2. hidden NaN/Inf risks,
3. incorrect shape assumptions,
4. missing tests,
5. overclaimed mathematical statements,
6. reproducibility problems,
7. unclear user-facing docs.

Return concrete comments and suggested patches where appropriate.
```

---

## 10. Erste GitHub Issues

Diese Issues kann Codex oder der Mensch direkt anlegen.

### Issue 1: Initialize repository

```text
Create initial Python repo structure for TEMPORA with AGENTS.md, README, pyproject.toml, CI, docs, src package, and smoke test.
```

### Issue 2: Implement synthetic trajectory dataclass

```text
Add TemporalDataset dataclass and shared validation utilities for synthetic temporal data.
```

### Issue 3: Add circle and torus generators

```text
Implement deterministic circle and torus trajectory generators with tests.
```

### Issue 4: Add Lorenz and Rossler generators

```text
Implement deterministic Lorenz and Rossler ODE trajectory generators with tests.
```

### Issue 5: Add time-warp/noise/masking augmentations

```text
Implement time warping, Gaussian noise, and missing-segment masking for temporal datasets.
```

### Issue 6: Implement contractive CTRNN core

```text
Implement PyTorch contractive CTRNN with positive D, recurrent W, tanh activation, input projection, and RK4 fallback.
```

### Issue 7: Implement spectral projection and contraction margin

```text
Implement projection of W into the contractive parameter set and add contraction margin metrics.
```

### Issue 8: Add Oja plasticity with projection

```text
Implement Oja-style plasticity and verify repeated updates preserve contraction constraints.
```

### Issue 9: Add TDA metrics

```text
Implement persistence diagrams and bottleneck/Wasserstein distances for input and latent point clouds.
```

### Issue 10: Add smoke benchmark

```text
Create a small end-to-end benchmark that trains TEMPORA on a circle dataset and writes metrics.json.
```

---

## 11. Metriken

TEMPORA soll nicht nur Accuracy reporten. Die Kernmetriken sind:

| Metrik | Zweck |
|---|---|
| Prediction MSE | Prüft, ob Dynamik gelernt wird. |
| Reconstruction MSE | Prüft, ob Eingabestruktur im Latentraum rekonstruierbar bleibt. |
| Contraction margin | Numerisches Stabilitätszertifikat. |
| Symmetric Jacobian max eigenvalue | Lokale Kontraktionsprüfung. |
| Largest Lyapunov estimate | Empirischer Divergenz-/Stabilitätshinweis. |
| H0/H1 persistence | Topologische Struktur der Punktwolke. |
| Bottleneck distance | Abstand zwischen Persistenzdiagrammen. |
| Wasserstein distance | Alternative TDA-Distanz. |
| Time-warp invariance score | Prüft Situation vs. Geschwindigkeit. |
| Noise robustness score | Prüft Stabilität unter Störung. |
| Missing-segment robustness | Prüft Stabilität bei partieller Beobachtung. |

---

## 12. Experiment-Protokoll

Jeder Experimentlauf muss speichern:

```text
outputs/<run_id>/
  config.yaml
  metrics.json
  report.md
  figures/
    input_trajectory.png
    latent_trajectory.png
    persistence_input.png
    persistence_latent.png
  checkpoints/
    model.pt
```

`metrics.json` soll mindestens enthalten:

```json
{
  "run_id": "example",
  "seed": 42,
  "dataset": "circle",
  "model": "contractive_ctrnn",
  "prediction_mse": null,
  "reconstruction_mse": null,
  "contraction_margin_min": null,
  "contraction_margin_final": null,
  "largest_lyapunov_estimate": null,
  "tda_bottleneck_h0": null,
  "tda_bottleneck_h1": null,
  "time_warp_invariance_score": null,
  "noise_robustness_score": null,
  "missing_segment_robustness_score": null
}
```

---

## 13. Baseline-Vergleich

Mindestens diese Modelle vergleichen:

| Modell | Zweck |
|---|---|
| TEMPORA Contractive CTRNN | Hauptmodell |
| GRU | Standardsequenzmodell |
| Unconstrained Neural ODE | ODE-Baseline ohne Kontraktionsprojektion |
| Reservoir | Einfaches dynamisches Referenzsystem |

Alle Modelle müssen dieselbe Evaluationsschnittstelle unterstützen:

```python
class TemporalModelProtocol(Protocol):
    def fit(self, dataset: TemporalDataset, seed: int) -> dict[str, float]: ...
    def encode(self, dataset: TemporalDataset) -> np.ndarray: ...
    def predict(self, dataset: TemporalDataset) -> np.ndarray: ...
```

---

## 14. Dokumentationsregeln

Jede Theorie-Datei muss diese Struktur haben:

```md
# Theorem X: Titel

## Statement

## Assumptions

## Proof sketch

## Implementation correspondence

## Empirical checks

## Limitations

## Related tests
```

Jede Experiment-Datei muss diese Struktur haben:

```md
# Experiment: Titel

## Purpose

## Dataset

## Models

## Metrics

## Procedure

## Expected failure modes

## Reproducibility notes
```

---

## 15. Nicht-Ziele für v0.1

Codex darf diese Dinge nicht in v0.1 verfolgen, außer der Mensch fordert sie explizit an:

```text
[ ] Keine realen Video-Datensätze.
[ ] Keine Mujoco-/Isaac-Sim-Abhängigkeit.
[ ] Kein großes Transformer-Modell.
[ ] Keine allgemeine Semantikbehauptung.
[ ] Keine GPU-Pflicht.
[ ] Kein verteilter Trainingsstack.
[ ] Kein riesiges Hyperparameter-Tuning.
[ ] Keine nicht-reproduzierbaren Notebook-only Ergebnisse.
```

---

## 16. Risiken und Gegenmaßnahmen

| Risiko | Gegenmaßnahme |
|---|---|
| Mathematische Claims sind zu stark | Claims an explizite Annahmen binden. |
| Plastizität zerstört Stabilität | Nach jedem Update Projektion erzwingen. |
| ODE-Solver erzeugt Artefakte | RK4-Fallback, Toleranztests, NaN-Prüfung. |
| TDA wird nur dekorativ | Baselines und Störungstests verpflichtend machen. |
| Codex baut zu große PRs | Kleine Phasen und Akzeptanzkriterien erzwingen. |
| Tests sind langsam | Smoke-Tests klein halten, große Experimente optional. |
| Ergebnisse sind nicht reproduzierbar | Seeds, Configs, Git-Commit und Dependencies speichern. |

---

## 17. 12-Wochen-Plan

| Woche | Ziel | Ergebnis |
|---:|---|---|
| 1 | Repo, CI, AGENTS.md, README | Codex-fähiges Projekt |
| 2 | Kreis-/Torus-/Lorenz-/Rössler-Daten | Synthetische Ground Truth |
| 3 | Contractive CTRNN | Erstes Modell |
| 4 | Projektion und Kontraktionszertifikat | Mathematischer Kern |
| 5 | Plastizität und Homöostase | Selbstorganisation |
| 6 | Mini-Training | End-to-End-Kreisexperiment |
| 7 | Persistente Homologie | TDA-Metriken |
| 8 | Bottleneck/Wasserstein + Diagramme | Topologie-Auswertung |
| 9 | Lyapunov und Invarianz | Stabilitätsauswertung |
| 10 | Baselines | Vergleichbarkeit |
| 11 | Experiment-Runner und Reports | Reproduzierbare Resultate |
| 12 | Release v0.1 | Forschungs-MVP |

---

## 18. Definition of Done für jeden PR

Ein PR ist nur fertig, wenn:

```text
[ ] Die Aufgabe ist klein und klar abgegrenzt.
[ ] Neue Funktionen haben Type Hints.
[ ] Neue Funktionen haben Tests.
[ ] Tests sind deterministisch.
[ ] Keine NaN-/Inf-Risiken ohne Prüfung.
[ ] Öffentliche Funktionen haben Docstrings.
[ ] Mathematische Annahmen sind dokumentiert.
[ ] README oder Docs wurden aktualisiert, falls nötig.
[ ] pytest läuft.
[ ] ruff läuft.
[ ] mypy läuft oder bekannte Probleme sind dokumentiert.
[ ] Codex berichtet geänderte Dateien und Testergebnisse.
```

---

## 19. Definition of Done für TEMPORA v0.1

```text
[ ] Repo installiert sauber.
[ ] CI läuft mit pytest, ruff und mypy.
[ ] Synthetische Datengeneratoren sind reproduzierbar.
[ ] Contractive CTRNN trainiert auf mindestens einem Dataset.
[ ] Kontraktionsmarge wird berechnet und geloggt.
[ ] Plastizitätsupdates verletzen Kontraktionsbedingung nicht.
[ ] Persistenzdiagramme werden für Input und Latent Space berechnet.
[ ] Bottleneck- oder Wasserstein-Distanz wird berichtet.
[ ] Lyapunov-Schätzung läuft.
[ ] Zeitverzerrungs-Invarianztest läuft.
[ ] Mindestens zwei Baselines sind implementiert.
[ ] Ein End-to-End-Benchmark erzeugt einen Report.
[ ] docs/theory enthält konservativ formulierte Theorem-Skizzen.
[ ] README erklärt Scope, Installation, Quickstart und Grenzen.
```

---

## 20. Wichtigste Codex-Regel

Wenn Codex unsicher ist, soll Codex nicht spekulieren, sondern:

1. den aktuellen Code inspizieren,
2. Annahmen explizit nennen,
3. den kleinsten sicheren nächsten Schritt wählen,
4. Tests hinzufügen,
5. keine nicht belegten Forschungsergebnisse behaupten.

---

## 21. Quellenhinweise für Codex-Konventionen

Diese Quellen sind nur Hintergrund für die Arbeitsweise mit Codex, nicht für die TEMPORA-Theorie:

- OpenAI Codex AGENTS.md Guide: https://developers.openai.com/codex/guides/agents-md
- OpenAI Codex CLI: https://developers.openai.com/codex/cli
- OpenAI Codex Cloud/Web: https://developers.openai.com/codex/cloud
- OpenAI Codex GitHub Integration: https://developers.openai.com/codex/integrations/github
- OpenAI Codex Best Practices: https://developers.openai.com/codex/learn/best-practices

---

## 22. Nächster konkreter Schritt

Lege dieses `masterplan.md` in die Repo-Root und starte Codex mit:

```text
Read masterplan.md completely. Then execute Phase 0 only: initialize the repository. Do not implement the model yet. Keep the PR small, add tests, and report results.
```
