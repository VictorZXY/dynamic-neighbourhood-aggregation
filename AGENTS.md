# AGENTS.md

## Project overview

- This repository contains research code for Learnable Dynamic Neighborhood Aggregation (LDNA) on graph-level prediction tasks.
- LDNA is the method studied in this repository. It uses canonical sorting of graph inputs before model execution and trains a learnable neighborhood aggregation model implemented in code as `LDNA` and `LDNAConv`.
- The main code supports training LDNA and several baseline GNN models on `ogbg-molhiv`, `ZINC`, and `MNISTSuperpixels`.

---

## Read first

Before doing any task, read:

- `docs/project_context.md`
- `docs/project_state.md`
- `docs/next_steps.md`

---

## Repo architecture

- `train.py`: main training entry point.
- `hyperparam_search_*.py`: Optuna search for LDNA on various datasets.
- `models/`: LDNA model code and baseline GNN model wrappers.
- `utils/`: dataset sorting, transforms, evaluators, logger, and resolver code.
- `configs/`: YAML experiment configs.
- `docs/`: agent-facing project notes.

Important architecture notes:

- Canonical sorting is done on the data side before model execution.
- Sorting is implemented in `utils/_utils.py` and applied from `utils/resolver.py`.

---

## Repo workflow

- Choose a YAML config from `configs/`.
- Run `train.py` with `--config <path>`.
- `utils/resolver.py` loads the dataset, applies preprocessing, and builds the requested model.
- Training runs for the configured number of epochs and runs.
- Metrics are printed during training and summarized at the end.
- Optional outputs are written to the configured checkpoint and log directories.

---

## Naming conventions

- Prefer following the existing repository naming in code, configs, and discussions of concrete modules.

---

## Documentation policy

- Use plain, factual language.
- Keep documents concise and structured.
- Do not invent features not present in the repository.
- Prefer explicit explanations over abstract descriptions.

---

## Task workflow

For each task:

1. Understand the task and relevant modules.
2. Provide a short plan.
3. Implement with minimal changes.
4. Keep consistency with existing style.
5. Validate if applicable.

---

## Handling ambiguity

If something is ambiguous:

1. Make the smallest reasonable assumption consistent with the current repo and docs.
2. Explicitly state the assumption.
3. If the assumption could affect architecture or interfaces, ask for confirmation before proceeding.

Prefer asking for clarification over making large or irreversible decisions.

---

## Task scoping

When project docs mention multiple current or future tasks, follow the task explicitly requested in the current user prompt.

Do not proactively continue to later roadmap items unless explicitly asked.

---

## Documentation updates

After completing a task, update relevant project docs when needed so they remain consistent with the current implementation and task status.

Keep documentation updates scoped:

- update only the docs affected by the completed task
- do not rewrite unrelated sections
- keep project docs concise and current

---

## Code explanation requirement

All non-trivial code changes must be accompanied by concise explanations.

Explain:

- what the code does
- why this design is used
- how it fits into the overall system in this repository
- how it interacts with other components in the repository, and with other repositories when that is actually relevant

Focus on system role and design rationale rather than line-by-line narration.

Avoid:

- pedantic walkthroughs
- restating obvious syntax
- overly long explanations without architectural insight

---

## Output format

When completing a task, include:

- assumptions
- changes made
- affected modules
- concise code explanation
- risks / follow-ups

Be concise and technical.
