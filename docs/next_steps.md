# Next Steps

## Right now

Merge `hyperparam_search_hiv.py` and `hyperparam_search_zinc.py` into a generic `hyperparam_search.py`:

- Include the dataset name as an argument for `train_LDNA`. You may feel free to reuse existing code such as 
  `utils.resolver.model_and_data_resolver(model_query='LDNA')`.
- Prefer minimal changes with clean code.
- Avoid broad refactors.

---

## High priority

- Expand `README.md` to include the project description, setup and usage.

---

## Future research directions (not immediate)

These are NOT current priorities.
Do NOT explore, implement, or optimize these unless explicitly asked.

- Add `ogbg-ppa` dataset
- Add `ogbg-code2` dataset
- Add `ogbn-products` dataset
- Add `ogbn-mag` dataset
- Add `CoraFull` dataset
- Add `Reddit` dataset
- Add `GraphSAINT` and `GraphSAINT` + `LDNA`
- Add `DGN` and `DGN` + `LDNA`

---

## Not now

- Large-scale refactors
- Schema redesign
