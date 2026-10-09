# Datasets

## `wellhead_integrity_real.csv` — real oil & gas wellhead process data

Real steady-state process snapshots from oil & gas wellheads, used to build and validate
the corrosion-risk detector on data no one designed for this project (see *Results on real
wellhead data* in the top-level README).

**Source.** The [`maisaian/oil-gas-leakage`](https://huggingface.co/datasets/maisaian/oil-gas-leakage)
dataset on Hugging Face (the `train` split, 10,292 rows). It is pulled through the public
Hugging Face **datasets-server rows API** (no authentication) and vendored here verbatim —
the file keeps the upstream column names exactly, so it stays a faithful copy.

**Columns (all `float64`).**

| Column                              | Unit     | Meaning                                                      |
| ----------------------------------- | -------- | ------------------------------------------------------------ |
| `Wellhead Temp. (C)`                | °C       | wellhead flowing temperature                                 |
| `Wellhead Press (psi)`              | psi      | wellhead operating pressure                                  |
| `MMCFD- gas`                        | MMscf/d  | gas production rate                                          |
| `BOPD (barrel of oil produced per day)` | bbl/day | oil production rate                                      |
| `BWPD (barrel of water produced per day)` | bbl/day | water production rate                                  |
| `BSW - basic solid and water (%)`   | %        | basic sediment & water fraction                              |
| `CO2 mol. (%) @ 25 C & 1 Atm.`      | mol %    | CO2 content of the produced gas                              |
| `Gas Grav.`                         | —        | gas specific gravity (air = 1)                               |
| `CR-corrosion defect`               | —        | measured internal corrosion rate (the integrity signal)      |

10,292 wells. Readings are physically sane: temperature 41–74 °C, pressure 382–2,317 psi,
oil 129–2,087 BOPD, water 41–9,314 BWPD, CO2 0.68–4.30 mol %. There are no missing values.

### How the target is derived

PipelineGuard frames integrity as **detection**, not exact-rate regression. `CR-corrosion
defect` is the measured corrosion rate; a well is labelled **`high_risk = 1`** when its
corrosion rate is at or above the **75th percentile of the fleet** (≈ 0.2328), i.e. the
worst quartile — the wells a reliability team prioritises for inspection. That threshold is
a fixed operating definition applied identically to every row.

`CR-corrosion defect` is used **only** to build this label. It is never fed to the model as
a feature, and no feature is a transform of it (every input is built from the eight process
sensors in `pipelineguard/features.py`). This is the leakage guard: the corrosion reading
that defines the target never leaks into the inputs that predict it.

### Regenerating

```bash
python -c "from pipelineguard.real_data import fetch_wellhead_data; print(fetch_wellhead_data())"
```

Needs outbound access to `datasets-server.huggingface.co`. The licence and terms of the
original `maisaian/oil-gas-leakage` dataset apply; this copy is vendored for reproducible
benchmarking only.
