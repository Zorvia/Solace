<div align="center">

<img src="assets/solace.svg" alt="Solace Logo" width="100">

# Solace

**An NNUE-powered, ultra-aggressive UCI chess engine built on Stockfish architecture. Structured aggression meets measurable strength.**

---

<p>
<img alt="Build Status" src="https://img.shields.io/github/actions/workflow/status/Zorvia/Solace/build.yml?style=for-the-badge&logo=github&logoColor=white&color=0a0e14&label=build">
<img alt="Version" src="https://img.shields.io/github/v/release/Zorvia/Solace?style=for-the-badge&logo=semantic-release&logoColor=white&color=0a0e14">
<img alt="License" src="https://img.shields.io/github/license/Zorvia/Solace?style=for-the-badge&logo=open-source-initiative&logoColor=white&color=020617">
<img alt="Language" src="https://img.shields.io/github/languages/top/Zorvia/Solace?style=for-the-badge&logo=c%2B%2B&logoColor=white&color=0a0e14">
<img alt="Contributors" src="https://img.shields.io/github/contributors/Zorvia/Solace?style=for-the-badge&logo=github&logoColor=white&color=0a0e14">
</p>
</div>

---

## Overview

Solace is a **deterministic, NNUE-powered UCI chess engine** designed for:

- **Aggressive Initiative:** Prefers dynamic imbalance and king pressure over static material.  
- **Structured Sacrifices:** Evaluates risk vs. compensation statistically.  
- **Performance & Accuracy:** Deterministic search, fully benchmarked, optimized in C++.

Solace is ideal for players and researchers exploring stylistic engine bias and neural evaluation tuning.

---

## Features

- **NNUE Evaluation:** Neural-network-driven position scoring.  
- **Dynamic Imbalance Handling:** Sacrifices and attacks calculated for maximum effectiveness.  
- **UCI-Compatible:** Works with any UCI-compliant GUI.  
- **Open Source (GPLv3):** Full source code, auditable and modifiable.  
- **Deterministic & Benchmarkable:** Reproducible search results.  
- **High Performance:** Optimized for speed and efficiency in C++.

---

## Quick Start

### Clone & Build

```bash
git clone https://github.com/Zorvia/Solace.git
cd Solace/src
make -j profile-build
````

### Run

```bash
./solace
uci
```

Expected output:

```text
id name Solace
uciok
```

> Solace has no built-in GUI; use any UCI-compatible interface.

---

## Architecture

```mermaid
flowchart LR
    A[Position] --> B[Search]
    B --> C[NNUE Evaluation]
    C --> D[Aggression Bias]
    D --> E[Move Selection]
    E --> B
```

* Evaluates positions prioritizing **initiative, king pressure, mobility, space, and long-term compensation**.
* Aggression is calculated and statistical, not random.

---

## Aggression Model

```mermaid
graph TD
    A[Material] -->|Balanced| B[Traditional Engine]
    A -->|Dynamic Imbalance| C[Solace]
    C --> D[Increased Sacrifice Rate]
    C --> E[Higher King Attack Score]
    C --> F[Lower Draw Tendency]
```

* Aggression is balanced with measurable statistical results.
* Sacrifices are executed only when pressure outweighs material.

---

## Validation Pipeline

```mermaid
flowchart TD
    A[Build] --> B[Unit Tests]
    B --> C[Self-Play]
    C --> D[Match vs Baseline]
    D --> E[Elo Analysis]
    E --> F{Improvement?}
    F -- Yes --> G[Promote Net]
    F -- No --> H[Refine]
```

* Each update is compiled, stress-tested, and evaluated through Elo self-play.
* Strength and aggression are validated, ensuring reliability.

---

## Project Structure

```
Solace/
├── assets/          # Logos, images, SVGs
├── src/             # Engine source code
├── tests/           # Unit tests and validation scripts
├── examples/        # Sample configuration and run scripts
├── docs/            # Documentation
├── Makefile
└── README.md
```

---

## Contributing

* Follow [CONTRIBUTING.md](CONTRIBUTING.md).
* Clear, readable commits and educational clarity are prioritized.
* All contributions welcome.

---

## License

**GPLv3** — See [LICENSE](LICENSE) for full terms.

---

## Version

**v1.0.0** (Alpha) — Fully functional but may contain minor issues.

**© 2025-2026 Zorvia. All Rights Reserved.**


Do you want me to do that next?
```
