# CSCN8020 Assignment 2 — Q-Learning Taxi

This repository contains the implementation for Assignment 2 (Q-Learning) — a taxi agent that learns to pick up and drop off passengers using Q-learning.

Files
- Albright_9053136_QLearning_Taxi.py — main experiment and trainer
- assignment2_utils.py — helper functions used by the agent and environment
- assignment2_outputs/ — produced outputs and logs (e.g. `qlearning_summary.csv`)

Requirements
- Python 3.8+
- Recommended: create a virtual environment and install dependencies if any (standard library only unless extras are added).

Setup
1. (Optional) Create and activate a virtual environment:

	For Windows (PowerShell):
	```
	python -m venv .venv
	.venv\\Scripts\\Activate.ps1
	```

2. Install any required packages (none required by default). If you add dependencies, record them in a `requirements.txt`.

Run
1. Run the main training/experiment:

```
python Albright_9053136_QLearning_Taxi.py
```

2. Results and summaries will be saved to the `assignment2_outputs/` directory (for example `qlearning_summary.csv`).

Notes
- Modify hyperparameters directly in `Albright_9053136_QLearning_Taxi.py` as needed.
- If you want me to add a `requirements.txt`, CI, or more detailed usage examples, tell me and I will add them.