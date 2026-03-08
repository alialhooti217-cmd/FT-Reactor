FT-Reactor/
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── training.yaml
│   └── user_job.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
├── notebooks/
│   └── ft_reactor_exploration.ipynb
├── src/
│   └── ft_reactor/
│       ├── __init__.py
│       ├── constants.py
│       ├── asf.py
│       ├── mass_balance.py
│       ├── energy.py
│       ├── model.py
│       ├── batch_runner.py
│       ├── surrogate.py
│       ├── optimizer.py
│       └── io_utils.py
├── tests/
│   ├── test_asf.py
│   ├── test_mass_balance.py
│   └── test_model.py
└── main.py
