# Usage Guide: Running and Extending the Cosmic Emanator

## Running Simulations
1. Install: `pip install -r requirements.txt`
2. Notebook: e.g., `jupyter notebook notebooks/entropy_sim.ipynb`—follow cells for results.
3. Custom: Import `src/tfnp_layer.py`; e.g.:
```python
from tfnp_layer import TFNPLayer
model = nn.Sequential(TFNPLayer(784, 128), nn.Linear(128, 10))
# Train as in MNIST demo
