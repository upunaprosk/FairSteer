# FairSteer
The code for FairSteer: Inference Time Debiasing for LLMs with Dynamic Activation Steering.

### Run the code

Step 1: Collect activations

```python
python get_activations.py
python get_activations_probes.py
python combine_activations_probe.py
```

Step 2: Get DSVs and probes

```python
python get_steering_vectors.py
python get_probes.py
```

Step 3: Select a layer

```python
python select_layer.py
```

Step 4: Evaluate

```python
python inference_time_debias.py
```

