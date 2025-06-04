# flearn

---

## 🧠 Core Modules

- `core/federated_averaging.py`  
  Implements Federated Averaging logic for distributed training.

- `core/train.py`  
  General-purpose training loop logic.

- `core/model_editing.py`  
  Logic for computing Fisher information, gradient masking, and model pruning.

- `core/training_params.py`  
  Configuration and validation for training jobs.

---

## 🛠️ Bin Modules

Scripts and executables for launching training, evaluation, and experiments.  
Located under `bin/`.

---

## 🏛️ Model Modules

Implementations of various deep learning architectures.  
Found under `models/`.

---

## Optim Modules

Implementation of custom optimizers, e.g. SparseSGD.
Found under `optim/`.

---

## 📂 Dataset Modules

Provides standardized PyTorch-style dataloaders for various common datasets.  
Located under `dataset/`.

---

## 🧰 Util Modules

Utility functions for logging, metric tracking, data preprocessing, and more.  
Found under `utils/`.

---

## 🧪 Test Modules

**PyTest** tests located in the `test/` directory.  

---

## ✅ Unit Tests

To run all tests:

```bash
pytest test/
