# Industry ML Pipeline: Splitting, Validation & Tuning Strategy

## The Core Mental Model

Forget the academic single-split or naked `cross_val_score`. In production, the question is always:
**"How do I get a reliable estimate of real-world performance without leaking information and without wasting compute?"**

The answer depends on three variables:
1. **Data type** (tabular, image, text, time-series)
2. **Dataset size** (< 5k, 5k–100k, > 100k)
3. **Task stage** (screening many models vs. tuning one final model)

---

## The Two Phases Everyone Skips in Kaggle

| Phase | Goal | Method |
|---|---|---|
| **Screening** | Quickly eliminate bad model families | Fast CV or lightweight hold-out |
| **Tuning** | Squeeze performance from the winner | Rigorous CV + nested validation |

Kaggle collapses both phases together and throws everything at cross-validation simultaneously. Production separates them deliberately to save time and avoid overfitting the tuning process itself.

---

## Decision Tree by Data Type & Size

---

### Case 1: Tabular Data — Small (< 5,000 rows)

**The real-world risk:** Variance is enormous. A single 80/20 split will lie to you. A random seed change can shift accuracy by 3–5%.

**Industry Pipeline:**

```
Raw Data
  │
  ▼
[1. Hard Hold-Out — 15-20%] ← Lock this. Never touch until final evaluation.
  │
  ▼
[2. Screening Phase — Remaining 80-85%]
    - Use Stratified K-Fold (k=5 or k=10) on training data only
    - Run 3–5 model families (LR, RF, XGB, SVM, LightGBM) with DEFAULT hyperparameters
    - Rank by mean CV score ± std deviation
    - Eliminate anything clearly worse (e.g., >1 std below the leader)
  │
  ▼
[3. Tuning Phase — Top 1–2 models only]
    - Use Repeated Stratified K-Fold (5×10 or 3×10) for stable estimates
    - Bayesian Optimization (Optuna) over a DEFINED search space
    - Budget: ~50–200 trials max
    - Track: mean CV score + std to detect overfitting the search
  │
  ▼
[4. Final Evaluation]
    - Retrain best config on full training data (80–85%)
    - Evaluate ONCE on the locked hold-out (15–20%)
    - This number is your production estimate.
```

**Key rule:** The hold-out is evaluated **exactly once**. If you evaluate it multiple times to pick a model, it becomes part of your training loop, and your estimate is optimistic garbage.

---

### Case 2: Tabular Data — Medium (5k – 100k rows)

**The real-world risk:** Less variance, but the temptation to skip screening still costs you hours of wasted compute.

**Industry Pipeline:**

```
Raw Data
  │
  ▼
[1. Hard Hold-Out — 15%]
  │
  ▼
[2. Screening Phase]
    - Stratified K-Fold (k=5) with default params
    - This time you can afford 5–10 model families
    - Screening budget: minutes, not hours
  │
  ▼
[3. Tuning Phase]
    - Standard Stratified K-Fold (k=5) is usually enough — variance is lower
    - Optuna or RandomizedSearch (100–300 trials)
    - Optional: Ensemble top-2 models if they have low correlation
  │
  ▼
[4. Final Model]
    - Retrain on full train set, evaluate once on hold-out
```

---

### Case 3: Tabular Data — Large (> 100k rows)

**The real-world risk:** Cross-validation becomes slow. The data itself provides enough variance stability.

**Industry Pipeline:**

```
Raw Data
  │
  ▼
[1. 70% Train / 15% Validation / 15% Test split]
  │                  (hold-out)
  ▼
[2. Screening Phase]
    - Single train/val split is sufficient at this scale
    - Run screening in hours, not days
  │
  ▼
[3. Tuning Phase]
    - Tune against the validation set
    - Optuna with early stopping
    - Watch for validation performance plateaus
  │
  ▼
[4. Final Model]
    - Retrain on train+val, evaluate once on test
```

> **Note:** At this scale, many teams skip k-fold entirely. Microsoft, Spotify, and similar companies use large fixed validation sets and care more about **distribution shift monitoring** than CV scores.

---

### Case 4: Deep Learning (any size)

Deep learning completely breaks the classic CV paradigm. Here's why:

- Training is stateful (weights evolve over epochs)
- One "fold" = one full training run = hours or days
- 10-fold CV = 10x compute budget — not feasible in production

**Industry Pipeline:**

```
Raw Data
  │
  ▼
[1. 80% Train / 10% Validation / 10% Test]
  │
  ▼
[2. Screening Phase — Architecture Search]
    - Train small proxy models on 10–20% of data
    - Compare architectures (CNN vs. ResNet vs. ViT for images)
    - Metric: validation loss trend, not final accuracy
    - Budget: few hours per candidate
  │
  ▼
[3. Hyperparameter Tuning]
    - Tune on full training data with early stopping on validation set
    - Key hyperparams: learning rate (most important), batch size, weight decay
    - Use LR finder (fastai/PyTorch Lightning) before tuning anything else
    - Tool: Optuna with Median Pruner (kills bad trials early)
    - Budget: 20–50 trials max
  │
  ▼
[4. Callbacks & Regularization]
    - EarlyStopping on val_loss (patience=5–10)
    - ReduceLROnPlateau or CosineAnnealingLR
    - Checkpoint the best epoch, not the last
  │
  ▼
[5. Final Evaluation]
    - Load best checkpoint, evaluate once on test set
```

**When DL teams DO use "folds":** Only in competitions or when dataset is < 10k and each sample is critical (medical imaging). They use 5-fold where each fold trains a full model, then **ensemble** all 5 predictions. This is Kaggle territory, not standard production.

---

### Case 5: Time-Series Data (any model type)

**The golden rule: Never use random splitting. Time must flow forward.**

**Industry Pipeline:**

```
Raw Data (sorted by time)
  │
  ▼
[1. Chronological Split]
    - Train: first 70% of timeline
    - Validation: next 15%
    - Test: last 15%
  │
  ▼
[2. Walk-Forward Validation (Expanding Window)]
    │
    ▼  Fold 1: Train [t0→t3] → Validate [t4]
    ▼  Fold 2: Train [t0→t4] → Validate [t5]
    ▼  Fold 3: Train [t0→t5] → Validate [t6]
    ...
    This mimics how the model will actually be updated in production.
  │
  ▼
[3. Tuning]
    - Tune against walk-forward mean score
    - Sliding window variant if data distribution shifts over time
  │
  ▼
[4. Final Evaluation]
    - Test set is the actual future — evaluate once
```

---

## Kaggle vs. Production: The Key Differences

| Dimension | Kaggle | Production |
|---|---|---|
| **Goal** | Maximize leaderboard score | Reliable generalization + deploy |
| **Hold-out** | Public/private LB (out of your control) | You own and lock it |
| **CV usage** | CV everything, every time | CV for screening; reduce for tuning |
| **Ensembling** | Always (stacking, blending) | Rarely (complexity vs. marginal gain) |
| **Tuning budget** | Unlimited (competitions run days/weeks) | Constrained (hours, cost-aware) |
| **Data leakage** | Often tolerated or exploited | Zero tolerance |
| **Splitting on time-series** | Often ignored (causes LB shake-up) | Always chronological |
| **Final metric** | Private leaderboard | Business KPI + monitored in production |
| **Model complexity** | Push to the max | Prefer simpler, interpretable, maintainable |

---

## The Production Decision Flowchart (Summary)

```
Start
 │
 ├─ Is it time-series? ──YES──► Walk-Forward CV + chronological split
 │
 └─ NO
     │
     ├─ Deep Learning? ──YES──► Fixed 80/10/10 + EarlyStopping + val set tuning
     │
     └─ Tabular
         │
         ├─ < 5k rows  ──► Repeated Stratified KFold (5×10) + locked hold-out
         ├─ 5k–100k    ──► Stratified KFold (k=5) + locked hold-out
         └─ > 100k     ──► Single large validation set + test hold-out
```

---

## What Leaks Performance (Common Mistakes)

1. **Fitting a scaler on the full dataset then splitting** — the scaler has seen test data.
2. **Doing feature selection before splitting** — you've used test information to choose features.
3. **Evaluating hold-out multiple times** — it is no longer a hold-out.
4. **Using CV score to report final performance** — always report hold-out, not CV mean.
5. **Ignoring class imbalance in splits** — always use `stratify=y` in `train_test_split`.
6. **Random split on time-series** — future data bleeds into training.

---

## The One Pipeline to Rule Them All (Tabular, Production Template)

```python
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import optuna

# Step 1: Lock hold-out first
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

# Step 2: Screening with default params (do this manually per model family)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Step 3: Tuning with Optuna (only on winning model)
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
    }
    model = Pipeline([("scaler", StandardScaler()), ("clf", XGBClassifier(**params))])
    scores = cross_val_score(model, X_dev, y_dev, cv=cv, scoring="roc_auc")
    return scores.mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

# Step 4: Final fit and ONE test evaluation
best_model = Pipeline([...best params...])
best_model.fit(X_dev, y_dev)
test_score = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
print(f"Production estimate: {test_score:.4f}")
# Never touch X_test again.
```

---

## Practical Rules to Memorize

- **Lock your test set first. Always.**
- **Screen with default params. Tune only the winner.**
- **CV is for estimation, not for reporting. Report hold-out.**
- **Pipelines prevent leakage. Use them.**
- **Time-series = no random splits. Ever.**
- **Deep learning = validate on a fixed val set, not k-fold.**
- **Kaggle tricks (leaky features, stacking) belong in Kaggle, not in code that ships.**
