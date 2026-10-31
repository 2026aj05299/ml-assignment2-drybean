# Dry Bean Variety Classification — ML Assignment 2

**Student:** Chaudhari Sandip Chandrakant Mamta  **BITS ID:** 2026AJ05296
**Live app:** <STREAMLIT_APP_URL>

## a. Problem statement

Identify which of seven registered dry-bean varieties a single bean belongs to, using 16 shape measurements extracted from a photograph of the bean (multi-class classification, 7 classes). Five classifiers — Logistic Regression, Decision Tree, K-Nearest Neighbours, Gaussian Naive Bayes and Support Vector Machine — are trained on the same data, compared on six evaluation metrics, and served through a Streamlit app that scores the saved models on an uploaded test file without retraining them.

## b. Dataset description

| Item | Detail |
|---|---|
| Name | Dry Bean Dataset (Koklu & Ozkan, 2020) |
| Source | UCI Machine Learning Repository, dataset id 602 (also on Kaggle) |
| Instances | 13,611 raw; 13,543 used after removing 68 exact duplicate rows (no missing values) |
| Features | 16 numeric (12 dimension features + 4 shape factors) |
| Target | `Class` — BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA |
| Class balance | DERMASON 26.2%, SIRA 19.5%, SEKER 15.0%, HOROZ 13.7%, CALI 12.0%, BARBUNYA 9.8%, BOMBAY 3.9% (moderately imbalanced) |

The beans were photographed with a high-resolution camera and each image was segmented to measure the bean's outline. In the formulas below, A = area, P = perimeter, L = major axis length, l = minor axis length.

| Feature | Meaning |
|---|---|
| Area (A) | Number of pixels inside the bean boundary |
| Perimeter (P) | Length of the bean boundary |
| MajorAxisLength (L) | Longest line that can be drawn through the bean |
| MinorAxisLength (l) | Longest line perpendicular to the major axis |
| AspectRation | L / l (column name as spelled in the source file) |
| Eccentricity | Eccentricity of the ellipse with the same moments as the bean |
| ConvexArea | Pixels in the smallest convex polygon enclosing the bean |
| EquivDiameter | Diameter of a circle with the same area: √(4A/π) |
| Extent | A / area of the bounding box |
| Solidity | A / ConvexArea (convexity) |
| roundness | 4πA / P² |
| Compactness | EquivDiameter / L |
| ShapeFactor1 | L / A |
| ShapeFactor2 | l / A |
| ShapeFactor3 | A / (π · (L/2)²) |
| ShapeFactor4 | A / (π · (L/2) · (l/2)) |

**Preprocessing**

- Missing-value check (none found) and removal of 68 duplicate rows.
- Stratified 80/20 train/test split (`random_state=42`): 10,834 training rows, 2,709 test rows. The test split is saved as `test_data.csv`.
- `StandardScaler` is fitted on training data only, inside each model's `Pipeline`, for Logistic Regression, KNN, Naive Bayes and SVM. Area is in the tens of thousands while ShapeFactor2 is about 0.001, so without scaling, distance- and margin-based models would be dominated by the size features. The Decision Tree uses raw features because its threshold splits are unaffected by scale.
- Hyperparameters are chosen with 5-fold stratified cross-validation on the **training set only** (`GridSearchCV`, scoring = macro F1). The test set is used exactly once, for the final numbers below.

## c. GitHub repository

Link: <https://github.com/2026aj05299/ml-assignment2-drybean>

```
ml-assignment2-drybean/
├── app.py                       # Streamlit app
├── requirements.txt
├── README.md
├── test_data.csv                # held-out 20% test split (16 features + Class)
├── .streamlit/config.toml       # app theme
├── data/
│   └── Dry_Bean_Dataset.csv     # full raw dataset (UCI id 602)
└── model/
    ├── train_models.py          # loading, cleaning, split, tuning, training, evaluation
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    ├── feature_columns.json     # feature list, class names, tuned hyperparameters
    └── metrics.csv              # test-set metrics for all five models
```

Reproduce:

```bash
pip install -r requirements.txt
python model/train_models.py
streamlit run app.py
```

## d. Models used

| Model | Configuration (best from 5-fold CV) |
|---|---|
| Logistic Regression | StandardScaler → LogisticRegression(C=10, max_iter=5000), multinomial softmax |
| Decision Tree | DecisionTreeClassifier(max_depth=10, min_samples_leaf=5) |
| KNN | StandardScaler → KNeighborsClassifier(n_neighbors=21, weights="distance") |
| Naive Bayes | StandardScaler → GaussianNB(var_smoothing=1e-3) |
| Support Vector Machine | StandardScaler → SVC(kernel="rbf", C=10, gamma="scale", probability=True) |

**Evaluation metrics (multi-class).** For each variety *k*, TPₖ, FPₖ and FNₖ are read from the 7 × 7 confusion matrix.

- Accuracy = (correct predictions) / (all predictions)
- Precisionₖ = TPₖ / (TPₖ + FPₖ); Recallₖ = TPₖ / (TPₖ + FNₖ); F1ₖ = 2·Precisionₖ·Recallₖ / (Precisionₖ + Recallₖ)
- Precision, Recall and F1 below are **macro averages**: the plain mean over the 7 varieties, so the small BOMBAY class counts as much as DERMASON.
- AUC: one-vs-rest ROC AUC for each variety from the predicted class probabilities, then macro-averaged.
- MCC (multi-class form, Gorodkin 2004): uses the whole confusion matrix; ranges from −1 to +1, with 0 = no better than chance. With c = correct predictions, s = total samples, pₖ = times *k* was predicted, tₖ = times *k* truly occurs:
  MCC = (c·s − Σ pₖtₖ) / √((s² − Σ pₖ²)(s² − Σ tₖ²))

### Comparison table (test set, n = 2,709)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9214 | **0.9949** | 0.9329 | 0.9309 | 0.9317 | 0.9050 |
| Decision Tree | 0.9051 | 0.9777 | 0.9204 | 0.9194 | 0.9197 | 0.8853 |
| KNN | 0.9192 | 0.9912 | 0.9334 | 0.9294 | 0.9309 | 0.9023 |
| Naive Bayes | 0.8974 | 0.9912 | 0.9088 | 0.9084 | 0.9076 | 0.8766 |
| Support Vector Machine | **0.9236** | 0.9944 | **0.9356** | **0.9340** | **0.9347** | **0.9076** |

Best value per metric in bold. Majority-class baseline (always predict DERMASON) accuracy = 0.262, so every model is far above chance.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Second-best overall (MCC 0.9050) and the **highest AUC (0.9949)**, so its probability ranking of varieties is the best of all five. Once the features are standardised the varieties are close to linearly separable; CV preferred weak regularisation (C = 10). Its weakest variety is SIRA (F1 0.876), and its most frequent error is DERMASON predicted as SIRA (53 beans). A strong result for the simplest and fastest-to-predict model. |
| Decision Tree | Fourth on MCC (0.8853) and the **lowest AUC (0.9777)**: a tree's leaf probabilities are coarse step values, so it ranks beans less smoothly than the other models. Its splits are axis-parallel, which approximates the slanted boundaries between similar-sized varieties poorly; it has the lowest SIRA F1 of any model (0.839). On the plus side it is fully interpretable — the most important splits are ShapeFactor1, MajorAxisLength, ShapeFactor3 and Perimeter — and needs no scaling. |
| KNN | Third (MCC 0.9023), within 0.003 of Logistic Regression. Distance weighting with k = 21 smooths out noisy neighbours, and its CALI F1 (0.933) is the best of all models. Scaling is essential here: without it, Area would dominate every distance. It stores the whole training set (largest saved model, about 1.5 MB) and does all its work at prediction time. |
| Naive Bayes | Lowest Accuracy (0.8974), F1 (0.9076) and MCC (0.8766), yet its AUC (0.9912) is still high, so it ranks varieties well but its hard predictions are worse. The reason is its independence assumption: 21 of the 120 feature pairs have absolute correlation above 0.9 (Area, Perimeter, ConvexArea and EquivDiameter all measure size), so the same evidence is counted several times and probabilities become over-confident. It makes the most DERMASON→SIRA errors (78) and has the weakest BARBUNYA F1 (0.833). It is by far the fastest to train. |
| Support Vector Machine | **Best on Accuracy (0.9236), Precision, Recall, F1 (0.9347) and MCC (0.9076)**, and second on AUC (0.9944). The RBF kernel draws curved boundaries, which helps on the overlapping DERMASON/SIRA region. The cost is training time: it was the slowest model by roughly an order of magnitude, because probability estimates require internal cross-validated calibration. |
| Overall winner | **Support Vector Machine** — best on five of six metrics. The margin is small: Logistic Regression and KNN are within 0.005 MCC, so Logistic Regression is a reasonable choice when speed or simplicity matters. All five models classify BOMBAY (the largest beans) almost perfectly (F1 ≥ 0.995), and for every model the main source of error is DERMASON vs SIRA, two varieties of similar size and shape. |

## Streamlit app features

- CSV upload of test data, with a downloadable sample `test_data.csv` in the sidebar and clear errors for missing columns or unknown labels
- Model selection dropdown (all five models) showing the tuned hyperparameters
- Six evaluation metrics for the selected model
- Confusion matrix (counts or per-variety share) and classification report
- "By variety" tab: per-variety F1/recall chart and the most frequent mix-up
- "Compare all models" tab: metrics table with the best model per metric highlighted, plus a grouped bar chart
- Row-level predictions with confidence, a filter for misclassified beans, and CSV download
