# DeforestationLDA&DecisionTrees
## LDA & Decision Trees: Classifying Countries at Critical Forest Loss Risk

---

### 📖 Purpose of Study

Can socioeconomic and demographic data alone identify which countries are actively destroying their forests — and do two fundamentally different algorithms agree on *why*?

This project is the third phase of an ongoing classification series. Building directly on the binary target variable and cleaned dataset established in the [Logistic Regression phase](https://rebeca-bc.github.io/DeforestationLogisticClassifier/), the focus here shifts to comparing two structurally different classifiers: **Linear Discriminant Analysis (LDA)** and a **Decision Tree**. The central question is not just which model performs better, but whether both independently converge on the same ecological story — and what that convergence means.

LDA approaches the problem geometrically, finding the linear combination of features that best separates the two classes. The Decision Tree approaches it as a sequence of binary rules. The fact that both methods, with entirely different mathematical foundations, arrive at the same primary predictor is one of the most important findings of this project.

---

### 📊 The Data

The dataset (`classified_deforestation_df.csv`) is inherited from the previous phase — a pre-processed combination of environmental and socioeconomic country-level records (n = 103 countries after filtering non-forested nations). The target variable and predictor features were constructed and validated in the prior repository.

**Target Variable:**
- `Deforestation_Critical`: Binary label — `1` (High Risk: annual deforestation rate > 0.501% of forested area) or `0` (Low Risk). Threshold sourced from Teo et al., PNAS 2024 [1]. Approximately **70% Class 0 / 30% Class 1**.

**Predictor Features (selected subset — full list in notebook):**
- `Infant Mortality`: Deaths per 1,000 live births — the single strongest predictor identified by both models.
- `Minimum Wage`, `Urban Population`, `Labor Force Participation (%)`: Demographic and economic pressure indicators.
- `Longitude`: Geographic position — captures tropical and equatorial belt effects not explained by economics alone.
- `Physicians per Thousand`, `Density`, `CPI`, `Gasoline Price`, `Armed Forces Size`, `Total Tax Rate`, `Agricultural Land (%)`, `Out of Pocket Health Expenditure`: Broader socioeconomic and institutional capacity proxies.

---

### 🛠 Main Conceptual Applications

The core purpose of this notebook is to apply two contrasting supervised classification methods to the same ecological problem and rigorously compare their behavior. The key technical applications shown are:

- **LDA Assumptions Audit:** Verifying multicollinearity (correlation threshold |r| > 0.80) and checking Gaussian class-conditional distributions via KDE plots before fitting — two assumptions LDA explicitly requires.
- **Discriminant Scaling Analysis:** Extracting `lda.scalings_` from a baseline fit to rank feature contributions to the LD1 axis, using this as a principled feature selection step before the final model.
- **Leakage-Free Scaling:** `StandardScaler` is fit exclusively on `X_train` and applied to `X_test` — re-fitted after each feature reduction step to prevent data leakage.
- **Wild Tree Baseline:** An unconstrained `DecisionTreeClassifier` (`max_depth=None`) is trained first to expose High Variance (train accuracy 1.0 vs. test 0.71) before any tuning.
- **Gini Feature Importance:** `feature_importances_` extracted from the wild tree to identify implicit feature selection by the algorithm — no manual pruning needed.
- **GridSearchCV with 5-Fold Cross-Validation:** Systematic hyperparameter search over `max_depth`, `min_samples_split`, and `min_samples_leaf`, evaluated across 5 folds to ensure the best parameters reflect genuine generalization.
- **Diagnostic Metrics:** Recall, Precision, F1-Score, Confusion Matrix, and ROC/AUC evaluated for both models with explicit attention to the ecological asymmetry between False Negatives and False Positives.

---

### 🚀 Key Findings

The full analysis, decisions, and narrative can be found in the notebook. The main findings are:

**Infant Mortality is a universal deforestation predictor.** Both LDA (highest scaling weight: $w = 1.46$) and the Decision Tree (root split, highest Gini importance) independently identified Infant Mortality as the dominant feature. This variable acts as a composite proxy for development level, institutional capacity, and environmental governance quality — and neither algorithm needed to be told so.

**Both models agree geographically.** Longitude appeared as a significant signal in both models, confirming that deforestation risk is not randomly distributed but regionally clustered — predominantly in tropical and equatorial zones. Geography captures something that socioeconomic variables alone cannot fully explain.

**LDA outperforms the Decision Tree on the minority class.** The optimized LDA achieved a Recall of 0.667 and AUC of 0.80 on the test set. The pruned Decision Tree, despite better interpretability, achieved a Recall of only 0.17 for High Risk countries — catching just 1 of 6 critical cases. With `max_depth=3`, the tree's simplicity came at the cost of sensitivity to the minority class.

**The Decision Tree's logic is highly interpretable.** The tree visualization reveals a coherent three-level rule: development level (Infant Mortality) → geographic region (Longitude) → secondary socioeconomic signal. One pure leaf node (`[0, 11]`) perfectly captures 11 High Risk countries with zero errors — a strong, clean signal that is easy to communicate to non-technical stakeholders.

**For ecological monitoring, LDA is the safer choice.** A missed high-risk deforestation country (False Negative) carries a far higher cost than a false alarm. LDA's higher recall makes it the more responsible operational model for this domain, despite the Decision Tree's interpretability advantage.

---

### 📁 Project Files

- `LDA&DecisionTrees.ipynb`: The full analysis — LDA pipeline, Decision Tree baseline, tuning, tree visualization, and comparative findings.
- `classified_deforestation_df.csv`: The dataset inherited from the previous project phase, cleaned and ready for modeling.

---

### References

[1] Teo, H. C., Sarira, T. V., Tan, A. R. P., Cheng, Y., & Koh, L. P. (2024). Charting the future of high forest low deforestation jurisdictions. *Proceedings of the National Academy of Sciences*, 121(37), e2306496121. https://doi.org/10.1073/pnas.2306496121

[2] Bhuva, L. (2025). Understanding Decision Trees and Hyperparameter Tuning in Machine Learning. *Medium*. https://medium.com/@lomashbhuva/understanding-decision-trees-and-hyperparameter-tuning-in-machine-learning-c0a4467a1e69
