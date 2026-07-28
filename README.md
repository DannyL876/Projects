# Project Notes: Probability of Default Model

# Objective
Build a model to predict the probability that a borrower will default on credit within two years, using historical borrower data.

# Dataset
- **Source:** "Give Me Some Credit" dataset (Kaggle)
- **Size:** ~150,000 borrower records
- **Target variable:** `SeriousDlqin2yrs` (1 = defaulted within 2 years, 0 = did not)

# Checking Data
- Checked for missing values across all columns
- Two columns had gaps: `MonthlyIncome` (~29,700 missing) and `NumberOfDependents` (~3,900 missing)
- Filled both using median imputation rather than mean, since income data is skewed by high earners, therefore median gives a more representative value

# Feature Selection (Initial Model)
Selected 8 features based on relevance to credit risk:
- RevolvingUtilizationOfUnsecuredLines
- Age
- NumberOfTime30-59DaysPastDueNotWorse
- DebtRatio
- MonthlyIncome
- NumberOfOpenCreditLinesAndLoans
- NumberOfTimes90DaysLate
- NumberOfDependents

# Method
- Split data 80/20 into training and test sets
- Trained a **logistic regression** model (standard baseline method for binary classification/credit risk problems)
- Evaluated using **AUC-ROC score** (measures how well the model separates defaulters from non-defaulters; 0.5 = random, 1.0 = perfect)

# Initial Result
- **AUC: 0.66** — better than random, but too weak to reliably assess credit risk in practice
- Tried scaling features (StandardScaler) to check if that improved things but little happened going from 0.662 to 0.666, thus suggesting scaling wasn't the limiting factor

# Identifying an Issue
Checked model coefficients to understand which features were driving predictions. I Found that `NumberOfTimes90DaysLate` had a negative coefficient which is very unlikely since more severe late payments should increase default risk, not decrease it.

**Diagnosis:** The issue was likely multicollinearity because `NumberOfTime30-59DaysPastDueNotWorse` and `NumberOfTimes90DaysLate` are correlated as someone 90 days late was often also 30-59 days late previously, so the model struggled to separate their individual effects, producing unlikely and counterintuitive coefficients.

# Fix
Combined both late-payment columns into a single binary feature: `EverLate` (1 if either column > 0, else 0).

# Improved Result
- **AUC improved from 0.66 to 0.79**
- Checked the coefficients of the new model: `EverLate` now shows a strong, clearly positive coefficient (2.14) which is a far more logical and interpretable result, confirming the fix worked

# Key Takeaways
- A simple correct diagnosis of a data issue improved model performance more than any parameter tuning would have
- Demonstrates the importance of checking not just overall performance (AUC) but also whether individual feature relationships make sense
- Reinforces that correlated features can distort logistic regression coefficients even when overall accuracy looks reasonable

# Limitations
- Logistic regression assumes linear relationships between features and default risk so may miss more complex patterns
- Did not test other model types (e.g. random forest, gradient boosting) which might capture non-linear relationships better
- Dataset is anonymised and historical so does not account for real world factors, would need to account for changing economic conditions not reflected in this data

# Possible Next Steps 
- Compare against a random forest or gradient boosting model
- Plot an ROC curve for visual comparison
- Test the interaction between features. For example someone with low income and high debt ratio is a much bigger risk than either factor alone suggests. 

