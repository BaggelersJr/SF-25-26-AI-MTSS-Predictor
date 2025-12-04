import numpy as np
import pandas as pd
np.random.seed(23)
N = 200
sex = np.random.choice([0, 1], size=N, p=[0.50, 0.50])
# 0 = female, 1 = male
acwr = np.random.normal(1.1, 0.25, N) #Keep unrounded for more variability
acwr = np.clip(acwr, 0.6, 2.0)
bmi = np.random.normal(20.5, 2.0, N) #Keep unrounded for more variability
bmi = np.clip(bmi, 16, 28)  
history = np.random.choice([0, 1], size=N, p=[0.75, 0.25])
rest_days = np.random.choice([0, 1, 2, 3, 4, 5], size=N, p=[.40, .35, .15, .05, .03, .02])
mileage = np.random.normal(30,7,N)
mileage = np.clip(mileage, 10, 60)
logits = (
        1.3 * (acwr - 1.0) +
        .15 * (bmi - 20) +
        .04 * mileage +
        1.7 * history +
        .5 * (1 - sex) +
        (-.6 * rest_days) -
        3.0
)
prob = 1/(1 + np.exp(-logits))
outcome = np.random.binomial(1, prob)

df = pd.DataFrame({
    "Sex":sex,
    "ACWR":acwr,
    "BMI":bmi,
    "Injury_History":history,
    "Rest_Days":rest_days,
    "Weekly_Mileage":mileage.round(1),
    "Injury":outcome
})

df.to_csv("synthetic_MTSS_dataset.csv", index=False)
df.head()
