import numpy as np
import pandas as pd
np.random.seed(23)
N = 200
sex = np.random.choice([0, 1], size=N, p=[0.50, 0.50])
acwr = np.random.normal(1.1, 0.25, N)
acwr = np.clip(acwr, 0.6, 2.0)
bmi = np.random.normal(20.5, 2.0, N)
bmi = np.clip(bmi, 16, 28)  
history = np.random.choice([0, 1], size=N, p=[0.75, 0.25])
rest_days = np.random.choice([0, 1, 2, 3, 4, 5], size=N, p=[.40, .35, .15, .05, .03, .02])
mileage = np.random.normal(30,7,N)
mileage = np.clip(mileage, 10, 60)