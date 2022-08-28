import numpy as np

for item in [4, 4, 4, 4, 4]:
    choces = np.random.choice(
                    4,
                    item, (0.05, 0.2, 0.05, 0.5, 0.05, 0.15))

print(choces)