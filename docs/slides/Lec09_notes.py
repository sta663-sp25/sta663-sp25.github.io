import numpy as np
import pandas as pd

## Exercise 1

df = pd.read_csv("https://sta663-sp25.github.io/slides/data/us_rent.csv")

( df
  .pivot(index="name", columns="variable", values="estimate")
  .assign(
    ratio = lambda d: d.income / d.rent
  )
  .sort_values("ratio", ascending=True)
)
