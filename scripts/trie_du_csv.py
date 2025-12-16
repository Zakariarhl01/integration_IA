import pandas as pd

df = pd.read_csv("../data/energiTech_maintenance_sample.csv")

df_sorted = df.sort_values(by=["turbine_id", "date_measure"])

df_sorted.to_csv("../data/energiTech_par_turbine.csv", index=False)

print("Fichier trié créé : energiTech_par_turbine.csv")
