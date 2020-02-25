import pandas as pd
import os

def savetable(filename, df):
    file_path = os.path.join(os.getcwd(), 'tables')
    os.makedirs(file_path, exist_ok=True)
    pd.DataFrame(df).to_latex(os.path.join(file_path, filename),escape = False, index = False)

# %% IMPORT

pd.set_option("display.precision", 4)
pd.set_option('display.float_format', lambda x: '%.3f' % x if (x > 10**-2) else '%.3e' % x)

df = pd.read_excel("DataLab3Student.xlsx", header=1, usecols="J:O")
df.dropna(inplace=True)

savetable("result.tex", df)

# %%
