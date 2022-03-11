# class to colorize , bold and underline output
class Color:
    PURPLE = '\033[95m'
    CYAN = '\33[96m'
    DARKCYAN = '\33[36m'
    BLUE = '\33[94m'
    GREEN = '\33[92m'
    YELLOW = '\33[93m'
    RED = '\33[91m'
    BOLD = '\33[1m'
    UNDERLINE = '\33[4m'
    END = '\33[0m'

import pandas as pd
import numpy as np
# define header for our data, the UCI dataset does not have a header
headers = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
url = "https://raw.githubusercontent.com/vbloise3/WhizLabsML/master/CensusIncome/CensusIncomeDataset.csv"
df = pd.read_csv(url, error_bad_lines=False, header=None, name=headers, na_values="null")
df.head(10)