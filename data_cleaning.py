import pandas as pd
import numpy as np
import re

def clean_data(df: pd.DataFrame, mapper: dict[str, str], to_split: str, columns: list[str]) -> pd.DataFrame:
    
    # Copy the missing from the translation column
    mask = df[to_split].isna()
    indices = df[mask].index
    df.loc[indices, to_split] = df.loc[indices, 'translation']
    df.loc[:, columns] = np.nan
    df = df.astype('object')

    # Normalize the fts column
    df.loc[:, to_split] = df.loc[:, to_split].str.strip("[]")
    
    # Splitting
    for i in df.index.values:
        # print(i)
        fts = df.at[i, to_split]
        for val in re.split('[, ]+', fts):
            if val.lower() in mapper:
                c = mapper[val.lower()]
                exists = df.at[i, c]
                if pd.isna(exists):
                    df.at[i, c] = val
                else:
                    df.at[i, c] = ','.join([exists, val])
            else:
                if fts != df.at[i, 'translation']:
                    print('Error!')
                    print(f'row: {i}\nfts: {fts}\nvalue: {val} has no matching column!')
    
    return df


def create_mapper(df: pd.DataFrame, cols: list[str]) -> dict[str, str]:
    mapper = {}
    for col in cols:
        for un in df[col].unique():
            if pd.isna(un) == False:
                for val in str.split(un, ','):
                    mapper[val.lower()] = col
    mapper['nitpalpel'] = 'binyan'
    mapper['nitpoel'] = 'binyan'
    mapper['hitpoel'] = 'binyan'
    mapper['hithaphel'] = 'binyan'
    mapper['passiveimperfect'] = 'tense'
    mapper['copulative'] = 'subtype'
    mapper['cohortativeheh'] = 'mood'
    return mapper


mikra = pd.read_csv('data/mikra.csv')
mishna = pd.read_csv('data/mishna.csv')
megilot = pd.read_csv('data/megilot.csv')
cols = ['pos', 'number', 'gender', 'tense', 'person', 'binyan', 'state', 'mood', 'subtype', 'other']
mapper = create_mapper(mikra, cols)
print("Before cleaning the mishna:")
mishna.info()
print("After cleaning")
mishna_fixed = clean_data(mishna, mapper, 'fts', cols)
mishna_fixed.info()
print("Before cleaning the megilot:")
megilot.info()
print("After cleaning")
megilot_fixed = clean_data(megilot, mapper, 'fts', cols)
megilot_fixed.info()

# Saving the fixed to csv
mishna_fixed.to_csv('data/mishna_fixed.csv', index=False, quotechar='"', quoting=1)
megilot_fixed.to_csv('data/megilot_fixed.csv', index=False, quotechar='"', quoting=1)

