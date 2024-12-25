import pandas as pd
import numpy as np
import re

def clean_data(df: pd.DataFrame, mapper: dict[str, str], to_split: str, columns: list[str]) -> pd.DataFrame:
    # Copy the missing from the translation column
    exclude = ['corrected inflection','and','lemma uncertain', 'spelling error/correction']
    mask = df[to_split].isna() & ~df['translation'].isin(exclude)
    df[to_split] = df[to_split].fillna(df.loc[mask, 'translation'])
    shared = list(set(df.columns.values).intersection(columns))
    df = df.drop(shared, axis=1)
    df = df.drop('verse', axis=1)
    
    # Splitting
    for c in columns:
        pat_txt = '|'.join(mapper[c])
        pat_txt = r'\b(?:' + pat_txt + r')\b'
        patt = re.compile(pat_txt, re.IGNORECASE)
        df[c] = (
            df[to_split].map(patt.findall, na_action='ignore')
            .map(lambda x: ','.join(x) if len(x) else np.nan, na_action='ignore')
            )
    return df


def create_mapper(df: pd.DataFrame, columns: list[str]) -> dict[str, str]:
    mapper = {a:[
        val for sub in df[c].unique().tolist() if pd.notna(sub)
        for val in re.split(r'[, ]+', sub)
    ] for a, c in zip(columns, columns)}

    mapper['binyan'].extend(['nitpalpel', 'nitpoel', 'hithaphel'])
    mapper['tense'].append('passiveimperfect')
    mapper['subtype'].append('copulative')
    mapper['mood'].append('cohortativeheh')

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
mishna_fixed.to_csv('data/mishna_fixed.csv', index=False)
megilot_fixed.to_csv('data/megilot_fixed.csv', index=False)

