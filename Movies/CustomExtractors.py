import pandas as pd

def __rating_equivalent(rating):
    if rating == 'GP' or rating == 'PGPG' or rating == 'M/PG':
        return 'PG'
    elif rating == 'G(Rating':
        return 'G'
    elif rating == 'Open':
        return 'R'
    
    return rating

def extract_numbers(data, columns = None):
    columns = columns if columns else data.columns
    new_data = data.copy()
    for col in columns:
        new_data[col] = new_data[col].str.extract('(\d+)')
        new_data[col] = new_data[col].astype('float64')
    return new_data

def extract_rating(data, columns = None):
    new_data = pd.DataFrame(data.copy())
    columns = columns if columns else new_data.columns
    for col in columns:
        new_data[col] = new_data[col].str.extract('(Not Rated|[^\s]+)')[0].map(__rating_equivalent)
    return new_data
