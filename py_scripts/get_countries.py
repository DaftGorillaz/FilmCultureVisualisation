import pandas as pd
from imdbinfo import get_movie
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import ValidationError
from pathlib import Path

def fetch_one(imdb_id):
    '''
    @ Params
    List of IMDB IDs 'tt000001'

    Returns: DataFrame of (imdb_id, [countries])
                DataFrame of (imdb_id, error_message)
    '''
    try:
        m = get_movie(imdb_id)
        countries = getattr(m, "countries", None) or []
        print(f'getting {m.title}')
        return {"movie_id": imdb_id, "countries": countries, "error": None}
    except ValidationError as e:
        return {"movie_id": imdb_id, "countries": [], "error": f"ValidationError: {e}"}
    except Exception as e:
        return {"movie_id": imdb_id, "countries": [], "error": repr(e)}

def get_countries_fast(imdb_ids, max_workers=16, checkpoint_path=None):
    # 1) de-duplicate and normalize to 'tt' format
    seen = set()
    norm_ids = []
    for raw in imdb_ids:
        s = str(raw)
        if not s.startswith("tt"):
            s = "tt" + s
        if s not in seen:
            seen.add(s)
            norm_ids.append(s)

    rows, errs = [], []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_one, iid): iid for iid in norm_ids}
        for fut in as_completed(futures):
            rec = fut.result()
            if rec["error"]:
                errs.append(rec)
            else:
                rows.append(rec)
            # optional: checkpoint every 5k results
            if checkpoint_path and len(rows) > 0 and len(rows) % 5000 == 0:
                pd.DataFrame(rows, columns=["movie_id", "countries"]).to_parquet(checkpoint_path, index=False)

    df = pd.DataFrame(rows)[["movie_id","countries"]]
    df_err = pd.DataFrame(errs)[["movie_id","error"]] if errs else pd.DataFrame(columns=["movie_id","error"])
    return df, df_err


imdb_basics_data = "datasets/imdb/title.basics.tsv.gz"

df_imdb_basics = pd.read_csv(imdb_basics_data, 
                 sep="\t",         # tab-separated
                 compression="gzip",  # gzip compression
                 na_values=[r'\N'],
                 keep_default_na=True
                ).convert_dtypes()

df = df_imdb_basics[['tconst', 'titleType', 'primaryTitle', 'originalTitle', 'isAdult', 'startYear', 'genres']]

df_filtered = df[
    (df['startYear'] >= 1990) & 
    (df['startYear'] <= 2023) &
    (df['isAdult'] == 0) &  #Remove Adult titles
    (df['titleType'].isin(['movie']))
    ].drop(columns=['isAdult','titleType'])

# Sampling from DataFrame

sample_size = 50000

df_sample = df_filtered.sample(sample_size, random_state=42)

imdb_ids = df_sample['tconst'].unique().tolist()

df_countries, df_errors = get_countries_fast(imdb_ids, max_workers=8, checkpoint_path="countries_checkpoint.parquet")

# Retry failed attempts


if not df_errors.empty:
    retry_ids = df_errors["movie_id"].tolist()
    df_ok2, df_err2 = get_countries_fast(retry_ids, max_workers=4)  # even gentler
    # merge results, keep first success per movie_id
    df_countries = (pd.concat([df_countries, df_ok2], ignore_index=True)
               .drop_duplicates("movie_id", keep="first"))
    df_err = df_err2  # remaining failures after retry

# Write to a single parquet file
out_dir = Path("outputs")
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / f'countries{len(df_countries)}.parquet'

df_countries.to_parquet(out_file, index=False, engine='pyarrow', compression='snappy')

print(f'Wrote {len(df_countries)}:, rows to {out_file.resolve()}')