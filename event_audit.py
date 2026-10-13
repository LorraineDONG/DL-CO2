from pathlib import Path
import numpy as np
import pandas as pd
import e2_data as base
TOP_EVENTS = 100
NO2_TIME_COL = 'no2_time'
TIME_ORIGIN = '1993-01-01'
def save_csv(df,path):
    df.to_csv(path,index=False,encoding='utf-8-sig')

def audit_events(raw,predictions,folder,label):
    folder.mkdir(parents=True,exist_ok=True)
    p=predictions.copy();p['error']=p.prediction-p[base.TARGET];p['abs_error']=p.error.abs()
    p['squared_error']=p.error**2;p=p.sort_values('abs_error',ascending=False).reset_index(drop=True)
    p['error_rank']=np.arange(1,len(p)+1)
    ids=p._row_id.to_numpy(dtype=int)
    original=raw.loc[ids].reset_index(drop=True).copy()
    # Prefix every original column: original timestamps/NO2/background remain intact.
    event=pd.concat([p.reset_index(drop=True),original.add_prefix('raw__')],axis=1)
    def numeric(c):
        return pd.to_numeric(original[c],errors='coerce') if c in original else pd.Series(np.nan,index=original.index)
    count=numeric('total_obs_count');event['audit_obs_count_le_1']=count.le(1)
    event['audit_obs_count_missing']=count.isna()
    event['audit_no2_missing_or_nonpositive']=numeric('no2_trop').le(0)|numeric('no2_trop').isna()
    delta=numeric(NO2_TIME_COL)-numeric('xco2_time_tai93')
    event['audit_no2_minus_xco2_seconds']=delta
    event['audit_no2_time_unresolved']=delta.isna()
    event['audit_abs_no2_time_over_3600s']=delta.abs().gt(3600)
    xt=numeric('xco2_time_tai93')
    event['audit_xco2_utc_assuming_elapsed1993']=pd.to_datetime(xt,unit='s',origin=TIME_ORIGIN,errors='coerce')
    bg=[c for c in raw.columns if any(s in str(c).lower() for s in ['background','baseline','bg_xco2','xco2_bg'])]
    save_csv(event,folder/f'{label}_all_events_raw.csv')
    save_csv(event.head(TOP_EVENTS),folder/f'{label}_top{TOP_EVENTS}_raw.csv')
    if 'total_obs_count' in original:
        g=pd.cut(count,[-np.inf,1,3,10,30,np.inf],labels=['<=1','2-3','4-10','11-30','>30'])
        stats=[]
        for name,sub in p.groupby(g,observed=True):
            stats.append(dict(obs_count_bin=str(name),**base.metrics(sub[base.TARGET],sub.prediction)))
        save_csv(pd.DataFrame(stats),folder/f'{label}_metrics_by_obs_count.csv')
    base.write_json(folder/f'{label}_audit_notes.json',dict(
        background_columns_found=bg,background_status='Original background construction cannot be validated from a table alone. Compare with original soundings and background-selection code.',
        no2_time_column=NO2_TIME_COL,no2_time_column_present=NO2_TIME_COL in raw,
        time_assumption=f'Elapsed UTC seconds since {TIME_ORIGIN}; verify preparation definitions before interpreting differences.',
        automatic_row_deletions=0,warning='Flags request source review, not exclusion. High labels/large errors are never deletion rules. Counts/timestamps/background are audit-only, never model inputs.'))


