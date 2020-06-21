from glob import glob
import boto3
import pandas as pd
from datetime import date, time, datetime
from collections import Counter

pd.options.mode.chained_assignment = None

debug = False

class DataManager():
    def __init__(self):
        if debug:
            esm_df = pd.read_csv("data/esm_data.csv")
            self.esm_df = esm_df
        else:
            self.s3 = boto3.resource('s3')
            self.bucket = self.s3.Bucket('mood-suggest')

            esm_csv = self.bucket.Object('esm_data.csv').get()
            self.esm_df = pd.read_csv(esm_csv['Body'])

        self.uids = self.esm_df.UID.unique()
        self.current = None
        self.data = None
        self.line_data = None
        self.n_pos, self.n_neg = 0,0

    def update(self, uid):
        if uid != self.current:
            self.current = uid
            self.line_data = self.generate_line_data(uid)
            self.data = self.generate_data(uid)

    def generate_data(self, uid):
        mood_data = self.esm_df.loc[self.esm_df.UID == uid]
        mood_data.loc[:,'timestamp'] = pd.to_datetime(mood_data['responseTime_unixtimestamp'], unit='s')
        mood_data.set_index('timestamp', drop=True, inplace=True)
        mood_data = mood_data.resample('100T').mean()
        mood_data['change'] = mood_data['Valence'] - mood_data['Valence'].shift(1)
        mood_data.query('abs(change) >= 1.5', inplace=True)

        app_df = self.get_data('extracted/P%d/AppUsageStat'%uid)
        app_df['startTime'] = pd.to_datetime(app_df['startTime'], unit='ms')
        app_df['lastTimeUsed'] = pd.to_datetime(app_df['endTime'], unit='ms')

        orig_freq = app_df.groupby('name').sum()['totalTimeForeground']

        subs = []
        pos, neg, meta = {}, {}, {}
        for i in range(len(mood_data)-1):
            ts = mood_data.index[i]
            ts2 = ts + pd.Timedelta(hours=3)
            sub  = app_df.loc[(app_df.index >= ts) & (app_df.index <= ts2)]
            sub_freq = sub.groupby('name').sum()['totalTimeForeground'].sort_values(ascending=False) * abs(mood_data.change[i])
            if mood_data.change[i] > 0:
                for i,v in sub_freq[:10].items():
                    if not i in pos:
                        pos[i] = v
                        meta[i] = {'times': [], 'durations': []}
                    else:
                        pos[i] += v
                    meta[i]['times'] += sub[sub.name == i].lastTimeUsed.unique().tolist()
                    meta[i]['durations'] += sub[sub.name == i].totalTimeForeground.unique().tolist()
            else:
                for i,v in sub_freq[:10].items():
                    if not i in neg:
                        neg[i] = v
                        meta[i] = {'times': [], 'durations': []}
                    else:
                        neg[i] += v
                    meta[i]['times'] += sub[sub.name == i].lastTimeUsed.unique().tolist()
                    meta[i]['durations'] += sub[sub.name == i].totalTimeForeground.unique().tolist()
            subs.append(sub)
        res = pd.concat(subs)
        res_freq = res.groupby('name').sum()['totalTimeForeground'].sort_values(ascending=False)
        ratios = orig_freq.div(res_freq)
        candidates = [x for x in res_freq.index[:15] if ratios[x] <= 5]
        candidate_ratios = [ratios[x] for x in candidates]
        types = []
        for cand in candidates:
            if cand in pos and cand in neg and pos[cand] > neg[cand]:
                types.append('Pos')
            elif cand in pos:
                types.append('Pos')
            else:
                types.append('Neg')
        return {'actions': candidates, 'types': types, 'meta': meta, 'effects': candidate_ratios}

    def generate_line_data(self, uid):
        mood_data = self.esm_df.loc[self.esm_df.UID == uid]
        mood_data.loc[:,'time'] = pd.to_datetime(mood_data['responseTime_unixtimestamp'], unit='s')
        line_data = mood_data.loc[:,['time','Valence']]

        df = self.get_data('extracted/P%d/AmbientLight'%uid)
        df.Brightness = df.Brightness.astype(float)
        df = df['Brightness'].resample('1H').mean()
        df.dropna(inplace=True)

        return (line_data, df)


    def get_data(self, path):
        df = None
        if debug:
            path = 'data/' + path + '*'
            for fn in glob(path):
              if df is None:
                df = pd.read_csv(fn)
              else:
                df = df.append(pd.read_csv(fn))
            df.reset_index(inplace=True, drop=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'],unit='ms')
            df.set_index('timestamp', inplace=True)
        else:
            for obj in self.bucket.objects.filter(Prefix=path):
              print('[INFO] Downloading:', obj.key)
              if df is None:
                df = pd.read_csv(obj.get()['Body'])
              else:
                df = df.append(pd.read_csv(obj.get()['Body']))
            df.reset_index(inplace=True, drop=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'],unit='ms')
            df.set_index('timestamp', inplace=True)
        return df
