"""Yambda-50M -> per-listen arrays for the sequential policy, the reward models and the evaluation.

Usage: python src/prep.py <raw_dir_with_multi_event.parquet> [out_dir=data]

Everything the later stages need is written into one npz (events sorted by user, then time) plus a
vocabulary file. Notes on the raw data (checked in the previous project on the same files):
  * listen timestamps mark the END of listening, on a 5-second grid; the start of a listen is
    approximately t_end - played_ratio * track_length;
  * played_ratio_pct can exceed 100 (rewinds) -> clipped to 100.

Global time split (no user split): test = last 7 days, val = the 7 days before, train = everything earlier.
Item vocabulary = the 50k most listened tracks in TRAIN only (no look-ahead); the rest map to UNK.

Per-listen labels (all strictly about the future of that listen, masked when the horizon crosses the
boundary of the listen's own split, so that a train label never peeks into val/test):
  played   played ratio in [0, 1]
  full     played ratio >= 0.5 ("listen+" in the Yambda paper)
  like     like of the same track between listen start and 1 hour after its end
  cont     session continues: next listen starts within 30 min after this one ends
  adopt    only for the first listen of a track by this user that was NOT organic (a recommendation):
           within 3 days the user listens to it again on their own (organic, >= 50%) or likes it
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

RAW = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(parents=True, exist_ok=True)
VOCAB = 50_000
D = 86_400
H_LIKE, H_CONT, H_ADOPT = 3_600, 1_800, 3 * D

ev = pl.read_parquet(RAW / "multi_event.parquet")
tmax = int(ev["timestamp"].max())
T_TEST, T_VAL = tmax - 7 * D, tmax - 14 * D

lis = ev.filter(pl.col("event_type") == "listen").select(
    "uid", "item_id", pl.col("timestamp").cast(pl.Int64).alias("t"), "is_organic",
    (pl.col("played_ratio_pct").clip(upper_bound=100).cast(pl.Float32) / 100).alias("played"),
    pl.col("track_length_seconds").cast(pl.Int32).alias("len"))
likes = ev.filter(pl.col("event_type") == "like").select("uid", "item_id", pl.col("timestamp").cast(pl.Int64).alias("t_like"))
del ev

# vocabulary from train period only
pop = lis.filter(pl.col("t") < T_VAL).group_by("item_id").agg(pl.len().alias("n")).sort(["n", "item_id"], descending=[True, False]).head(VOCAB)
pop = pop.with_row_index("tok", offset=2).with_columns(pl.col("tok").cast(pl.Int32))
lis = lis.join(pop.select("item_id", "tok"), on="item_id", how="left").with_columns(pl.col("tok").fill_null(1))

lis = lis.sort(["uid", "t", "item_id"])
lis = lis.with_columns(
    (pl.col("t") - (pl.col("played") * pl.col("len")).round().cast(pl.Int64)).alias("t_start"),
    pl.when(pl.col("t") < T_VAL).then(0).when(pl.col("t") < T_TEST).then(1).otherwise(2).cast(pl.Int8).alias("split"),
)
lis = lis.with_columns(
    pl.when(pl.col("split") == 0).then(T_VAL).when(pl.col("split") == 1).then(T_TEST).otherwise(tmax + 1).alias("bound"))

# session continuation
lis = lis.with_columns(pl.col("t_start").shift(-1).over("uid").alias("next_start"))
lis = lis.with_columns(
    pl.when(pl.col("t") + H_CONT >= pl.col("bound")).then(-1)
    .when(pl.col("next_start").is_not_null() & (pl.col("next_start") - pl.col("t") <= H_CONT)).then(1)
    .otherwise(0).cast(pl.Int8).alias("cont"))

# first listen of the (user, item) pair
lis = lis.with_columns((pl.col("t").rank("ordinal").over(["uid", "item_id"]) == 1).cast(pl.Int8).alias("first"))
lis = lis.with_row_index("eid")

# like within [start, end + 1h]
lk = lis.select("eid", "uid", "item_id", "t_start", "t").join(likes, on=["uid", "item_id"], how="inner") \
    .filter((pl.col("t_like") >= pl.col("t_start")) & (pl.col("t_like") <= pl.col("t") + H_LIKE)) \
    .select("eid").unique().with_columns(pl.lit(1, pl.Int8).alias("lk"))
lis = lis.join(lk, on="eid", how="left").with_columns(
    pl.when(pl.col("t") + H_LIKE >= pl.col("bound")).then(-1).otherwise(pl.col("lk").fill_null(0)).cast(pl.Int8).alias("like")).drop("lk")

# adoption of a recommended new track
cand = lis.filter((pl.col("first") == 1) & (pl.col("is_organic") == 0)).select("eid", "uid", "item_id", "t")
later_org = lis.filter((pl.col("is_organic") == 1) & (pl.col("played") >= 0.5)).select("uid", "item_id", pl.col("t").alias("t2"))
later = pl.concat([later_org, likes.rename({"t_like": "t2"})])
ad = cand.join(later, on=["uid", "item_id"], how="inner").filter(
    (pl.col("t2") > pl.col("t")) & (pl.col("t2") <= pl.col("t") + H_ADOPT)).select("eid").unique() \
    .with_columns(pl.lit(1, pl.Int8).alias("ad"))
lis = lis.join(ad, on="eid", how="left").with_columns(
    pl.when((pl.col("first") == 0) | (pl.col("is_organic") == 1) | (pl.col("t") + H_ADOPT >= pl.col("bound"))).then(-1)
    .otherwise(pl.col("ad").fill_null(0)).cast(pl.Int8).alias("adopt")).drop("ad")
lis = lis.sort("eid")

# item side tables indexed by token
art = pl.read_parquet(RAW / "artist_item_mapping.parquet").unique("item_id", keep="first")
pop = pop.join(art, on="item_id", how="left")
pop = pop.with_columns(pl.col("artist_id").fill_null(-1).rank("dense").cast(pl.Int32).alias("artist"))  # 1.. dense
ilen = lis.filter(pl.col("tok") > 1).group_by("tok").agg(pl.col("len").median().alias("ilen"))
pop = pop.join(ilen, on="tok", how="left").sort("tok")
V = VOCAB + 2
tok_item = np.zeros(V, np.int64); tok_item[2:] = pop["item_id"].to_numpy()
tok_artist = np.zeros(V, np.int32); tok_artist[2:] = pop["artist"].to_numpy()
tok_len = np.full(V, 200, np.float32); tok_len[2:] = pop["ilen"].fill_null(200).to_numpy()
tok_pop = np.zeros(V, np.float32); tok_pop[2:] = pop["n"].to_numpy()

uid = lis["uid"].to_numpy()
users, start = np.unique(uid, return_index=True)
offsets = np.append(start, len(uid)).astype(np.int64)
np.savez(OUT / "events.npz",  # split into data/ev/*.npy by src/split_npz.py
         uid=uid.astype(np.int32), t=lis["t"].to_numpy().astype(np.int64), tok=lis["tok"].to_numpy().astype(np.int32),
         organic=lis["is_organic"].to_numpy().astype(np.int8), played=lis["played"].to_numpy().astype(np.float32),
         len=lis["len"].to_numpy().astype(np.int32), split=lis["split"].to_numpy().astype(np.int8),
         full=(lis["played"].to_numpy() >= 0.5).astype(np.int8), like=lis["like"].to_numpy(), cont=lis["cont"].to_numpy(),
         adopt=lis["adopt"].to_numpy(), first=lis["first"].to_numpy(), offsets=offsets, users=users)
np.savez(OUT / "vocab.npz", tok_item=tok_item, tok_artist=tok_artist, tok_len=tok_len, tok_pop=tok_pop)
meta = dict(tmax=tmax, t_val=T_VAL, t_test=T_TEST, vocab=VOCAB, n_events=len(uid), n_users=len(users),
            n_artists=int(tok_artist.max()) + 1)
(OUT / "meta.json").write_text(json.dumps(meta, indent=1))
print(meta)
s = lis.group_by("split").agg(pl.len(), (pl.col("tok") == 1).mean().alias("unk"), pl.col("played").mean(),
                              pl.col("is_organic").mean(),
                              pl.col("like").filter(pl.col("like") >= 0).mean().alias("like"),
                              pl.col("cont").filter(pl.col("cont") >= 0).mean().alias("cont"),
                              pl.col("adopt").filter(pl.col("adopt") >= 0).mean().alias("adopt"),
                              (pl.col("adopt") >= 0).sum().alias("n_adopt")).sort("split")
print(s)
