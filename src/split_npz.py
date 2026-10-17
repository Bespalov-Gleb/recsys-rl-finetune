"""data/events.npz -> data/ev/<column>.npy (memory-mapped by common.events()) + per-event user index."""
import numpy as np

from common import DATA

z = np.load(DATA / "events.npz")
(DATA / "ev").mkdir(exist_ok=True)
for k in z.files:
    np.save(DATA / "ev" / f"{k}.npy", z[k])
off = z["offsets"]
np.save(DATA / "ev" / "uid_idx.npy", np.repeat(np.arange(len(off) - 1), np.diff(off)).astype(np.int32))
