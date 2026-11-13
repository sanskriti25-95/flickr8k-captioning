"""Flickr8k image captioning: CNN encoder + LSTM decoder (Keras 3 / TensorFlow).

Subcommands:  features | train | eval | baseline
  python cap.py features --enc inception
  python cap.py train    --enc inception
  python cap.py eval     --enc inception
  python cap.py baseline --enc inception

Design ideas taken from (rewritten, not copied): Brownlee's cleaning + startseq/endseq +
feature caching, dabasajay's merge architecture + beam search + BLEU-1..4 evaluation,
keras.io's Keras 3 idioms (set_random_seed, tf.data), TF tutorial's feature caching.
"""
import argparse, json, os, re, sys, time
from collections import Counter
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import keras
from keras import layers

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
IMG_DIR = DATA / "images" / "Flicker8k_Dataset"
TXT_DIR = DATA / "text"
OUT = ROOT / "outputs"
SEED = 42
START, END = "startseq", "endseq"

ENCODERS = {  # name -> (constructor, input size, preprocess fn)
    "inception": ("InceptionV3", 299, "inception_v3"),
    "resnet50": ("ResNet50", 224, "resnet50"),
}


# ----------------------------------------------------------------------------- data
def clean(caption: str) -> str:
    words = re.sub(r"[^a-z ]", " ", caption.lower()).split()
    return " ".join(w for w in words if len(w) > 1)


def load_captions():
    """image filename -> list of cleaned captions (no start/end tokens)."""
    caps = {}
    for line in (TXT_DIR / "Flickr8k.token.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        key, cap = line.split("\t")
        caps.setdefault(key.split("#")[0], []).append(clean(cap))
    return caps


def load_split(name):  # train | dev | test  (official Flickr_8k.*Images.txt)
    f = TXT_DIR / f"Flickr_8k.{name}Images.txt"
    return [l.strip() for l in f.read_text().splitlines() if l.strip()]


class Vocab:
    def __init__(self, words):
        self.itos = ["<pad>", "<unk>"] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def build(cls, captions, min_count=5):
        c = Counter(w for cap in captions for w in cap.split())
        words = [START, END] + sorted(w for w, n in c.items() if n >= min_count and w not in (START, END))
        return cls(words)

    def encode(self, cap):
        return [self.stoi.get(w, 1) for w in cap.split()]

    def decode(self, ids):
        return " ".join(self.itos[i] for i in ids if i > 1 and self.itos[i] not in (START, END))

    def save(self, p):
        Path(p).write_text(json.dumps(self.itos))

    @classmethod
    def load(cls, p):
        v = cls([]); v.itos = json.loads(Path(p).read_text())
        v.stoi = {w: i for i, w in enumerate(v.itos)}; return v


def make_pairs(names, caps, vocab, max_len, feat_index):
    """Progressive (image, prefix) -> next word pairs, precomputed as arrays."""
    ii, xs, ys = [], [], []
    for n in names:
        for cap in caps[n]:
            seq = vocab.encode(f"{START} {cap} {END}")
            for t in range(1, len(seq)):
                ii.append(feat_index[n])
                xs.append(seq[:t][-max_len:] if t <= max_len else seq[t - max_len:t])
                ys.append(seq[t])
    X = np.zeros((len(xs), max_len), np.int32)
    for r, s in enumerate(xs):
        X[r, : len(s)] = s
    return np.array(ii, np.int32), X, np.array(ys, np.int32)


# ----------------------------------------------------------------------------- features
def cmd_features(a):
    import tensorflow as tf
    ctor, size, pp = ENCODERS[a.enc]
    OUT.mkdir(exist_ok=True)
    dest = OUT / f"features_{a.enc}.npz"
    if dest.exists():
        print("features exist, skipping:", dest); return
    keras.utils.set_random_seed(SEED)
    base = getattr(keras.applications, ctor)(include_top=False, weights="imagenet", pooling="avg")
    prep = getattr(keras.applications, pp).preprocess_input
    names = sorted(set(load_split("train") + load_split("dev") + load_split("test")))

    def load(n):
        img = tf.io.decode_jpeg(tf.io.read_file(str(IMG_DIR) + "/" + n), channels=3)
        return prep(tf.image.resize(img, (size, size)))

    ds = tf.data.Dataset.from_tensor_slices(names).map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(32).prefetch(2)
    t0 = time.time()
    feats = base.predict(ds, verbose=2)
    np.savez_compressed(dest, names=np.array(names), feats=feats.astype(np.float32))
    print(f"saved {feats.shape} in {time.time()-t0:.0f}s -> {dest}")


def load_features(enc):
    z = np.load(OUT / f"features_{enc}.npz")
    names, feats = list(z["names"]), z["feats"]
    return {n: i for i, n in enumerate(names)}, feats


# ----------------------------------------------------------------------------- model
def build_model(vocab_size, max_len, feat_dim, units=256, drop=0.5):
    """Merge architecture: image branch + LSTM text branch are added, then decoded."""
    img_in = keras.Input((feat_dim,), name="image_features")
    x1 = layers.Dense(units, activation="relu")(layers.Dropout(drop)(img_in))
    seq_in = keras.Input((max_len,), dtype="int32", name="prefix")
    x2 = layers.Embedding(vocab_size, units, mask_zero=True)(seq_in)
    x2 = layers.LSTM(units)(layers.Dropout(drop)(x2))
    x = layers.add([x1, x2])
    x = layers.Dense(units, activation="relu")(x)
    out = layers.Dense(vocab_size, activation="softmax")(x)
    return keras.Model([img_in, seq_in], out)


def cmd_train(a):
    import tensorflow as tf
    keras.utils.set_random_seed(SEED)
    caps = load_captions()
    train, dev = load_split("train"), load_split("dev")
    vocab = Vocab.build([c for n in train for c in caps[n]])
    max_len = max(len(c.split()) for n in train for c in caps[n]) + 1  # +startseq
    idx, feats = load_features(a.enc)
    tr = make_pairs(train, caps, vocab, max_len, idx)
    dv = make_pairs(dev, caps, vocab, max_len, idx)
    print(f"vocab={len(vocab.itos)} max_len={max_len} train_pairs={len(tr[2])} dev_pairs={len(dv[2])}")
    F = tf.constant(feats)

    def ds(p, shuffle):
        d = tf.data.Dataset.from_tensor_slices(p)
        if shuffle:
            d = d.shuffle(100_000, seed=SEED)
        return d.batch(a.batch).map(lambda i, x, y: ((tf.gather(F, i), x), y)).prefetch(2)

    model = build_model(len(vocab.itos), max_len, feats.shape[1])
    model.compile("adam", "sparse_categorical_crossentropy")
    model.summary()
    cb = [keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
          keras.callbacks.CSVLogger(OUT / f"history_{a.enc}.csv")]
    model.fit(ds(tr, True), validation_data=ds(dv, False), epochs=a.epochs, callbacks=cb, verbose=2)
    model.save(OUT / f"model_{a.enc}.keras")
    vocab.save(OUT / f"vocab_{a.enc}.json")
    json.dump({"max_len": max_len}, open(OUT / f"meta_{a.enc}.json", "w"))


# ----------------------------------------------------------------------------- decoding
class Captioner:
    def __init__(self, enc):
        self.model = keras.models.load_model(OUT / f"model_{enc}.keras")
        self.vocab = Vocab.load(OUT / f"vocab_{enc}.json")
        self.max_len = json.load(open(OUT / f"meta_{enc}.json"))["max_len"]
        self.s, self.e = self.vocab.stoi[START], self.vocab.stoi[END]

    def _probs(self, f, seqs):
        X = np.zeros((len(seqs), self.max_len), np.int32)
        for r, s in enumerate(seqs):
            s = s[-self.max_len:]; X[r, : len(s)] = s
        F = np.repeat(f[None], len(seqs), 0)
        return self.model([F, X], training=False).numpy()

    def greedy_batch(self, feats):
        n = len(feats); seqs = [[self.s] for _ in range(n)]; done = np.zeros(n, bool)
        for _ in range(self.max_len):
            X = np.zeros((n, self.max_len), np.int32)
            for r, s in enumerate(seqs):
                X[r, : len(s)] = s[: self.max_len]
            p = self.model([feats, X], training=False).numpy()
            p[:, :2] = 0  # never emit pad/unk
            nxt = p.argmax(1)
            for r in range(n):
                if not done[r]:
                    seqs[r].append(int(nxt[r])); done[r] = nxt[r] == self.e
            if done.all(): break
        return [self.vocab.decode(s[1:]) for s in seqs]

    def beam(self, f, k=3):
        return self.beam_batch(f[None], k)[0]

    def beam_batch(self, feats, k=3, chunk=400):
        """Beam search over many images at once: one model call per word position."""
        out = []
        for c in range(0, len(feats), chunk):
            out += self._beam_chunk(feats[c:c + chunk], k)
        return out

    def _beam_chunk(self, feats, k):
        N, L = len(feats), self.max_len
        seqs = np.zeros((N, k, L), np.int32); seqs[:, :, 0] = self.s
        scores = np.full((N, k), -np.inf); scores[:, 0] = 0.0   # only one live beam at start
        lens = np.ones((N, k)); done = np.zeros((N, k), bool)
        F = np.repeat(feats, k, 0)
        for t in range(1, L):
            p = self.model([F, seqs.reshape(N * k, L)], training=False).numpy().reshape(N, k, -1)
            logp = np.log(p + 1e-12); logp[:, :, :2] = -np.inf     # never emit pad/unk
            logp[done] = -np.inf; logp[done, 0] = 0.0               # finished beams: frozen, pad only
            V = logp.shape[2]
            cand = (scores[:, :, None] + logp).reshape(N, k * V)
            top = np.argpartition(-cand, k, axis=1)[:, :k]
            src, word = top // V, top % V
            rows = np.arange(N)[:, None]
            seqs, lens, done = seqs[rows, src].copy(), lens[rows, src].copy(), done[rows, src].copy()
            scores = cand[rows, top]
            seqs[:, :, t] = np.where(done, 0, word)
            lens += ~done
            done |= word == self.e
            if done.all(): break
        best = np.argmax(scores / lens, axis=1)  # length-normalised
        return [self.vocab.decode(seqs[i, best[i], 1:].tolist()) for i in range(N)]


def bleu(refs, hyps):
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    R = [[r.split() for r in rs] for rs in refs]; H = [h.split() for h in hyps]
    sm = SmoothingFunction().method1
    return {f"BLEU-{n}": corpus_bleu(R, H, weights=tuple([1 / n] * n), smoothing_function=sm) for n in (1, 2, 3, 4)}


def cmd_eval(a):
    caps = load_captions(); test = load_split("test")
    idx, feats = load_features(a.enc); cp = Captioner(a.enc)
    refs = [caps[n] for n in test]; F = feats[[idx[n] for n in test]]
    res = {}
    t = time.time(); g = cp.greedy_batch(F); res["greedy"] = bleu(refs, g); print("greedy", res["greedy"], f"{time.time()-t:.0f}s")
    t = time.time(); b = cp.beam_batch(F, a.k); res[f"beam{a.k}"] = bleu(refs, b); print(f"beam{a.k}", res[f"beam{a.k}"], f"{time.time()-t:.0f}s")
    res["samples"] = [{"image": n, "greedy": g[i], f"beam{a.k}": b[i], "refs": refs[i][:2]} for i, n in enumerate(test[:15])]
    res["distinct_captions_beam"] = len(set(b)); res["avg_len_beam"] = float(np.mean([len(x.split()) for x in b]))
    json.dump(res, open(OUT / f"eval_{a.enc}.json", "w"), indent=1)


# ----------------------------------------------------------------------------- baseline
def cmd_baseline(a):
    """Retrieval baselines: caption of the nearest training image (cosine on CNN features)."""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    caps = load_captions(); train, test = load_split("train"), load_split("test")
    idx, feats = load_features(a.enc)
    norm = lambda M: M / np.linalg.norm(M, axis=1, keepdims=True)
    Tr = norm(feats[[idx[n] for n in train]]); Te = norm(feats[[idx[n] for n in test]])
    S = Te @ Tr.T; sm = SmoothingFunction().method1
    nn1, cons = [], []
    for i in range(len(test)):
        top = np.argsort(-S[i])[:5]
        nn1.append(caps[train[top[0]]][0])
        pool = [c for j in top for c in caps[train[j]]]
        sc = [np.mean([sentence_bleu([o.split()], c.split(), smoothing_function=sm) for o in pool if o is not c]) for c in pool]
        cons.append(pool[int(np.argmax(sc))])
    refs = [caps[n] for n in test]
    res = {"nn1": bleu(refs, nn1), "knn5_consensus": bleu(refs, cons)}
    print(res); json.dump(res, open(OUT / f"baseline_{a.enc}.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["features", "train", "eval", "baseline"])
    ap.add_argument("--enc", default="inception", choices=list(ENCODERS))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    globals()["cmd_" + a.cmd](a)
