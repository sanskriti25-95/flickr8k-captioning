# Image Captioning on Flickr8k: CNN encoder + LSTM decoder

| Field | Details |
|---|---|
| Name | Sanskriti Parakh |
| Registration number | 23FE10ITE000298 |
| Course | Deep Learning – Case Study |
| Section | B |
| Programme | B.Tech CSE |
| Department | Information Technology |
| University | Manipal University Jaipur |
| Live demo | https://flickr8k-captioning-jp9tfcuw88xxt5vjxsgykw.streamlit.app/ |
| Source code | https://github.com/sanskriti25-95/flickr8k-captioning |

## Summary
A pretrained CNN turns each image into a 2048-number feature vector. An LSTM language model, conditioned on that vector, generates the caption one word at a time. We compare two encoders (InceptionV3 and ResNet50) and two decoding strategies (greedy and beam search, k=3), and measure both against a retrieval baseline that writes nothing new. The best model is **ResNet50 with beam search**: BLEU-1/2/3/4 = **0.538 / 0.357 / 0.232 / 0.141** on the official Flickr8k test set.

## Data
- **Source:** the official Flickr8k release (`Flickr8k_Dataset.zip`, `Flickr8k_text.zip` from github.com/jbrownlee/Datasets). The Kaggle fallback was not needed.
- **Split:** the **official split** from `Flickr_8k.trainImages.txt` / `devImages.txt` / `testImages.txt`: 6000 / 1000 / 1000 images, each with 5 captions from `Flickr8k.token.txt`. The archive holds 8091 images, but only the 8000 listed in the split files are used. Using the official split keeps results comparable with published work.
- **Cleaning:** lowercase, keep letters only, drop one-character tokens, wrap each caption in `startseq … endseq`.
- **Vocabulary:** built from training captions only, keeping words seen ≥ 5 times. That gives 2527 tokens including `<pad>`, `<unk>`, start and end. The maximum prefix length is 34.
- **Training pairs:** each caption is expanded into (image, prefix) → next-word pairs, giving 307,078 training and 51,733 dev pairs.

## Method
| Component | Choice |
|---|---|
| Encoder | ImageNet InceptionV3 (299×299) or ResNet50 (224×224), `include_top=False`, global average pooling → 2048-d. Frozen; features computed once and cached to `outputs/features_*.npz`. |
| Decoder | "Merge" architecture. Image branch: Dropout(0.5) → Dense(256, ReLU). Text branch: Embedding(256, `mask_zero`) → Dropout(0.5) → LSTM(256). The two are added, then Dense(256, ReLU) → softmax over the vocabulary. 2.41 M parameters. |
| Training | Adam, sparse categorical cross-entropy, batch 512, at most 8 epochs, early stopping on dev loss (patience 2, best weights restored), `keras.utils.set_random_seed(42)`, `tf.data` input pipeline. CPU only, about 5 min per epoch. |
| Decoding | Greedy, and beam search with k=3 and length-normalised scoring. Beam search is vectorised across images, so one model call scores every live beam of every image at a given word position, and all 1000 test images decode in about 30 s. `<pad>`/`<unk>` are never emitted. |
| Metric | Corpus BLEU-1..4 (NLTK, smoothing method 1) against all 5 reference captions. |

**Baselines (no caption generation).** Cosine similarity on the same CNN features:
- **NN-1:** the first caption of the most similar training image.
- **kNN-5 consensus:** pool the 25 captions of the 5 nearest training images and pick the one with the highest mean BLEU against the others.

## Results (official test set, 1000 images)

| Encoder | Method | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|---|---|---|---|---|---|
| InceptionV3 | NN-1 baseline | 0.350 | 0.181 | 0.098 | 0.053 |
| InceptionV3 | kNN-5 consensus baseline | 0.431 | 0.253 | 0.151 | 0.087 |
| InceptionV3 | LSTM, greedy | 0.506 | 0.313 | 0.188 | 0.108 |
| InceptionV3 | LSTM, beam k=3 | 0.526 | 0.343 | 0.217 | 0.130 |
| ResNet50 | NN-1 baseline | 0.361 | 0.189 | 0.100 | 0.051 |
| ResNet50 | kNN-5 consensus baseline | 0.427 | 0.251 | 0.151 | 0.091 |
| ResNet50 | LSTM, greedy | 0.516 | 0.326 | 0.200 | 0.118 |
| **ResNet50** | **LSTM, beam k=3** | **0.538** | **0.357** | **0.232** | **0.141** |

| | InceptionV3 | ResNet50 |
|---|---|---|
| Best dev loss (epoch) | 3.228 (6) | 3.291 (5) |
| Distinct beam captions / 1000 | 789 | 815 |
| Mean beam caption length (words) | 8.8 | 8.6 |

**Observations**
- The generative model clearly beats retrieval. Beam search with ResNet50 raises BLEU-4 from 0.091 (best baseline) to 0.141, a 55% relative gain.
- Beam search beats greedy for both encoders, by about +0.02 BLEU-1 and +0.02 BLEU-4.
- ResNet50 gives slightly better BLEU than InceptionV3 even though its dev loss is higher. Lower next-word loss does not fully predict caption-level quality.
- Both models overfit quickly. Training loss keeps falling while dev loss bottoms out at epoch 5–6, so early stopping matters.

**Sample captions (ResNet50, beam k=3)**

| Test image | Generated | One human reference |
|---|---|---|
| 3385593926_d3e9c21170.jpg | two brown dogs are playing in the snow | the dogs are in the snow in front of fence |
| 2677656448_6b7e7702af.jpg | small boy is jumping into pool | brown and white dog swimming towards some in the pool |
| 311146855_0b65fdb169.jpg | man in red shirt and blue boots is dancing on the sidewalk | man and woman in festive costumes dancing |
| 1258913059_07c613f7ff.jpg | two people are sitting on bench in front of an audience | couple of people sit outdoors at table with an umbrella and talk |

The captions are fluent and get the scene type right (snow, pool, dancing, sitting), but they often get object identity or count wrong (the "boy" in the pool is a dog).

## Comparison with existing implementations
The closest reference, dabasajay/Image-Caption-Generator, reports a best validation BLEU-1/2/3/4 of about 0.606 / 0.359 / 0.249 / 0.130, for InceptionV3 with beam k=3. Our best model gives 0.538 / 0.357 / 0.232 / **0.141**: BLEU-2 and BLEU-3 are about level and BLEU-4 is slightly higher. Our BLEU-1 is lower, probably because our captions are short (about 8.7 words) and BLEU's brevity penalty hits unigram precision hardest. Their numbers are on a validation set rather than the official test split, and smoothing and tokenisation differ, so the comparison is approximate. yashk2810/image-captioning reports only a training loss, not BLEU. Brownlee's tutorial reports a single BLEU score from a small experiment, which is not directly comparable. The Keras and TensorFlow tutorials report accuracy and loss, not BLEU.

**What we did differently:**
- Everything is written for current **Keras 3 / TensorFlow 2.20**: functional API, `keras.utils.set_random_seed`, `tf.data`, `.keras` model format. The deprecated `Tokenizer`, `fit_generator` and `layers.merge` used by the older references are not used.
- The **official Flickr8k train/dev/test split** is used, with dev-loss early stopping and test BLEU reported once.
- **Nearest-neighbour retrieval baselines** show how much the generative model adds.
- An **encoder comparison** (InceptionV3 vs ResNet50) uses identical decoders and settings.
- **Vectorised beam search** runs over all images at once, so a full beam-search evaluation takes about 30 s instead of hours.
- A **Streamlit demo** (`app.py`) lets you upload any image and choose the encoder and beam width.

## Limitations and next steps
- Single run per configuration (seed 42), so there are no error bars. The InceptionV3 vs ResNet50 gap (+0.011 BLEU-4) is small and may be within seed noise.
- Global-pooled features discard spatial layout. An attention decoder over the 7×7 or 8×8 feature map would probably help with counting and object identity.
- BLEU alone is a coarse measure. METEOR, CIDEr or SPICE would give a fuller picture.
- The encoder is frozen and trained on CPU only. Fine-tuning the encoder or training longer with more regularisation could improve results.

## How to reproduce
```bash
python cap.py features --enc inception      # and --enc resnet50
python cap.py train    --enc inception --epochs 8 --batch 512
python cap.py baseline --enc inception
python cap.py eval     --enc inception
streamlit run app.py
```
The demo is also deployed on Streamlit Community Cloud at https://flickr8k-captioning-jp9tfcuw88xxt5vjxsgykw.streamlit.app/ (Python 3.12, CPU). Upload a photo, choose the encoder and beam width, and it returns a caption.

The dataset goes in `data/images/Flicker8k_Dataset/` and `data/text/`. Outputs (features, models, vocabularies, training histories, `eval_*.json`, `baseline_*.json`) are written to `outputs/`.

## References
1. dabasajay, *Image-Caption-Generator*: https://github.com/dabasajay/Image-Caption-Generator. Main reference for the merge architecture, beam search (k=3) and BLEU-1..4 evaluation; source of the comparison numbers.
2. yashk2810, *image-captioning*: https://github.com/yashk2810/image-captioning. Reference for InceptionV3 features and beam-search inference.
3. J. Brownlee, "How to Develop a Deep Learning Photo Caption Generator from Scratch", Machine Learning Mastery: https://machinelearningmastery.com/develop-a-caption-generation-model-in-keras/. Reference for caption cleaning, `startseq/endseq`, cached features and progressive (prefix → next word) training pairs.
4. A. K. Nain, "Image Captioning", Keras examples: https://keras.io/examples/vision/image_captioning/. Used only for Keras 3 idioms (`set_random_seed`, `tf.data`, early stopping); its Transformer model was not used.
5. TensorFlow, "Image captioning with visual attention": https://www.tensorflow.org/text/tutorials/image_captioning. Reference for the Flickr8k loader and for caching features once.
6. M. Hodosh, P. Young, J. Hockenmaier, "Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics", JAIR 47, 2013. The Flickr8k dataset and its official split.

No reference code was copied verbatim. All code was rewritten for Keras 3.
