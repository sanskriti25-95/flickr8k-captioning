# Image captioning (CNN + LSTM, Flickr8k)

Deep Learning case study by Sanskriti Parakh (B.Tech CSE, Section B, Manipal University Jaipur).

Upload a photo and get a one-sentence caption. A frozen ImageNet CNN (InceptionV3 or ResNet50) encodes the image, and an LSTM trained on the 6,000 Flickr8k training images writes the caption, using greedy decoding or beam search.

Best test-set score (ResNet50, beam k=3): BLEU-1/2/3/4 = 0.538 / 0.357 / 0.232 / 0.141. The full write-up is in [report.pdf](report.pdf).

The app works best on Flickr-style photos of people, dogs and outdoor activities. Selfies, indoor close-ups and other scenes unlike the training data get rough captions.

Run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```
