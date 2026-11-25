# Kiswahili Sentiment Analysis with AfroXLMR

Mini project for AI Applications. We fine-tune a pretrained multilingual model to classify Kiswahili tweets as **positive**, **negative** or **neutral**, and provide a small web app to try it.

- **Group members:** Wendpouire Nafissa & Vunabandi Fabiola

- **Course / institution:** Bsc Satistics and Data science, Strathmore University
- **Trained model (Hugging Face):** https://huggingface.co/nafissaoued/swahili-sentiment-afroxlmr
- **Code and data:** this repository

## 1. Problem

Kiswahili is spoken by tens of millions of people across East Africa, and a lot of it appears on social media. Someone who does not read Kiswahili cannot tell what people are saying about a topic. This project builds a classifier that gives a quick positive / negative / neutral signal for a Kiswahili text. Possible uses are social media monitoring and customer-feedback analysis.

## 2. Data

- **Source:** AfriSenti, Swahili subset (`swa`), Hugging Face dataset `HausaNLP/AfriSenti-Twitter`. Tweets annotated by native speakers as positive, negative or neutral.
- **License / use:** the dataset page lists non-commercial terms (and is inconsistent about the exact license), so this project is for **non-commercial academic use only**.
- **Files in `data/`:**

| File | Rows | Description |
|---|---|---|
| `afrisenti_swa_train.csv` | 1,810 | original training split |
| `afrisenti_swa_validation.csv` | 453 | original validation split |
| `afrisenti_swa_test.csv` | 748 | original test split (never modified) |
| `afrisenti_swa_train_clean.csv` | 1,799 | cleaned training split (used for training) |
| `afrisenti_swa_validation_clean.csv` | 450 | cleaned validation split (used for model selection) |

Each split has one row fewer than the counts on the dataset page; we report the numbers we actually obtained.

**Label distribution (original train):** neutral 1,072 (59%), positive 547 (30%), negative 191 (11%). Validation and test have the same proportions.

### Data preparation

1. Checked label balance, tweet length (mean about 17 words, maximum 45) and empty tweets (none).
2. Found 8 duplicate rows in the training set. Three tweets appeared with **conflicting labels** (for example, the same customer-service reply labelled neutral once and positive twice). We removed all 7 rows belonging to these three tweets, because their correct label is undefined.
3. Kept one copy of the 4 other repeated tweets (4 rows removed).
4. Found 3 tweets in both train and validation and removed them from validation. There is no overlap between train and test.
5. The test set was left unchanged.

Result: training 1,810 to 1,799 tweets (neutral 1,067, positive 541, negative 191); validation 453 to 450.

## 3. Model and training

- **Base model:** [`Davlan/afro-xlmr-base`](https://huggingface.co/Davlan/afro-xlmr-base), an XLM-RoBERTa model further trained on African languages, including Kiswahili.
- **Task head:** a sequence-classification layer with 3 outputs.
- **Settings:** 5 epochs, batch size 16, learning rate 2e-5, weight decay 0.01, maximum length 128, fp16, seed 42. The epoch with the best validation macro F1 is kept. On validation, macro F1 rose from 0.38 (epoch 1) to about 0.52 (epochs 3 to 5), while validation loss increased slightly after epoch 2, a sign of mild overfitting.
- **Hardware:** Google Colab, T4 GPU, about 8 minutes of training.
- **Baseline for comparison:** TF-IDF (1-2 word n-grams) with logistic regression and balanced class weights.
- **Reproduce:** open `notebooks/swahili-sentiment.ipynb` in Google Colab with a GPU runtime and run the cells in order.

## 4. Evaluation

All numbers below are on the untouched **test set** (748 tweets). The main metric is **macro F1**, because the classes are imbalanced: a model that always predicts "neutral" would get 59% accuracy but only about 0.25 macro F1.

| Model | Macro F1 | Accuracy | F1 negative | F1 neutral | F1 positive |
|---|---|---|---|---|---|
| TF-IDF + logistic regression | 0.463 | 0.54 | 0.30 | 0.63 | 0.45 |
| **AfroXLMR-base fine-tuned (seed 42)** | **0.585** | 0.62 | 0.59 | 0.70 | 0.46 |

Confusion matrix, seed 42 (rows = true class, columns = predicted class):

| | pred negative | pred neutral | pred positive |
|---|---|---|---|
| **true negative** | 48 | 27 | 5 |
| **true neutral** | 27 | 318 | 99 |
| **true positive** | 8 | 117 | 99 |

### Error analysis

- Opposite-sentiment confusions are rare (5 negative tweets predicted positive, 8 the other way round).
- About three quarters of the errors are between neutral and positive (99 + 117 of 283). Typical cases are polite customer-service replies, questions that contain positive words, and upbeat announcements labelled neutral.
- Several identical tweets carry different labels in the original data, so some of the remaining error comes from label noise.
- Positive-class F1 did not improve over the baseline.
- Quick sanity check with three hand-written sentences: positive and neutral examples were classified correctly, but "Hii ni mbaya sana, sijafurahi" ("this is very bad, I am not happy") was predicted neutral with low confidence. This matches the lower recall for the negative class.

## 5. Demo app

The app is built with Streamlit and loads the model from Hugging Face. The user pastes a Kiswahili comment and gets the predicted label, its confidence and a bar chart of all three class scores. It warns when the top score is below 50%.

```
pip install -r requirements.txt
streamlit run app.py
```

The first analysis downloads the model (about 1.1 GB); later runs use the local cache.

Using the model directly in Python:

```python
from transformers import pipeline

clf = pipeline("text-classification",
               model="nafissaoued/swahili-sentiment-afroxlmr", top_k=None)
print(clf("Huduma ilikuwa nzuri sana, asante!"))
```

## 6. Limitations

- The data is Twitter text, largely from news and customer-service accounts. Performance on other kinds of text (for example TikTok comments, which mix Kiswahili, English and slang) has not been tested and is likely to be lower.
- The negative class is small (80 test tweets), so its F1 changes noticeably from run to run.
- Labels are noisy, which limits the achievable score.
- No group member reads Kiswahili, so we used the dataset labels as given and could not verify them ourselves.
- The model should not be used for decisions about people. Non-commercial academic use only, following the dataset terms.

## 7. Repository structure

```
notebooks/   Colab notebook: data preparation, baseline, fine-tuning, evaluation
data/        CSV files listed above
app.py       Streamlit demo
requirements.txt
README.md
```

## References

- Muhammad, S. H. et al. (2023). *AfriSenti: A Twitter Sentiment Analysis Benchmark for African Languages.* arXiv:2302.08956.
- Muhammad, S. H. et al. (2023). *SemEval-2023 Task 12: Sentiment Analysis for African Languages (AfriSenti-SemEval).* arXiv:2304.06845.
- Model card: `Davlan/afro-xlmr-base`, Hugging Face.
