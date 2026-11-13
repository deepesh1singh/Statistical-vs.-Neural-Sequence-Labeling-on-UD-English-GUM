# CS60075 NLP Assignment II — Project 2: Statistical vs. Neural Sequence Labeling

**Dataset 2.2 — UD English GUM (part-of-speech tagging)**
**Group 5:** Devashri Deulkar (21CS10021), Pranav Vijaykumar (23EE10048), Deepesh Singh (22CS30020)

This repository contains a single Google Colab notebook that implements all three
tasks end to end, plus the written report.

| Task | What it is | Test accuracy |
|------|------------|--------------:|
| Task 1 | First-order HMM, add-λ smoothing (λ = 0.1), suffix + capitalisation model for unknown words, log-space Viterbi | 92.436 % |
| Task 2 | Character + word BiLSTM-CRF (reproduction of Lample et al., 2016) | 96.806 % |
| Task 3 | Neural-emission HMM: Task 2 encoder + calibrated softmax emissions, Task 1 transitions and Viterbi decoder reused unchanged | 95.852 % |

Accuracy is per-token accuracy on the official `test` split (28,397 tokens).

---

## 1. Contents of this package

```
code/
  Group_5_NLP_Assignment_2.ipynb   # all three tasks, in separate notebook sections
  task1/README.md                  # pointer to the Task 1 cells
  task2/README.md                  # pointer to the Task 2 cells
  task3/README.md                  # pointer to the Task 3 cells
README.md                          # this file
requirements.txt                   # pinned Python dependencies
report.pdf                         # written report (methods, results, discussion)
```

The notebook is organised, top to bottom, as:

1. **Initial Setup** — dependency install, imports, configuration (seeds, hyper-parameters, output paths)
2. **Data** — loading UD English GUM v2.18, split statistics, preprocessing
3. **Task 1 — First-order HMM tagger** (model, training/fit, validation accuracy)
4. **Shared neural data pipeline** — vocabularies, batching, GloVe initialisation
5. **Task 2 — Character + word BiLSTM-CRF** (model, training, validation history)
6. **Task 3 — Neural-emission HMM** (softmax fine-tuning, temperature calibration, hybrid decoding)
7. **Final held-out test evaluation and qualitative analysis** (all three systems on `test`)
8. **Save** — writes every result file and zips the output directory

Each section is a clearly delimited group of cells with a Markdown heading, so each task's
contribution can be inspected and run independently (see §4).

---

## 2. Environment setup

The notebook was developed and run on a **Google Colab GPU runtime** (CUDA). A GPU is
strongly recommended for Tasks 2 and 3; Task 1 runs on CPU in well under a minute.

### Option A — Google Colab (recommended, minimal setup)

1. Open `code/Group_5_NLP_Assignment_2.ipynb` in Google Colab.
2. Set **Runtime → Change runtime type → GPU**.
3. **Runtime → Run all.** The first cell installs the dependencies
   (`datasets`, `gensim`, `pyarrow`, `tqdm`); everything else is pre-installed on Colab.

### Option B — local machine

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook code/Group_5_NLP_Assignment_2.ipynb
```

Python 3.10+ and a CUDA-capable GPU are assumed for the neural tasks. Outputs are
written next to the notebook under `project2_gum_outputs/` when `/content` does not exist.

---

## 3. Data and pretrained resources (downloaded automatically)

Nothing needs to be downloaded by hand; the notebook fetches everything on first run and
caches it.

| Resource | Source | How it is obtained |
|----------|--------|--------------------|
| UD English GUM, splits `train` / `dev` / `test` | Hugging Face hub, `universal-dependencies/universal_dependencies`, config `en_gum`, **revision `2.18`** | `load_dataset("universal-dependencies/universal_dependencies", "en_gum", revision="2.18")` |
| 100-d GloVe word vectors (GloVe 6B, Wikipedia + Gigaword) | gensim-data, model name `glove-wiki-gigaword-100` | `gensim.downloader.load("glove-wiki-gigaword-100")` |

The dataset revision is pinned so that both the data and the train/dev/test partition are
frozen. Split sizes at v2.18: train 11,314 sentences / 200,223 tokens; dev 1,575 / 28,119;
test 1,464 / 28,397. Only the `tokens` and `upos` columns are used; `upos` integer ids are
mapped back to the 17 Universal POS tag names.

No data files or model checkpoints are included in this archive.

---

## 4. Running the tasks

### All three tasks in sequence

**Runtime → Run all.** On a Colab GPU the full pipeline (Task 1 fit, Task 2 training with
early stopping, Task 3 fine-tuning and calibration, final evaluation) completes in roughly
an hour; with the saved checkpoints present it completes in a few minutes.

### Task 1 only (no GPU needed)

Run the cells in the sections **Initial Setup**, **Data** and **Task 1**. The last Task 1
cell fits the HMM on `train` and prints validation accuracy.

### Task 2 only

Run **Initial Setup**, **Data**, **Shared neural data pipeline** and **Task 2**. The training
cell trains the BiLSTM-CRF with early stopping on `dev` (patience 5) and saves a checkpoint
plus the per-epoch history.

### Task 3

Task 3 needs the Task 1 transition tables and the Task 2 encoder, so either run those
sections first in the same session, or place compatible checkpoints in
`project2_gum_outputs/checkpoints/` (see §5) and then run the **Task 3** section. The
section fine-tunes a softmax tagger from the Task 2 encoder, selects a temperature on `dev`
NLL, and plugs the calibrated emissions into the unchanged Task 1 HMM decoder.

### Final evaluation and saving

Run **Final held-out test evaluation and qualitative analysis** (requires all three models
in memory) and then **Save**. The test split is touched only in this section.

---

## 5. Where results and checkpoints are written

All outputs go to `OUTPUT_DIR`, which is `/content/project2_gum_outputs/` on Colab and
`./project2_gum_outputs/` elsewhere:

| File | Contents |
|------|----------|
| `checkpoints/task2_bilstm_crf.pt` | Task 2 weights, training history, best dev accuracy, metadata |
| `checkpoints/task3_neural_emissions.pt` | Task 3 softmax tagger weights, history, selected temperature, calibration curve |
| `results.json` | every headline metric (Task 1/2/3 test accuracy, best dev accuracies, temperature, GloVe coverage, …) |
| `configuration.json` | all hyper-parameters |
| `task_comparison.csv`, `task_comparison.png` | Task 1 vs Task 2 vs Task 3 table and bar chart |
| `task2_training_history.csv`, `task3_training_history.csv` | per-epoch learning rate, losses, dev accuracy |
| `task3_temperature_calibration.csv` | dev NLL for each candidate temperature |
| `qualitative_examples.csv` | the automatically selected test sentences with all three systems' tags |
| `test_predictions.jsonl` | per-token predictions of all three systems for every test sentence |
| `environment_freeze.txt` | `pip freeze` of the runtime that produced the results |

The **Save** cell also zips the whole directory as `Project2_UD_English_GUM_outputs.zip`
(at `/content/` on Colab) for download.

Checkpoints are re-used automatically when they match the current dataset revision,
vocabulary sizes and tag list; delete the `checkpoints/` folder to force re-training.

---

## 6. Reproducibility

* All random seeds (Python `random`, NumPy, PyTorch CPU and CUDA) are fixed to **42**;
  cuDNN runs in deterministic mode with benchmarking disabled.
* The dataset revision (`2.18`) and the GloVe model name are pinned in the configuration cell.
* Mini-batch composition is drawn from a seeded RNG that is re-seeded per epoch.
* `environment_freeze.txt` records the exact package versions of the run that produced the
  reported numbers.

Exact bit-for-bit reproduction of the neural results additionally depends on the GPU model
and CUDA/cuDNN versions of the runtime.

---

## 7. Key hyper-parameters

| Setting | Value |
|---------|-------|
| HMM smoothing λ | 0.1 |
| Unknown-word model | suffixes of length 3 → 2 → 1 + capitalisation pattern; uniform over open classes as last resort |
| Char embedding / char-LSTM hidden | 25 / 25 per direction |
| Word embedding | GloVe 100-d, fine-tuned (94.75 % vocabulary coverage) |
| Word BiLSTM | 1 layer, 100 hidden per direction |
| Dropout | 0.5 before and after the word BiLSTM |
| Optimiser | SGD, momentum 0.9, lr 0.015 decayed as lr₀ / (1 + 0.05·(epoch − 1)), grad-norm clip 5.0 |
| Batch size / epochs / early stopping | 32 / max 50 / patience 5 on dev accuracy |
| Task 3 temperature grid | 151 values in [0.25, 4.0], selected by dev NLL (chosen T = 1.175) |

---

## 8. References

* Lample, Ballesteros, Subramanian, Kawakami, Dyer. *Neural Architectures for Named Entity
  Recognition.* NAACL 2016. arXiv:1603.01360. Reference code: https://github.com/glample/tagger
* de Marneffe, Manning, Nivre, Zeman. *Universal Dependencies.* Computational Linguistics 47(2), 2021.
* Zeldes. *The GUM Corpus.* Language Resources and Evaluation 51(3), 2017.
* Pennington, Socher, Manning. *GloVe: Global Vectors for Word Representation.* EMNLP 2014.
