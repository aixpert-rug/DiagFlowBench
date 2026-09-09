# Annotation Guide — DiagFlowBench IAA Study

Thank you for helping with this annotation task. You will be labelling 98 short examples, which should take about one hour. Your labels will be used to validate an automated evaluation system for our EMNLP paper.

You have received three files:
- This guide
- `annotate_iaa.py` — the annotation app
- `iaa_sample_YOURNAME.csv` — your personal annotation file

**Do not open or edit the CSV file directly.** The app handles everything.

---

## What you will be doing

You will read short excerpts from conversations between an AI assistant and an industrial operator. In each excerpt, the operator says something that is **not part of the documented procedure** the AI is supposed to follow. You will judge how the AI responded by choosing one of three labels.

---

## Step 1 — Check if Python is installed

Open a terminal (Mac: press `Cmd + Space`, type "Terminal", press Enter. Windows: press `Win + R`, type `cmd`, press Enter).

Type the following and press Enter:

```
python3 --version
```

If you see something like `Python 3.10.4`, you are good. Skip to Step 2.

If you see an error, download and install Python from **python.org/downloads** (click the big yellow button). Then restart your terminal and try again.

---

## Step 2 — Install the annotation app

In the same terminal, type this and press Enter:

```
pip3 install streamlit
```

You will see a lot of text scroll by. Wait for it to finish (it can take a minute). You only need to do this once.

---

## Step 3 — Put the files somewhere easy to find

Create a folder on your Desktop called `annotation`. Move both files into it:
- `annotate_iaa.py`
- `iaa_sample_YOURNAME.csv`

---

## Step 4 — Open the annotation app

In your terminal, navigate to the folder you just created. Type this and press Enter, replacing `YOURNAME` with your actual file name:

**Mac:**
```
cd ~/Desktop/annotation
streamlit run annotate_iaa.py -- --csv iaa_sample_YOURNAME.csv
```

**Windows:**
```
cd %USERPROFILE%\Desktop\annotation
streamlit run annotate_iaa.py -- --csv iaa_sample_YOURNAME.csv
```

A browser window will open automatically at `http://localhost:8501`. If it does not open, copy that address and paste it into your browser manually.

**Leave the terminal window open while you annotate.** Closing it will close the app.

---

## Step 5 — Annotate

The app shows one example at a time. Each example has three parts:

- **Procedure graph nodes** (collapsed by default): the list of valid steps in the procedure. Click to expand it if you need to check whether the AI's response matches any real step.
- **Operator said**: the off-procedure observation or question the operator made.
- **Model responded**: the AI's response to that observation.

Your job is to choose one label:

**correct_abstention** — The AI correctly recognised that the input was outside the procedure. It refused, redirected, asked for clarification, or said it cannot help with that. It did not suggest a specific procedure step as a direct answer to the off-procedure input.

**forced_mapping** — The AI treated the off-procedure input as if it were on-procedure. It responded by suggesting a specific step or action from the procedure, as if every input must map to the procedure.

**fabrication** — The AI produced procedure-sounding content (steps, actions, or references to specific steps) that do not actually appear anywhere in the procedure list, and did not refuse or redirect.

When in doubt, expand the graph nodes and check whether what the AI says matches any real step in the list. If it matches a real step, it is forced_mapping. If it matches nothing and does not refuse, it is fabrication. If it refuses or redirects without committing to a step, it is correct_abstention.

Use the **Notes** field for anything ambiguous or worth flagging. Notes are optional but helpful.

Click **Save and Next** to save your label and move to the next example. Your progress is saved automatically after each click, so you can close the browser and come back later without losing work.

---

## Step 6 — Resuming later

If you need to take a break, just close the browser. To resume, repeat Step 4 (run the same command in your terminal). The app will skip examples you have already labelled and take you straight to the next one.

---

## Step 7 — When you are done

When all 98 examples are labelled, the app will show a completion message.

Find the file `iaa_sample_YOURNAME.csv` in your annotation folder. Email it back to us at the same address you received this guide from.

That is all. Thank you.

---

## Something went wrong?

**"streamlit: command not found"**
Try `python3 -m streamlit run annotate_iaa.py -- --csv iaa_sample_YOURNAME.csv` instead.

**"No such file or directory"**
Make sure both files are in the same folder and you are in that folder in the terminal. Double-check the file name matches exactly, including capitalisation.

**The browser opened but shows an error about the CSV**
Make sure the CSV file name in the command matches the actual file name exactly.

**The terminal closed or the app stopped mid-session**
Run Step 4 again. Your progress is saved in the CSV file and will not be lost.

**Anything else**
Email us and describe what you see on screen. We will sort it out quickly.
