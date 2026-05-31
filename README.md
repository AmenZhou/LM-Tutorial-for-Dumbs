# ML/DS Learning Arena

Game-based interactive tutorials for learning machine learning and data science fundamentals.

## Phase 1 — Data Foundations & Math Intuition

Open `index.html` in a browser to start. No build step — pure HTML/CSS/JS.

| Week | Game | Concepts |
|------|------|----------|
| Week 1 | Data Foundations Sandbox | NumPy vectorization, Pandas wrangling, matrix multiplication, gradient descent |
| Week 2 | Cloud Cost Optimizer Game | Feature scaling, loss landscapes, gradient descent arena |
| Week 3 | Data Detective Game | EDA, log transforms, scatter forensics, correlation heatmap |
| Week 4 | Capstone Pipeline Simulator | End-to-end preprocessing: imputation → transform → prune → scale |

`phase1/week4/capstone.py` — runnable Python version of the capstone pipeline (requires numpy, pandas, matplotlib, seaborn).

## Usage

```bash
open index.html          # macOS
# or just double-click index.html in Finder
```

For the Python capstone:
```bash
pip install numpy pandas matplotlib seaborn
python phase1/week4/capstone.py
```
