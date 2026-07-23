# Student Graduation Risk & Degree Classification Prediction

A statistical machine learning project predicting university student graduation
likelihood and final degree classification (First Class, Second Class Upper/Lower,
Third Class — using the Nigerian grading system), built as a portfolio project
for graduate school applications in Statistics/Data Science.

## Motivation

During internships at Aero Contractors Nigeria, I worked on the build team for
Excel-based revenue reporting and flight operations data systems. That experience
highlighted the gap between manual, spreadsheet-based reporting and what's
possible with proper statistical modeling — motivating this project as a step
toward deeper, code-based statistical analysis.

## Project overview

Since real institutional student records weren't accessible for this project,
I designed and built a **realistic simulated dataset** with known, true
underlying relationships between predictors (attendance rate, entry CGPA, study
hours, financial stress, entry mode) and two outcomes: graduation likelihood and
final class of degree. Building the data-generating process myself — rather
than using an off-the-shelf dataset — meant I could verify each model actually
recovered the true relationships I built in, which is the core validation
approach used throughout this project.

## Methods

- **Logistic Regression** — baseline model, interpreted via odds ratios and
  95% confidence intervals (via bootstrap resampling)
- **Bayesian Logistic Regression** — implemented from scratch using a
  Metropolis-Hastings MCMC sampler (no external Bayesian library), producing
  posterior distributions and credible intervals for each coefficient
- **Random Forest** and **Gradient Boosting** — compared against the logistic
  baseline using test-set AUC and variable importance
- **Calibration analysis** — checking whether predicted probabilities matched
  actual observed outcome rates, not just ranking ability (AUC)
- **Multinomial Logistic Regression** — extended to predict probability across
  all four degree classifications, not just a binary graduation outcome

## Key findings

- All models correctly recovered the true relationships built into the
  simulation, with attendance rate emerging as the strongest predictor across
  every method (logistic regression, Random Forest, Gradient Boosting, and the
  Bayesian model all agreed on this ranking)
- Frequentist (bootstrap) and Bayesian (MCMC) confidence/credible intervals
  converged closely, as expected given a weakly informative prior
- Logistic regression slightly outperformed the tree-based models on this
  dataset (AUC ~0.85 vs ~0.83), consistent with the underlying relationships
  being genuinely additive/linear by design — a useful reminder that more
  complex models don't automatically win

## A debugging note

Midway through this project, one predictor's estimated effect came out wildly
different from its true, known value. Rather than ignore it, I traced the
cause to a subtle simulation bug: re-seeding the random number generator with
the *same* seed value across separate pipeline steps created a spurious
correlation between otherwise-unrelated random draws. Using distinct seeds per
pipeline stage resolved it. I've kept this process visible in the notebook
rather than a "clean" final version only, since diagnosing this kind of issue
is a real and important part of doing this work honestly.

## Repository structure

- `Grad_track.ipynb` — full analysis notebook: data simulation, all four models,
  bootstrap and Bayesian inference, calibration analysis
- `app.py` — interactive Streamlit dashboard for live predictions
- `README.md` — this file

## Running this project

**Notebook:** open `Grad_track.ipynb` in Google Colab or Jupyter and run all
cells in order.

**Dashboard:** 
[pip install streamlit scikit-learn pandas numpy
streamlit run app.py ]
This launches a local interactive dashboard where you can adjust student
variables (attendance, CGPA, study hours, entry mode, financial stress) and see
live predictions across all four models, along with odds ratio interpretation
and class-of-degree probabilities.

## Disclaimer

This project uses entirely simulated data for demonstration purposes. No real
student records were used at any point.