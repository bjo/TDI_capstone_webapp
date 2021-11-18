# TDI Capstone deliverable - Project Menrva

Brian Jo, last updated on 11/18/21

This capstone deliverable consists of this github repo, and a Flask app deployed with heroku:
https://projectmenrva.herokuapp.com/

## Project description
The main goal of Project Menrva is to identify the underlying mechanism of common behavioral risks,
such as smoking, binge drinking, lack of exercise, lack of vaccination, etc.
I gathered datasets from the [CDC](https://www.cdc.gov/brfss/) for behavioral risks and 
related covariates, and from and [MSU](https://cspp.ippsr.msu.edu/) for public policies.

The Project Menrva dashboard has four portions:
1. Highlight the relationship between behavioral risks and common covariates (such as income, education, etc.)
2. Geographic understanding of prevalence of behavioral risks, and how they changed over time
3. State public policies that are correlated with the behavioral risks.
4. Causal inference with time-series data using Granger Causality.

With these understandings, we want to be able to pinpoint opportunities to influence populations
to engage less in negative behaviors.

## Skills highlight - ETL and reporting/visualization
1. Wrangled ~15G of data, including 8.2M survey responses across 21 years. (Extract)
2. Use of machine learning and statistical modeling packages such as
numpy, pandas, scikit-learn, statsmodels (Transform)
3. Use of SQL, with relevant data loaded and served using a database solution,
[elephantSQL](https://www.elephantsql.com/) (Load)
4. Development of a flask webapp and deployment with heroku (Reporting and visualization)


## Notebooks
1. Notes for data preprocessing pipelines for BRF and state policy data: https://github.com/bjo/TDI_capstone_webapp/blob/main/notebooks/Project_Menrva_preprocessing.ipynb
2. Notes for main data pipelines for statistical analysis and visualization: https://github.com/bjo/TDI_capstone_webapp/blob/main/notebooks/Project_Menrva_generate_sumstat_viz.ipynb
3. Notes for flask app, writing to DB and loading data from DB: https://github.com/bjo/TDI_capstone_webapp/blob/main/notebooks/Capstone%20project%20dev%20notes.ipynb

## Notable findings
This project highlights how socioeconomic and demographic factors, such as income as education,
are highly protective when it comes to common behavioral risks. Meanwhile, it also shows that despite
worsening health trends overall, we are able to observe that this trend is reversed in certain states.
We can also see that simple correlational studies between behavioral risks and state policies are often
confounded, as expected. Perhaps the most notable finding is that the state policies most strongly
associated causally (with time-series analysis) are not necessary policies that are directly related to the 
behavioral risk (such as alcohol taxes for binge drinking), but rather state policies that affect the 
general population (such as unemployment spending, education spending, etc.)