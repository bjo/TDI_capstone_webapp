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


## Notebooks and scripts
- Notes for data preprocessing pipelines for BRF and state policy data: 
https://github.com/bjo/TDI_capstone_webapp/blob/main/notebooks/Project_Menrva_preprocessing.ipynb
(also refer to https://github.com/bjo/TDI_capstone_webapp/blob/main/bin/preprocessing_scripts.py)
- Notes for main data pipelines for statistical analysis and visualization:
https://github.com/bjo/TDI_capstone_webapp/blob/main/notebooks/Project_Menrva_generate_sumstat_viz.ipynb
(also refer to https://github.com/bjo/TDI_capstone_webapp/blob/main/bin/generate_sumstat_viz.py)
- Notes for flask app, writing to DB and loading data from DB: 
https://github.com/bjo/TDI_capstone_webapp/blob/main/notebooks/Capstone%20project%20dev%20notes.ipynb
(also refer to https://github.com/bjo/TDI_capstone_webapp/blob/main/bin/create_table.py)

## Notable findings
This project highlights how socioeconomic and demographic factors, such as income as education,
are highly protective when it comes to common behavioral risks. Meanwhile, we can see that various
health-related behavioral risks are improving, notably pneumonia vaccination and healthcare coverage.
These improving health trends are more pronounced in certain regions (such as Puerto Rico), but 
sometimes they are reversed as well.
We can also see that simple correlational studies between behavioral risks and state policies are often
confounded, as expected. For an example, look at the relationship between lack of exercise and
anecpi (miscellaneous spending). There is no clear reason why these should be correlated, and 
the confounding appears to come from the fact that certain states (most notably, Alaska), have high rates of both,
confounding this study (even after correcting for state-specific effects in the behavioral risk).
Perhaps the most notable finding is that the state policies that are causally most strongly
associated (according to time-series analysis using Granger causality) are not necessary policies that
are directly related to the behavioral risk (such as alcohol taxes for binge drinking), 
but rather state policies that affect the general population more (such as unemployment spending, education spending, etc.)