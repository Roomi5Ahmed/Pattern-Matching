# Random Data to Classifier Training Pipeline
Pattern Matching mini project.

## Setup and run
    pip install -r requirements.txt
    python pipeline.py            # full run (~1-2 min), 30 repetitions per setting
    python pipeline.py --quick    # fast run, 8 repetitions

Tables (CSV) and figures (PNG) are written to `results/`.

## Pipeline stages
1. Data generation: 3 classes in 2-D with controllable separation, correlation,
   covariance structure, class priors, training-set size and distribution family
   (gaussian, laplace, uniform, student_t, lognormal, bimodal).
2. Parameter estimation: MLE of class means, covariances and priors,
   compared against the true generating parameters (Exp A).
3. Classifiers: Gaussian Bayes written from scratch (QDA, LDA, Naive),
   plus k-NN and Logistic Regression. A Gaussian "Oracle" uses the true parameters.
4. Analysis: experiments B-H (separation, families, covariance, correlation,
   training size, imbalance, normality tests).
