# Model Lifecycle and Evaluation

## Supported input

The workspace implementation accepts fitted binary scikit-learn estimators with numeric features, `.skops` model serialization and class labels `[0, 1]`. A workspace includes a historical reference dataset, incoming observations and, when performing supervised evaluation or retraining, labels and a historical validation cohort. Feature and identifier columns are specific to the workspace.

## Monitoring and retraining

1. Validate model compatibility and dataset schema.
2. Assess population drift between reference and incoming feature distributions.
3. Evaluate the champion against available ground-truth labels.
4. Clone and retrain a compatible estimator using designated labelled training records.
5. Compare champion and challenger against the same historical and incoming holdouts.
6. Apply improvement thresholds and regression guardrails.
7. Retain candidate artifacts and results; require explicit promotion of an eligible candidate.

A challenger may fail the acceptance checks. Historical validation data supplied by a user cannot by itself prove that the data was never used to train the uploaded model.
