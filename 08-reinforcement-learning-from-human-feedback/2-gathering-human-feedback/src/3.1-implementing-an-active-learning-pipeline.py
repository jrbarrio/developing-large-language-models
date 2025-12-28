from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling


# Create the active learner object
learner = ActiveLearner(
    # Set the estimator 
    estimator=LogisticRegression(),
    # Set the query strategy
    query_strategy=uncertainty_sampling,
    # Pass the labeled data
    X_training=X_labeled, y_training=y_labeled
)