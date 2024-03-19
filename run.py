import logging
logging.basicConfig(level=logging.INFO)

from pipeline import Pipeline

pipe = Pipeline()
for task in ["random", "sentence_length", "tree_depth", "top_constituents", "tense", "subject_number", "object_number"]: 
    for seed in [1, 2, 3]:
        pipe.initialize_task_seed(task, seed, batch_size=32)
        pipe.train(until_val_loss_goes_up=True, num_epochs=50)
        pipe.find_importance_scores()