import pickle
from pathlib import Path


def save_model_pickle(model, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as file:
        pickle.dump(model, file)

    return str(output_path)


def load_model_pickle(input_path):
    with open(input_path, "rb") as file:
        model = pickle.load(file)

    return model