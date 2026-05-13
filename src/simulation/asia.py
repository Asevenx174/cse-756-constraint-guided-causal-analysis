from pgmpy.example_models import load_model

def load_asia_model():
    return load_model("bnlearn/asia")


def simulate_asia_samples(n_samples=10000, seed=42):
    asia_model = load_asia_model()
    asia_samples = asia_model.simulate(n_samples=n_samples, seed=seed)
    return asia_samples