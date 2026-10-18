
import numpy as np
import pandas as pd


class BoltzmannMachine:
    """A simplified RBM for learning product preference patterns."""

    def __init__(self, n_visible, n_hidden, learning_rate=0.1):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate

        # Initialize model parameters
        self.weights = np.random.normal(
            0, 0.1, (n_visible, n_hidden)
        )
        self.visible_bias = np.zeros(n_visible)
        self.hidden_bias = np.zeros(n_hidden)


    @staticmethod
    def sigmoid(x):
        """Convert activation values into probabilities."""
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def hidden_probabilities(self, visible_data):
        """Estimate hidden-unit activation probabilities."""
        activation = (
            np.dot(visible_data, self.weights)
            + self.hidden_bias
        )
        return self.sigmoid(activation)

    def visible_probabilities(self, hidden_data):
        """Estimate visible-unit activation probabilities."""
        activation = (
            np.dot(hidden_data, self.weights.T)
            + self.visible_bias
        )
        return self.sigmoid(activation)

    
    def train(self, training_data, epochs=100):
        """Train the RBM using one-step Contrastive Divergence."""
        n_samples = training_data.shape[0]

        for epoch in range(epochs):
            # Positive phase: learn from actual preferences
            positive_hidden_probs = self.hidden_probabilities(
                training_data
            )

            positive_hidden_states = (
                np.random.random(positive_hidden_probs.shape)
                < positive_hidden_probs
            ).astype(float)

            # Negative phase: reconstruct preferences
            # Negative phase: reconstruct and sample visible states
            negative_visible_probs = self.visible_probabilities(
             positive_hidden_states
            )

            negative_visible_states = (
                np.random.random(negative_visible_probs.shape)
                < negative_visible_probs
            ).astype(float)

            negative_hidden_probs = self.hidden_probabilities(
                negative_visible_states
            )

            # Update weights using positive and negative associations
            positive_associations = np.dot(
                training_data.T, positive_hidden_probs
            )
            negative_associations = np.dot(
                negative_visible_states.T, negative_hidden_probs
            )

            self.weights += self.learning_rate * (
                positive_associations - negative_associations
            ) / n_samples

            # Update visible and hidden biases
            self.visible_bias += self.learning_rate * np.mean(
                training_data - negative_visible_states, axis=0
            )

            self.hidden_bias += self.learning_rate * np.mean(
                positive_hidden_probs - negative_hidden_probs, axis=0
            )

        print(f"Training completed for {epochs} epochs.")



    def recommend(self, user_preferences, product_names, top_n=3):
        """Recommend unselected products with the highest estimated probabilities."""
        user_preferences = np.asarray(user_preferences, dtype=float)

        if user_preferences.shape != (self.n_visible,):
            raise ValueError(
                f"Expected {self.n_visible} product preferences."
            )

        if len(product_names) != self.n_visible:
            raise ValueError(
                "The number of product names must match visible units."
            )

        if not 1 <= top_n <= self.n_visible:
            raise ValueError(
                f"top_n must be between 1 and {self.n_visible}."
            )

        # Infer latent preferences and reconstruct product probabilities
        hidden_probs = self.hidden_probabilities(user_preferences)
        product_probs = self.visible_probabilities(hidden_probs)

        # Exclude products already positively selected
        unselected = np.flatnonzero(user_preferences == 0)
        ranked = unselected[
            np.argsort(product_probs[unselected])[::-1]
        ]

        return [
            (product_names[i], float(product_probs[i]))
            for i in ranked[:top_n]
        ]


    


if __name__ == "__main__":
    dataset = pd.read_csv("dataset/user_preferences.csv")
    user_preferences = dataset.drop(columns=["user_id"]).to_numpy(
        dtype=float
    )

    np.random.seed(42)

    model = BoltzmannMachine(
        n_visible=user_preferences.shape[1],
        n_hidden=3
    )

    print("Dataset shape:", user_preferences.shape)
    print("Visible units:", model.n_visible)
    print("Hidden units:", model.n_hidden)
    print("Weight matrix shape:", model.weights.shape)

    hidden_probs = model.hidden_probabilities(user_preferences)
    reconstructed_probs = model.visible_probabilities(hidden_probs)

    print("Hidden probabilities shape:", hidden_probs.shape)
    print("Reconstructed probabilities shape:", reconstructed_probs.shape)
    print("Sample hidden probabilities:", hidden_probs[0])
    print("Sample reconstructed preferences:", reconstructed_probs[0])


    weights_before_training = model.weights.copy()

    model.train(user_preferences, epochs=100)

    print(
        "Weights changed after training:",
        not np.allclose(weights_before_training, model.weights)
    )

    trained_hidden_probs = model.hidden_probabilities(user_preferences)
    print("Trained hidden probabilities shape:", trained_hidden_probs.shape)


    product_names = dataset.drop(columns=["user_id"]).columns.tolist()

    user_index = 0
    user_vector = user_preferences[user_index]

    recommendations = model.recommend(
        user_vector, product_names, top_n=3
    )

    print(f"\nRecommendations for user {dataset.iloc[user_index]['user_id']}:")
    for product, probability in recommendations:
        print(f"{product}: {probability:.4f}")

