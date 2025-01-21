from BayesianClassifier import BayesianClassifier
from scipy.stats import norm
import numpy as np

# Sanity Check and Test of the Bayesian Classifier


# Create Gaussian Distribution Likelihood Functions
def create_likelihood(mean, std):
    return lambda x: norm.pdf(x, mean, std)

def generate_test_data(n_samples, means, stds, priors):
    
    samples = []
    labels = []

    for i, (mean, std, prior) in enumerate(zip(means, stds, priors)):
        n_class_samples = int(n_samples * prior)
        class_samples = np.random.normal(mean, std, n_class_samples)
        samples.append(class_samples)
        labels.append(np.ones(n_class_samples) * (i + 1))
    
    X = np.concatenate(samples)
    y = np.concatenate(labels)
    
    return X, y

def main():
    
    means = [0, 2, 4]
    stds = [1, 1, 1]
    
    likelihoods = [create_likelihood(mean, std) for mean, std in zip(means, stds)]
    priors = [0.3, 0.3, 0.4]
    
    classifier = BayesianClassifier(likelihoods, priors)
    
    X_test, y_test = generate_test_data(1000, means, stds, priors)
    
    # Evaluate Classifier:
    accuracy = classifier.evaluate_classifier(X_test, y_test)
    print(f"Classification accuracy: {accuracy:.4f}")
    
    # Calculate theoretical error rate
    error_rate = classifier.p_error()
    print(f"Theoretical error rate: {error_rate:.4f}")
    
    test_points = [1, 2, 3]
    for point in test_points:
        class_label, confidence, error = classifier.classify(point)
        print(f"\nPoint {point:.1f}:")
        print(f"  Classified as class {class_label}")
        print(f"  Confidence: {confidence:.4f}")
        print(f"  Probability of error: {error:.4f}")
    
    #Plot likelihoods & posteriors:
    classifier.plot_likelihoods(range_=(-10, 10))
    classifier.plot_posteriors(range_=(-10, 10))
    

if __name__ == "__main__":
    main()