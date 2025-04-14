from BayesianClassifier import BayesianClassifier
from scipy.stats import multivariate_normal, norm
import numpy as np

# Sanity Check and Test of the Bayesian Classifier

def create_likelihood_1d(mean, std):
    return lambda x: norm.pdf(x, mean, std)

def create_likelihood_2d(mean, cov):
    return lambda x: multivariate_normal.pdf(x, mean, cov)

def generate_test_data_1d(n_samples, means, stds, priors):
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

def generate_test_data_2d(n_samples, means, covs, priors):
    samples = []
    labels = []
    
    for i, (mean, cov, prior) in enumerate(zip(means, covs, priors)):
        n_class_samples = int(n_samples * prior)
        class_samples = np.random.multivariate_normal(mean, cov, n_class_samples)
        samples.append(class_samples)
        labels.append(np.ones(n_class_samples) * (i + 1))
    
    X = np.concatenate(samples)
    y = np.concatenate(labels)

    return X, y

def test_1d():
        
    means = [0, 2, 4]
    stds = [1, 1, 1]
    
    likelihoods = [create_likelihood_1d(mean, std) for mean, std in zip(means, stds)]
    priors = [0.3, 0.3, 0.4]
    
    classifier = BayesianClassifier(likelihoods, priors, input_dim=1)
    
    X_test, y_test = generate_test_data_1d(1000, means, stds, priors)
    
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
    
    # Plot likelihoods & posteriors:
    classifier.plot_likelihoods(range_=(-10, 10))
    classifier.plot_posteriors(range_=(-10, 10))
    classifier.plot_decision_boundaries(range_=(-10, 10))

def test_2d():
    
    means = [
        np.array([0, 0]),
        np.array([4, 4]),
        np.array([0, 8])
    ]
    
    covs = [
        np.array([[2, 0.5], [0.5, 1]]),
        np.array([[1, -0.3], [-0.3, 1]]),
        np.array([[2, 0], [0, 2]])
    ]
    
    priors = [0.3, 0.3, 0.4]
    
    # Create likelihood functions
    likelihoods = [create_likelihood_2d(mean, cov) for mean, cov in zip(means, covs)]
    
    # Create classifier with input_dim=2
    classifier = BayesianClassifier(likelihoods, priors, input_dim=2)
    
    # Generate test data
    n_samples = 1000
    X_test, y_test = generate_test_data_2d(n_samples, means, covs, priors)
    
    # Evaluate classifier
    accuracy = classifier.evaluate_classifier(X_test, y_test)
    print(f"Classification accuracy: {accuracy:.4f}")
    
    # Test some specific points
    test_points = [
        np.array([1, 1]),
        np.array([4, 4]),
        np.array([0, 7])
    ]
    
    for point in test_points:
        class_label, confidence, error = classifier.classify(point)
        print(f"\nPoint {point}:")
        print(f"  Classified as class {class_label}")
        print(f"  Confidence: {confidence:.4f}")
        print(f"  Probability of error: {error:.4f}")
    
    # Plot likelihoods, posteriors and decision boundaries
    x_range = [-4, 8]
    y_range = [-4, 12]
    
    classifier.plot_likelihoods(range_=[x_range, y_range])
    classifier.plot_posteriors(range_=[x_range, y_range])
    classifier.plot_decision_boundaries(range_=[x_range, y_range])

def main():
    print("Testing 1D Bayesian Classifier")
    print("==============================")
    test_1d()
    
    print("\nTesting 2D Bayesian Classifier")
    print("==============================")
    test_2d()

if __name__ == "__main__":
    main()