from BayesianClassifier import BayesianClassifier
from scipy.stats import norm
import numpy as np

# Sanity Check and Test of the Bayesian Classifier


# Create Gaussian Distribution Likelihood Functions
def create_likelihood(mean, std):
    return lambda x: norm.pdf(x, mean, std)

def generate_test_data(n_samples, mean1, std1, mean2, std2):
    
    class1_samples = np.random.normal(mean1, std1, n_samples)
    
    class2_samples = np.random.normal(mean2, std2, n_samples)
    
    X = np.concatenate([class1_samples, class2_samples])
    y = np.concatenate([np.ones(n_samples), np.ones(n_samples)*2])
    
    return X, y


def main():
    
    mean1, std1 = 0, 1
    mean2, std2 = 2, 1
    
    likelihood1 = create_likelihood(mean1, std1)
    likelihood2 = create_likelihood(mean2, std2)
    
    classifier = BayesianClassifier(likelihood_class1=likelihood1,
                                    prior_class1=0.5,
                                    likelihood_class2=likelihood2,
                                    prior_class2=0.5)
    
    X_test, y_test = generate_test_data(1000, mean1, std1, mean2, std2)
    
    # Evaluate Classifier:
    accuracy = classifier.evaluate_classifier(X_test, y_test)
    print(f"Classification accuracy: {accuracy:.4f}")
    
    # Calculate theoretical error rate
    error_rate = classifier.p_error()
    print(f"Theoretical error rate: {error_rate:.4f}")
    
    # Classify 1 point:
    point = 0.5
    class_label, confidence, error = classifier.classify(point)
    print(f"Point {point} classified into class {class_label}, with confidence: {confidence:.4f} and probability of error: {error:.4f}")
    
    #Plot likelihoods:
    classifier.plot_likelihoods(range=(-10, 10))
    classifier.plot_posteriors(range=(-10, 10))
    

if __name__ == "__main__":
    main()