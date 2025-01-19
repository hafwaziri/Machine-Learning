import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

class BayesianClassifier:
    
    # Dichotomizer that uses Bayesian Decision Theory
    
    def __init__(self, likelihood_class1, prior_class1, likelihood_class2, prior_class2):
        
        if not (0 <= prior_class1 <= 1 and 0 <= prior_class2 <= 1):
            raise ValueError("Prior must be between 0 and 1")
        if not (prior_class1 + prior_class2 == 1):
            raise ValueError("Priors must sum to 1")
        
        self.likelihood_class1 = likelihood_class1
        self.prior_class1 = prior_class1
        self.likelihood_class2 = likelihood_class2
        self.prior_class2 = prior_class2
    
    # Overall Probability of x
    def evidence(self, x):
        return (self.likelihood_class1(x) * self.prior_class1) + (self.likelihood_class2(x) * self.prior_class2)
    
    
    # Bayes Formula
    def posterior(self, x, c):
        if self.evidence(x) == 0:
            return 0.0
        
        if (c == 1):
            return (self.likelihood_class1(x) * self.prior_class1)/self.evidence(x)
        elif (c ==2):
            return (self.likelihood_class2(x) * self.prior_class2)/self.evidence(x)
        else:
            raise ValueError("Class c must be 1 or 2")

    # Function returns probability of error after observing x
    def p_error_x(self, x):
        return min(self.posterior(x, 1), self.posterior(x, 2))
    
    # Function that calculates the average error
    def p_error(self):
        def integrand(x):
            return self.p_error_x(x) * self.evidence(x)

        result, _ = quad(integrand, -np.inf, np.inf)
        return result

    # Function to classify an observation based on Bayes Decision Rule (Eq (8) from Duda et el)
    #Tie breaker - choose class 1
    def classify(self, x):
        predicted_class = 1 if self.likelihood_class1(x) * self.prior_class1 >= self.likelihood_class2(x) * self.prior_class2 else 2
        confidence = abs(self.posterior(x, 1) - self.posterior(x, 2))
        error = self.p_error_x(x)
        
        return predicted_class, confidence, error
    
    
    def evaluate_classifier(self, x_test, y_true):
        
        predictions = [self.classify(x)[0] for x in x_test]
        
        accuracy = np.mean(np.array(predictions) == y_true)
        
        return accuracy
    
    def plot_likelihoods(self, range):
        
        x = np.linspace(range[0], range[1], 1000)
        likelihood1_values = [self.likelihood_class1(xi) for xi in x]
        likelihood2_values = [self.likelihood_class2(xi) for xi in x]
        
        plt.figure(figsize=(18,8))
        plt.plot(x, likelihood1_values, 'b-', label="Class 1")
        plt.plot(x, likelihood2_values, 'r-', label="Class 2")
        plt.xlabel('x')
        plt.xticks(np.arange(range[0], range[1] + 0.5, 0.5))
        plt.ylabel('Probability Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('likelihoods.png', bbox_inches='tight')
        plt.close()
        
    def plot_posteriors(self, range):
        x = np.linspace(range[0], range[1], 1000)
        posterior1_values = [self.posterior(xi, 1) for xi in x]
        posterior2_values = [self.posterior(xi, 2) for xi in x]
        
        plt.figure(figsize=(18,8))
        plt.plot(x, posterior1_values, 'b-', label="Class 1")
        plt.plot(x, posterior2_values, 'r-', label="Class 2")
        plt.xlabel('x')
        plt.xticks(np.arange(range[0], range[1] + 0.5, 0.5))
        plt.ylabel('Posteriors')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('posteriors.png', bbox_inches='tight')
        plt.close()