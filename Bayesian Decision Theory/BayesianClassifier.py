import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

class BayesianClassifier:
    
    # Polychotomizer that uses Bayesian Decision Theory
    
    def __init__(self, likelihoods, priors, loss_matrix=None):
        
        if not all(0 <= prior <= 1 for prior in priors):
            raise ValueError("Prior must be between 0 and 1")
        if not (sum(priors) == 1):
            raise ValueError("Priors must sum to 1")
        if (len(priors) != len(likelihoods)):
            raise ValueError("Number of likelihood functions and priors must be the same")
    
        self.likelihoods = likelihoods
        self.priors = priors
        self.num_classes = len(priors)
        
        # Default 0-1 Loss if loss_matrix not provided
        if loss_matrix is None:
            self.loss_matrix = np.zeros((self.num_classes, self.num_classes))
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    if i != j:
                        self.loss_matrix[i, j] = 1.0
        else:
            self.loss_matrix = loss_matrix
    
    # Overall Probability of x
    def evidence(self, x):
        return sum(likelihood(x) * prior for likelihood,prior in zip(self.likelihoods, self.priors))
    
    
    # Bayes Formula
    def posterior(self, x, c):
        if not 1 <= c <= self.num_classes:
            raise ValueError(f"Class must be between 1 and {self.num_classes}")
        if self.evidence(x) == 0:
            return 0.0
        
        return (self.likelihoods[c - 1](x) * self.priors[c - 1])/self.evidence(x)

    # Function returns probability of error after observing x
    def p_error_x(self, x):
        return 1 - max(self.posterior(x, c) for c in range(1, self.num_classes + 1))
    
    # Function that calculates the average error
    def p_error(self):
        def integrand(x):
            return self.p_error_x(x) * self.evidence(x)

        result, _ = quad(integrand, -np.inf, np.inf)
        return result
    
    def conditional_risk(self, x, action):
        return sum(self.loss_matrix[action - 1, j] * self.posterior(x, j+1) for j in range(self.num_classes))

    # Function to classify an observation based on conditional risk. If loss_matrix is zero one loss then its the same as Bayes Decision Rule (Eq (8) from Duda et el)
    #Tie breaker - choose class 1
    def classify(self, x):
        
        risks = [self.conditional_risk(x, c) for c in range(1, self.num_classes + 1)]
        
        posteriors = [self.posterior(x, c) for c in range(1, self.num_classes + 1)]
        predicted_class = np.argmin(risks) + 1
        sorted_risks = sorted(risks)
        if self.num_classes == 1:
                confidence = 1.0
        else:
            if sorted_risks[0] == sorted_risks[1]:
                confidence = 0.0
            else:
                confidence = sorted_risks[1] - sorted_risks[0]
        error = self.p_error_x(x)
        
        return predicted_class, confidence, error
    
    
    # Function to calculate the minimum overall risk
    def bayes_risk(self):
        def integrand(x):
            min_risk = min(self.conditional_risk(x,a) for a in range(1, self.num_classes + 1))
            return min_risk * self.evidence(x)
        
        result, _ = quad(integrand, -np.inf, np.inf)
        return result
    
    def evaluate_classifier(self, x_test, y_true):
        
        if len(x_test) != len(y_true):
            raise ValueError("X_test and y_true must have the same length.")
        
        predictions = [self.classify(x)[0] for x in x_test]
        
        accuracy = np.mean(np.array(predictions) == y_true)
        
        return accuracy
    
    def plot_likelihoods(self, range_):
        
        x = np.linspace(range_[0], range_[1], 1000)

        plt.figure(figsize=(18,8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_classes))
        for i, (likelihood, color) in enumerate(zip(self.likelihoods, colors)):
            likelihood_values = [likelihood(xi) for xi in x]
            plt.plot(x, likelihood_values, color=color, label=f"Class {i+1}")
            
        plt.xlabel('x')
        plt.xticks(np.arange(range_[0], range_[1] + 0.5, 0.5))
        plt.ylabel('Probability Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('likelihoods.png', bbox_inches='tight')
        plt.close()
        
    def plot_posteriors(self, range_):
        x = np.linspace(range_[0], range_[1], 1000)
        
        plt.figure(figsize=(18,8))

        colors = plt.cm.tab10(np.linspace(0, 1, self.num_classes))
        for i, color in enumerate(colors):
            posterior_values = [self.posterior(xi, i+1) for xi in x]
            plt.plot(x, posterior_values, color=color, label=f"Class {i+1}")
        
        plt.xlabel('x')
        plt.xticks(np.arange(range_[0], range_[1] + 0.5, 0.5))
        plt.ylabel('Posteriors')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('posteriors.png', bbox_inches='tight')
        plt.close()