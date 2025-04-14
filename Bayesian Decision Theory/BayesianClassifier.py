import numpy as np
from scipy.integrate import nquad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plots


class BayesianClassifier:
    
    # Polychotomizer that uses Bayesian Decision Theory
    
    def __init__(self, likelihoods, priors, loss_matrix=None, input_dim=1):
        
        if not all(0 <= prior <= 1 for prior in priors):
            raise ValueError("Prior must be between 0 and 1")
        if not (sum(priors) == 1):
            raise ValueError("Priors must sum to 1")
        if (len(priors) != len(likelihoods)):
            raise ValueError("Number of likelihood functions and priors must be the same")
    
        self.likelihoods = likelihoods
        self.priors = priors
        self.num_classes = len(priors)
        self.min_error_rate_classification = False
        self.input_dims = input_dim
        
        # Default 0-1 Loss if loss_matrix not provided
        if loss_matrix is None:
            self.min_error_rate_classification = True
            self.loss_matrix = np.zeros((self.num_classes, self.num_classes))
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    if i != j:
                        self.loss_matrix[i, j] = 1.0
        else:
            self.loss_matrix = loss_matrix
            
        # Threshold value for 2 category classification using likelihood rations
        if self.num_classes == 2:
            self.threshold = self._compute_threshold()
    
    
    def _compute_threshold(self):
        if self.num_classes != 2:
            raise ValueError("Threshold computation is only applicable for 2-category classification")
        
        lambda_11 = self.loss_matrix[0, 0]
        lambda_12 = self.loss_matrix[0, 1]
        lambda_21 = self.loss_matrix[1, 0]
        lambda_22 = self.loss_matrix[1, 1]
        
        P_omega1 = self.priors[0]
        P_omega2 = self.priors[1]
        
        threshold = ((lambda_12 - lambda_22) * P_omega2) / ((lambda_21 - lambda_11) * P_omega1)
        return threshold
    
    def likelihood_ratio_decision_rule(self, x):
        if self.num_classes != 2:
            raise ValueError("This method is only for 2-Category Classification")
        
        likelihood_ratio = self.likelihoods[0](x) / self.likelihoods[1](x)
        
        if likelihood_ratio > self.threshold:
            return 1
        else:
            return 2
    
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
        def integrand(*x):
            x_point = np.array(x)
            return self.p_error_x(x_point) * self.evidence(x_point)

        bounds = [(-np.inf, np.inf)] * self.input_dims
        result, _ = nquad(integrand, bounds)
        return result
    
    #Incase of zero-one loss function the conditional risk is 1 - posterior of the correct class
    def conditional_risk(self, x, action):
        if self.min_error_rate_classification == True:
            return 1 - self.posterior(x, action)
        return sum(self.loss_matrix[action - 1, j] * self.posterior(x, j+1) for j in range(self.num_classes))

    #Discriminant function for min-error-rate classification
    def discriminant_function(self, x, c):
        if not 1 <= c <= self.num_classes:
            raise ValueError(f"Classes must be between 1 and {self.num_classes}")
        
        log_likelihood = np.log(self.likelihoods[c -1](x) + 1e-10)
        log_prior = np.log(self.priors[c-1] + 1e-10)
        
        return log_likelihood + log_prior
            
    # Function to classify an observation based on conditional risk. If loss_matrix is zero one loss then its the same as Bayes Decision Rule (Eq (8) from Duda et el)
    #Tie breaker - choose class 1
    def classify(self, x):
        
        if self.min_error_rate_classification:
            discriminant_values = [self.discriminant_function(x, c) for c in range(1, self.num_classes + 1)]
            predicted_class = np.argmax(discriminant_values) + 1
            sorted_values = sorted(discriminant_values, reverse=True)
            confidence = sorted_values[0] - sorted_values[1]
        else:
            risks = [self.conditional_risk(x, c) for c in range(1, self.num_classes + 1)]    
            predicted_class =  np.argmin(risks) + 1     
            sorted_values = sorted(risks)
            confidence = sorted_values[1] - sorted_values[0]
        
        posteriors = [self.posterior(x, c) for c in range(1, self.num_classes + 1)]
        
        if self.num_classes == 1:
                confidence = 1.0
        else:
            if sorted_values[0] == sorted_values[1]:
                confidence = 0.0
        error = self.p_error_x(x)
        
        return predicted_class, confidence, error
    
    
    # Function to calculate the minimum overall risk
    def bayes_risk(self):
        def integrand(*x):
            x_point = np.array(x)
            min_risk = min(self.conditional_risk(x_point,a) for a in range(1, self.num_classes + 1))
            return min_risk * self.evidence(x_point)
        
        bounds = [(-np.inf, np.inf)] * self.input_dims
        result, _ = quad(integrand, bounds)
        return result
    
    def evaluate_classifier(self, x_test, y_true):
        
        if len(x_test) != len(y_true):
            raise ValueError("X_test and y_true must have the same length.")
        
        predictions = [self.classify(x)[0] for x in x_test]
        
        accuracy = np.mean(np.array(predictions) == y_true)
        
        return accuracy
    
    def plot_likelihoods(self, range_):
        if self.input_dims == 1:
            x = np.linspace(range_[0], range_[1], 1000)
            plt.figure(figsize=(18, 8))
            colors = plt.cm.tab10(np.linspace(0, 1, self.num_classes))
            for i, (likelihood, color) in enumerate(zip(self.likelihoods, colors)):
                likelihood_values = [likelihood(np.array([xi])) for xi in x]
                plt.plot(x, likelihood_values, color=color, label=f"Class {i + 1}")
            plt.xlabel('x')
            plt.ylabel('Probability Density')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('likelihoods_1d.png', bbox_inches='tight')
            plt.show()
            plt.close()

        elif self.input_dims == 2:
            x = np.linspace(range_[0][0], range_[0][1], 100)
            y = np.linspace(range_[1][0], range_[1][1], 100)
            X, Y = np.meshgrid(x, y)
            fig = plt.figure(figsize=(18, 8))
            colors = plt.cm.tab10(np.linspace(0, 1, self.num_classes))

            for i, (likelihood, color) in enumerate(zip(self.likelihoods, colors)):
                Z = np.array([[likelihood(np.array([x_val, y_val])) for x_val, y_val in zip(x_row, y_row)]
                            for x_row, y_row in zip(X, Y)])
                ax = fig.add_subplot(1, self.num_classes, i+1, projection='3d')
                ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
                ax.set_title(f'Likelihood for Class {i + 1}')
                ax.set_xlabel('x1')
                ax.set_ylabel('x2')
                ax.set_zlabel('P(x|ω)')
            
            plt.savefig('likelihoods_2d.png', bbox_inches='tight')
            plt.show()
            plt.close()

    def plot_posteriors(self, range_):
        if self.input_dims == 1:
            x = np.linspace(range_[0], range_[1], 1000)
            plt.figure(figsize=(18, 8))
            colors = plt.cm.tab10(np.linspace(0, 1, self.num_classes))
            for i, color in enumerate(colors):
                posterior_values = [self.posterior(np.array([xi]), i + 1) for xi in x]
                plt.plot(x, posterior_values, color=color, label=f"Class {i + 1}")
            plt.xlabel('x')
            plt.ylabel('Posterior Probability')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('posteriors_1d.png', bbox_inches='tight')
            plt.show()
            plt.close()

        elif self.input_dims == 2:
            x = np.linspace(range_[0][0], range_[0][1], 100)
            y = np.linspace(range_[1][0], range_[1][1], 100)
            X, Y = np.meshgrid(x, y)
            fig = plt.figure(figsize=(18, 8))
            colors = plt.cm.tab10(np.linspace(0, 1, self.num_classes))

            for i, color in enumerate(colors):
                Z = np.array([[self.posterior(np.array([x_val, y_val]), i + 1) for x_val, y_val in zip(x_row, y_row)]
                            for x_row, y_row in zip(X, Y)])
                ax = fig.add_subplot(1, self.num_classes, i+1, projection='3d')
                ax.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='none')
                ax.set_title(f'Posterior for Class {i + 1}')
                ax.set_xlabel('x1')
                ax.set_ylabel('x2')
                ax.set_zlabel('P(ω|x)')

            plt.savefig('posteriors_2d.png', bbox_inches='tight')
            plt.show()
            plt.close()

    def plot_decision_boundaries(self, range_, resolution=500):
        if self.input_dims == 1:
            x = np.linspace(range_[0], range_[1], resolution)
            decisions = np.array([self.classify(np.array([xi]))[0] for xi in x])
            
            plt.figure(figsize=(18, 8))
            
            for c in range(1, self.num_classes + 1):
                mask = (decisions == c)
                plt.fill_between(x, 0, 1, where=mask, 
                                color=plt.cm.tab10(c-1), alpha=0.2,
                                label=f'Class {c} Region')
            
            colors = plt.cm.tab10(np.linspace(0, 1, self.num_classes))
            for i, (likelihood, color) in enumerate(zip(self.likelihoods, colors)):
                likelihood_values = [likelihood(np.array([xi])) for xi in x]
                plt.plot(x, likelihood_values, color=color, 
                        label=f"Class {i+1} Likelihood")
            
            boundaries = []
            for i in range(len(x)-1):
                if decisions[i] != decisions[i+1]:
                    boundary = (x[i] + x[i+1])/2
                    boundaries.append(boundary)
                    plt.axvline(x=boundary, color='k', linestyle='--', alpha=0.5)
            
            plt.xlabel('x')
            plt.ylabel('Probability Density')
            plt.title('Decision Boundaries')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, None)
            plt.savefig('decision_boundaries_1d.png', bbox_inches='tight')
            plt.show()
            plt.close()
            
        elif self.input_dims == 2:
            x1 = np.linspace(range_[0][0], range_[0][1], resolution)
            x2 = np.linspace(range_[1][0], range_[1][1], resolution)
            X1, X2 = np.meshgrid(x1, x2)
            
            decisions = np.zeros_like(X1)
            for i in range(X1.shape[0]):
                for j in range(X1.shape[1]):
                    point = np.array([X1[i,j], X2[i,j]])
                    decisions[i,j] = self.classify(point)[0]
            
            plt.figure(figsize=(10, 8))
            
            cmap = plt.cm.get_cmap('tab10', self.num_classes)
            
            plt.contourf(X1, X2, decisions, levels=np.arange(0.5, self.num_classes+1.5), 
                        cmap=cmap, alpha=0.3)
            
            plt.contour(X1, X2, decisions, levels=np.arange(0.5, self.num_classes+1.5),
                        colors='k', linewidths=0.5, linestyles='solid')
            
            patches = [plt.Rectangle((0,0),1,1, fc=cmap(i)) 
                    for i in range(self.num_classes)]
            plt.legend(patches, [f'Class {i+1}' for i in range(self.num_classes)],
                    loc='upper right')
            
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.title('Decision Boundaries')
            plt.grid(True, alpha=0.3)
            plt.savefig('decision_boundaries_2d.png', bbox_inches='tight')
            plt.show()
            plt.close()
            
        else:
            raise ValueError("Decision boundary plotting only supported for 1D or 2D data")
