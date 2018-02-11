import simulator
from strategies import *
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import csv


class Experiment:
    def __init__(self):
        self.writer = None
        self.result = dict()
        self.i = 0

    # Get training and test data from the simulated sample.
    @staticmethod
    def get_data(sample, sim, test_s, train_s):
        # Total number of instances to use
        total_s = test_s + train_s
        # Get outcome labels for both treatment conditions
        labels1 = sim.apply_treatments(1)
        labels0 = sim.apply_treatments(0)
        # Potential outcome labels for the test set
        t_labels_t = labels1[:test_s]
        t_labels_u = labels0[:test_s]
        # Transform data from utility and treatment effect to outcome and treatment
        data1 = sample[:, sim.observed].copy()
        data0 = sample[:, sim.observed].copy()
        data1[:, sim.d_treatment_ix] = 1
        data0[:, sim.d_treatment_ix] = 0
        # Get training and test sets, as well as the outcome labels for the training set.
        train_labels = np.concatenate([labels0[test_s:total_s], labels1[test_s:total_s]], axis=0)
        train_data = np.concatenate([data0[test_s:total_s, :], data1[test_s:total_s, :]], axis=0)
        test_data = data0[:test_s, :]
        return train_data, train_labels, test_data, t_labels_t, t_labels_u

    # Evaluate the quality of the outcome classifier
    @staticmethod
    def eval_classifier(classifier, data, labels):
        predictions = classifier.predict_proba(data)[:, 1]
        fpr, tpr, thresholds = metrics.roc_curve(labels, predictions)
        auc = metrics.auc(fpr, tpr)
        return auc

    # Run a specific scenario for a list of classifiers and strategies
    def run_scenario(self, sim, classifiers, strategies, *data):
        t_ix = sim.d_treatment_ix
        for c_type, params in classifiers:
            c_name = c_type.__name__
            if params is not None:
                for key in params:
                    c_name += key + str(params[key])
            self.result["C_Type"] = c_name
            # Get training and test sets
            train_data, train_labels, test_data, test_l_t, test_l_u = data[0], data[1], data[2], data[3], data[4]
            # Train the classifiers
            if c_type == PerfectClassifier:
                model_t = PerfectClassifier(sim)
                model_u = PerfectClassifier(sim)
            else:
                model_t, model_u = Strategy.create_classifiers(c_type, train_data, train_labels, t_ix, params)
            # Evaluate how good are the classifiers at predicting the outcomes
            test_data[:, t_ix] = 1
            auc1 = Experiment.eval_classifier(model_t, test_data, test_l_t)
            test_data[:, t_ix] = 0
            auc0 = Experiment.eval_classifier(model_u, test_data, test_l_u)
            self.result["T_AUC"] = auc1
            self.result["U_AUC"] = auc0
            self.result["AVG_AUC"] = (auc0 + auc1)/2
            # Get Complier labels
            test_labels = test_l_t & ~test_l_u
            # Test other strategies
            for s_type in strategies:
                self.result["S_Type"] = s_type.__name__
                # Evaluate strategy
                strategy = s_type(model_t, model_u, t_ix)
                predictions = strategy.get_scores(test_data)
                fpr, tpr, thresholds = metrics.roc_curve(test_labels, predictions)
                auc = metrics.auc(fpr, tpr)
                self.result["AUC"] = auc
                self.write()

    # Run experiments for a list of scenarios
    def run_scenarios(self, rate_pairs, train_sizes, noise_rates, classifiers, strategies, test_size):
        for target_rate, avg_effect in rate_pairs:
            self.result["Rate"] = target_rate
            self.result["Effect"] = avg_effect
            # Create a new simulator with 10 variables (all of them observed)
            sim = simulator.Simulator(10, 10, target_rate, avg_effect)
            max_size = max(train_sizes) + test_size
            for noise_t, noise_u in noise_rates:
                self.result["Noise_T"] = noise_t
                self.result["Noise_U"] = noise_u
                sample = sim.load_sample("data/simulated_data.csv", max_size, noise_t, noise_u)
                np.random.shuffle(sample)
                for sample_size in train_sizes:
                    self.result["Size"] = sample_size
                    data = Experiment.get_data(sample, sim, test_size, sample_size)
                    self.run_scenario(sim, classifiers, strategies, *data)

    # Write result to the output file
    def write(self):
        self.i += 1
        print(self.i)
        self.writer.writerow(self.result.values())

    def test_analytical_results(self):
        # First element of every tuple is the base rate. Second element is the avg. treatment effect.
        rates = [(.08, .12), (.80, .12), (.33, .33), (.01, .01)]
        # First element of every tuple is noise for the treated. Second element is the noise for the untreated.
        noises = [(i * 5 / 100, i * 5 / 100) for i in range(21)]
        # Training size. It's zero for the perfect classifier
        training_size = [0]
        # This classifier uses the data generating model
        classifiers = [(PerfectClassifier, None)]
        # All three causal classification approaches
        strategies = [TargetLeastLikely, TargetMostLikely, TargetMaxTreatment]
        test_size = 10000
        self.run_scenarios(rates, training_size, noises, classifiers, strategies, test_size)

    def test_knn_example(self):
        rates = [(.05, .05)]
        noises = [(0, 0)]
        training_size = [1000]
        neighbors = [1, 2, 4, 20, 450, 700, 900]
        classifiers = [(KNeighborsClassifier, {'n_neighbors': n}) for n in neighbors]
        strategies = [TargetMaxTreatment]
        test_size = 10000
        self.run_scenarios(rates, training_size, noises, classifiers, strategies, test_size)

    def run_experiment(self, exp_id, f_name):
        np.random.seed(0)
        self.result = dict()
        self.i = 0
        with open(f_name, 'w', newline='') as csv_file:
            self.writer = csv.writer(csv_file)
            columns = ["Rate", "Effect", "Noise_T", "Noise_U", "Size", "C_Type",
                       "T_AUC", "U_AUC", "AVG_AUC", "S_Type", "AUC"]
            self.writer.writerow(columns)
            if exp_id == "ANALYTICAL":
                self.test_analytical_results()
            elif exp_id == "KNN":
                self.test_knn_example()
        with open(f_name, 'r') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                print('\t\t'.join(row))


ex = Experiment()
ex.run_experiment("KNN", 'data/knn_experiment.csv')
ex.run_experiment("ANALYTICAL", 'data/analytical_experiment.csv')

