import numpy as np


class Strategy:
    def __init__(self, classifier_t, classifier_u, treatment_ix):
        self.classifier_t = classifier_t
        self.classifier_u = classifier_u
        self.treatment_ix = treatment_ix

    # Scores that rank which observations should be preferred to treat
    def get_scores(self, observations):
        prob_scores = np.zeros(len(observations))
        return prob_scores

    @staticmethod
    def create_classifier(classifier_type, data, labels, params):
        arg_vals = {'random_state': 0, 'probability': True, 'n_neighbors': 5}
        for key in params:
            arg_vals[key] = params[key]
        init_args = classifier_type.__init__.__code__.co_varnames
        args = dict()
        for arg_name in arg_vals:
            if arg_name in init_args:
                args[arg_name] = arg_vals[arg_name]
        classifier = classifier_type(**args)
        classifier = classifier.fit(data, labels)
        return classifier

    @staticmethod
    def create_classifiers(classifier_type, data, labels, treatment_ix, params = None):
        t_ixs = data[:, treatment_ix] == 1
        if params is None:
            params = dict()
        classifier_t = Strategy.create_classifier(classifier_type, data[t_ixs, :], labels[t_ixs], params)
        classifier_u = Strategy.create_classifier(classifier_type, data[~t_ixs, :], labels[~t_ixs], params)
        return classifier_t, classifier_u


class TargetMaxTreatment(Strategy):
    def get_scores(self, observations):
        observations[:, self.treatment_ix] = 1
        p_treated = self.classifier_t.predict_proba(observations)[:, 1]
        observations[:, self.treatment_ix] = 0
        p_untreated = self.classifier_u.predict_proba(observations)[:, 1]
        return p_treated - p_untreated


class TargetMostLikely(Strategy):
    def get_scores(self, observations):
        observations[:, self.treatment_ix] = 1
        p_treated = self.classifier_t.predict_proba(observations)[:, 1]
        return p_treated


class TargetLeastLikely(Strategy):
    def get_scores(self, observations):
        observations[:, self.treatment_ix] = 0
        p_untreated = self.classifier_u.predict_proba(observations)[:, 1]
        return 1 - p_untreated


class PerfectClassifier:
    def __init__(self, sim):
        self.sim = sim

    def get_val(self, observations, ix):
        sim = self.sim
        sample_ixs = np.argsort(sim.observed)
        sample = observations[:, sample_ixs]
        return np.array([sim.eval_node(ix, obs) for obs in sample])

    def get_utility(self, observations):
        return self.get_val(observations, self.sim.utility_ix) - self.sim.min_u

    def get_effect(self, observations):
        return self.get_val(observations, self.sim.effect_ix) + self.sim.fixed_effect

    def predict_proba(self, observations):
        sim = self.sim
        utility = self.get_utility(observations)
        effect = self.get_effect(observations)
        treated = observations[:, sim.d_treatment_ix]
        prob1 = utility + effect * treated > 0
        return np.vstack((1 - prob1, prob1)).T