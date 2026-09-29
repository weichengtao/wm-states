"""Objective-invariance and holdout checks for calibration scale diagnosis."""
import unittest

import numpy as np
from sklearn.utils.class_weight import compute_sample_weight

from scripts.next import validate_calibration_scale as validation
from scripts.next.decoder_models import make_logistic_calibration_cv_splits


class CalibrationScaleValidationTest(unittest.TestCase):
    def test_training_normalization_has_weighted_zero_mean_unit_variance(self):
        margins = np.array([-2., -.4, .2, 1., 3.])
        weights = np.array([.5, .5, 1., 2., 3.])
        transformed, center, scale = validation.normalize_training_margins(margins, weights)
        self.assertAlmostEqual(np.average(transformed, weights=weights), 0)
        self.assertAlmostEqual(np.average(transformed**2, weights=weights), 1)
        np.testing.assert_allclose(transformed * scale + center, margins)

    def test_affine_margin_normalization_preserves_platt_objective(self):
        labels = np.array([0, 0, 0, 1, 1])
        margins = np.array([-2., -.4, .2, 1., 3.])
        weights = compute_sample_weight('balanced', labels)
        normalized, center, scale = validation.normalize_training_margins(margins, weights)
        original = validation.sigmoid_objective(margins, labels, weights, -.7, .2)
        transformed = validation.sigmoid_objective(normalized, labels, weights, -.7 * scale, .2 - .7 * center)
        self.assertAlmostEqual(original['loss'], transformed['loss'])
        np.testing.assert_allclose(validation.expit(-(-.7 * margins + .2)),
                                   validation.expit(-(-.7 * scale * normalized + .2 - .7 * center)))

    def test_platt_objective_gradient_matches_finite_difference(self):
        y = np.array([0, 0, 0, 1, 1])
        margins = np.array([-2., -.4, .2, 1., 3.])
        weights = compute_sample_weight('balanced', y)
        a, b, eps = -.6, .15, 1e-6
        value = validation.sigmoid_objective(margins, y, weights, a, b)
        for key, direction in [('gradient_a', (eps, 0)), ('gradient_b', (0, eps))]:
            forward = validation.sigmoid_objective(margins, y, weights, a + direction[0], b + direction[1])['loss']
            backward = validation.sigmoid_objective(margins, y, weights, a - direction[0], b - direction[1])['loss']
            self.assertAlmostEqual(value[key], (forward - backward) / (2 * eps), places=7)

    def test_constant_margins_and_invalid_weights_are_rejected(self):
        for margins, weights in [(np.full(5, .1), np.ones(5)), (np.arange(5), np.zeros(5)),
                                 (np.array([0, np.nan]), np.ones(2))]:
            with self.assertRaises(ValueError):
                validation.normalize_training_margins(margins, weights)

    def test_heldout_activity_and_labels_cannot_change_fitted_calibration(self):
        rng = np.random.default_rng(21)
        y = np.tile([0, 1], 20)
        X = rng.normal(size=(40, 4)) + y[:, None] * .2
        train, test = np.arange(30), np.arange(30, 40)
        splits, _ = make_logistic_calibration_cv_splits(y[train], np.arange(30), 5, 8)
        first, arrays1 = validation.diagnose_one_c(X, y, train, test, splits, 8, .01)
        X[test] = rng.normal(size=(10, 4)) * 5
        y[test] = 1 - y[test]
        second, arrays2 = validation.diagnose_one_c(X, y, train, test, splits, 8, .01)
        for key in ('base_coefficient_l2_norm', 'production_sigmoid_a', 'production_sigmoid_b',
                    'normalized_sigmoid_a', 'normalized_sigmoid_b', 'training_normalization_center',
                    'training_normalization_scale', 'production_training_platt_objective',
                    'normalized_training_platt_objective'):
            self.assertEqual(first[key], second[key])
        np.testing.assert_array_equal(arrays1['training_oof_margin'], arrays2['training_oof_margin'])
        self.assertNotEqual(first['heldout_raw']['brier'], second['heldout_raw']['brier'])
        self.assertAlmostEqual(first['production_optimizer']['objective'], first['production_training_platt_objective']['loss'])
        self.assertAlmostEqual(first['normalized_optimizer']['objective'], first['normalized_training_platt_objective']['loss'])


if __name__ == '__main__':
    unittest.main()
