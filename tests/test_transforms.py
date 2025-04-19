import numpy as np
from src.data.transforms import softmax_with_temperature


def test_softmax_with_temperature():
    """Test softmax with temperature scaling."""
    # Test logits
    logits = np.array([[1.0, 2.0], [3.0, 1.0]])
    
    # Temperature = 1 (standard softmax)
    probs_t1 = softmax_with_temperature(logits, temperature=1.0)
    expected_t1 = np.array([
        [1.0 / (1.0 + np.exp(1.0)), np.exp(1.0) / (1.0 + np.exp(1.0))],
        [np.exp(2.0) / (np.exp(2.0) + 1.0), 1.0 / (np.exp(2.0) + 1.0)]
    ])
    np.testing.assert_allclose(probs_t1, expected_t1, rtol=1e-5)
    
    # Check row sums to 1
    assert np.allclose(np.sum(probs_t1, axis=1), 1.0)
    
    # Higher temperature (T=2) should make distribution softer
    probs_t2 = softmax_with_temperature(logits, temperature=2.0)
    
    # Softer distribution means closer to uniform
    diffs_t1 = np.abs(probs_t1[:, 0] - probs_t1[:, 1])
    diffs_t2 = np.abs(probs_t2[:, 0] - probs_t2[:, 1])
    assert np.all(diffs_t2 < diffs_t1)
    
    # Check row sums to 1
    assert np.allclose(np.sum(probs_t2, axis=1), 1.0) 