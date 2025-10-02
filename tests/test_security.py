"""Tests for security and compliance module."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from automl.security import (
    ModelEncryption,
    AuditLogger,
    PrivacyGuard,
    ComplianceChecker,
)


def test_model_encryption() -> None:
    """Test model encryption/decryption."""
    pytest.importorskip("cryptography")
    
    # Create and train a model
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Encrypt model
    encryption = ModelEncryption()
    
    with TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "encrypted_model.pkl"
        
        # Save encrypted
        encryption.save_encrypted_model(model, model_path)
        assert model_path.exists()
        
        # Load encrypted
        loaded_model = encryption.load_encrypted_model(model_path)
        
        # Should make same predictions
        original_pred = model.predict(X)
        loaded_pred = loaded_model.predict(X)
        np.testing.assert_array_equal(original_pred, loaded_pred)


def test_audit_logger() -> None:
    """Test audit logging."""
    with TemporaryDirectory() as tmpdir:
        logger = AuditLogger(log_dir=Path(tmpdir))
        
        # Log some events
        logger.log_model_training(
            user="alice",
            model_id="model-123",
            dataset_info={"name": "test", "samples": 1000},
        )
        
        logger.log_prediction(
            user="bob",
            model_id="model-123",
            n_samples=10,
        )
        
        logger.log_data_access(
            user="charlie",
            dataset_id="data-456",
            access_type="read",
        )
        
        # Query logs
        all_logs = logger.get_logs()
        assert len(all_logs) == 3
        
        # Filter by user
        alice_logs = logger.get_logs(user="alice")
        assert len(alice_logs) == 1
        assert alice_logs[0].user == "alice"


def test_privacy_guard() -> None:
    """Test differential privacy."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    guard = PrivacyGuard(epsilon=1.0)
    
    # Add noise
    noised = guard.add_noise(data)
    
    # Should be different but similar
    assert not np.array_equal(data, noised)
    assert np.abs(data - noised).mean() < 2.0  # Reasonable noise


def test_gradient_privacy() -> None:
    """Test gradient privatization."""
    gradients = np.random.randn(10)
    
    guard = PrivacyGuard(epsilon=1.0)
    private_grads = guard.privatize_gradients(gradients, clip_norm=1.0)
    
    # Should be clipped and noised
    assert np.linalg.norm(private_grads) <= 2.0  # Clipped + noise


def test_compliance_checker() -> None:
    """Test GDPR compliance checking."""
    checker = ComplianceChecker()
    
    # Check compliant system
    compliant = checker.check_gdpr_compliance(
        has_consent=True,
        has_right_to_erasure=True,
        has_data_portability=True,
        has_explainability=True,
    )
    assert compliant
    
    # Check non-compliant system
    non_compliant = checker.check_gdpr_compliance(
        has_consent=True,
        has_right_to_erasure=False,  # Missing
        has_data_portability=True,
        has_explainability=False,  # Missing
    )
    assert not non_compliant


def test_fairness_check() -> None:
    """Test model fairness checking."""
    # Create predictions with disparity
    predictions = np.array([0, 1, 1, 1, 0, 0, 1, 1])
    sensitive = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    checker = ComplianceChecker()
    
    # With high threshold, should pass
    fair = checker.check_model_fairness(
        predictions,
        sensitive,
        threshold=0.5,
    )
    assert fair


def test_compliance_report() -> None:
    """Test compliance report generation."""
    checker = ComplianceChecker()
    
    # Run some checks
    checker.check_gdpr_compliance(
        has_consent=True,
        has_right_to_erasure=True,
        has_data_portability=False,
        has_explainability=True,
    )
    
    # Generate report
    report = checker.generate_compliance_report()
    
    assert "COMPLIANCE REPORT" in report
    assert "consent" in report
    assert "NON-COMPLIANT" in report  # Due to missing data_portability
