"""Security and compliance features for production ML.

Includes model encryption, differential privacy, audit logging,
and GDPR compliance tools.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "ModelEncryption",
    "AuditLogger",
    "PrivacyGuard",
    "ComplianceChecker",
]

logger = logging.getLogger(__name__)


@dataclass
class AuditLogEntry:
    """Single audit log entry."""
    
    timestamp: datetime
    user: str
    action: str
    resource: str
    details: dict[str, Any] = field(default_factory=dict)
    ip_address: str | None = None
    success: bool = True
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


class AuditLogger:
    """Comprehensive audit logging for ML operations.
    
    Tracks all model training, predictions, and data access
    for compliance and security auditing.
    """
    
    def __init__(self, log_dir: Path | None = None) -> None:
        """Initialize audit logger.
        
        Args:
            log_dir: Directory to store audit logs
        """
        self.log_dir = log_dir or Path("./audit_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_log_file = self.log_dir / f"audit_{datetime.now():%Y%m%d}.jsonl"
    
    def log(
        self,
        user: str,
        action: str,
        resource: str,
        details: dict[str, Any] | None = None,
        ip_address: str | None = None,
        success: bool = True,
    ) -> None:
        """Log an audit entry.
        
        Args:
            user: User performing action
            action: Action type (train, predict, access, etc.)
            resource: Resource being accessed
            details: Additional details
            ip_address: IP address of user
            success: Whether action succeeded
        """
        entry = AuditLogEntry(
            timestamp=datetime.now(),
            user=user,
            action=action,
            resource=resource,
            details=details or {},
            ip_address=ip_address,
            success=success,
        )
        
        # Append to log file
        with open(self.current_log_file, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")
        
        logger.debug(f"Audit: {user} {action} {resource}")
    
    def log_model_training(
        self,
        user: str,
        model_id: str,
        dataset_info: dict[str, Any],
        success: bool = True,
    ) -> None:
        """Log model training event."""
        self.log(
            user=user,
            action="model_training",
            resource=model_id,
            details={
                "dataset": dataset_info,
                "timestamp": datetime.now().isoformat(),
            },
            success=success,
        )
    
    def log_prediction(
        self,
        user: str,
        model_id: str,
        n_samples: int,
        ip_address: str | None = None,
    ) -> None:
        """Log prediction request."""
        self.log(
            user=user,
            action="prediction",
            resource=model_id,
            details={"n_samples": n_samples},
            ip_address=ip_address,
            success=True,
        )
    
    def log_data_access(
        self,
        user: str,
        dataset_id: str,
        access_type: str = "read",
    ) -> None:
        """Log data access."""
        self.log(
            user=user,
            action=f"data_{access_type}",
            resource=dataset_id,
        )
    
    def get_logs(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        user: str | None = None,
        action: str | None = None,
    ) -> list[AuditLogEntry]:
        """Query audit logs.
        
        Args:
            start_date: Filter by start date
            end_date: Filter by end date
            user: Filter by user
            action: Filter by action type
        
        Returns:
            List of matching audit entries
        """
        entries = []
        
        # Read all log files in date range
        for log_file in sorted(self.log_dir.glob("audit_*.jsonl")):
            with open(log_file) as f:
                for line in f:
                    data = json.loads(line)
                    data["timestamp"] = datetime.fromisoformat(data["timestamp"])
                    entry = AuditLogEntry(**data)
                    
                    # Apply filters
                    if start_date and entry.timestamp < start_date:
                        continue
                    if end_date and entry.timestamp > end_date:
                        continue
                    if user and entry.user != user:
                        continue
                    if action and entry.action != action:
                        continue
                    
                    entries.append(entry)
        
        return entries


class ModelEncryption:
    """Encrypt/decrypt models for secure storage.
    
    Protects model intellectual property and prevents
    unauthorized model access.
    """
    
    def __init__(self, encryption_key: bytes | None = None) -> None:
        """Initialize model encryption.
        
        Args:
            encryption_key: 32-byte encryption key (None = generate)
        """
        try:
            from cryptography.fernet import Fernet
            self.Fernet = Fernet
        except ImportError:
            raise ImportError("cryptography required: pip install cryptography")
        
        if encryption_key is None:
            encryption_key = Fernet.generate_key()
        
        self.cipher = Fernet(encryption_key)
        self.encryption_key = encryption_key
    
    def encrypt_model(self, model_bytes: bytes) -> bytes:
        """Encrypt model bytes.
        
        Args:
            model_bytes: Serialized model
        
        Returns:
            Encrypted model bytes
        """
        return self.cipher.encrypt(model_bytes)
    
    def decrypt_model(self, encrypted_bytes: bytes) -> bytes:
        """Decrypt model bytes.
        
        Args:
            encrypted_bytes: Encrypted model
        
        Returns:
            Decrypted model bytes
        """
        return self.cipher.decrypt(encrypted_bytes)
    
    def save_encrypted_model(
        self,
        model: Any,
        path: Path,
        serializer: str = "joblib",
    ) -> None:
        """Save model with encryption.
        
        Args:
            model: Model to save
            path: Output path
            serializer: "joblib" or "pickle"
        """
        import pickle
        
        # Serialize model
        if serializer == "joblib":
            import joblib
            model_bytes = joblib.dumps(model)
        else:
            model_bytes = pickle.dumps(model)
        
        # Encrypt
        encrypted = self.encrypt_model(model_bytes)
        
        # Save
        path.write_bytes(encrypted)
        logger.info(f"Saved encrypted model to {path}")
    
    def load_encrypted_model(
        self,
        path: Path,
        serializer: str = "joblib",
    ) -> Any:
        """Load encrypted model.
        
        Args:
            path: Model path
            serializer: "joblib" or "pickle"
        
        Returns:
            Loaded model
        """
        import pickle
        
        # Load encrypted bytes
        encrypted = path.read_bytes()
        
        # Decrypt
        model_bytes = self.decrypt_model(encrypted)
        
        # Deserialize
        if serializer == "joblib":
            import joblib
            model = joblib.loads(model_bytes)
        else:
            model = pickle.loads(model_bytes)
        
        logger.info(f"Loaded encrypted model from {path}")
        return model


class PrivacyGuard:
    """Differential privacy for model training.
    
    Adds noise to protect individual data points while
    preserving overall model utility.
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5) -> None:
        """Initialize privacy guard.
        
        Args:
            epsilon: Privacy budget (smaller = more privacy)
            delta: Probability of privacy breach
        """
        self.epsilon = epsilon
        self.delta = delta
    
    def add_noise(
        self,
        data: np.ndarray,
        sensitivity: float = 1.0,
    ) -> np.ndarray:
        """Add Laplacian noise for differential privacy.
        
        Args:
            data: Data to privatize
            sensitivity: Global sensitivity
        
        Returns:
            Noised data
        """
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale, size=data.shape)
        return data + noise
    
    def privatize_gradients(
        self,
        gradients: np.ndarray,
        clip_norm: float = 1.0,
    ) -> np.ndarray:
        """Privatize gradients for DP-SGD.
        
        Args:
            gradients: Model gradients
            clip_norm: Gradient clipping norm
        
        Returns:
            Privatized gradients
        """
        # Clip gradients
        norm = np.linalg.norm(gradients)
        if norm > clip_norm:
            gradients = gradients * (clip_norm / norm)
        
        # Add noise
        return self.add_noise(gradients, sensitivity=clip_norm)


class ComplianceChecker:
    """Check ML system compliance with regulations.
    
    Supports GDPR, CCPA, and other data protection regulations.
    """
    
    def __init__(self) -> None:
        """Initialize compliance checker."""
        self.checks_passed: dict[str, bool] = {}
    
    def check_gdpr_compliance(
        self,
        has_consent: bool,
        has_right_to_erasure: bool,
        has_data_portability: bool,
        has_explainability: bool,
    ) -> bool:
        """Check GDPR compliance.
        
        Args:
            has_consent: User consent for data processing
            has_right_to_erasure: Right to be forgotten implemented
            has_data_portability: Data export capability
            has_explainability: Model explainability available
        
        Returns:
            True if compliant
        """
        checks = {
            "consent": has_consent,
            "right_to_erasure": has_right_to_erasure,
            "data_portability": has_data_portability,
            "explainability": has_explainability,
        }
        
        self.checks_passed.update(checks)
        
        compliant = all(checks.values())
        
        if compliant:
            logger.info("✅ GDPR compliance: PASSED")
        else:
            failed = [k for k, v in checks.items() if not v]
            logger.warning(f"❌ GDPR compliance: FAILED - {failed}")
        
        return compliant
    
    def check_model_fairness(
        self,
        predictions: np.ndarray,
        sensitive_attributes: np.ndarray,
        threshold: float = 0.1,
    ) -> bool:
        """Check model fairness across groups.
        
        Args:
            predictions: Model predictions
            sensitive_attributes: Protected attributes
            threshold: Max allowed disparity
        
        Returns:
            True if fair
        """
        # Calculate accuracy per group
        unique_groups = np.unique(sensitive_attributes)
        group_accuracies = []
        
        for group in unique_groups:
            mask = sensitive_attributes == group
            group_acc = predictions[mask].mean()
            group_accuracies.append(group_acc)
        
        # Check disparity
        disparity = max(group_accuracies) - min(group_accuracies)
        
        fair = disparity <= threshold
        
        if fair:
            logger.info(f"✅ Fairness check: PASSED (disparity: {disparity:.3f})")
        else:
            logger.warning(f"❌ Fairness check: FAILED (disparity: {disparity:.3f})")
        
        return fair
    
    def generate_compliance_report(self) -> str:
        """Generate compliance report.
        
        Returns:
            Compliance report as string
        """
        lines = [
            "=" * 80,
            "COMPLIANCE REPORT",
            "=" * 80,
            f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
            "",
            "Checks:",
        ]
        
        for check, passed in self.checks_passed.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            lines.append(f"  {check}: {status}")
        
        overall = "COMPLIANT" if all(self.checks_passed.values()) else "NON-COMPLIANT"
        lines.extend([
            "",
            f"Overall Status: {overall}",
            "=" * 80,
        ])
        
        return "\n".join(lines)
