# Security Policy

## Supported Versions

Currently supported versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please follow these steps:

### 1. **Do Not** Open a Public Issue

Please do not open a public GitHub issue for security vulnerabilities.

### 2. Report Privately

Send an email to: **[jainampatel1673@gmail.com]**

Include the following information:
- Type of vulnerability
- Full description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### 3. Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Varies by severity
  - Critical: Within 7 days
  - High: Within 14 days
  - Medium: Within 30 days
  - Low: Within 90 days

### 4. Disclosure Policy

- We will acknowledge your report within 48 hours
- We will provide regular updates on our progress
- We will notify you when the vulnerability is fixed
- We will credit you in the security advisory (unless you prefer to remain anonymous)

## Security Best Practices

When using AutoML:

1. **Input Validation**: Always validate input data before processing
2. **Model Serialization**: Only load models from trusted sources
3. **Dependency Updates**: Keep dependencies up to date
4. **Environment Isolation**: Use virtual environments or containers
5. **Secrets Management**: Never commit API keys or credentials
6. **Access Control**: Restrict access to model artifacts and logs

## Known Security Considerations

### Model Deserialization

When loading models with `joblib` or `pickle`, be aware that malicious models can execute arbitrary code. Only load models from trusted sources.

```python
# Safe: Load from your own saved model
from automl.utils.serialization import load_model
model = load_model("models/my_model.joblib")

# Unsafe: Loading untrusted models
# DO NOT do this with untrusted sources
model = joblib.load("untrusted_model.joblib")  # ⚠️ Dangerous!
```

### Configuration Files

YAML configuration files are parsed with `pyyaml`. Ensure configuration files come from trusted sources only.

### GPU Usage

When using GPU acceleration, ensure proper resource limits are set to prevent DoS attacks through resource exhaustion.

## Dependencies

We monitor dependencies for known vulnerabilities using:
- GitHub Dependabot
- Safety checks in CI/CD
- Regular dependency updates

To check dependencies manually:
```bash
uv run pip-audit
```

## Contact

For security concerns: **jainampatel1673@gmail.com**

For general issues: Open a GitHub issue

---

Thank you for helping keep AutoML secure! 🔒
