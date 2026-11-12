# Google Cloud Managed MLflow Plugin (`google-cloud-managed-mlflow`)

> **Disclaimer**: This is not an officially supported Google product. This
> project is not eligible for the
> [Google Open Source Software Vulnerability Rewards Program](https://goo.gle/vulnz).

`google-cloud-managed-mlflow` is the open-source Python client plugin that
integrates standard [MLflow](https://mlflow.org/) clients with Google Cloud
Managed Service for MLflow (`mlflow.googleapis.com`) and Google Cloud Storage
(GCS).

## Overview

Installing `google-cloud-managed-mlflow` registers standard MLflow Python entry
points for:

- **Experiment Tracking**: Routing MLflow tracking API requests to regional
  Google Cloud Managed Service for MLflow endpoints.
- **Model Registry**: Managing registered models, versions, and aliases via
  Google Cloud Managed Service for MLflow.
- **Artifact Storage**: Reading and writing experiment and model artifacts in
  Google Cloud Storage using short-lived, downscoped credentials.
- **Authentication**: Authenticating requests with Google Cloud Application
  Default Credentials (ADC).

## Installation

Install the package with `pip`:

```bash
pip install google-cloud-managed-mlflow
```

## Contributing

Contributions to this project are welcome and encouraged. Please see
[CONTRIBUTING.md](https://github.com/GoogleCloudPlatform/google-cloud-managed-mlflow/blob/main/CONTRIBUTING.md) for instructions on signing the Google
Contributor License Agreement (<https://cla.developers.google.com/>), submitting
pull requests, and reviewing our
[Open Source Community Guidelines](https://opensource.google/conduct/).

## Security Policy

To report a security vulnerability, please follow the instructions in
[SECURITY.md](https://github.com/GoogleCloudPlatform/google-cloud-managed-mlflow/blob/main/SECURITY.md) and submit your report via
<https://goo.gle/vulnz> (<https://g.co/vulnz>).

## License & Terms of Service

This project is licensed under the Apache License, Version 2.0 — see
[LICENSE](https://github.com/GoogleCloudPlatform/google-cloud-managed-mlflow/blob/main/LICENSE) for full details.

Your use of Google Cloud services with this plugin is governed by the
[Google Cloud Terms of Service](https://cloud.google.com/terms).
