# Security and Limitations

The local multi-model registry is a development application for a trusted operator. Its administration interface and workspace inference API do not authenticate individual users. Keep the service bound to localhost and never expose the registry administration feature to the open internet.

The `.skops` file format and type checks reduce some model-serialization risks but do not guarantee that a model is safe to execute. Production user-upload workflows require isolated workers, authentication, upload and inference quotas, resource limits, audit logging, and durable managed storage.

The example datasets are synthetic and not intended for consequential decisions. The local model registry is independent of the reference pipeline's MLflow configuration and may not be shared with a public Render or Streamlit deployment.
