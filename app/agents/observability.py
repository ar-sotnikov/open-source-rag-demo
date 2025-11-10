"""Phoenix Observability Integration"""

import os
from contextlib import contextmanager

from phoenix.otel import register

collector_endpoint = os.getenv("COLLECTOR_ENDPOINT", "phoenix:4317")
tracer_provider = register(
    project_name="agentic-rag",
    endpoint=collector_endpoint,
    auto_instrument=True,
)
tracer = tracer_provider.get_tracer(__name__)


@contextmanager
def trace_span(name: str, attributes: dict | None = None):
    """Context manager for creating spans"""
    with tracer.start_as_current_span(name) as span:
        if attributes and span:
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(key, str(value))
        yield span
