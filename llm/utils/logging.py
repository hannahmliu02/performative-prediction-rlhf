from __future__ import annotations

import structlog

_configured = False


def get_logger(name: str) -> structlog.BoundLogger:
    global _configured
    if not _configured:
        structlog.configure(
            processors=[
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.dev.ConsoleRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
        )
        _configured = True
    return structlog.get_logger(name)
