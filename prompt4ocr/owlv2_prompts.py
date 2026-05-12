"""Text queries for OWLv2 zero-shot detection on GIẤY GỬI TIỀN TIẾT KIỆM–style bank slips.

OWLv2 is CLIP-style: short English "a photo of …" phrases usually generalize better than
raw field names. We still add a few Vietnamese phrases for printed headings that appear
literally on the form.
"""

from __future__ import annotations

# OWLv2's CLIP text encoder caps each query at 16 BPE tokens. Keep phrases short.
# Single-image batch: one inner list with all queries (see HF Owlv2Processor docs).
PROFILE_FULL: list[str] = [
    # Header / branding
    "a bank logo",
    "a printed form title",
    "a printed date field",
    # Customer block
    "a section heading on a form",
    "printed customer name",
    "a national ID number",
    "an issue date",
    "a CIF customer number",
    "a deposit amount in digits",
    "amount written in words",
    "currency code VND",
    "a savings product code",
    "a term length",
    "payment method text",
    "a bank account number",
    # Cash table
    "a cash denomination table",
    "a money table with columns",
    # Declarations / signatures
    "a confirmation paragraph",
    "a handwritten signature",
    "a signature line",
    # Bank-only footer block
    "a bank stamp",
    "a journal entry number",
    "a serial number",
    "an account opening date",
    "an interest rate with percent",
    "a maturity date",
    "a debit and credit table",
    # Staff
    "a teller name or code",
    "a supervisor stamp",
]

# Fewer queries: faster, less duplicate boxes; good for smoke tests.
PROFILE_COMPACT: list[str] = [
    "a bank logo",
    "a printed form title",
    "a customer information block",
    "a deposit amount in digits",
    "a cash denomination table",
    "a handwritten signature",
    "a bank stamp",
    "a debit and credit table",
]


def text_labels_for_batch(queries: list[str]) -> list[list[str]]:
    """Format queries for a single-image batch."""
    return [list(queries)]


def queries_for_profile(name: str) -> list[str]:
    key = name.strip().lower()
    if key in ("compact", "small", "fast"):
        return list(PROFILE_COMPACT)
    if key in ("full", "default", ""):
        return list(PROFILE_FULL)
    raise ValueError(f"Unknown prompt profile: {name!r} (use 'full' or 'compact')")
