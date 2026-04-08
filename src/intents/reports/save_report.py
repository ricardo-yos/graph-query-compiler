"""
Report Persistence Utility
==========================

Provides a helper function to save diagnostic reports as JSON files.

Ensures:
- directory creation if it does not exist
- readable JSON formatting
- UTF-8 encoding support
"""

import json
from config.paths import REPORTS_DIR


def save_report(report, filename):
    """
    Saves a report dictionary to a JSON file.

    Parameters
    ----------
    report : dict
        Report content to be saved.
    filename : str
        Name of the output JSON file.
    """

    # Full output path
    output_path = REPORTS_DIR / filename

    # Create directory if it does not exist
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Write JSON file with readable formatting
    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2,          # improves readability
            ensure_ascii=False # preserves accents
        )

    # Confirmation message
    print(f"Report saved to: {output_path}")
