from pathlib import Path

import pandas as pd


def main() -> None:
    # Placeholder for real ingestion. Reads the latest daily transactions if available.
    input_path = Path("daily_transactions.csv")
    if not input_path.exists():
        return

    df = pd.read_csv(input_path)
    output_path = Path("daily_transactions_normalized.csv")
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
