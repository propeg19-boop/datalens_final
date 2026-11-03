"""
data/sample/generate_reference_data.py
Generates the reference coffee shop dataset and sample datasets according to PRD specifications.
"""
from pathlib import Path
import numpy as np
import pandas as pd


def generate_coffee_shop_sample(n_rows: int = 1000) -> pd.DataFrame:
    """
    Generate representative coffee shop sales dataset matching PRD Appendix A:
    Columns:
    transaction_id, transaction_date, transaction_time, transaction_qty,
    store_id, store_location, product_id, unit_price, product_category,
    product_type, product_detail.
    """
    np.random.seed(42)

    stores = ["Store-1", "Store-2", "Store-3"]
    locations = {
        "Store-1": "Lower Manhattan",
        "Store-2": "Astoria",
        "Store-3": "Hell's Kitchen",
    }
    categories = ["Coffee", "Tea", "Drinking Chocolate", "Bakery", "Flavours"]
    product_types = ["Gourmet brewed coffee", "Barista Espresso", "Hot Chocolate", "Scone", "Biscotti"]

    store_choices = np.random.choice(stores, size=n_rows)
    store_locs = [locations[s] for s in store_choices]

    dates = pd.date_range("2025-01-01", "2025-06-30", freq="min")
    random_dates = np.random.choice(dates, size=n_rows)

    times = [d.strftime("%H:%M:%S") for d in pd.to_datetime(random_dates)]
    dates_str = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(random_dates)]

    # transaction_qty: 1 to 6
    qtys = np.random.choice([1, 2, 3, 4, 5, 6], size=n_rows, p=[0.70, 0.18, 0.06, 0.03, 0.02, 0.01]).astype(float)
    
    # product_id: 1 to 80
    product_ids = np.random.randint(1, 81, size=n_rows).astype(float)

    # unit_price: base prices between 2.0 and 5.0, with premium items creating realistic right skew
    base_prices = np.random.choice([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], size=n_rows)
    # Inject ~3% premium products ($15 - $28) to create the documented legitimate skew
    premium_mask = np.random.rand(n_rows) < 0.03
    base_prices[premium_mask] = np.random.choice([15.0, 18.5, 22.0, 28.0], size=premium_mask.sum())

    df = pd.DataFrame({
        "transaction_id": [f"TX-{100000 + i}" for i in range(n_rows)],
        "transaction_date": dates_str,
        "transaction_time": times,
        "transaction_qty": qtys,
        "store_id": store_choices,
        "store_location": store_locs,
        "product_id": product_ids,
        "unit_price": base_prices,
        "product_category": np.random.choice(categories, size=n_rows),
        "product_type": np.random.choice(product_types, size=n_rows),
        "product_detail": [f"Detail-{int(pid)}" for pid in product_ids],
    })

    # Inject tiny missing counts matching PRD Appendix A
    if n_rows >= 10:
        df.loc[0, "transaction_qty"] = np.nan
        df.loc[1:2, "product_id"] = np.nan
        df.loc[3:5, "unit_price"] = np.nan

    return df


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    coffee_df = generate_coffee_shop_sample(n_rows=2500)
    csv_path = out_dir / "coffeeshopsales_sample.csv"
    coffee_df.to_csv(csv_path, index=False)
    print(f"Generated sample coffee shop dataset at: {csv_path} (shape: {coffee_df.shape})")

    # Also save an Excel version to test Excel loading
    excel_path = out_dir / "coffeeshopsales_sample.xlsx"
    coffee_df.to_excel(excel_path, index=False)
    print(f"Generated Excel sample dataset at: {excel_path}")
