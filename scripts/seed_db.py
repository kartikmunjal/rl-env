"""
seed_db.py — Deterministic SQLite database seeder for sql-rl-env.

Creates data/ecommerce.db with a small but expressive e-commerce schema:
  - customers   (50 rows)
  - products    (20 rows)
  - orders      (200 rows)
  - order_items (600 rows, ~3 per order)

Usage:
    python scripts/seed_db.py [--db_path data/ecommerce.db] [--seed 42]

Author: Kartik Munjal
"""

import argparse
import json
import random
import sqlite3
from pathlib import Path
from datetime import date, timedelta


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id  INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    email        TEXT UNIQUE,
    city         TEXT,
    signup_date  DATE,
    tier         TEXT CHECK(tier IN ('bronze', 'silver', 'gold'))
);

CREATE TABLE IF NOT EXISTS products (
    product_id   INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    category     TEXT CHECK(category IN ('electronics', 'clothing', 'books', 'home')),
    price        REAL NOT NULL,
    stock        INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS orders (
    order_id     INTEGER PRIMARY KEY,
    customer_id  INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date   DATE NOT NULL,
    status       TEXT CHECK(status IN ('pending', 'shipped', 'delivered', 'cancelled'))
);

CREATE TABLE IF NOT EXISTS order_items (
    item_id      INTEGER PRIMARY KEY,
    order_id     INTEGER NOT NULL REFERENCES orders(order_id),
    product_id   INTEGER NOT NULL REFERENCES products(product_id),
    quantity     INTEGER NOT NULL DEFAULT 1,
    unit_price   REAL NOT NULL
);
"""

# ---------------------------------------------------------------------------
# Reference data (deterministic — do not use rng for these)
# ---------------------------------------------------------------------------
CITIES = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]
TIERS = ["bronze", "silver", "gold"]
TIER_WEIGHTS = [0.5, 0.3, 0.2]
CATEGORIES = ["electronics", "clothing", "books", "home"]
STATUSES = ["pending", "shipped", "delivered", "cancelled"]
STATUS_WEIGHTS = [0.15, 0.20, 0.55, 0.10]

FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael",
    "Linda", "William", "Barbara", "David", "Susan", "Richard", "Jessica",
    "Joseph", "Sarah", "Thomas", "Karen", "Charles", "Lisa",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
]

PRODUCT_TEMPLATES = {
    "electronics": [
        ("Wireless Headphones", 79.99), ("Bluetooth Speaker", 49.99),
        ("USB-C Hub", 34.99), ("Smart Watch", 199.99), ("Tablet Stand", 24.99),
    ],
    "clothing": [
        ("Cotton T-Shirt", 19.99), ("Denim Jacket", 89.99),
        ("Running Shoes", 64.99), ("Winter Scarf", 29.99), ("Sports Cap", 14.99),
    ],
    "books": [
        ("Python Programming", 39.99), ("Data Science Handbook", 44.99),
        ("Machine Learning Basics", 49.99), ("Clean Code", 34.99),
        ("Design Patterns", 42.99),
    ],
    "home": [
        ("Coffee Maker", 59.99), ("Air Purifier", 129.99),
        ("Desk Lamp", 27.99), ("Storage Organizer", 18.99),
        ("Bamboo Cutting Board", 22.99),
    ],
}


# ---------------------------------------------------------------------------
# Seed functions
# ---------------------------------------------------------------------------
def seed_customers(rng: random.Random, n: int = 50) -> list[dict]:
    rows = []
    used_emails: set[str] = set()
    base_date = date(2022, 1, 1)
    for i in range(1, n + 1):
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        name = f"{first} {last}"
        email_base = f"{first.lower()}.{last.lower()}"
        email = f"{email_base}@example.com"
        suffix = 1
        while email in used_emails:
            email = f"{email_base}{suffix}@example.com"
            suffix += 1
        used_emails.add(email)
        city = rng.choice(CITIES)
        days_offset = rng.randint(0, 730)
        signup = base_date + timedelta(days=days_offset)
        tier = rng.choices(TIERS, weights=TIER_WEIGHTS)[0]
        rows.append({
            "customer_id": i,
            "name": name,
            "email": email,
            "city": city,
            "signup_date": signup.isoformat(),
            "tier": tier,
        })
    return rows


def seed_products() -> list[dict]:
    rows = []
    pid = 1
    for cat, items in PRODUCT_TEMPLATES.items():
        for name, base_price in items:
            rows.append({
                "product_id": pid,
                "name": name,
                "category": cat,
                "price": base_price,
                "stock": 0,  # will be filled below
            })
            pid += 1
    return rows


def seed_orders(rng: random.Random, n_customers: int, n_orders: int = 200) -> list[dict]:
    rows = []
    base_date = date(2023, 1, 1)
    for i in range(1, n_orders + 1):
        cid = rng.randint(1, n_customers)
        days_offset = rng.randint(0, 729)
        order_date = base_date + timedelta(days=days_offset)
        status = rng.choices(STATUSES, weights=STATUS_WEIGHTS)[0]
        rows.append({
            "order_id": i,
            "customer_id": cid,
            "order_date": order_date.isoformat(),
            "status": status,
        })
    return rows


def seed_order_items(
    rng: random.Random, orders: list[dict], products: list[dict]
) -> list[dict]:
    rows = []
    iid = 1
    for order in orders:
        n_items = rng.choices([1, 2, 3, 4], weights=[0.3, 0.4, 0.2, 0.1])[0]
        chosen_products = rng.sample(products, min(n_items, len(products)))
        for prod in chosen_products:
            qty = rng.randint(1, 4)
            rows.append({
                "item_id": iid,
                "order_id": order["order_id"],
                "product_id": prod["product_id"],
                "quantity": qty,
                "unit_price": prod["price"],
            })
            iid += 1
    return rows


def update_stock(rng: random.Random, products: list[dict]) -> None:
    for p in products:
        p["stock"] = rng.randint(0, 100)


# ---------------------------------------------------------------------------
# NL vocabulary builder (bag-of-words vocab for NL embedding)
# ---------------------------------------------------------------------------
def build_nl_vocab(query_files: list[Path], vocab_dim: int = 128) -> dict:
    """
    Collect all unique words from NL queries across all tasks.
    Truncate/pad to vocab_dim entries. Save as word -> index mapping.
    """
    import json as _json

    words: set[str] = set()
    for qf in query_files:
        if not qf.exists():
            continue
        queries = _json.loads(qf.read_text())
        for q in queries:
            for tok in q["nl"].lower().split():
                tok = tok.strip(".,?!;:")
                if tok:
                    words.add(tok)

    sorted_words = sorted(words)[:vocab_dim - 2]
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, w in enumerate(sorted_words):
        vocab[w] = i + 2
    return vocab


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(db_path: str, seed: int, vocab_path: str) -> None:
    rng = random.Random(seed)

    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    if db_file.exists():
        db_file.unlink()
        print(f"[seed_db] Removed existing database at {db_file}")

    conn = sqlite3.connect(str(db_file))
    conn.executescript(SCHEMA_SQL)

    customers = seed_customers(rng, n=50)
    products = seed_products()
    update_stock(rng, products)
    orders = seed_orders(rng, n_customers=50, n_orders=200)
    order_items = seed_order_items(rng, orders, products)

    conn.executemany(
        "INSERT INTO customers VALUES (:customer_id,:name,:email,:city,:signup_date,:tier)",
        customers,
    )
    conn.executemany(
        "INSERT INTO products VALUES (:product_id,:name,:category,:price,:stock)",
        products,
    )
    conn.executemany(
        "INSERT INTO orders VALUES (:order_id,:customer_id,:order_date,:status)",
        orders,
    )
    conn.executemany(
        "INSERT INTO order_items VALUES (:item_id,:order_id,:product_id,:quantity,:unit_price)",
        order_items,
    )
    conn.commit()
    conn.close()

    print(
        f"[seed_db] Created {db_file} with "
        f"{len(customers)} customers, {len(products)} products, "
        f"{len(orders)} orders, {len(order_items)} order_items."
    )

    # Build and save NL vocabulary
    task_query_dir = Path("configs/tasks")
    query_files = list(task_query_dir.glob("task_*.json"))
    vocab = build_nl_vocab(query_files, vocab_dim=128)
    vocab_file = Path(vocab_path)
    vocab_file.parent.mkdir(parents=True, exist_ok=True)
    vocab_file.write_text(json.dumps(vocab, indent=2))
    print(f"[seed_db] Saved NL vocabulary ({len(vocab)} tokens) to {vocab_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the ecommerce SQLite database")
    parser.add_argument("--db_path", default="data/ecommerce.db")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab_path", default="configs/nl_vocab.json")
    args = parser.parse_args()
    main(args.db_path, args.seed, args.vocab_path)
