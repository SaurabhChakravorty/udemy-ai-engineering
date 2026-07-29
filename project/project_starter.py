import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
import json
import re
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union, Optional, Tuple
from sqlalchemy import create_engine, Engine
from smolagents import ToolCallingAgent, OpenAIServerModel, tool

# Create an SQLite database
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
db_engine = create_engine(f"sqlite:///{os.path.join(_SCRIPT_DIR, 'munder_difflin.db')}")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv(os.path.join(_SCRIPT_DIR, "quote_requests.csv"))
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv(os.path.join(_SCRIPT_DIR, "quotes.csv"))
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################

# Set up and load your env parameters and instantiate your model.
dotenv.load_dotenv(os.path.join(_SCRIPT_DIR, ".env"))

_model = None


def get_model() -> OpenAIServerModel:
    """Create the LLM model once, loading the API key from the local .env file."""
    global _model
    if _model is not None:
        return _model
    api_key = os.getenv("UDACITY_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing API key. Add UDACITY_OPENAI_API_KEY or OPENAI_API_KEY to project/.env"
        )
    _model = OpenAIServerModel(
        model_id="gpt-4o-mini",
        api_base="https://openai.vocareum.com/v1",
        api_key=api_key,
    )
    return _model

CATALOG_BY_NAME = {item["item_name"]: item for item in paper_supplies}

BULK_DISCOUNT_RATES = {"small": 0.05, "medium": 0.10, "large": 0.15}


def _validate_item_name(item_name: str) -> str:
    if item_name not in CATALOG_BY_NAME:
        raise ValueError(
            f"Unknown item '{item_name}'. Use list_catalog_items() for exact names."
        )
    return item_name


def _parse_items_json(items_json: str) -> List[Dict]:
    items = json.loads(items_json)
    if not isinstance(items, list):
        raise ValueError("items_json must be a JSON list of {item_name, quantity} objects.")
    for entry in items:
        _validate_item_name(entry["item_name"])
        entry["quantity"] = int(entry["quantity"])
    return items


def _line_subtotals(items: List[Dict]) -> List[Dict]:
    lines = []
    for entry in items:
        unit_price = CATALOG_BY_NAME[entry["item_name"]]["unit_price"]
        subtotal = unit_price * entry["quantity"]
        lines.append({**entry, "unit_price": unit_price, "subtotal": subtotal})
    return lines


def _apply_bulk_discount(subtotal: float, total_units: int, order_size: str) -> float:
    rate = BULK_DISCOUNT_RATES.get(order_size.lower(), 0.05)
    if total_units >= 1000:
        rate += 0.05
    if subtotal >= 100:
        rate = max(rate, 0.10)
    discounted = subtotal * (1 - rate)
    if discounted >= 100:
        return round(discounted / 5) * 5
    return round(discounted, 2)


def _parse_customer_deadline(request: str) -> Optional[str]:
    """Extract a requested delivery deadline from natural language."""
    match = re.search(
        r"(?:deliver(?:ed|y)?\s+by|needed\s+by|ensure\s+delivery\s+by|by)\s+"
        r"([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
        request,
        re.IGNORECASE,
    )
    if not match:
        return None
    try:
        deadline = datetime.strptime(match.group(1).replace(",", ""), "%B %d %Y")
        return deadline.strftime("%Y-%m-%d")
    except ValueError:
        return None


UNSUPPORTED_PRODUCT_PATTERNS = [
    (r"\ba3\b", "A3 paper (not in catalog)"),
    (r"\ba5\b", "A5 paper (not in catalog)"),
    (r"balloon", "Balloons (not in catalog)"),
    (r"\bticket", "Tickets (not in catalog)"),
    (r"\bcardboard\b", "Cardboard (not in catalog)"),
]


def _detect_unsupported_products(request: str) -> List[str]:
    """Flag catalog items the company does not sell."""
    unsupported = []
    request_lower = request.lower()
    for pattern, label in UNSUPPORTED_PRODUCT_PATTERNS:
        if re.search(pattern, request_lower):
            unsupported.append(label)
    return unsupported


# Keyword → exact catalog name (checked in order; first match wins)
_CATALOG_KEYWORD_MAP = [
    (r"decorative adhesive tape|washi tape", "Decorative adhesive tape (washi tape)"),
    (r"large poster|poster board|24\s*[x\"]\s*36|24x36", "Large poster paper (24x36 inches)"),
    (r"party streamer|streamer", "Party streamers"),
    (r"construction paper", "Construction paper"),
    (r"heavyweight paper|heavy cardstock|heavy-weight cardstock", "Heavyweight paper"),
    (r"glossy a4|a4 glossy|glossy paper", "Glossy paper"),
    (r"matte a4|a4 matte|matte paper", "Matte paper"),
    (r"recycled cardstock|recycled paper", "Recycled paper"),
    (r"colored paper|colour paper|assorted colors", "Colored paper"),
    (r"standard copy|standard printer|printer paper|printing paper|copy paper", "Standard copy paper"),
    (r"paper napkin|napkin", "Paper napkins"),
    (r"paper cup|biodegradable cup", "Paper cups"),
    (r"paper plate|biodegradable plate", "Paper plates"),
    (r"table cover", "Table covers"),
    (r"envelope|kraft paper envelope", "Envelopes"),
    (r"flyer", "Flyers"),
    (r"poster paper", "Poster paper"),
    (r"cardstock|card stock", "Cardstock"),
    (r"\ba4 paper|\ba4 white|\ba4 printing|\ba4 printer", "A4 paper"),
    (r"glossy paper", "Glossy paper"),
]


def _programmatic_map_request(request: str) -> List[Dict]:
    """Fallback mapper: extract quantities and map descriptions to catalog items."""
    items: List[Dict] = []
    seen: set = set()
    patterns = [
        r"(\d[\d,]*)\s*(?:sheets?|rolls?|reams?|packs?|units?|flyers?|posters?|tickets?|napkins?|cups?|plates?|covers?|envelopes?)\s*(?:of\s+)?(.+?)(?=\n-|\n\n|,|\n|$|\.)",
        r"(\d[\d,]*)\s+(?:sheets?|rolls?|reams?|packs?|units?)\s+(?:of\s+)?(.+?)(?=\n-|\n\n|,|\n|$|\.)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, request, re.IGNORECASE | re.MULTILINE):
            qty = int(match.group(1).replace(",", ""))
            desc = match.group(2).strip().lower()
            item_name = None
            for keyword_pattern, catalog_name in _CATALOG_KEYWORD_MAP:
                if re.search(keyword_pattern, desc, re.IGNORECASE):
                    item_name = catalog_name
                    break
            if item_name and item_name in CATALOG_BY_NAME:
                key = (item_name, qty)
                if key not in seen:
                    seen.add(key)
                    items.append({"item_name": item_name, "quantity": qty})
    return items


def _safe_agent_run(agent: ToolCallingAgent, prompt: str, fallback: str) -> str:
    """Run an agent safely; return fallback text if the LLM call fails."""
    try:
        return str(agent.run(prompt))
    except Exception as exc:
        print(f"WARN: {agent.name} failed ({exc}). Using fallback response.")
        return fallback


"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""


# Tools for inventory agent

@tool
def list_catalog_items() -> List[str]:
    """Return exact catalog item names available for quoting and ordering."""
    return [item["item_name"] for item in paper_supplies]


@tool
def get_item_unit_price(item_name: str) -> float:
    """Return the unit price for a catalog item.

    Args:
        item_name: Exact catalog item name.

    Returns:
        Unit price in dollars.
    """
    return CATALOG_BY_NAME[_validate_item_name(item_name)]["unit_price"]


@tool
def check_stock_for_item(item_name: str, as_of_date: str) -> Dict:
    """Check current stock for one item as of a date.

    Args:
        item_name: Exact catalog item name.
        as_of_date: Inventory cutoff date in YYYY-MM-DD format.

    Returns:
        Dictionary with item_name, current_stock, and as_of_date.
    """
    item_name = _validate_item_name(item_name)
    stock_df = get_stock_level(item_name, as_of_date)
    return {
        "item_name": item_name,
        "current_stock": int(stock_df["current_stock"].iloc[0]),
        "as_of_date": as_of_date,
    }


@tool
def check_all_inventory(as_of_date: str) -> Dict[str, int]:
    """Return all in-stock items and quantities as of a date.

    Args:
        as_of_date: Inventory cutoff date in YYYY-MM-DD format.

    Returns:
        Mapping of item names to stock quantities.
    """
    return get_all_inventory(as_of_date)


@tool
def reorder_stock(item_name: str, quantity_needed: int, order_date: str) -> str:
    """Place a supplier stock order when inventory is insufficient.

    Args:
        item_name: Exact catalog item name.
        quantity_needed: Units required for the customer order.
        order_date: Date the reorder is placed in YYYY-MM-DD format.

    Returns:
        Status message describing stock level or reorder details.
    """
    item_name = _validate_item_name(item_name)
    quantity_needed = int(quantity_needed)
    current_stock = int(get_stock_level(item_name, order_date)["current_stock"].iloc[0])
    shortfall = max(0, quantity_needed - current_stock)
    if shortfall == 0:
        return f"Sufficient stock for {item_name} ({current_stock} available)."

    reorder_qty = shortfall + max(100, int(shortfall * 0.2))
    unit_price = CATALOG_BY_NAME[item_name]["unit_price"]
    total_cost = reorder_qty * unit_price
    cash = get_cash_balance(order_date)
    if cash < total_cost:
        return (
            f"Cannot reorder {item_name}: need ${total_cost:.2f}, "
            f"but only ${cash:.2f} cash available on {order_date}."
        )

    delivery_date = get_supplier_delivery_date(order_date, reorder_qty)
    create_transaction(item_name, "stock_orders", reorder_qty, total_cost, delivery_date)
    return (
        f"Reordered {reorder_qty} units of {item_name} for ${total_cost:.2f}. "
        f"Expected delivery: {delivery_date}."
    )


@tool
def ensure_inventory_for_order(items_json: str, as_of_date: str) -> str:
    """Check stock for all line items and reorder anything that is short.

    Args:
        items_json: JSON list like [{"item_name": "A4 paper", "quantity": 200}].
        as_of_date: Inventory cutoff date in YYYY-MM-DD format.

    Returns:
        Summary of stock checks and any reorders placed.
    """
    items = _parse_items_json(items_json)
    messages = []
    for entry in items:
        item_name = entry["item_name"]
        needed = entry["quantity"]
        current = int(get_stock_level(item_name, as_of_date)["current_stock"].iloc[0])
        if current >= needed:
            messages.append(f"{item_name}: {current} in stock (need {needed}). OK.")
        else:
            messages.append(reorder_stock(item_name, needed, as_of_date))
    return "\n".join(messages)


# Tools for quoting agent

@tool
def search_historical_quotes(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """Search prior quotes and customer requests for pricing guidance.

    Args:
        search_terms: Keywords to match against past requests and quotes.
        limit: Maximum number of historical quotes to return.

    Returns:
        List of matching historical quote records.
    """
    return search_quote_history(search_terms, limit=limit)


@tool
def calculate_quote(items_json: str, order_size: str) -> Dict:
    """Calculate a quote with bulk discounts.

    Args:
        items_json: JSON list like [{"item_name": "A4 paper", "quantity": 200}].
        order_size: Customer order size: small, medium, or large.

    Returns:
        Quote breakdown including line_items, subtotal, discount, and total_amount.
    """
    items = _parse_items_json(items_json)
    lines = _line_subtotals(items)
    subtotal = sum(line["subtotal"] for line in lines)
    total_units = sum(line["quantity"] for line in lines)
    total_amount = _apply_bulk_discount(subtotal, total_units, order_size)
    discount = subtotal - total_amount
    return {
        "line_items": lines,
        "subtotal": round(subtotal, 2),
        "discount_applied": round(discount, 2),
        "total_amount": total_amount,
        "order_size": order_size,
        "total_units": total_units,
    }


@tool
def estimate_delivery_date(order_date: str, total_units: int) -> str:
    """Estimate supplier delivery date for a given order size.

    Args:
        order_date: Order placement date in YYYY-MM-DD format.
        total_units: Total units in the order.

    Returns:
        Estimated delivery date in YYYY-MM-DD format.
    """
    return get_supplier_delivery_date(order_date, int(total_units))


# Tools for ordering agent

@tool
def get_available_cash(as_of_date: str) -> float:
    """Return cash balance as of a date.

    Args:
        as_of_date: Financial cutoff date in YYYY-MM-DD format.

    Returns:
        Available cash balance in dollars.
    """
    return get_cash_balance(as_of_date)


@tool
def fulfill_sales_order(items_json: str, total_amount: float, transaction_date: str) -> str:
    """Fulfill a customer order by recording sales transactions.

    Args:
        items_json: JSON list like [{"item_name": "A4 paper", "quantity": 200}].
        total_amount: Discounted order total in dollars.
        transaction_date: Sale date in YYYY-MM-DD format.

    Returns:
        Fulfillment status message for each line item.
    """
    items = _parse_items_json(items_json)
    lines = _line_subtotals(items)
    subtotal = sum(line["subtotal"] for line in lines)
    if subtotal <= 0:
        return "Cannot fulfill order: subtotal is zero."

    scale = float(total_amount) / subtotal
    results = []
    for line in lines:
        item_name = line["item_name"]
        quantity = line["quantity"]
        available = int(get_stock_level(item_name, transaction_date)["current_stock"].iloc[0])
        if available < quantity:
            results.append(
                f"FAILED {item_name}: need {quantity}, only {available} available on {transaction_date}."
            )
            continue
        line_total = round(line["subtotal"] * scale, 2)
        create_transaction(item_name, "sales", quantity, line_total, transaction_date)
        results.append(
            f"Sold {quantity} x {item_name} for ${line_total:.2f} on {transaction_date}."
        )

    if any(r.startswith("FAILED") for r in results):
        return "Partial fulfillment:\n" + "\n".join(results)
    return "Order fulfilled:\n" + "\n".join(results)


@tool
def get_financial_snapshot(as_of_date: str) -> Dict:
    """Return cash, inventory value, and total assets as of a date.

    Args:
        as_of_date: Financial cutoff date in YYYY-MM-DD format.

    Returns:
        Dictionary with cash_balance, inventory_value, and total_assets.
    """
    report = generate_financial_report(as_of_date)
    return {
        "cash_balance": report["cash_balance"],
        "inventory_value": report["inventory_value"],
        "total_assets": report["total_assets"],
    }


@tool
def generate_company_financial_report(as_of_date: str) -> Dict:
    """Generate a full financial and inventory report for health checks.

    Args:
        as_of_date: Report cutoff date in YYYY-MM-DD format.

    Returns:
        Full report including inventory_summary and top_selling_products.
    """
    return generate_financial_report(as_of_date)


@tool
def check_order_feasibility(
    items_json: str,
    request_date: str,
    customer_deadline: str,
) -> Dict:
    """Evaluate whether an order can be fulfilled by the customer's deadline.

    Args:
        items_json: JSON list like [{"item_name": "A4 paper", "quantity": 200}].
        request_date: Date the order is placed in YYYY-MM-DD format.
        customer_deadline: Required delivery date in YYYY-MM-DD format.

    Returns:
        Dictionary with feasible flag, stock_ready flag, and blocking reasons.
    """
    items = _parse_items_json(items_json)
    reasons = []
    stock_ready = True
    latest_required_delivery = request_date

    for entry in items:
        item_name = entry["item_name"]
        quantity = entry["quantity"]
        available = int(get_stock_level(item_name, request_date)["current_stock"].iloc[0])
        if available >= quantity:
            continue

        stock_ready = False
        shortfall = quantity - available
        reorder_qty = shortfall + max(100, int(shortfall * 0.2))
        unit_price = CATALOG_BY_NAME[item_name]["unit_price"]
        reorder_cost = reorder_qty * unit_price
        cash = get_cash_balance(request_date)

        if cash < reorder_cost:
            reasons.append(
                f"Insufficient cash (${cash:.2f}) to reorder {reorder_qty} units of "
                f"{item_name} (cost ${reorder_cost:.2f})."
            )

        supplier_delivery = get_supplier_delivery_date(request_date, reorder_qty)
        latest_required_delivery = max(latest_required_delivery, supplier_delivery)
        if supplier_delivery > customer_deadline:
            reasons.append(
                f"{item_name}: supplier can deliver by {supplier_delivery}, "
                f"which misses the customer deadline of {customer_deadline}."
            )

    feasible = len(reasons) == 0
    return {
        "feasible": feasible,
        "stock_ready": stock_ready,
        "reasons": reasons,
        "latest_supplier_delivery": latest_required_delivery,
        "customer_deadline": customer_deadline,
    }


# Set up your agents and create an orchestration agent that will manage them.

class QuotingAgent(ToolCallingAgent):
    """Generates customer quotes using catalog pricing and historical data."""

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[
                list_catalog_items,
                get_item_unit_price,
                search_historical_quotes,
                calculate_quote,
                estimate_delivery_date,
            ],
            model=model,
            name="quoting_agent",
            max_steps=12,
            description=(
                "Autonomous quoting agent for Munder Difflin paper supplies. You MUST use "
                "your tools to plan and execute each quote: discover catalog names, search "
                "historical quotes for guidance, build a validated items_json, and call "
                "calculate_quote for official pricing. Reason about which tools to call and "
                "in what order; do not invent prices or catalog names."
            ),
        )


class InventoryAgent(ToolCallingAgent):
    """Checks stock levels and places supplier reorders when needed."""

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[
                list_catalog_items,
                check_stock_for_item,
                check_all_inventory,
                reorder_stock,
                ensure_inventory_for_order,
                estimate_delivery_date,
            ],
            model=model,
            name="inventory_agent",
            max_steps=12,
            description=(
                "Autonomous inventory agent. Check current stock with check_stock_for_item, "
                "determine supplier delivery with estimate_delivery_date, place reorders with "
                "reorder_stock when short, and use ensure_inventory_for_order for consolidated "
                "checks. Reason about cash constraints and customer deadlines before recommending "
                "READY, REORDERED, or CANNOT_FULFILL."
            ),
        )


class OrderingAgent(ToolCallingAgent):
    """Fulfills confirmed customer orders and records sales transactions."""

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[
                check_stock_for_item,
                fulfill_sales_order,
                get_available_cash,
                get_financial_snapshot,
                generate_company_financial_report,
                check_order_feasibility,
                estimate_delivery_date,
            ],
            model=model,
            name="ordering_agent",
            max_steps=12,
            description=(
                "Autonomous ordering agent. Use check_order_feasibility to evaluate delivery "
                "and cash constraints, verify stock with check_stock_for_item, and call "
                "fulfill_sales_order only when the order is feasible and inventory is ready. "
                "Return FULFILLED, DELAYED, or REJECTED with clear reasoning based on tool "
                "outputs — do not guess feasibility."
            ),
        )


class OrchestratorAgent(ToolCallingAgent):
    """Coordinates quoting, inventory, and ordering agents for each customer request."""

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[
                list_catalog_items,
                get_financial_snapshot,
                check_all_inventory,
            ],
            model=model,
            name="orchestrator_agent",
            max_steps=4,
            description=(
                "Coordinates the Munder Difflin multi-agent workflow. Delegates quoting, "
                "inventory, and fulfillment tasks to worker agents; does not execute business "
                "operations directly."
            ),
        )
        self.quoting_agent = QuotingAgent(model)
        self.inventory_agent = InventoryAgent(model)
        self.ordering_agent = OrderingAgent(model)

    def _extract_labeled_section(self, text: str, label: str) -> Optional[str]:
        """Extract text after a labeled marker such as CUSTOMER_MESSAGE:."""
        pattern = rf"{re.escape(label)}:\s*(.+?)(?=\n[A-Z_]+:|\Z)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _extract_items_json(self, text: str) -> Optional[str]:
        """Extract a JSON item list from agent output."""
        labeled = self._extract_labeled_section(text, "ITEMS_JSON")
        if labeled:
            try:
                parsed = json.loads(labeled)
                if isinstance(parsed, list) and parsed:
                    return json.dumps(parsed)
            except json.JSONDecodeError:
                pass

        candidates = re.findall(r"\[[\s\S]*?\"item_name\"[\s\S]*?\]", text)
        for blob in reversed(candidates):
            try:
                parsed = json.loads(blob)
                if isinstance(parsed, list) and parsed and "item_name" in parsed[0]:
                    return json.dumps(parsed)
            except json.JSONDecodeError:
                continue

        match = re.search(r"\{[\s\S]*\"line_items\"[\s\S]*\}", text)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict) and "line_items" in parsed:
                    items = [
                        {"item_name": li["item_name"], "quantity": li["quantity"]}
                        for li in parsed["line_items"]
                    ]
                    return json.dumps(items)
            except json.JSONDecodeError:
                pass
        return None

    def _extract_quote_data(self, text: str) -> Optional[Dict]:
        """Extract quote breakdown JSON from quoting agent output."""
        labeled = self._extract_labeled_section(text, "QUOTE_JSON")
        if labeled:
            try:
                parsed = json.loads(labeled)
                if isinstance(parsed, dict) and "total_amount" in parsed:
                    return parsed
            except json.JSONDecodeError:
                pass

        for match in re.finditer(r"\{[\s\S]*?\"total_amount\"[\s\S]*?\}", text):
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict) and "total_amount" in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue
        return None

    def _build_search_terms(
        self, request: str, job: str, event: str
    ) -> List[str]:
        terms = [
            t
            for t in re.findall(r"[A-Za-z]{3,}", request.lower())
            if t
            not in {
                "the", "and", "for", "with", "need", "would", "like",
                "order", "sheets", "paper", "request", "following", "supplies",
            }
        ][:8]
        if job:
            terms.append(job.replace("_", " "))
        if event:
            terms.append(event)
        return terms

    def _run_quoting_workflow(
        self,
        request: str,
        order_size: str,
        job: str,
        event: str,
        request_date: str,
        search_terms: List[str],
        customer_deadline: str,
    ) -> Tuple[str, Optional[str], Optional[Dict]]:
        """Delegate the full quote workflow to the quoting agent."""
        quote_prompt = f"""
You are the Quoting Agent for Munder Difflin paper supplies.

TASK: Produce a complete quote for this customer request. You MUST use your tools —
do not guess prices or catalog names.

Customer request:
{request}

Context:
- Job type: {job}
- Order size category: {order_size}
- Event type: {event}
- Request date: {request_date}
- Customer delivery deadline: {customer_deadline}
- Search terms for history: {search_terms}

WORKFLOW (select and call tools as needed):
1. Call list_catalog_items() to see exact catalog names.
2. Map each requested product to EXACT catalog item names (substitute close matches).
3. Call search_historical_quotes({json.dumps(search_terms)}) for pricing guidance.
4. Build items_json as a JSON array string, e.g. '[{{"item_name": "Glossy paper", "quantity": 200}}]'.
5. Call calculate_quote(items_json, "{order_size}") for the official quote breakdown.
6. Optionally call estimate_delivery_date if helpful for delivery planning.

FINAL ANSWER must include these labeled sections:
ITEMS_JSON: <JSON array passed to calculate_quote>
QUOTE_JSON: <full JSON output from calculate_quote>
CUSTOMER_MESSAGE: <friendly quote message with total, discount, and line items>
"""
        quote_response = _safe_agent_run(self.quoting_agent, quote_prompt, fallback="")
        items_json = self._extract_items_json(quote_response)
        quote_data = self._extract_quote_data(quote_response)

        if not items_json:
            fallback_items = _programmatic_map_request(request)
            if fallback_items:
                items_json = json.dumps(fallback_items)
                retry_prompt = f"""
The line items for this order are already mapped. Call calculate_quote(
{items_json!r}, "{order_size}") and return ITEMS_JSON, QUOTE_JSON, and CUSTOMER_MESSAGE.
"""
                quote_response = _safe_agent_run(
                    self.quoting_agent, retry_prompt, fallback=quote_response
                )
                items_json = self._extract_items_json(quote_response) or items_json
                quote_data = self._extract_quote_data(quote_response)

        customer_message = self._extract_labeled_section(quote_response, "CUSTOMER_MESSAGE")
        if customer_message:
            quote_response = customer_message
        elif quote_data:
            quote_response = (
                f"Thank you for your order! Your quote total is "
                f"${quote_data['total_amount']:.2f} "
                f"(includes a bulk discount of ${quote_data['discount_applied']:.2f}). "
                f"Delivery is scheduled by {customer_deadline}."
            )

        return quote_response, items_json, quote_data

    def _run_inventory_workflow(
        self,
        items_json: str,
        request_date: str,
        customer_deadline: str,
    ) -> Tuple[str, str]:
        """Delegate inventory checks and reorders to the inventory agent."""
        inventory_prompt = f"""
You are the Inventory Agent for Munder Difflin.

TASK: Ensure we can supply this order. Use your tools to check stock, reorder if needed,
and decide whether the order is feasible by the customer deadline.

Order line items (JSON): {items_json}
Order date (inventory as-of date): {request_date}
Customer delivery deadline: {customer_deadline}

WORKFLOW (reason and select tools):
1. For each line item, call check_stock_for_item(item_name, "{request_date}").
2. When stock is insufficient, call reorder_stock(item_name, quantity_needed, "{request_date}").
3. Call ensure_inventory_for_order({items_json!r}, "{request_date}") for a consolidated summary.
4. Use estimate_delivery_date when you need supplier lead times vs. {customer_deadline}.

FINAL ANSWER must include:
INVENTORY_STATUS: READY | REORDERED | CANNOT_FULFILL
INVENTORY_DETAILS: summary of tool outputs and your reasoning
CUSTOMER_MESSAGE: 2-3 sentence inventory update for the customer

If cash or delivery deadline makes fulfillment impossible, set INVENTORY_STATUS to
CANNOT_FULFILL and start CUSTOMER_MESSAGE with "ORDER REJECTED:".
"""
        full_response = _safe_agent_run(
            self.inventory_agent, inventory_prompt, fallback=""
        )
        customer_message = self._extract_labeled_section(
            full_response, "CUSTOMER_MESSAGE"
        )
        if customer_message:
            return full_response, customer_message
        details = self._extract_labeled_section(full_response, "INVENTORY_DETAILS")
        return full_response, details or full_response

    def _inventory_blocks_order(self, full_inventory_response: str) -> bool:
        response_lower = full_inventory_response.lower()
        if re.search(r"inventory_status:\s*cannot_fulfill", response_lower):
            return True
        if "order rejected" in response_lower:
            return True
        return False

    def _run_ordering_workflow(
        self,
        items_json: str,
        quote_data: Dict,
        request_date: str,
        customer_deadline: str,
        inventory_response: str,
    ) -> str:
        """Delegate feasibility checks and fulfillment to the ordering agent."""
        order_prompt = f"""
You are the Ordering Agent for Munder Difflin.

TASK: Decide whether to fulfill this order and record the sale if appropriate.
You MUST use your tools — do not guess feasibility or inventory levels.

Order line items (JSON): {items_json}
Quote total amount: ${quote_data["total_amount"]:.2f}
Request date: {request_date}
Customer deadline: {customer_deadline}

Inventory agent report:
{inventory_response}

WORKFLOW (reason and select tools):
1. Call check_order_feasibility({items_json!r}, "{request_date}", "{customer_deadline}").
2. Call check_stock_for_item for any item whose availability is unclear.
3. If feasible and stock is on hand today, call fulfill_sales_order(
   {items_json!r}, {quote_data["total_amount"]}, "{request_date}").
4. If feasible but stock arrives later (per inventory report), set ORDER_STATUS to DELAYED
   and explain — do NOT call fulfill_sales_order until stock is available.
5. If not feasible (deadline, cash, or stock), set ORDER_STATUS to REJECTED — do NOT fulfill.
6. After fulfillment, optionally call get_available_cash or get_financial_snapshot.

FINAL ANSWER must include:
ORDER_STATUS: FULFILLED | DELAYED | REJECTED
FULFILLMENT_DETAILS: key tool outputs
CUSTOMER_MESSAGE: final customer-facing fulfillment message

If rejecting, include "ORDER REJECTED:" in CUSTOMER_MESSAGE.
If fulfilled, include "Order fulfilled" or sold line details from fulfill_sales_order.
If delayed, mention queued/awaiting supplier delivery.
"""
        full_response = _safe_agent_run(self.ordering_agent, order_prompt, fallback="")
        customer_message = self._extract_labeled_section(
            full_response, "CUSTOMER_MESSAGE"
        )
        fulfillment_details = self._extract_labeled_section(
            full_response, "FULFILLMENT_DETAILS"
        )
        if fulfillment_details and customer_message:
            if "order fulfilled" in fulfillment_details.lower() or fulfillment_details.lower().startswith("sold "):
                return f"{fulfillment_details}\n{customer_message}"
            return customer_message
        if customer_message:
            return customer_message
        return fulfillment_details or full_response

    def process_request(
        self,
        request: str,
        job: str,
        order_size: str,
        event: str,
        request_date: str,
    ) -> str:
        search_terms = self._build_search_terms(request, job, event)

        unsupported = _detect_unsupported_products(request)
        unsupported_note = ""
        if unsupported:
            unsupported_note = (
                f"Note: the following requested products are not in our catalog and were "
                f"excluded: {', '.join(unsupported)}.\n\n"
            )

        snapshot = get_financial_snapshot(request_date)
        print(
            f"Pre-flight check on {request_date}: "
            f"cash=${snapshot['cash_balance']:.2f}, "
            f"inventory=${snapshot['inventory_value']:.2f}"
        )

        customer_deadline = _parse_customer_deadline(request) or request_date

        quote_response, items_json, quote_data = self._run_quoting_workflow(
            request, order_size, job, event, request_date, search_terms, customer_deadline
        )
        if not items_json or not quote_data:
            if unsupported:
                return (
                    "ORDER REJECTED: We cannot fulfill this request because it includes "
                    f"products we do not carry: {', '.join(unsupported)}. "
                    "Please revise your order using our paper supply catalog."
                )
            return (
                "ORDER REJECTED: We could not map your request to catalog items. "
                "Please specify products using our available paper supply catalog."
            )

        inventory_full, inventory_response = self._run_inventory_workflow(
            items_json, request_date, customer_deadline
        )
        if self._inventory_blocks_order(inventory_full):
            return (
                f"{unsupported_note}"
                f"{quote_response}\n\n"
                f"--- Inventory ---\n{inventory_response}"
            )

        order_response = self._run_ordering_workflow(
            items_json, quote_data, request_date, customer_deadline, inventory_full
        )

        return (
            f"{unsupported_note}"
            f"{quote_response}\n\n"
            f"--- Inventory ---\n{inventory_response}\n\n"
            f"--- Fulfillment ---\n{order_response}"
        )

    def classify_response(self, response: str) -> str:
        """Classify order outcome for evaluation reporting."""
        response_lower = response.lower()
        if "order rejected" in response_lower:
            return "rejected"
        if "order fulfilled" in response_lower or "sold " in response_lower:
            return "fulfilled"
        if "partial fulfillment" in response_lower:
            return "partial"
        if any(kw in response_lower for kw in ("queued", "awaiting", "delay", "not yet available")):
            return "delayed"
        return "quoted"


# Run your test scenarios by writing them here. Make sure to keep track of them.

def _parse_request_dates(date_values: List[str]) -> List[Optional[datetime]]:
    """Parse MM/DD/YY request dates using plain Python (avoids pandas parsing issues)."""
    parsed: List[Optional[datetime]] = []
    for value in date_values:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            parsed.append(None)
            continue
        dt = None
        for fmt in ("%m/%d/%y", "%m/%d/%Y"):
            try:
                dt = datetime.strptime(str(value).strip(), fmt)
                break
            except ValueError:
                continue
        parsed.append(dt)
    return parsed


def run_test_scenarios():
    os.chdir(_SCRIPT_DIR)

    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv(
            os.path.join(_SCRIPT_DIR, "quote_requests_sample.csv")
        )
        quote_requests_sample["request_date"] = _parse_request_dates(
            quote_requests_sample["request_date"].tolist()
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############
    llm = get_model()
    orchestrator = OrchestratorAgent(llm)

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        try:
            response = orchestrator.process_request(
                request=row["request"],
                job=str(row["job"]),
                order_size=str(row["need_size"]),
                event=str(row["event"]),
                request_date=request_date,
            )
            order_status = orchestrator.classify_response(response)
        except Exception as exc:
            response = f"ORDER REJECTED: System error processing request — {exc}"
            order_status = "rejected"
            print(f"ERROR: {exc}")

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response[:500]}{'...' if len(response) > 500 else ''}")
        print(f"Order Status: {order_status}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "order_status": order_status,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    status_counts = pd.Series([r["order_status"] for r in results]).value_counts()
    print("\n===== ORDER STATUS SUMMARY =====")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    print("\nResults saved to test_results.csv")
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
