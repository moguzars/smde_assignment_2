import simpy
import pandas as pd
import random
from datetime import datetime, timedelta

# -------------------- Configuration --------------------
UNIT_SALE_PRICE = 100
DELIVERY_INTERVAL = 1
TRANSPORT_COST_PER_DELIVERY = 1
SIM_DAYS = 60
STORAGE_COST_PER_UNIT_PER_DAY = 1

# Define storage tiers
STORAGE_TIERS = {
    "small":  {"capacity": 100, "monthly_rent": 500},
    "medium": {"capacity": 200, "monthly_rent": 1000},
    "large":  {"capacity": 400, "monthly_rent": 2000},
}

# -------------------- Load Sales Data --------------------
sales_df = pd.read_csv("Walmart_Sales.csv")
sales_df["Date"] = pd.to_datetime(sales_df["Date"], format="%d-%m-%Y")
sales_df["Store"] = sales_df["Store"].astype(str)

# Keep only last 2 months of data
sales_df = sales_df.sort_values("Date")
last_date = sales_df["Date"].max()
two_months_ago = last_date - pd.DateOffset(weeks=8)
sales_df = sales_df[sales_df["Date"] >= two_months_ago]

START_DATE = datetime.strptime(two_months_ago.strftime("%Y-%m-%d"), "%Y-%m-%d").date()

# Compute weekly customer arrival intervals
sales_df["Weekly_Units"] = sales_df["Weekly_Sales"] / UNIT_SALE_PRICE
sales_df["Arrival_Interval"] = 7 / sales_df["Weekly_Units"]

# Create lookup dict: (store, week_start) -> arrival_interval
arrival_lookup = {
    (row["Store"], row["Date"].date()): row["Arrival_Interval"]
    for _, row in sales_df.iterrows()
}

# -------------------- Store Class --------------------
class Store:
    def __init__(self, env, store_id, storage_type):
        self.env = env
        self.store_id = str(store_id)
        self.name = f"Store {store_id}"
        self.storage_type = storage_type

        # Set tier attributes
        self.capacity = STORAGE_TIERS[storage_type]["capacity"]
        self.monthly_rent = STORAGE_TIERS[storage_type]["monthly_rent"]

        # State variables
        self.inventory = int(self.capacity / 2)
        self.total_sales = 0
        self.total_revenue = 0
        self.total_transport_cost = 0
        self.total_stockouts = 0
        self.lost_profit = 0
        self.total_storage_cost = 0
        self.total_rent_paid = 0
        self.last_day_checked = -1

        self.action = env.process(self.run())

    def get_arrival_interval(self, current_date):
        week_start = current_date - timedelta(days=current_date.weekday()-4)
        return arrival_lookup.get((self.store_id, week_start), 99999)

    def run(self):
        while True:
            sim_day = int(self.env.now)
            current_date = START_DATE + timedelta(days=sim_day)

            # Track storage and rent costs daily
            if sim_day != self.last_day_checked:
                self.last_day_checked = sim_day
                self.total_storage_cost += self.inventory * STORAGE_COST_PER_UNIT_PER_DAY
                if sim_day % 30 == 0:
                    self.total_rent_paid += self.monthly_rent

            # Get customer arrival interval for this date
            arrival_interval = self.get_arrival_interval(current_date)
            yield self.env.timeout(random.expovariate(1.0 / arrival_interval))

            # Handle customer
            if self.inventory > 0:
                self.inventory -= 1
                self.total_sales += 1
                self.total_revenue += UNIT_SALE_PRICE
            else:
                self.total_stockouts += 1
                self.lost_profit += UNIT_SALE_PRICE

# -------------------- Delivery Process --------------------
def delivery(env, stores):
    while True:
        yield env.timeout(DELIVERY_INTERVAL)
        for store in stores:
            store.inventory = store.capacity
            store.total_transport_cost += TRANSPORT_COST_PER_DELIVERY

# -------------------- Simulation Runner --------------------
def simulate(store_configs):
    env = simpy.Environment()
    stores = [Store(env, store_id, storage_type) for store_id, storage_type in store_configs]
    env.process(delivery(env, stores))
    env.run(until=SIM_DAYS)

    # Reporting
    total_revenue = sum(s.total_revenue for s in stores)
    total_transport = sum(s.total_transport_cost for s in stores)
    total_storage = sum(s.total_storage_cost for s in stores)
    total_rent = sum(s.total_rent_paid for s in stores)
    total_stockouts = sum(s.total_stockouts for s in stores)
    total_lost_profit = sum(s.lost_profit for s in stores)

    final_profit = total_revenue - total_transport - total_storage - total_rent

    print(f"--- Simulation Results after {SIM_DAYS} days ---")
    for s in stores:
        print(f"{s.name} ({s.storage_type}) | Sales: {s.total_sales}, "
              f"Revenue: {s.total_revenue:.2f}, Stockouts: {s.total_stockouts}, "
              f"Lost Profit: {s.lost_profit:.2f}, Transport: {s.total_transport_cost}, "
              f"Storage Cost: {s.total_storage_cost:.2f}, Rent Paid: {s.total_rent_paid}")

    print(f"\nTOTAL Revenue: €{total_revenue:.2f}")
    print(f"TOTAL Transport Cost: €{total_transport:.2f}")
    print(f"TOTAL Storage Cost: €{total_storage:.2f}")
    print(f"TOTAL Rent Paid: €{total_rent:.2f}")
    print(f"TOTAL Stockouts: {total_stockouts}")
    print(f"TOTAL Lost Profit: €{total_lost_profit:.2f}")
    print(f"FINAL Profit: €{final_profit:.2f}")

# Example usage: (store_id, storage_type)
simulate([
    ("1", "large"),
    ("1", "large"),
    ("1", "large")
])
