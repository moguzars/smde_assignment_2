import simpy
import pandas as pd
import random
from datetime import datetime, timedelta

# Load the dataset
sales_df = pd.read_csv("Walmart_Sales.csv")
sales_df["Date"] = pd.to_datetime(sales_df["Date"],format="%d-%m-%Y")
sales_df["Store"] = sales_df["Store"].astype(str)

# Sort by date
sales_df = sales_df.sort_values("Date")

# Filter for the last 2 months
last_date = sales_df["Date"].max()
two_months_ago = last_date - pd.DateOffset(weeks=8)
sales_df = sales_df[sales_df["Date"] >= two_months_ago]

# Convert sales (€) to units, assume 1 unit = €100
sales_df["Weekly_Units"] = sales_df["Weekly_Sales"] / 100
sales_df["Arrival_Interval"] = 7 / sales_df["Weekly_Units"]

# Create arrival lookup dictionary for store-week
arrival_lookup = {
    (row["Store"], row["Date"].date()): row["Arrival_Interval"]
    for _, row in sales_df.iterrows()
}

# Simulation parameters
SIM_DAYS = 60
UNIT_SALE_PRICE = 100
DELIVERY_INTERVAL = 7
DELIVERY_QUANTITY = 100
INITIAL_INVENTORY = 50
TRANSPORT_COST_PER_DELIVERY = 100
START_DATE = datetime.strptime("2012-09-14", "%Y-%m-%d").date()


class Store:
    def __init__(self, env, store_id):
        self.env = env
        self.store_id = str(store_id)
        self.name = f"Store {store_id}"
        self.inventory = INITIAL_INVENTORY
        self.total_sales = 0
        self.total_revenue = 0
        self.total_transport_cost = 0
        self.total_stockouts = 0
        self.lost_profit = 0
        self.action = env.process(self.run())

    def get_arrival_interval(self, current_day):
        week_start = current_day - timedelta(days=current_day.weekday()-4)
        return arrival_lookup.get((self.store_id, week_start), 99999)  # very high if missing

    def run(self):
        while True:
            x = self.env.now
            current_day = START_DATE + timedelta(days=int(self.env.now))
            arrival_interval = self.get_arrival_interval(current_day)
            demand = 1

            if self.inventory >= demand:
                self.inventory -= 1
                self.total_sales += 1
                self.total_revenue += UNIT_SALE_PRICE
            else:
                self.total_stockouts += 1
                self.lost_profit += UNIT_SALE_PRICE
            x = random.expovariate(1.0 / arrival_interval)
            yield self.env.timeout(random.expovariate(1.0 / arrival_interval))  # ✅ Advance time here

    def handle_customer(self):
        yield self.env.timeout(0)
        demand = 1
        if self.inventory >= demand:
            self.inventory -= 1
            self.total_sales += 1
            self.total_revenue += UNIT_SALE_PRICE
        else:
            self.total_stockouts += 1
            self.lost_profit += UNIT_SALE_PRICE

def delivery(env, stores):
    while True:
        yield env.timeout(DELIVERY_INTERVAL)
        for store in stores:
            store.inventory = DELIVERY_QUANTITY
            store.total_transport_cost += TRANSPORT_COST_PER_DELIVERY

def simulate(unique_stores):
    env = simpy.Environment()
    stores = [Store(env, store_id) for store_id in unique_stores]
    env.process(delivery(env, stores))
    env.run(until=SIM_DAYS)

    total_revenue = sum(s.total_revenue for s in stores)
    total_transport = sum(s.total_transport_cost for s in stores)
    total_profit = total_revenue - total_transport
    total_stockouts = sum(s.total_stockouts for s in stores)
    total_lost_profit = sum(s.lost_profit for s in stores)

    print(f"--- Simulation Results for {SIM_DAYS} days ---")
    for s in stores:
        print(f"{s.name} | Sales: {s.total_sales}, Revenue: {s.total_revenue}, "
              f"Stockouts: {s.total_stockouts}, Lost Profit: {s.lost_profit}, "
              f"Transport: {s.total_transport_cost}")

    print(f"TOTAL Revenue: {total_revenue}")
    print(f"TOTAL Transport Cost: {total_transport}")
    print(f"TOTAL Profit: {total_profit}")
    print(f"TOTAL Stockouts: {total_stockouts}")
    print(f"TOTAL Lost Profit: {total_lost_profit}")

# Example run (replace with your real store IDs)
simulate(sales_df["Store"].unique())
