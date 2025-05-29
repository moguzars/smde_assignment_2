import simpy
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# -------------------- Configuration --------------------
UNIT_SALE_PRICE = 100
DELIVERY_INTERVAL = 1
SIM_DAYS = 60
STORAGE_COST_PER_UNIT_PER_DAY = 1

# Define storage tiers
STORAGE_TIERS = {
    "small":  {"capacity": 100, "monthly_rent": 500},
    "medium": {"capacity": 200, "monthly_rent": 1000},
    "large":  {"capacity": 400, "monthly_rent": 2000},
}



# Distances from distribution center (DC) for each store
num_elements = 45
min_km = 10
max_km = 500
mean = (min_km + max_km) / 2
std_dev = (max_km - min_km) / 6
raw_distances = np.random.normal(loc=mean, scale=std_dev, size=num_elements)
clipped_distances = np.clip(raw_distances, min_km, max_km)

DELIVERY_QUANTITY = 10000
INITIAL_INVENTORY = 10000 # Average weekly sales overall
TRANSPORT_COST_PER_DELIVERY = 100
START_DATE = datetime.strptime("2012-09-14", "%Y-%m-%d").date()
NUM_TRUCKS = 2
DELIVERY_TIME_MEAN = 1.5
DELIVERY_TIME_STD = 0.3
REORDER_POINT = 5000
TRANSPORT_COST_PER_DELIVERY_BASE = 100
TRANSPORT_COST_PER_UNIT_LOAD = 0.01
TRANSPORT_COST_PER_KM = 0.5
DELIVERY_CHECK_INTERVAL = 1
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


STORE_DISTANCES_FROM_DC = {
    store_id: dist
    for store_id, dist in zip(sales_df["Store"].unique(), clipped_distances.tolist())
}
AVG_FUEL_PRICE = sales_df["Fuel_Price"].mean()


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
class DistributionCenter:
    def __init__(self, env, stores, trucks, distances=STORE_DISTANCES_FROM_DC):
        self.env = env
        self.stores = stores
        self.trucks = trucks
        self.distances = distances
        self.total_transport_cost = 0
        self.action = env.process(self.periodic_check_and_order())
        
    def check_if_stock_below_reorder_point(self, store):
        return store.inventory < REORDER_POINT

    def _execute_delivery(self, store):
        # print(f"[{self.env.now:.2f}]: DC is preparing an order for {store.name} (Inv: {store.inventory}). Requesting truck...")
        
        with self.trucks.request() as request:
            yield request # Wait until a truck is available
            
            print(f"[{self.env.now:.2f}]: Truck acquired for {store.name}. Beginning delivery...")

            # Calculate actual delivery time (one way)
            delivery_travel_time = max(0.1, np.random.normal(DELIVERY_TIME_MEAN, DELIVERY_TIME_STD))
            
            # Simulate travel to store
            yield self.env.timeout(delivery_travel_time)

            # Update store inventory and costs
            distance = self.distances.get(store.store_id, 0)
            transport_cost = (TRANSPORT_COST_PER_DELIVERY_BASE +
                              (distance * TRANSPORT_COST_PER_KM) +
                              (DELIVERY_QUANTITY * TRANSPORT_COST_PER_UNIT_LOAD))
            
            store.inventory += DELIVERY_QUANTITY
            store.total_transport_cost += transport_cost
            self.total_transport_cost += transport_cost # DC also tracks total transport cost
            
            print(f"[{self.env.now:.2f}]: Delivery completed for {store.name}. New inventory: {store.inventory}. Cost: {transport_cost:.2f}")

            # Simulate return trip for the truck
            yield self.env.timeout(delivery_travel_time)
            print(f"[{self.env.now:.2f}]: Truck returned to DC from {store.name}.")

    def periodic_check_and_order(self):
        while True:
            yield self.env.timeout(DELIVERY_CHECK_INTERVAL)
            print(f"[{self.env.now:.2f}]: DC performing periodic stock check for all stores.")

            for store in self.stores:
                if self.check_if_stock_below_reorder_point(store):
                    # Start a new process for each delivery request
                    # This allows multiple deliveries to be in progress or queued
                    self.env.process(self._execute_delivery(store))

# -------------------- Simulation Runner --------------------
def simulate(store_configs):
    env = simpy.Environment()
    stores = [Store(env, store_id, storage_type) for store_id, storage_type in store_configs]
    trucks = simpy.Resource(env, capacity=NUM_TRUCKS)
    dc = DistributionCenter(env, stores=stores, trucks=trucks) 
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
