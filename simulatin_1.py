import simpy
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta


# -------------------- Load Sales Data --------------------
sales_df = pd.read_csv("Walmart_Sales.csv")
sales_df["Date"] = pd.to_datetime(sales_df["Date"], format="%d-%m-%Y")
sales_df["Store"] = sales_df["Store"].astype(str)

START_DATE = datetime.strptime("05-02-2010", "%d-%m-%Y").date()

# Weekly_Units
sales_df["Weekly_Units"] = sales_df["Weekly_Sales"] / 100

fuel_lookup = {
    (row["Store"], row["Date"].date()): row["Fuel_Price"]
    for _, row in sales_df.iterrows()
}

units_lookup = {
    (row["Store"], row["Date"].date()): row["Weekly_Units"]
    for _, row in sales_df.iterrows()
}

# -------------------- Calculate customer arrival interval --------------------
# Generate Arrival_Interval column (7 days/week sales)
sales_df["Arrival_Interval"] = 7 / sales_df["Weekly_Units"]

# Create an arrival interval dictionary
arrival_lookup = {
    (row["Store"], row["Date"].date()): row["Arrival_Interval"]
    for _, row in sales_df.iterrows()
}

# -------------------- Configuration --------------------
UNIT_SALE_PRICE = 100
DELIVERY_INTERVAL = 7
SIM_DAYS = 100  # For full simulation it is 1000 days
STORAGE_COST_PER_UNIT_PER_DAY = 1

# Define storage tiers
STORAGE_TIERS = {
    "small":  {"capacity": 100, "monthly_rent": 500},
    "medium": {"capacity": 200, "monthly_rent": 1000},
    "large":  {"capacity": 400, "monthly_rent": 2000},
}

TRANSPORT_TIERS = {
    "low": {"trucks": 45, "truck_speed": 50, "fuel_per_km": 0.06, "delivery_quantity": 20},
    "high": {"trucks": 90, "truck_speed": 60, "fuel_per_km": 0.05, "delivery_quantity": 40}
}


# Distances from distribution center (DC) for each store
num_elements = 45
min_km = 10
max_km = 500
mean = (min_km + max_km) / 2
std_dev = (max_km - min_km) / 6
raw_distances = np.random.normal(loc=mean, scale=std_dev, size=num_elements)
clipped_distances = np.clip(raw_distances, min_km, max_km)


INITIAL_INVENTORY = 60000 # four weeks sales overall
TRANSPORT_COST_PER_DELIVERY = 100
NUM_TRUCKS = 2
DELIVERY_TIME_MEAN = 1.5
DELIVERY_TIME_STD = 0.3

TRANSPORT_COST_PER_DELIVERY_BASE = 100
TRANSPORT_COST_PER_UNIT_LOAD_PER_KM = 0.0001
TRANSPORT_COST_PER_KM = 0.5


STORE_DISTANCES_FROM_DC = {
    store_id: dist
    for store_id, dist in zip(sales_df["Store"].unique(), clipped_distances.tolist())
}
AVG_FUEL_PRICE = sales_df["Fuel_Price"].mean()

# Compute weekly customer arrival intervals
sales_df["Weekly_Units"] = sales_df["Weekly_Sales"] / 100
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
        self.inventory = min(int(self.capacity / 2), self.capacity)  # Make sure don't exceed capacity
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
        week_start = current_date - timedelta(days=current_date.weekday() - 4)
        # Use week_start directly
        weekly_units = units_lookup.get((self.store_id, week_start), 0)
        return 7 / weekly_units if weekly_units != 0 else 99999

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
        def __init__(self, env, stores, trucks, transport_type, order_policy, distances=STORE_DISTANCES_FROM_DC):
            self.env = env
            self.stores = stores
            self.trucks = trucks
            self.transport_type = transport_type  # Record the current transport level
            self.truck_speed = TRANSPORT_TIERS[transport_type]["truck_speed"]
            self.fuel_per_km = TRANSPORT_TIERS[transport_type]["fuel_per_km"]
            self.capacity = TRANSPORT_TIERS[transport_type]["delivery_quantity"]  # Get delivery volume from tier
            self.distances = distances
            self.total_transport_cost = 0
            self.order_policy = order_policy
            self.action = env.process(self.periodic_check_and_order())

        def check_model(self, store):
            current_date = START_DATE + timedelta(days=int(self.env.now))
            week_start = current_date - timedelta(days=current_date.weekday() - 4)
            distance = self.distances[store.store_id]
            weekly_units = self.get_weekly_units(store.store_id, week_start)
            hourly_units = weekly_units / (7 * 24)
            delivery_travel_time = distance / self.truck_speed
            safety_stock = hourly_units * (24 + delivery_travel_time)  # Safety stock = hourly demand × (lead time + shipping time)
            return store.inventory < safety_stock  # Determine whether it is below safety stock


        def get_fuel_price(self, store, current_date):
            week_start = current_date - timedelta(days=current_date.weekday() - 4)
            return fuel_lookup.get((store, week_start), 99999)

        def get_weekly_units(self, store, current_date):
            week_start = current_date - timedelta(days=current_date.weekday()-4)
            return units_lookup.get((store, week_start), 99999)


        def _execute_delivery(self, store):
             with self.trucks.request() as request:
                yield request
                current_date = START_DATE + timedelta(days=int(self.env.now))
                week_start = current_date - timedelta(days=current_date.weekday() - 4)
                distance = self.distances[store.store_id]
                fuel_price = fuel_lookup.get((store.store_id, week_start), AVG_FUEL_PRICE)
                # Use the current level's delivery capacity (self.capacity) and
                # the warehouse's remaining capacity to determine the actual delivery quantity
                delivery_quantity = min(self.capacity, store.capacity - store.inventory)
                delivery_travel_time = np.random.normal(
                    loc=distance / self.truck_speed,
                    scale=DELIVERY_TIME_STD
                )
                delivery_travel_time = max(0, delivery_travel_time)
                # Shipping cost calculation
                transport_cost = (
                    TRANSPORT_COST_PER_DELIVERY_BASE +
                    distance * (fuel_price * self.fuel_per_km + delivery_quantity * TRANSPORT_COST_PER_UNIT_LOAD_PER_KM)
                )

             yield self.env.timeout(delivery_travel_time)  
             store.inventory += delivery_quantity
             store.total_transport_cost += transport_cost
             self.total_transport_cost += transport_cost

        def periodic_check_and_order(self):
            while True:
                yield self.env.timeout(7)  # Change to weekly inspection
                print(f"[{self.env.now:.2f}]: DC performing weekly stock check for all stores.")

                for store in self.stores:
                    if self.order_policy == "model" and self.check_model(store):
                        self.env.process(self._execute_delivery(store))


# -------------------- Simulation Runner --------------------
def simulate(store_configs, transport_type, order_policy):
    env = simpy.Environment()
    stores = [Store(env, store_id, storage_type) for store_id, storage_type in store_configs]
    trucks = simpy.Resource(env, capacity=TRANSPORT_TIERS[transport_type]["trucks"])
    dc = DistributionCenter(env, stores=stores, trucks=trucks, transport_type=transport_type, order_policy=order_policy)
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

simulate([
    ("1", "large"),
    ("2", "medium"),
    ("3", "small")
], "high", "model")

