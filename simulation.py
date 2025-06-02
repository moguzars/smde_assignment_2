import simpy
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# -------------------- Load Sales Data --------------------
sales_df = pd.read_csv("Walmart_Sales.csv")
sales_df["Date"] = pd.to_datetime(sales_df["Date"], format="%d-%m-%Y")
sales_df["Store"] = sales_df["Store"].astype(str)

# -------------------- MLP coefficientsicients from R --------------------
coefficients = {
    'intercept': 6.210e+05,
    'Holiday_Flag': 1.803e+04,
    'Temperature': -455.2,
    'Fuel_Price': -27860.0,
    'CPI': 2804.0,
    'Unemployment': -18150.0,
    'Last_Week_Sales': 0.378,

    # Store-specific coefficientsicients
    'Store2': 234600.0, 'Store3': -732800.0, 'Store4': 546000.0, 'Store5': -794300.0,
    'Store6': -11220.0, 'Store7': -544600.0, 'Store8': -442400.0, 'Store9': -666400.0,
    'Store10': 491500.0, 'Store11': -137600.0, 'Store12': 20780.0, 'Store13': 506900.0,
    'Store14': 411100.0, 'Store15': -342800.0, 'Store16': -613300.0, 'Store17': -198100.0,
    'Store18': -44100.0, 'Store19': 168400.0, 'Store20': 357500.0, 'Store21': -490200.0,
    'Store22': -102600.0, 'Store23': 66870.0, 'Store24': 119100.0, 'Store25': -514900.0,
    'Store26': -120600.0, 'Store27': 368600.0, 'Store28': 217000.0, 'Store29': -364700.0,
    'Store30': -691900.0, 'Store31': -94760.0, 'Store32': -169300.0, 'Store33': -530800.0,
    'Store34': -83710.0, 'Store35': -158800.0, 'Store36': -719700.0, 'Store37': -633800.0,
    'Store38': -371900.0, 'Store39': -62090.0, 'Store40': -197300.0, 'Store41': -143600.0,
    'Store42': -350800.0, 'Store43': -506000.0, 'Store44': -557300.0, 'Store45': -374200.0
}


START_DATE = datetime.strptime("05-02-2010", "%d-%m-%Y").date()

# Compute weekly customer arrival intervals
sales_df["Weekly_Units"] = sales_df["Weekly_Sales"] / 100
sales_df["Arrival_Interval"] = 7 / sales_df["Weekly_Units"]

# Create lookup dict: (store, week_start) -> arrival_interval
arrival_lookup = {
    (row["Store"], row["Date"].date()): row["Arrival_Interval"]
    for _, row in sales_df.iterrows()
}

fuel_lookup = {
    (row["Store"], row["Date"].date()): row["Fuel_Price"]
    for _, row in sales_df.iterrows()
}

units_lookup = {
    (row["Store"], row["Date"].date()): row["Weekly_Units"]
    for _, row in sales_df.iterrows()
}

# -------------------- Configuration --------------------
UNIT_SALE_PRICE = 100
DELIVERY_INTERVAL = 1
SIM_DAYS = 100  # For full simulation it is 1000 days
STORAGE_COST_PER_UNIT_PER_DAY = 1

# Define storage tiers
STORAGE_TIERS = {
    "small":  {"capacity": 10000, "monthly_rent": 5000},
    "medium": {"capacity": 20000, "monthly_rent": 10000},
    "large":  {"capacity": 40000, "monthly_rent": 20000},
}

TRANSPORT_TIERS = {
    "low": {"trucks": 45, "truck_speed": 50, "fuel_per_km": 0.06, "delivery_quantity": 20},
    "high": {"trucks": 90, "truck_speed": 60, "fuel_per_km": 0.05, "delivery_quantity": 40}
}


# Distances from distribution center (DC) for each store
np.random.seed(42)
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
NUM_TRUCKS = 2
DELIVERY_TIME_MEAN = 1.5
DELIVERY_TIME_STD = 0.3
REORDER_POINT = 5000
TRANSPORT_COST_PER_DELIVERY_BASE = 100
TRANSPORT_COST_PER_UNIT_LOAD_PER_KM = 0.0001
TRANSPORT_COST_PER_KM = 0.5
DELIVERY_CHECK_INTERVAL = 1

STORE_DISTANCES_FROM_DC = {
    store_id: dist
    for store_id, dist in zip(sales_df["Store"].unique(), clipped_distances.tolist())
}
AVG_FUEL_PRICE = sales_df["Fuel_Price"].mean()

def predict_weekly_sales(store_id, holiday_flag, temperature, fuel_price, cpi, unemployment, last_week_sales):
    intercept = coefficients['intercept']
    store_coef = coefficients.get(f'Store{store_id}', 0)
    return intercept + store_coef + \
           coefficients['Holiday_Flag'] * holiday_flag + \
           coefficients['Temperature'] * temperature + \
           coefficients['Fuel_Price'] * fuel_price + \
           coefficients['CPI'] * cpi + \
           coefficients['Unemployment'] * unemployment + \
           coefficients['Last_Week_Sales'] * last_week_sales

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
        self.last_week_predicted_sales = None  # for using in subsequent predictions
        self.action = env.process(self.run())

    def get_arrival_interval(self, current_date):
        week_start = current_date - timedelta(days=current_date.weekday() - 4)
        try:
            features = sales_df[
                (sales_df['Store'] == self.store_id) &
                (sales_df['Date'].dt.date == week_start)
                ].iloc[0]
            if self.last_week_predicted_sales is None:
                self.last_week_predicted_sales = features['Weekly_Sales']

            predicted_sales = predict_weekly_sales(
                store_id=int(self.store_id),
                holiday_flag=features['Holiday_Flag'],
                temperature=features['Temperature'],
                fuel_price=features['Fuel_Price'],
                cpi=features['CPI'],
                unemployment=features['Unemployment'],
                last_week_sales=self.last_week_predicted_sales
            )
            units = predicted_sales / 100
            if units > 0:
                return 7 / units
        except Exception as e:
            print(e)
            pass
        return 99999  # fallback in case of missing data

    def run(self):
        arrival_interval = self.get_arrival_interval(START_DATE)
        while True:
            sim_day = int(self.env.now)
            current_date = START_DATE + timedelta(days=sim_day)

            # Track storage and rent costs daily
            if sim_day != self.last_day_checked:
                self.last_day_checked = sim_day
                self.total_storage_cost += self.inventory * STORAGE_COST_PER_UNIT_PER_DAY
                if sim_day % 30 == 0:
                    self.total_rent_paid += self.monthly_rent
                total_sales_list[f'Store{self.store_id}'].append(self.total_stockouts)
                # Get customer arrival interval for this date
                if sim_day % 7 == 0:
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
        self.truck_speed = TRANSPORT_TIERS[transport_type]["truck_speed"]
        self.fuel_per_km = TRANSPORT_TIERS[transport_type]["fuel_per_km"]
        self.capacity = TRANSPORT_TIERS[transport_type]["delivery_quantity"]
        self.distances = distances
        self.total_transport_cost = 0
        self.order_policy = order_policy
        self.action = env.process(self.periodic_check_and_order())
        
    def check_if_stock_below_reorder_point(self, store):
        return store.inventory <=0

    # def check_model(self, store):
    #     current_day = int(self.env.now)
    #     current_date = START_DATE + timedelta(days=current_day)
    #     week_start = current_date - timedelta(days=current_date.weekday() - 4)
    #
    #     # Lookup inputs
    #     key = (store.store_id, week_start)
    #     holiday = 1 if key in arrival_lookup and arrival_lookup[key] < 99999 else 0
    #     fuel_price = fuel_lookup.get(key, AVG_FUEL_PRICE)
    #     last_week_sales = units_lookup.get(key, 0) * 100  # convert back to € to match model input
    #     cpi = sales_df.loc[
    #         (sales_df["Store"] == store.store_id) & (sales_df["Date"].dt.date == week_start), "CPI"].mean()
    #     unemployment = sales_df.loc[
    #         (sales_df["Store"] == store.store_id) & (sales_df["Date"].dt.date == week_start), "Unemployment"].mean()
    #     temperature = sales_df.loc[
    #         (sales_df["Store"] == store.store_id) & (sales_df["Date"].dt.date == week_start), "Temperature"].mean()
    #
    #     # Fallbacks
    #     cpi = cpi if not pd.isna(cpi) else sales_df["CPI"].mean()
    #     unemployment = unemployment if not pd.isna(unemployment) else sales_df["Unemployment"].mean()
    #     temperature = temperature if not pd.isna(temperature) else sales_df["Temperature"].mean()
    #
    #     # Linear regression formula (MLP)
    #     store_id = int(store.store_id)
    #     y_pred = coefficientsicients["intercept"]
    #     y_pred += coefficientsicients.get(f"Store{store_id}", 0)
    #     y_pred += coefficientsicients["Holiday_Flag"] * holiday
    #     y_pred += coefficientsicients["Temperature"] * temperature
    #     y_pred += coefficientsicients["Fuel_Price"] * fuel_price
    #     y_pred += coefficientsicients["CPI"] * cpi
    #     y_pred += coefficientsicients["Unemployment"] * unemployment
    #     y_pred += coefficientsicients["Last_Week_Sales"] * last_week_sales
    #
    #     weekly_predicted_units = y_pred / 100  # Convert € back to unit demand
    #     hourly_demand = (weekly_predicted_units / 7) / 24
    #
    #     # Lead time buffer: 1 day + travel time
    #     distance = self.distances.get(store.store_id, 0)
    #     delivery_lead_time = 24 + (distance / self.truck_speed)
    #     print(store.inventory)
    #     # Will they run out before new stock can arrive?
    #     threshold = hourly_demand * delivery_lead_time
    #     return store.inventory < threshold

    def get_fuel_price(self, store, current_date):
        week_start = current_date - timedelta(days=current_date.weekday()-4)
        return arrival_lookup.get((store, week_start), 99999)
    
    def get_weekly_units(self, store, current_date):
        week_start = current_date - timedelta(days=current_date.weekday()-4)
        return units_lookup.get((store, week_start), 99999)


    def _execute_delivery(self, store):
        with self.trucks.request() as request:
            yield request
            print(f"[{self.env.now:.2f}]: Truck acquired for {store.name}. Beginning delivery...")
            distance = self.distances.get(store.store_id, 0)
            delivery_travel_time = distance/self.truck_speed
            
            yield self.env.timeout(delivery_travel_time/24)
            
            current_date = START_DATE + timedelta(days=int(self.env.now))
            fuel_price = self.get_fuel_price(store.store_id, current_date)
            delivery_quantity = min(DELIVERY_QUANTITY, store.capacity - store.inventory) # this logic isnt completely sensical either, also some calculation needed probably.
            transport_cost = (TRANSPORT_COST_PER_DELIVERY_BASE +
                              (distance * (fuel_price * self.fuel_per_km 
                                           + delivery_quantity * TRANSPORT_COST_PER_UNIT_LOAD_PER_KM )))
            
            
            store.inventory += delivery_quantity
            store.total_transport_cost += transport_cost
            self.total_transport_cost += transport_cost
            
            # print(f"[{self.env.now:.2f}]: Delivery completed for {store.name}. New inventory: {store.inventory}. Cost: {transport_cost:.2f}")
            yield self.env.timeout(delivery_travel_time/24)
            # print(f"[{self.env.now:.2f}]: Truck returned to DC from {store.name}.")

    def periodic_check_and_order(self):
        while True:
            yield self.env.timeout(DELIVERY_CHECK_INTERVAL)
            print(f"[{self.env.now:.2f}]: DC performing periodic stock check for all stores.")

            for store in self.stores:
                if self.order_policy == "half" and self.check_if_stock_below_reorder_point(store):
                    self.env.process(self._execute_delivery(store))
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

total_sales_list = {
    'Store1':[],
    'Store2':[],
    'Store3':[],
}
# Example usage: (store_id, storage_type)
simulate([
        ("1", "large"),
        ("2", "large"),
        ("3", "large"),
],
    "low",
    "half"
)
