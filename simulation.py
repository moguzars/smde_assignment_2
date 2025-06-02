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
UNIT_SALE_PRICE = 1
SIM_DAYS = 31  # For full simulation it is 1000 days
STORAGE_COST_PER_UNIT_PER_DAY = 0.001
TRANSPORT_COST_PER_DELIVERY_BASE = 100
TRANSPORT_COST_PER_UNIT_LOAD_PER_KM = 0.00001
DELIVERY_CHECK_INTERVAL = 1


# Define storage tiers
STORAGE_TIERS = {
    "low":  {"capacity": 10000, "monthly_rent":  5000},
    "medium": {"capacity": 20000, "monthly_rent":  9000},
    "high":  {"capacity": 40000, "monthly_rent": 17000},
}


TRUCK_CAPACITIES = {
    "low" : {"delivery_quantity": 2000, "monthly_rent": 1000, "truck_gas_consumption": 1 },
    "high": {"delivery_quantity": 4000, "monthly_rent": 1800, "truck_gas_consumption": 1.7}
}

TRUCK_NUMBERS = {
    "low" : {"trucks": 30},
    "high": {"trucks": 50}
}



# Distances from distribution center (DC) for each store
np.random.seed(42)
num_elements = 45
min_km = 50
max_km = 2000
mean = (min_km + max_km) / 2
std_dev = (max_km - min_km) / 6
raw_distances = np.random.normal(loc=mean, scale=std_dev, size=num_elements)
clipped_distances = np.clip(raw_distances, min_km, max_km)
STORE_DISTANCES_FROM_DC = {
    store_id: dist
    for store_id, dist in zip(sales_df["Store"].unique(), clipped_distances.tolist())
}

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
                if sim_day % 7 == 0:
                    arrival_interval = self.get_arrival_interval(current_date)

            if sim_day % 7 == 0 and sim_day != self.last_day_checked:
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
    def __init__(self, env, stores, trucks, capacities, distances=STORE_DISTANCES_FROM_DC):
        self.env = env
        self.stores = stores
        self.trucks = trucks
        self.truck_speed = 60
        self.fuel_per_km = TRUCK_CAPACITIES[capacities]["truck_gas_consumption"]
        self.truck_capacity = TRUCK_CAPACITIES[capacities]["delivery_quantity"]
        self.distances = distances
        self.total_transport_cost = 0
        self.action = env.process(self.periodic_check_and_order())
        self.total_storage_cost = 0
        
    def check_if_stock_below_reorder_point(self, store):
        return store.inventory <=0

    def get_fuel_price(self, store, current_date):
        week_start = current_date - timedelta(days=current_date.weekday()-4)
        return arrival_lookup.get((store, week_start), 99999)
    
    def get_weekly_units(self, store, current_date):
        week_start = current_date - timedelta(days=current_date.weekday()-4)
        return units_lookup.get((store, week_start), 99999)


    def _execute_delivery(self, store):
        if (self.trucks.count >= self.trucks.capacity):
            # print(f"[{self.env.now:.2f}]: No available trucks for store {store.store_id}")
            return
        
        with self.trucks.request() as request:
            yield request
            distance = self.distances.get(store.store_id, 0)
            delivery_travel_time = distance/self.truck_speed
            # print(f"[{self.env.now:.2f}]: Truck acquired for {store.name}. Beginning delivery for time {delivery_travel_time}")
            
            yield self.env.timeout(delivery_travel_time/24)
            
            current_date = START_DATE + timedelta(days=int(self.env.now))
            fuel_price_per_km = self.get_fuel_price(store.store_id, current_date)/10
            transport_cost = (TRANSPORT_COST_PER_DELIVERY_BASE +
                              (distance * (fuel_price_per_km * self.fuel_per_km) +
                              (distance/2) * (self.truck_capacity * TRANSPORT_COST_PER_UNIT_LOAD_PER_KM )))
            
            
            store.inventory += self.truck_capacity
            store.total_transport_cost += transport_cost
            self.total_transport_cost += transport_cost
            
            yield self.env.timeout(delivery_travel_time/24)

    def periodic_check_and_order(self):
        while True:
            yield self.env.timeout(1)
            sim_day = int(self.env.now)

            # print(f"[{self.env.now:.2f}]: DC performing periodic stock check for all stores.")

            store_metrics = []
            for store in self.stores:
                if (store.capacity - store.inventory < self.truck_capacity):
                    continue
                store_metrics.append((store, store.inventory))
                # if self.order_policy == "distance":
                #     store_metrics.append((store, self.distances[store.store_id]))
                # if self.order_policy == "predicted_sales":

            sorted_by_second=sorted(store_metrics, key=lambda tup: tup[1])
            for store_id, _ in sorted_by_second:
                self.env.process(self._execute_delivery(store_id))

            for store in self.stores:
                self.total_storage_cost += store.inventory * STORAGE_COST_PER_UNIT_PER_DAY

            

# -------------------- Simulation Runner --------------------
def simulate(store_type, truck_capacities, truck_numbers):
    env = simpy.Environment()
    stores = [Store(env, store_id, store_type) for store_id in range(1,46)]
    trucks = simpy.Resource(env, capacity=TRUCK_NUMBERS[truck_numbers]["trucks"])
    dc = DistributionCenter(env, stores=stores, trucks=trucks, capacities=truck_capacities) 
    env.run(until=SIM_DAYS)
    

    # Reporting
    total_revenue = sum(s.total_revenue for s in stores)
    total_transport = dc.total_transport_cost
    total_storage = dc.total_storage_cost
    store_rent = (len(stores) * STORAGE_TIERS[store_type]["monthly_rent"] * SIM_DAYS)/30
    truck_rent = (TRUCK_NUMBERS[truck_numbers]["trucks"] * TRUCK_CAPACITIES[truck_capacities]["monthly_rent"] * SIM_DAYS) / 30
    total_rent = store_rent + truck_rent
    total_stockouts = sum(s.total_stockouts for s in stores)
    total_lost_profit = sum(s.lost_profit for s in stores)

    final_profit = total_revenue - total_transport - total_storage - total_rent
    net_profit   = final_profit - total_lost_profit

    # print(f"--- Simulation Results after {SIM_DAYS} days ---")
    # for s in stores:
    #     print(f"{s.name} ({s.storage_type}) | Sales: {s.total_sales}, "
    #           f"Revenue: {s.total_revenue:.2f}, Stockouts: {s.total_stockouts}, "
    #           f"Lost Profit: {s.lost_profit:.2f}, Transport: {s.total_transport_cost}, "
    #           f"Storage Cost: {s.total_storage_cost:.2f}, Rent Paid: {s.total_rent_paid}")

    print(f"\nTOTAL Revenue: €{total_revenue:.2f}")
    print(f"TOTAL Transport Cost: €{total_transport:.2f}")
    print(f"TOTAL Storage Cost: €{total_storage:.2f}")
    print(f"TOTAL Rent Paid: €{total_rent:.2f}")
    print(f"TOTAL Stockouts: {total_stockouts}")
    print(f"TOTAL Lost Profit: €{total_lost_profit:.2f}")
    print(f"FINAL Profit: €{final_profit:.2f}")
    print(f"NET Profit: €{net_profit:.2f}")

total_sales_list = {
    'Store1':[],
    'Store2':[],
    'Store3':[],
}
# Example usage: (store_id, storage_type)
simulate(
    "low", #store capacity
    "high",   #truck capacities
    "low",   #truck numbers
)
