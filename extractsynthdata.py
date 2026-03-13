import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def generate_synthetic_data(days=180, seed=42):
    """
    Generate synthetic energy consumption data for four devices:
      - energy_oven: simulating a triangular pulse (peak between 1200 and 3000)
      - energy_dish_washer: simulating a half-sinusoid (peak between 1000 and 3200)
      - energy_washing_machine: simulating a rectangular (step) pulse (peak between 1500 and 4000)
      - energy_fridge_freezer: a continuously active signal that now produces a noise-like signal
                                with abrupt changes (e.g., from 10 to 100 or 400 to 200) instead of a fixed cycle.
    Each non-fridge device produces roughly 4 cycles per day at random times (with no overlapping
    cycles for that device in a given day).
    The energy_mains is computed as the sum of all devices.
    """
    np.random.seed(seed)
    minutes_per_day = 1440
    total_minutes = days * minutes_per_day
    datetimes = pd.date_range(start="2024-01-01", periods=total_minutes, freq='T')
    
    # Preallocate arrays for each device:
    energy_dish_washer       = np.zeros(total_minutes)
    energy_washing_machine   = np.zeros(total_minutes)
    energy_oven              = np.zeros(total_minutes)
    energy_fridge_freezer    = np.zeros(total_minutes)
    
    # --- Helper functions ---
    def get_non_overlapping_intervals(n_intervals, day_length, min_duration, max_duration):
        """
        For a given day (day_length minutes) return a list of n_intervals tuples:
          (start, end, duration)
        ensuring that cycles do not overlap.
        """
        intervals = []
        attempts = 0
        while len(intervals) < n_intervals and attempts < 1000:
            duration = np.random.randint(min_duration, max_duration + 1)
            start = np.random.randint(0, day_length - duration)
            end = start + duration
            if any(not (end <= s or start >= e) for (s, e, _) in intervals):
                attempts += 1
                continue
            intervals.append((start, end, duration))
            attempts += 1
        intervals.sort(key=lambda x: x[0])
        return intervals

    def generate_triangular_pattern(duration):
        """Generate a triangular pulse with a random peak between 1200 and 3000."""
        peak = np.random.uniform(1200, 3000)
        half = int(np.ceil(duration / 2))
        asc = np.linspace(0, peak, half, endpoint=True)
        desc = np.linspace(peak, 0, duration - half, endpoint=True) if duration - half > 0 else np.array([])
        return np.concatenate([asc, desc])
    
    def generate_dishwasher_pattern(duration):
        """Generate a half-sinusoid pulse (0 to pi) with a random peak between 1000 and 3200."""
        peak = np.random.uniform(1000, 3200)
        return np.sin(np.linspace(0, np.pi, duration)) * peak
    
    def generate_washing_machine_pattern(duration):
        """
        Generate a rectangular pulse.
        The consumption “jumps” from 0 to a random peak (between 1500 and 4000) and stays constant,
        then drops back to 0 when the cycle ends.
        """
        peak = np.random.uniform(1500, 4000)
        return np.full(duration, peak)
    
    # --- Generate daily data ---
    for day in range(days):
        day_start = day * minutes_per_day
        day_end = day_start + minutes_per_day
        
        # For each non-fridge device, choose 4 non-overlapping cycles per day.
        intervals_triangular = get_non_overlapping_intervals(4, minutes_per_day, 30, 120)
        intervals_dishwasher  = get_non_overlapping_intervals(4, minutes_per_day, 30, 120)
        intervals_washing     = get_non_overlapping_intervals(4, minutes_per_day, 30, 120)
        
        # Insert the triangular cycles:
        for (start, end, duration) in intervals_triangular:
            pattern = generate_triangular_pattern(duration)
            energy_oven[day_start + start: day_start + end] = pattern
        
        # Insert the dish washer cycles (half-sine pulses):
        for (start, end, duration) in intervals_dishwasher:
            pattern = generate_dishwasher_pattern(duration)
            energy_dish_washer[day_start + start: day_start + end] = pattern
        
        # Insert the washing machine cycles (rectangular pulses):
        for (start, end, duration) in intervals_washing:
            pattern = generate_washing_machine_pattern(duration)
            energy_washing_machine[day_start + start: day_start + end] = pattern
        
        # Για το ψυγείο/καταψύκτη: δημιουργούμε ένα σήμα με απότομες αλλαγές τιμών.
        # Διαιρούμε την ημέρα σε διαστήματα (π.χ. κάθε 10 λεπτά) και για κάθε διάστημα επιλέγουμε
        # είτε μια χαμηλή τιμή (από 10 έως 100) είτε μια υψηλή τιμή (από 200 έως 400).
        segment_length = 10  # λεπτά
        n_segments = minutes_per_day // segment_length
        low_segment_values = np.random.uniform(10, 100, size=n_segments)
        high_segment_values = np.random.uniform(200, 400, size=n_segments)
        choices = np.random.choice([0, 1], size=n_segments, p=[0.5, 0.5])
        segment_values = np.where(choices == 0, low_segment_values, high_segment_values)
        fridge_pattern = np.repeat(segment_values, segment_length)
        remainder = minutes_per_day - n_segments * segment_length
        if remainder > 0:
            extra_value = np.random.uniform(10, 400, size=1)
            extra_pattern = np.repeat(extra_value, remainder)
            fridge_pattern = np.concatenate([fridge_pattern, extra_pattern])
        # Προσθέτουμε λίγο ελαφρύ θόρυβο για ομαλοποίηση των μεταβάσεων (προαιρετικό)
        fridge_pattern += np.random.normal(0, 5, size=minutes_per_day)
        fridge_pattern = np.clip(fridge_pattern, 0, None)
        energy_fridge_freezer[day_start: day_end] = fridge_pattern

    # Compute the mains consumption as the sum of all devices:
    energy_mains = (energy_dish_washer + energy_washing_machine +
                    energy_oven + energy_fridge_freezer)
    
    # Assemble the full DataFrame:
    df = pd.DataFrame({
        'datetime': datetimes,
        'energy_mains': energy_mains,
        'energy_dish_washer': energy_dish_washer,
        'energy_oven': energy_oven,
        'energy_washing_machine': energy_washing_machine,
        'energy_fridge_freezer': energy_fridge_freezer
    })
    
    return df

def save_synthetic_datasets(base_path="NILMv2/DATA/SyntheticData/data", days=180):
    """
    Generates the synthetic dataset and splits it into train (60%), validation (20%) and test (20%)
    CSV files. A plot (with all features) is saved for each dataset.
    """
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    df = generate_synthetic_data(days=days, seed=42)
    total_records = len(df)
    train_size = int(0.6 * total_records)
    val_size = int(0.2 * total_records)
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]
    
    train_df.to_csv(os.path.join(base_path, "train.csv"), index=False)
    val_df.to_csv(os.path.join(base_path, "validation.csv"), index=False)
    test_df.to_csv(os.path.join(base_path, "test.csv"), index=False)
    
    print(f"Saved train dataset with {len(train_df)} records (~{len(train_df) / 1440:.2f} days).")
    print(f"Saved validation dataset with {len(val_df)} records (~{len(val_df) / 1440:.2f} days).")
    print(f"Saved test dataset with {len(test_df)} records (~{len(test_df) / 1440:.2f} days).")
    
    # Plot each dataset (using all features except datetime):
    for name, data in zip(["Train", "Validation", "Test"], [train_df, val_df, test_df]):
        plt.figure(figsize=(10, 5))
        for column in data.columns[1:]:
            plt.plot(data['datetime'], data[column], label=column)
        plt.xlabel('Datetime')
        plt.ylabel('Energy Consumption')
        plt.title(f'{name} Dataset Features')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(base_path, f"{name.lower()}_plot.png"))
        plt.close()
        print(f"Saved {name} dataset plot with all features.")

if __name__ == "__main__":
    save_synthetic_datasets()
