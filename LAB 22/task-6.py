import sys

#!/usr/bin/env python3
"""
task-6.py

Interactive helper to estimate GPU power consumption, energy cost, and CO2 emissions
for cloud server GPUs. Prompts user for details but accepts common GPU presets.

Usage: run the script and follow prompts.
"""


GPU_PRESETS = {
    "a100": 400,   # watts (approx)
    "v100": 300,
    "t4": 70,
    "p4": 75,
    "rtx3090": 350,
    "rtx4090": 450,
    "custom": None
}

def ask(prompt, cast=float, default=None):
    while True:
        raw = input(f"{prompt}" + (f" [{default}]" if default is not None else "") + ": ").strip()
        if raw == "" and default is not None:
            return default
        try:
            return cast(raw)
        except Exception:
            print("Invalid input — try again.")

def choose_gpu():
    print("Select GPU preset or type 'custom' to enter watts manually:")
    for k in GPU_PRESETS:
        print(" -", k, ("(" + str(GPU_PRESETS[k]) + " W)" ) if GPU_PRESETS[k] else "")
    while True:
        choice = input("GPU model: ").strip().lower()
        if choice in GPU_PRESETS:
            if GPU_PRESETS[choice] is None:
                watts = ask("Enter GPU power draw (watts, typical TDP)", float)
                return watts, choice
            return GPU_PRESETS[choice], choice
        # allow direct numeric entry
        try:
            watts = float(choice)
            return watts, "custom"
        except Exception:
            print("Unknown GPU model. Choose a preset or enter a numeric watts value.")

def parse_number(raw, cast=float, default=None):
    if raw is None:
        return default
    raw = str(raw).replace(",", "").strip()
    try:
        return cast(raw)
    except Exception:
        return default

def format_kwh(x):
    return f"{x:,.3f} kWh"

def format_money(x, currency="$"):
    return f"{currency}{x:,.2f}"

def main():
    print("GPU Cloud Consumption Estimator\n")
    watts, gpu_name = choose_gpu()
    num_gpus = ask("Number of GPUs", int, 1)
    hours = ask("Total hours of runtime", float, 1.0)
    utilization_pct = ask("Average GPU utilization (0-100%)", float, 50.0)
    utilization = max(0.0, min(utilization_pct / 100.0, 1.0))
    price_per_gpu_hour = ask("Cloud cost per GPU-hour (enter 0 if unknown)", float, 0.0)
    electricity_cost = ask("Electricity cost per kWh (local) in your currency", float, 0.13)
    carbon_intensity = ask("Grid carbon intensity (gCO2 per kWh). Use 0 if unknown", float, 475.0)

    # Calculations
    # kWh = (W * utilization * hours) / 1000 per GPU
    kwh_per_gpu = watts * utilization * hours / 1000.0
    total_kwh = kwh_per_gpu * num_gpus
    energy_cost = total_kwh * electricity_cost
    cloud_compute_cost = price_per_gpu_hour * hours * num_gpus
    total_cost = energy_cost + cloud_compute_cost
    co2_g = total_kwh * carbon_intensity
    co2_kg = co2_g / 1000.0

    print("\n--- Estimation Results ---")
    print(f"GPU preset: {gpu_name} ({watts} W)")
    print(f"GPUs: {num_gpus}, Hours: {hours}, Utilization: {utilization_pct}%")
    print(f"Energy per GPU: {format_kwh(kwh_per_gpu)}")
    print(f"Total energy: {format_kwh(total_kwh)}")
    print(f"Estimated electricity cost: {format_money(energy_cost)}")
    print(f"Estimated cloud compute cost: {format_money(cloud_compute_cost)}")
    print(f"Estimated total cost: {format_money(total_cost)}")
    print(f"Estimated CO2 emissions: {co2_kg:,.3f} kg CO2 ({co2_g:,.0f} gCO2)")

    # Simple suggestions
    print("\nSuggestions to reduce cost & emissions:")
    if utilization_pct < 10:
        print(" - GPU utilization low: consider reducing instance size or batching jobs.")
    if price_per_gpu_hour > 0 and price_per_gpu_hour > 5.0:
        print(" - Cloud GPU cost is high: check for spot/preemptible instances or committed discounts.")
    if watts > 250:
        print(" - High-power GPU: consider newer efficient models if possible.")
    print(" - Use region with cleaner grid or renewable-backed instances to lower CO2.")
    print("\nDone.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(1)