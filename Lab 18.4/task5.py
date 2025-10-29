import requests
import json
import os

def get_weather_by_city(city_name):
    api_key = input("Enter your OpenWeatherMap API key: ")

    base_url = 'https://api.openweathermap.org/data/2.5/weather'
    params = {
        'q': city_name,
        'appid': api_key,
        'units': 'metric'
    }

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("cod") == "404":
            print("Error: City not found. Please enter a valid city.")
            return

        summary = {
            "city": data['name'],
            "temp": data['main']['temp'],
            "humidity": data['main']['humidity'],
            "weather": data['weather'][0]['description'].capitalize()
        }

        # Display formatted output
        print("\nWeather Report:")
        for key, value in summary.items():
            print(f"{key.capitalize()}: {value}")

        # Append to results.json
        filename = 'results.json'
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                existing_data = json.load(f)
        else:
            existing_data = []

        existing_data.append(summary)

        with open(filename, 'w') as f:
            json.dump(existing_data, f, indent=4)

        print(f"\nWeather data appended to '{filename}'")

    except requests.exceptions.Timeout:
        print("Error: Request timed out. Please check your network connection.")
    except requests.exceptions.HTTPError:
        print("Error: Could not connect to API. Check your API key or city name.")
    except requests.exceptions.RequestException:
        print("Error: Could not connect to API. Check your API key or network connection.")

# Example usage
get_weather_by_city("London")