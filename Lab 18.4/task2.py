import requests
import json

def get_weather():
    api_key = input("Enter your OpenWeatherMap API key: ")
    city_name = input("Enter city name: ")

    base_url = 'https://api.openweathermap.org/data/2.5/weather'
    params = {
        'q': city_name,
        'appid': api_key,
        'units': 'metric'
    }

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        weather_data = response.json()

        # Print to console
        print("\nWeather Data:")
        print(json.dumps(weather_data, indent=4))

        # Save to JSON file
        with open('weather_output.json', 'w') as f:
            json.dump(weather_data, f, indent=4)
        print("\nWeather data saved to 'weather_output.json'")

    except requests.exceptions.Timeout:
        print("Error: Request timed out. Please check your network connection.")
    except requests.exceptions.HTTPError:
        print("Error: Could not connect to API. Check your API key or city name.")
    except requests.exceptions.RequestException:
        print("Error: Could not connect to API. Check your API key or network connection.")

# Run the function
get_weather()