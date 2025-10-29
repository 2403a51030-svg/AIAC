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
        data = response.json()

        summary = {
            "City": data['name'],
            "Temperature": f"{data['main']['temp']}°C",
            "Humidity": f"{data['main']['humidity']}%",
            "Weather": data['weather'][0]['description'].capitalize()
        }

        # Print to console
        print("\nWeather Report:")
        for key, value in summary.items():
            print(f"{key}: {value}")

        # Save to JSON file
        with open('weather_summary.json', 'w') as f:
            json.dump(summary, f, indent=4)
        print("\nSummary saved to 'weather_summary.json'")

    except requests.exceptions.Timeout:
        print("Error: Request timed out. Please check your network connection.")
    except requests.exceptions.HTTPError:
        print("Error: Could not connect to API. Check your API key or city name.")
    except requests.exceptions.RequestException:
        print("Error: Could not connect to API. Check your API key or network connection.")

# Run the function
get_weather()