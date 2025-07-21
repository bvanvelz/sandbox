import requests


def send_request(endpoint, method='GET', params=None,
                 data=None, headers=None):
    try:
        response = requests.request(method, endpoint, params=params,
                                    data=data, headers=headers)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None


endpoint = "http://127.0.0.1:5000/api/v1/key_values"
response = send_request(endpoint)

if response is not None:
    print("API response:")
    print(response.json())
else:
    print("Failed to fetch data from the API.")
