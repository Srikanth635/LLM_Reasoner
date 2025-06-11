import requests

url = 'http://localhost:8081/generate'



data = {
    'instruction': "pour honey from the mug into sink"
}

response = requests.post(url, json=data)  # or data=data for form-data
print(response.text)