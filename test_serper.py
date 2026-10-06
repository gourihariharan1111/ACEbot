from dotenv import load_dotenv
import os
import requests
import json

load_dotenv()

serper_key = os.environ.get("SERPER_API_KEY")

url = "https://google.serper.dev/search"

payload = json.dumps({
    "q": 'site:www-old.cev.eu/Competition-Area/competition.aspx "EuroVolley 2026 Women"'
})

headers = {
    "X-API-KEY": serper_key,
    "Content-Type": "application/json"
}

response = requests.post(
    url,
    headers=headers,
    data=payload
)

print(response.status_code)
print(response.text)