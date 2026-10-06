
import gradio as gr
from google import genai
from dotenv import load_dotenv
import os
import requests
import json
from bs4 import BeautifulSoup
from google.genai import types

load_dotenv()

gemini_key = os.environ.get("GEMINI_API_KEY")
serper_key = os.environ.get("SERPER_API_KEY")

if not gemini_key:
    raise ValueError("GEMINI_API_KEY is missing from your .env file.")

if not serper_key:
    raise ValueError("SERPER_API_KEY is missing from your .env file.")

client = genai.Client(api_key=gemini_key)

default_model = "gemini-3.8-flash"

def search_competition(competition_name):

    query = (
        f'site:www-old.cev.eu/Competition-Area/competition.aspx '
        f'"{competition_name}"'
    )

    search_url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query
    })

    headers = {
        "X-API-KEY": serper_key,
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            search_url,
            headers=headers,
            data=payload,
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

    except requests.RequestException as e:
        return None, f"Search request failed: {e}"

    except json.JSONDecodeError:
        return None, "Could not decode the search response."

    top_results = data.get("organic", [])[:5]

    if not top_results:
        return None, (
            f"Could not find a CEV competition page for "
            f"'{competition_name}'."
        )

    return top_results, None

def select_competition_url(competition_name, search_results):

    search_context = ""

    for i, result in enumerate(search_results):

        search_context += (
            f"Result {i + 1}:\n"
            f"Title: {result.get('title')}\n"
            f"Link: {result.get('link')}\n"
            f"Snippet: {result.get('snippet')}\n\n"
        )

    prompt = f"""
You are helping identify the correct CEV volleyball competition page.

The user searched for:

"{competition_name}"

Here are the search results:

{search_context}

Your task is to determine which result is the official CEV competition
page for the competition the user requested.

Rules:

1. If one result clearly corresponds to the requested competition,
return:

{{
    "status": "clear",
    "url": "the_correct_url"
}}

2. If multiple different competitions could reasonably match the search,
return:

{{
    "status": "ambiguous",
    "options": [
        "Competition name 1",
        "Competition name 2"
    ]
}}

3. If no result clearly matches, return:

{{
    "status": "not_found"
}}

Return ONLY valid JSON.
Do not include markdown.
Do not include any explanation.
"""

    try:

        response = client.models.generate_content(
            model=default_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        return json.loads(response.text)

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }

def get_competition_page_text(url):

    try:

        response = requests.get(
            url,
            timeout=15,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/120.0 Safari/537.36"
                )
            }
        )

        response.raise_for_status()

    except requests.RequestException as e:

        return None, f"Could not retrieve the CEV page: {e}"

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    # Remove elements that normally do not contain useful information.
    for element in soup([
        "script",
        "style",
        "noscript"
    ]):
        element.decompose()

    clean_text = soup.get_text(
        separator=" ",
        strip=True
    )

    if not clean_text:
        return None, "The CEV page contained no readable text."

    return clean_text, None


def extract_competition_details(
    competition_name,
    competition_text
):

    prompt = f"""
You are extracting structured information from an official CEV
volleyball competition webpage.

Competition requested:

"{competition_name}"

Below is the text extracted from the CEV competition page:

---------------- BEGIN DATA ----------------

{competition_text}

----------------- END DATA -----------------

Return ONLY a valid JSON object with exactly these keys:

{{
    "competition": "string",
    "final_standings": [
        {{
            "rank": integer,
            "country": "string"
        }}
    ],
    "dream_team": [
        {{
            "award": "string",
            "player": "string",
            "country": "string or null"
        }}
    ]
}}

IMPORTANT RULES:

1. final_standings must contain the final ranking of countries/teams
   if it is present in the data.

2. Use the actual final ranking, not group-stage rankings,
   semifinal rankings, or temporary standings.

3. dream_team must contain the players and their specific awards or
   positions when the Dream Team information is present.

4. Examples of Dream Team awards include:
   - Best Setter
   - Best Outside Spiker
   - Best Opposite
   - Best Middle Blocker
   - Best Libero
   - MVP

5. Do NOT guess information.

6. If final standings cannot be found, return:
   "final_standings": []

7. If the Dream Team cannot be found, return:
   "dream_team": []

8. If the player's country is not explicitly available, use null.

9. Preserve the spelling of names exactly as they appear in the source
   where possible.

10. Only extract information supported by the supplied webpage text.

Return ONLY JSON.
"""

    try:

        response = client.models.generate_content(
            model=default_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        return json.loads(response.text), None

    except json.JSONDecodeError:

        return None, "Gemini returned invalid JSON."

    except Exception as e:

        return None, f"Gemini extraction failed: {e}"

def save_competition_details(competition_data):

    filename = "competition_database.json"

    # Read existing database if it exists.
    if os.path.exists(filename):

        try:

            with open(
                filename,
                "r",
                encoding="utf-8"
            ) as file:

                database = json.load(file)

        except (json.JSONDecodeError, FileNotFoundError):

            database = []

    else:

        database = []

    # Avoid creating duplicate entries for the same competition.
    competition_name = competition_data.get(
        "competition"
    )

    database = [
        item
        for item in database
        if item.get("competition") != competition_name
    ]

    database.append(competition_data)

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            database,
            file,
            indent=4,
            ensure_ascii=False
        )

def get_competition_details(competition_name):

    if not competition_name or not competition_name.strip():

        return {
            "error": "Please enter a competition name."
        }

    competition_name = competition_name.strip()


    search_results, error = search_competition(
        competition_name
    )

    if error:

        return {
            "error": error
        }

    url_data = select_competition_url(
        competition_name,
        search_results
    )

    if url_data.get("status") == "ambiguous":

        return {
            "status": "ambiguous",
            "options": url_data.get("options", [])
        }

    if url_data.get("status") == "not_found":

        return {
            "error": (
                f"Could not identify the CEV competition "
                f"page for '{competition_name}'."
            )
        }

    if url_data.get("status") == "error":

        return {
            "error": url_data.get(
                "message",
                "Unknown Gemini error."
            )
        }

    if url_data.get("status") != "clear":

        return {
            "error": "Could not determine the correct competition page."
        }

    competition_url = url_data.get("url")

    if not competition_url:

        return {
            "error": "Gemini did not return a competition URL."
        }

    competition_text, error = get_competition_page_text(
        competition_url
    )

    if error:

        return {
            "error": error
        }

    competition_data, error = extract_competition_details(
        competition_name,
        competition_text
    )

    if error:

        return {
            "error": error
        }

    competition_data["url"] = competition_url

    save_competition_details(
        competition_data
    )

    return competition_data

def gradio_search(competition_name):

    result = get_competition_details(
        competition_name
    )

    return json.dumps(
        result,
        indent=4,
        ensure_ascii=False
    )


demo = gr.Interface(
    fn=gradio_search,
    inputs=gr.Textbox(
        label="Competition name",
        placeholder="e.g. EuroVolley 2026 Women"
    ),
    outputs=gr.Code(
        label="Competition details",
        language="json"
    ),
    title="CEV Competition Search",
    description=(
        "Search for a CEV competition and retrieve "
        "the final standings and Dream Team."
    )
)

if __name__ == "__main__":
    demo.launch()

