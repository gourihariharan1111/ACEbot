import os
import json
import time
import requests
import gradio as gr

from pathlib import Path
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

MODEL_NAME = "gemini-3.6-flash"
DATABASE_FILE = "competition_database.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

if not SERPER_API_KEY:
    raise ValueError("SERPER_API_KEY not found in .env")


client = genai.Client(api_key=GEMINI_API_KEY)

def ask_gemini(prompt, max_retries=3):

    for attempt in range(1, max_retries + 1):

        try:

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )

            return response.text

        except Exception as e:

            print(
                f"Gemini error "
                f"(attempt {attempt}/{max_retries}): {e}"
            )

            if attempt < max_retries:
                time.sleep(attempt * 5)

    return None

def search_competition(competition_name):

    url = "https://google.serper.dev/search"

    query = (
        f'site:www-old.cev.eu/Competition-Area/competition.aspx '
        f'"{competition_name}"'
    )

    payload = {
        "q": query
    }

    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }

    try:

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=30
        )

        response.raise_for_status()

        return response.json().get("organic", [])

    except Exception as e:

        print(f"Serper error: {e}")

        return []


def select_competition_url(
    competition_name,
    search_results
):

    if not search_results:
        return None

    results_text = "\n\n".join(
        [
            f"TITLE: {r.get('title')}\n"
            f"URL: {r.get('link')}\n"
            f"SNIPPET: {r.get('snippet', '')}"
            for r in search_results[:5]
        ]
    )

    prompt = f"""
You are selecting the official CEV competition page.

Requested competition:

{competition_name}

Search results:

{results_text}

Choose the official CEV competition page.

Return ONLY valid JSON:

{{
    "selected_url": "URL",
    "competition": "competition name",
    "confidence": "high"
}}

Rules:

- URL must be from www-old.cev.eu
- Prefer Competition.aspx
- Do not invent a URL
- If there is no clear match, selected_url must be null
"""

    response = ask_gemini(prompt)

    if not response:
        return None

    try:

        data = json.loads(response)

        return data.get("selected_url")

    except json.JSONDecodeError:

        print("Invalid Gemini JSON:")
        print(response)

        return None

def get_page_text(url):

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=30
        )

        response.raise_for_status()

    except Exception as e:

        print(f"CEV request error: {e}")

        return None

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    for element in soup([
        "script",
        "style",
        "noscript"
    ]):

        element.decompose()

    lines = [
        line.strip()
        for line in soup.get_text(
            separator="\n"
        ).splitlines()
        if line.strip()
    ]

    return "\n".join(lines)

def get_standings_url(competition_url):

    base_url = competition_url.split("&PID=")[0]

    if "?" in base_url:
        return base_url + "&PID=-2"

    return base_url + "?PID=-2"

def extract_competition_details(
    competition_name,
    competition_text,
    standings_text,
    competition_url
):

    prompt = f"""
Extract structured information from official CEV pages.

Competition:
{competition_name}

Competition URL:
{competition_url}


{competition_text}

{standings_text}

Extract:

1. Final competition standings.

Use ONLY the final overall standings.

Do NOT use:

- Pool standings
- Group standings
- Qualification standings
- Match results
- Intermediate standings


2. Dream Team.

Extract every Dream Team / Best Player position.

For each player return:

- award
- player
- country

Return ONLY valid JSON:

{{
    "competition": "{competition_name}",
    "url": "{competition_url}",
    "final_standings": [
        {{
            "rank": 1,
            "country": "country name"
        }}
    ],
    "dream_team": [
        {{
            "award": "MVP",
            "player": "player name",
            "country": "country name or null"
        }}
    ]
}}

Rules:

- Preserve CEV spelling.
- Do not invent players.
- Do not invent countries.
- Rank must be an integer.
- Sort final standings by rank.
- Return JSON only.
"""

    response = ask_gemini(prompt)

    if not response:
        return None

    try:

        return json.loads(response)

    except json.JSONDecodeError:

        print("Gemini returned invalid JSON:")
        print(response)

        return None

def save_competition_details(data):

    path = Path(DATABASE_FILE)

    if path.exists():

        try:

            with open(
                path,
                "r",
                encoding="utf-8"
            ) as f:

                database = json.load(f)

            if not isinstance(database, list):
                database = []

        except Exception:

            database = []

    else:

        database = []

    competition_name = data.get("competition")

    # Replace existing competition
    database = [
        item
        for item in database
        if item.get("competition")
        != competition_name
    ]

    database.append(data)

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            database,
            f,
            indent=4,
            ensure_ascii=False
        )

def format_result(data):

    if not data:
        return "❌ No data was extracted."

    output = []

    output.append(
        f"# {data.get('competition', 'Unknown competition')}"
    )

    output.append("")

    output.append("## Final Standings")

    output.append("")

    for item in data.get(
        "final_standings",
        []
    ):

        rank = item.get("rank")
        country = item.get("country")

        output.append(
            f"{rank}. {country}"
        )

    output.append("")

    output.append("## Dream Team")

    output.append("")

    for player in data.get(
        "dream_team",
        []
    ):

        award = player.get("award")
        name = player.get("player")
        country = player.get("country")

        if country:
            output.append(
                f"- **{award}:** {name} ({country})"
            )
        else:
            output.append(
                f"- **{award}:** {name}"
            )

    return "\n".join(output)

def get_competition_details(competition_name, progress=gr.Progress()):
    """
    Complete CEV competition pipeline for Gradio.
    """

    if not competition_name or not competition_name.strip():
        return "⚠️ Please enter a competition name."

    competition_name = competition_name.strip()

    progress(0.05, desc="Starting...")

    print("\n" + "=" * 60)
    print(f"COMPETITION: {competition_name}")
    print("=" * 60)

    progress(0.10, desc="Searching CEV through Google...")

    search_results = search_competition(
        competition_name
    )

    if not search_results:
        return (
            "❌ **No CEV competition was found.**\n\n"
            "Check the competition name and try again."
        )

    print(f"Found {len(search_results)} search results.")

    progress(
        0.25,
        desc="Identifying the correct CEV competition..."
    )

    competition_url = select_competition_url(
        competition_name,
        search_results
    )

    if not competition_url:
        return (
            "❌ **Could not identify the CEV competition page.**"
        )

    print(f"Competition URL: {competition_url}")

    progress(
        0.40,
        desc="Downloading CEV competition page..."
    )

    competition_text = get_page_text(
        competition_url
    )

    if not competition_text:
        return (
            "❌ **Could not download the CEV competition page.**"
        )

    progress(
        0.55,
        desc="Downloading final standings..."
    )

    standings_url = get_standings_url(
        competition_url
    )

    print(f"Standings URL: {standings_url}")

    standings_text = get_page_text(
        standings_url
    )

    if not standings_text:
        return (
            "❌ **Could not download the final standings page.**"
        )

    progress(
        0.70,
        desc="Extracting standings and Dream Team..."
    )

    data = extract_competition_details(
        competition_name,
        competition_text,
        standings_text,
        competition_url
    )

    if not data:
        return (
            "❌ **Gemini could not extract the competition data.**"
        )

    progress(
        0.90,
        desc="Saving competition data..."
    )

    save_competition_details(data)

    print("Competition saved successfully.")

    progress(
        1.0,
        desc="Complete!"
    )

    return format_result(data)

with gr.Blocks(
    title="CEV Volleyball Competition Bot"
) as demo:

    gr.Markdown(
        """
# CEV Volleyball Competition Bot

Enter a CEV competition name to retrieve:

- Final standings
- Dream Team
- Saved competition data
        """
    )

    with gr.Row():

        competition_input = gr.Textbox(
            label="Competition name",
            placeholder="Example: EuroVolley 2026 Women",
            scale=4
        )

        search_button = gr.Button(
            "Get Competition Details",
            variant="primary",
            scale=1
        )

    result_output = gr.Markdown(
        value="Enter a competition name above.",
        label="Results"
    )

    # Button click
    search_button.click(
        fn=get_competition_details,
        inputs=competition_input,
        outputs=result_output,
        show_progress="full"
    )

    # Press Enter also works
    competition_input.submit(
        fn=get_competition_details,
        inputs=competition_input,
        outputs=result_output,
        show_progress="full"
    )

if __name__ == "__main__":

    print("\nStarting CEV Volleyball Bot...")

    demo.launch(
        inbrowser=True,
        show_error=True
    )


