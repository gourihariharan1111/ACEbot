import os
import json
import time
from datetime import datetime
from operator import itemgetter
import requests
import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes

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

MODEL_NAME = "gemini-3.5-flash-lite"
DATABASE_FILE = "competition_database.json"
PLAYER_DB_FILE = "player_database.json"
MATCH_DB_FILE = "match_database.json"

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

# ============================================================
# CEV PLAYER DETAILS PIPELINE
# ============================================================

def search_player(player_name):

    url = "https://google.serper.dev/search"

    query = (
        f"site:www-old.cev.eu/Competition-Area/PlayerDetails.aspx {player_name}"
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

        data = response.json()
        organic_results = data.get("organic", [])

        if organic_results:
            return organic_results

        fallback_payload = {"q": f"Volleyball player {player_name}",
                            "autocorrect": True}
        fallback_response = requests.post(url, headers=headers, json=fallback_payload, timeout=30)
        fallback_data = fallback_response.json()
        
        if "spelling" in fallback_data and "text" in fallback_data["spelling"]:
            return f"DID_YOU_MEAN:{fallback_data['spelling']['text']}"
        
        fallback_organic = fallback_data.get("organic", [])
        if fallback_organic:
            first_title = fallback_organic[0].get("title", "")
            clean_name = first_title.split(" - ")[0].split(" (@")[0].split(" | ")[0].split(" • ")[0].strip()
            
            if clean_name:
                return f"DID_YOU_MEAN:{clean_name}"
        
        return []

    except Exception as e:

        print(f"Serper error: {e}")

        return []

def select_player_url(player_name,search_results):

    if not search_results:
        return None

    results_text = "\n\n".join(
        [
            f"Result {i+1}:\n"
            f"Title: {r.get('title')}\n"
            f"Link: {r.get('link')}\n"
            f"Snippet: {r.get('snippet', '')}"
            for i, r in enumerate(search_results)
        ]
    )

    prompt = f"""
    You are an assistant routing search results. The user searched for the volleyball player "{player_name}".
    Here are the top search results:

    {results_text}

    Rules:
    1. If the results clearly point to ONE specific player that the user most likely intended, return this JSON: {{"status": "clear", "url": "the_url_here"}}
    2. If the search is ambiguous because the results show multiple DIFFERENT players (for example, different first names with the searched surname), return this JSON: {{"status": "ambiguous", "options": ["Name 1", "Name 2", "Name 3"]}}

    Return ONLY valid JSON and do not include any other text.
    """

    response = ask_gemini(prompt)

    if not response:
        return None

    try:

        return json.loads(response)

    except json.JSONDecodeError:

        print("Invalid Gemini JSON:")
        print(response)

        return None


def get_player_page_text(url):

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
    except Exception as e:
        print(f"CEV request error: {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text(separator=' ', strip=True)

    player_id = None
    team_id = None

    if "?" in url:
        query_string = url.split("?", 1)[1]
        for pair in query_string.split("&"):
            if "=" in pair:
                key, val = pair.split("=", 1)
                if key.lower() == "playerid":
                    player_id = val
                elif key.lower() == "teamid":
                    team_id = val

    if player_id and team_id:
        photo_url = f"https://www-old.cev.eu/Modules/PlayerPhoto/TeamPlayer_Photo_115_160.aspx?ID={player_id}&TeamID={team_id}"
        text += f"\n[PHOTO_URL: {photo_url}]"

    return text, player_id

def extract_player_details(player_name,player_id,clean_text):

    prompt = f"""
    Analyze the following text data regarding the volleyball player '{player_name}'.
    Return the result as a valid JSON object. It must ONLY contain the following keys and no surrounding explanation or markdown:
    - cev_id (int, {player_id})
    - full_name (str, the player's full and correctly spelled name)
    - position (str, the designated field position of the player, or null if missing)
    - height (int, the physical height of the player in centimeters, or null if missing)
    - country (list of strings, nationalities or countries the player represents, capturing both if dual nationality is listed)
    - clubs (list of strings, all clubs the player has played for mentioned in the data)
    - competitions (list of strings, only the competitions listed under the player's history, National teams and Club teams. Format EVERY entry strictly as "Period/Competition (Team)" from the data. Do not invent competitions or duplicate entries.
    - photo_url (str, the URL of the player's photo if present at the end of the text, otherwise null)

    If any detail is missing from the text data, set its value to null or an empty list ([]). Do not guess or hallucinate any information.

    Data to analyze:
    {clean_text}
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


def save_player_details(data):

    path = Path(PLAYER_DB_FILE)

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

    player_name = data.get("full_name")

    # Replace existing player
    database = [
        item
        for item in database
        if item.get("full_name")
        != player_name
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


def format_player_result(data):
    if not data:
        return "❌ No data was extracted."

    output = []
    output.append(f"### {data.get('full_name', 'Unknown Player')}\n")
    output.append(f"### Player {data.get('cev_id', 'Unknown')}\n")
    
    if data.get("photo_url"):
        output.append(f"![Player Photo]({data.get('photo_url')})\n")

    pos = data.get("position")
    if pos:
        output.append(f"**Position:** {pos}\n")
        
    height = data.get("height")
    if height:
        output.append(f"**Height:** {height} cm\n")

    countries = data.get('country', [])
    if isinstance(countries, list) and countries:
        country_str = " / ".join(countries)
    elif isinstance(countries, str):
        country_str = countries
    else:
        country_str = "Unknown"
        
    output.append(f"**Country:** {country_str}\n")
    output.append("**Clubs:**")

    clubs = data.get("clubs", [])
    if clubs:
        for club in clubs:
            output.append(f"* {club}")
    else:
        output.append("* None listed")

    output.append("\n**Competitions:**")

    comps = data.get("competitions", [])
    if comps:
        for comp in comps:
            output.append(f"* {comp}")
    else:
        output.append("* None listed")

    return "\n".join(output)


def get_player_details(player_name,progress=gr.Progress()):

    if not player_name or not player_name.strip():
        return "⚠️ Please enter a player name."

    player_name = player_name.strip()

    progress(0.05, desc="Starting...")

    print("\n" + "=" * 60)
    print(f"PLAYER: {player_name}")
    print("=" * 60)

    progress(0.10, desc="Searching CEV through Google...")

    search_results = search_player(
        player_name
    )

    if isinstance(search_results, str) and search_results.startswith("DID_YOU_MEAN:"):
        corrected_name = search_results.split(":")[1]
        return f"❌ Did you mean: {corrected_name}?"

    if not search_results:
        return (
            f"❌ **No player found.**\n\n"
            f"Could not find any player named '{player_name}'."
        )

    print(f"Found {len(search_results)} search results.")

    progress(
        0.25,
        desc="Identifying the correct CEV profile..."
    )

    url_data = select_player_url(
        player_name,
        search_results
    )

    if not url_data:
        return (
            "❌ **Could not process the search results.**"
        )

    if url_data.get("status") == "ambiguous":

        options_list = url_data.get("options", [])

        if options_list:

            options_str = ", ".join(options_list)

            return (
                f"⚠️ The search returned multiple matches. "
                f"Did you mean one of these: {options_str}?"
            )

        else:

            return (
                "⚠️ The search was too ambiguous. "
                "Please provide the player's full name."
            )

    elif url_data.get("status") == "clear" and url_data.get("url"):

        link = url_data.get("url")

        print(f"Player URL: {link}")

    else:

        return (
            f"❌ Could not determine the correct player profile "
            f"for '{player_name}'."
        )

    progress(
        0.40,
        desc="Downloading CEV player page..."
    )

    page_text, player_id = get_player_page_text(
        link
    )

    if not page_text:
        return (
            "❌ **Could not download the player page.**"
        )

    progress(
        0.70,
        desc="Extracting player details..."
    )

    player_data = extract_player_details(
        player_name,
        player_id,
        page_text
    )

    if not player_data:
        return (
            "❌ **Could not extract the player data.**"
        )

    progress(
        0.90,
        desc="Saving player data..."
    )

    save_player_details(player_data)

    print("Player saved successfully.")

    progress(
        1.0,
        desc="Complete!"
    )

    return format_player_result(player_data)

# ============================================================
# CEV COMPETITION PIPELINE
# ============================================================

def search_competition(competition_name):

    url = "https://google.serper.dev/search"

    query = (
        f'site:www-old.cev.eu/Competition-Area/competition.aspx '
        f'{competition_name}'
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


def select_competition_url(competition_name,search_results):

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

def extract_competition_details(competition_name,competition_text,standings_text,competition_url):
    query_string = competition_url.split("?")[1].split("&")
    for pair in query_string:
        if "=" in pair:
            key, val = pair.split("=", 1)
            if key.lower() == "id":
                competition_id = val


    prompt = f"""
    Extract structured information from official CEV pages.

    Competition ID:
    {competition_id}

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
    - team

    Return ONLY valid JSON:

    {{
        "competition_ID": {competition_id},
        "competition": "CEV ZEREN Group Champions League Volley 2026 | Women",
        "url": "{competition_url}",
        "final_standings": [
            {{
                "rank": 1,
                "team": "team name"
            }}
        ],
        "dream_team": [
            {{
                "award": "MVP",
                "player": "player name",
                "team": "team name or null"
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

    output.append(f"# Competition {data.get('competition_ID', 'Unknown')}")

    output.append("")

    output.append("## Final Standings")

    output.append("")

    for item in data.get("final_standings",[]):

        rank = item.get("rank")
        team = item.get("team")

        output.append(
            f"{rank}. {team}"
        )

    output.append("")

    if data.get("dream_team") != []:

        output.append("## Dream Team")

        output.append("")

        for player in data.get(
            "dream_team",
            []
        ):

            award = player.get("award")
            name = player.get("player")
            team = player.get("team")

            if team:
                output.append(
                    f"- **{award}:** {name} ({team})"
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

# ============================================================
# CEV MATCH DETAILS PIPELINE
# ============================================================

# Given a match name, returns results (Organic / Did you mean / Empty list)
def search_match(match_name):

    url = "https://google.serper.dev/search"
    query = (f"site:www-old.cev.eu/Competition-Area/MatchStatistics.aspx {match_name} ")
    payload = {"q": query}
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        organic_results = data.get("organic", [])

        if organic_results:
            return organic_results

        fallback_payload = {"q": f"site:www-old.cev.eu/Competition-Area/MatchPage.aspx {match_name}", "autocorrect": True}
        fallback_response = requests.post(url, headers=headers, json=fallback_payload, timeout=30)
        fallback_data = fallback_response.json()
        
        if "spelling" in fallback_data and "text" in fallback_data["spelling"]:
            return f"DID_YOU_MEAN:{fallback_data['spelling']['text']}"
        
        fallback_organic = fallback_data.get("organic", [])
        if fallback_organic:
            first_title = fallback_organic[0].get("title", "")
            clean_name = first_title.split(" - ")[0].split(" (@")[0].split(" | ")[0].split(" • ")[0].strip()
            
            if clean_name:
                return f"DID_YOU_MEAN:{clean_name}"
        
        return []

    except Exception as e:
        print(f"Serper error: {e}")
        return []

# Given match name and search results, go to URL
def select_match_url(match_name, search_results):

    if not search_results:
        return None

    results_text = "\n\n".join(
        [
            f"Result {i+1}:\n"
            f"Title: {r.get('title')}\n"
            f"Link: {r.get('link')}\n"
            f"Snippet: {r.get('snippet', '')}"
            for i, r in enumerate(search_results)
        ]
    )

    prompt = f"""
    You are an assistant routing search results. The user searched for the volleyball match "{match_name}".
    Here are the top search results:

    {results_text}

    Rules:
    1. If the results clearly point to ONE specific match that the user most likely intended, return this JSON: {{"status": "clear", "url": "the_url_here"}}
    2. If the search is ambiguous because the results show multiple DIFFERENT matches (for example, same opponents in different competitions or different stage of competition), return this JSON: {{"status": "ambiguous", "options": ["Name 1", "Name 2", "Name 3"]}}

    Return ONLY valid JSON and do not include any other text.
    """

    response = ask_gemini(prompt)
    if not response:
        return None
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        print("Invalid Gemini JSON:")
        print(response)
        return None

# Transform URL (str) into text (str)
def get_match_page_text(url):

    match_id = None
    query_string = url.split("?")[1].split("&")
    for pair in query_string:
        if "=" in pair:
            key, val = pair.split("=", 1)
            if (key.lower() == "mid" and "MatchPage" in url) or (key.lower() == "id" and "MatchStatistics" in url):
                match_id = val

    if "setN=" in url and "setN=0" not in url:
        url = url.replace("setN=","")

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
    except Exception as e:
        print(f"CEV request error: {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text(separator=' ', strip=True)



    return text, match_id

# Extract the match details given the match name and the clean text
def extract_match_details(match_name, match_id, clean_text):

    prompt = f"""
    Analyze the following text data regarding the volleyball match '{match_name}'.
    Return the result as a valid JSON object. It must ONLY contain the following keys and no surrounding explanation or markdown:
    - matchID ({match_id})
    - name (str, such as "CEV Trendyol EuroVolley 2026 | Women Quarter Final matches in CZE  (WQF-04)")
    - dateTime (str, reformat as "2026-09-02T19:00")
    - spectators (int, return null if the field is not found)
    - stadium (str, such as "Brno Exhibition Centre - The Pavilion P - BRNO")
    - city (str, such as "Brno")
    - matchDuration (int, such as 111)
    - referees (list of str)

    - teamA
    -- name (str, such as "Italy")
    -- coach (str, such as VELASCO Julio)
    -- setsWon (int, such as 3)
    -- receptionPos (int)
    -- receptionExc (int)
    -- attackExc (int)

    - teamB
    -- name (str, such as "Sweden")
    -- coach (str, such as MICELLI Lorenzo)
    -- setsWon (int, such as 1)
    -- receptionPos (int)
    -- receptionExc (int)
    -- attackExc (int)

    - set1, set2, set3, set4, set5 (one for each set. set4 / set5 should be null with nothing if they do not exist)
    -- duration (int)
    -- pointsWon
    --- A (int)
    --- B (int)
    -- receptionPos
    --- A (int)
    --- B (int)
    -- receptionExc
    --- A (int)
    --- B (int)
    -- attackExc
    --- A (int)
    --- B (int)
    - playerStats (for each player across both teams.)
    -- jerseyNo (int)
    -- name (str, such as 'HAAK Anna')
    -- team (str, such as 'Sweden')
    -- pointsTot (int, return 0 if there is '-', look at 'Tot' and NOT 'BP' or 'W-L')
    -- reception
    --- posPercent (int, return null if there is '.')
    --- excPercent (int, return null if there is '.')
    -- attack
    --- total (int, return 0 if there is '-')
    --- error (int, return 0 if there is '.')
    --- blocked (int, return 0 if there is '.')
    --- excellent (int, return 0 if there is '.')
    --- excPercent (int, return null if there is '.')
    -- block (int, return 0 if there is '.')

    If any detail is missing from the text data, set its value to null or an empty list ([]). Do not guess or hallucinate any information.

    Data to analyze:
    {clean_text}
    """

    # Ask Gemini for the response
    response = ask_gemini(prompt)

    # If there is no response, there is no response
    if not response:
        return None

    # Load Response if there is one
    try:
        return json.loads(response)
    
    # If exception occurs, there is no response
    except json.JSONDecodeError:
        print("Gemini returned invalid JSON:")
        print(response)
        return None

# Save the match data into the database file
def save_match_details(data):

    path = Path(MATCH_DB_FILE)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                database = json.load(f)
            if not isinstance(database, list):
                database = []
        except Exception:
            database = []
    else:
        database = []

    match_id = data.get("matchID")

    # Replace existing match
    database = [item for item in database if item.get("matchID") != match_id]
    database.append(data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(database, f, indent=4, ensure_ascii=False)

# Format from Pythonized JSON match data to Markdown Output
def format_match_result(data):
    if not data:
        return "❌ No data was extracted."
    
    dateAndTime = datetime.strptime(data["dateTime"], "%Y-%m-%dT%H:%M")
    dateAndTime = dateAndTime.strftime("%d %B, %Y, %H:%M")
    # If the duration is 0, it also means there won't be player statistics and team statistics
    duration = data["matchDuration"]

    teamA = data["teamA"]
    teamB = data["teamB"]
    referees = data["referees"]
    sets = [data["set1"], data["set2"], data["set3"]]
    if data["set4"]:
        sets.append(data["set4"])
        if data["set5"]:
            sets.append(data["set5"])
    teamAname = teamA["name"]
    teamBname = teamB["name"]
    players = sorted(data["playerStats"], key=lambda d: (d['team'], d["pointsTot"], d["attack"]["total"]), reverse=True)

    output = []

    # MATCH HEADER
    output.append(f"# Match {data["matchID"]}\n")
    output.append(f"## {data["name"]}\n")
    output.append(f"## {teamAname} vs {teamBname}\n")
    output.append(f"\n")
    # MATCH DETAILS
    output.append(f"# Match details")
    output.append(f"| Date and Time | Spectators | Stadium | City | Duration | Referees |")
    output.append(f"| :---: | :---: | :---: | :---: | :---: | :---: |")

    match_detail_string = f"| {dateAndTime} | {data["spectators"]} | {data["stadium"]} "
    match_detail_string += f"| {data["city"]} | {(str(duration) + " minutes") if duration else "Unknown"} | {referees[0]} and {referees[1]} |"
    output.append(match_detail_string)

    # RESULTS
    output.append(f"# Results\n")
    output.append(f"## {teamAname} {teamA["setsWon"]}-{teamB["setsWon"]} {teamBname}\n")
    point_result_string = ""
    for i, setN in enumerate(sets):
        point_result_string += f"{setN["pointsWon"]["A"]}-{setN["pointsWon"]["B"]}, "
    point_result_string = point_result_string[:-2] # Crop the last ", " away
    output.append(point_result_string)

    # DETAILED STATISTICS BY SET
    if duration:
        output.append(f"# Detailed Team Statistics\n")
        output.append(f"| Set | Duration | Points | Reception Positivity | Reception Excellence | Attack Excellence |")
        output.append(f"| :---: | :---: | :---: | :---: | :---: | :---: |")
        for i, setN in enumerate(sets):
            set_detail_string = f"| {i+1} "
            set_detail_string += f"| {setN["duration"]} mins "
            set_detail_string += f"| {setN["pointsWon"]["A"]}-{setN["pointsWon"]["B"]} "
            set_detail_string += f"| {setN["receptionPos"]["A"]}% vs {setN["receptionPos"]["B"]}% "
            set_detail_string += f"| {setN["receptionExc"]["A"]}% vs {setN["receptionExc"]["B"]}% "
            set_detail_string += f"| {setN["attackExc"]["A"]}% vs {setN["attackExc"]["B"]}% |"
            output.append(set_detail_string)

    # PLAYERS STATISTICS
    if duration:
        output.append(f"# Players' Statistics\n")
        output.append(f"ATK = Attack, REC = Reception, Pos = Positivity, Exc = Excellence, Err = Error, ATK Blk = Blocked Attacks\n")
        output.append(f"| Team | Jersey # | Name | Points | ATK | ATK Exc | ATK Err | ATK Blk | ATK Exc% | BLOCK | REC Pos% | REC Exc% |")
        output.append(f"| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for player in players:
            player_detail_string = f"| {player["team"] or ""}"
            player_detail_string += f"| {player["jerseyNo"] or ""}"
            player_detail_string += f"| {player["name"] or ""}"
            player_detail_string += f"| {player["pointsTot"] or ""}"
            player_detail_string += f"| {player["attack"]["total"] or ""}"
            player_detail_string += f"| {player["attack"]["excellent"] or ""}"
            player_detail_string += f"| {player["attack"]["error"] or ""}"
            player_detail_string += f"| {player["attack"]["blocked"] or ""}"
            player_detail_string += f"| {player["attack"]["excPercent"] or ""}"
            player_detail_string += f"| {player["block"] or ""}"
            player_detail_string += f"| {player["reception"]["posPercent"] or ""}"
            player_detail_string += f"| {player["reception"]["excPercent"] or ""} |"
            output.append(player_detail_string)
    # IF PLAYER STATISTICS ARE NOT AVAILABLE, PRINT A PLAYER LIST INSTEAD
    else:
        output.append(f"# Player List\n")
        output.append(f"## {teamAname}\n")
        for player in players:
            if player["team"] == teamAname:
                output.append(f"{player["name"]}\n")
        output.append(f"## {teamBname}\n")
        for player in players:
            if player["team"] == teamBname:
                output.append(f"{player["name"]}\n")

    return "\n".join(output)

# OVERALL FUNCTION: execute the above 6 functions in order
def get_match_details(match_name,progress=gr.Progress()):

    if not match_name or not match_name.strip():
        return "⚠️ Please enter a match name."

    match_name = match_name.strip()

    progress(0.05, desc="Starting...")
    print("\n" + "=" * 60)
    print(f"MATCH: {match_name}")
    print("=" * 60)

    progress(0.10, desc="Searching CEV through Google...")
    search_results = search_match(match_name)
    if isinstance(search_results, str) and search_results.startswith("DID_YOU_MEAN:"):
        corrected_name = search_results.split(":")[1]
        return f"❌ Did you mean: {corrected_name}?"
    if not search_results:
        return f"❌ **No match found.**\n\n Could not find any match named '{match_name}'."
    print(f"Found {len(search_results)} search results.")

    progress(0.25, desc="Identifying the correct CEV match...")
    url_data = select_match_url(match_name, search_results)
    if not url_data:
        return ("❌ **Could not process the search results.**")
    if url_data.get("status") == "ambiguous":
        options_list = url_data.get("options", [])
        if options_list:
            options_str = ", ".join(options_list)
            return (f"⚠️ The search returned multiple matches. Did you mean one of these: {options_str}?")
        else:
            return ("⚠️ The search was too ambiguous. Please provide the match's full name, including year, gender and/or competition.")
    elif url_data.get("status") == "clear" and url_data.get("url"):
        link = url_data.get("url")
        print(f"Match URL: {link}")
    else:
        return (f"❌ Could not determine the correct match profile for '{match_name}'.")

    progress(0.40,desc="Downloading CEV match page...")
    page_text, match_id = get_match_page_text(link)
    if not page_text:
        return ("❌ **Could not download the match page.**")

    progress(0.70,desc="Extracting match details...")
    match_data = extract_match_details(match_name, match_id, page_text)
    if not match_data:
        return ("❌ **Could not extract the match data.**")

    progress(0.90,desc="Saving match data...")
    save_match_details(match_data)
    print("Match saved successfully.")

    progress(1.0,desc="Complete!")
    return format_match_result(match_data)

# ============================================================
# GRADIO INTERFACE
# ============================================================

with gr.Blocks(title="ACEBot: Your Guide for European Volleyball!") as demo:
    
    gr.Markdown("# ACEBot: Your Guide for European Volleyball!\n\nChoose what you want to search for below.")

    with gr.Tabs():
        
        # ==========================================
        # TAB 1: PLAYER SEARCH (Jacob)
        # ==========================================
        with gr.TabItem("Search Player"):
            
            gr.Markdown("**Enter a player name to retrieve:** Country, Clubs, Competitions")
            
            player_input = gr.Textbox(
                label="Player Name",
                placeholder="E.g., Isabelle Haak"
            )
            player_button = gr.Button(
                "Get Player Details",
                variant="primary"
            )
            player_output = gr.Markdown(
                value="Results will appear here..."
            )
            
            player_button.click(fn=get_player_details, inputs=player_input, outputs=player_output)
            player_input.submit(fn=get_player_details, inputs=player_input, outputs=player_output)

        # ==========================================
        # TAB 2: COMPETITION SEARCH (Gouri)
        # ==========================================
        with gr.TabItem("Search Competition"):
            
            gr.Markdown("**Enter a CEV competition name to retrieve:** Final standings, Dream Team, Saved competition data")
            
            comp_input = gr.Textbox(
                label="Competition Name",
                placeholder="E.g., EuroVolley 2026 Women"
            )
            comp_button = gr.Button(
                "Get Competition Details",
                variant="primary"
            )
            comp_output = gr.Markdown(
                value="Results will appear here..."
            )
            
            comp_button.click(fn=get_competition_details, inputs=comp_input, outputs=comp_output, show_progress="full")
            comp_input.submit(fn=get_competition_details, inputs=comp_input, outputs=comp_output, show_progress="full")

        # ==========================================
        # TAB 3: MATCH SEARCH (Mikael)
        # ==========================================
        with gr.TabItem("Search Match"):

            gr.Markdown("**Enter match details to retrieve:** Match info, Results for the match and sets, Match statistics")
            
            match_input = gr.Textbox(
                label="Match Details",
                placeholder="E.g., Sweden vs Italy, EuroVolley Women 2026 (Please specify the year, the gender and the competition if possible!)"
            )
            match_button = gr.Button(
                "Get Match Details",
                variant="primary"
            )
            match_output = gr.Markdown(
                value="Results will appear here..."
            )
            
            match_button.click(fn=get_match_details, inputs=match_input, outputs=match_output)
            match_input.submit(fn=get_match_details, inputs=match_input, outputs=match_output)

# == GRADIO THEME ==
# By Mikael

class Blaagult(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.yellow,
        secondary_hue: colors.Color | str = colors.blue,
        neutral_hue: colors.Color | str = colors.blue,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Sofia Sans"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            body_background_fill="linear-gradient(45deg, *primary_500, *primary_200)",
            body_background_fill_dark="linear-gradient(45deg, *secondary_900, *secondary_700)",
            button_primary_background_fill="linear-gradient(90deg, *secondary_400, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_200, *secondary_400)",
            button_primary_text_color="white",
            button_primary_background_fill_dark="linear-gradient(90deg, *primary_600, *primary_800)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *primary_400, *primary_600)",
            slider_color="*secondary_300",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="32px",
        )

blaagult = Blaagult()

if __name__ == "__main__":

    print("\nStarting CEV Volleyball Bot...")

    demo.launch(
        inbrowser=True,
        show_error=True,
        theme=blaagult
    )


