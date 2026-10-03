"""Flight search for the travel agent (AviationStack API). Plain Python, no LangChain."""
import os
import re
import time

import airportsdata
import certifi
import pycountry
import requests
from dotenv import load_dotenv

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

API_KEY = os.getenv("AVIATIONSTACK_API_KEY")
DEFAULT_ORIGIN = os.getenv("DEFAULT_ORIGIN_IATA", "")  # used when no origin is given
BASE_URL = "https://api.aviationstack.com/v1/flights"  # if your plan rejects HTTPS, try http://

AIRPORTS = airportsdata.load("IATA")  # {"LHE": {"name": ..., "city": ..., "country": "PK", ...}}

# Your full alias dict (USA/UK/UAE/Turkiye etc.) lives in country_aliases.py, moved over unchanged.
try:
    from country_aliases import COUNTRY_ALIASES
except ImportError:
    COUNTRY_ALIASES = {}

AIRPORTS_PER_SIDE = 2   # airports tried per country (the first entry in HUBS is the main one)
MAX_API_CALLS = 4       # hard cap per search: the free plan only allows ~100 requests a month
CACHE_TTL = 600         # seconds; identical searches reuse the previous API result
_cache = {}

# AviationStack can't filter by country, so a country is mapped to its main airports.
# Countries missing here fall back to a heuristic (airports with "International" in the name).
HUBS = {
    "PK": ["KHI", "LHE", "ISB"], "IN": ["DEL", "BOM", "BLR"], "BD": ["DAC", "CGP"], "LK": ["CMB"], "NP": ["KTM"],
    "AF": ["KBL"], "MV": ["MLE"], "KZ": ["ALA", "NQZ"], "UZ": ["TAS"],
    "AE": ["DXB", "AUH", "SHJ"], "SA": ["JED", "RUH", "DMM"], "QA": ["DOH"], "KW": ["KWI"], "BH": ["BAH"],
    "OM": ["MCT"], "JO": ["AMM"], "LB": ["BEY"], "IL": ["TLV"], "IQ": ["BGW"], "IR": ["IKA", "THR"],
    "TR": ["IST", "SAW", "ESB"], "EG": ["CAI", "HRG"],
    "GB": ["LHR", "LGW", "MAN"], "IE": ["DUB"], "DE": ["FRA", "MUC", "BER"], "FR": ["CDG", "ORY"],
    "IT": ["FCO", "MXP"], "ES": ["MAD", "BCN"], "PT": ["LIS", "OPO"], "NL": ["AMS"], "BE": ["BRU"],
    "CH": ["ZRH", "GVA"], "AT": ["VIE"], "GR": ["ATH"], "PL": ["WAW"], "RU": ["SVO", "DME"],
    "SE": ["ARN"], "NO": ["OSL"], "DK": ["CPH"], "FI": ["HEL"],
    "US": ["JFK", "LAX", "ORD", "ATL"], "CA": ["YYZ", "YVR", "YUL"], "MX": ["MEX", "CUN"],
    "BR": ["GRU", "GIG"], "AR": ["EZE"], "CL": ["SCL"], "CO": ["BOG"], "PE": ["LIM"],
    "CN": ["PEK", "PVG", "CAN"], "HK": ["HKG"], "JP": ["NRT", "HND", "KIX"], "KR": ["ICN"], "TW": ["TPE"],
    "SG": ["SIN"], "MY": ["KUL"], "TH": ["BKK"], "ID": ["CGK", "DPS"], "VN": ["SGN", "HAN"], "PH": ["MNL"],
    "AU": ["SYD", "MEL"], "NZ": ["AKL"],
    "ZA": ["JNB", "CPT"], "NG": ["LOS"], "KE": ["NBO"], "ET": ["ADD"], "MA": ["CMN"],
}

# A city name maps to one preferred airport (checked before the airport database,
# so "London" gives Heathrow instead of whichever "London" airport ranks first).
CITY_MAIN_AIRPORT = {
    "dhaka": "DAC", "delhi": "DEL", "new delhi": "DEL", "mumbai": "BOM", "kolkata": "CCU",
    "chennai": "MAA", "bangalore": "BLR", "bengaluru": "BLR", "karachi": "KHI", "lahore": "LHE",
    "islamabad": "ISB", "tokyo": "NRT", "osaka": "KIX", "kyoto": "KIX", "seoul": "ICN",
    "beijing": "PEK", "shanghai": "PVG", "hong kong": "HKG", "manila": "MNL", "jakarta": "CGK",
    "bali": "DPS", "hanoi": "HAN", "ho chi minh city": "SGN", "singapore": "SIN",
    "kuala lumpur": "KUL", "bangkok": "BKK", "dubai": "DXB", "abu dhabi": "AUH", "doha": "DOH",
    "jeddah": "JED", "riyadh": "RUH", "istanbul": "IST", "cairo": "CAI", "new york": "JFK",
    "los angeles": "LAX", "chicago": "ORD", "san francisco": "SFO", "miami": "MIA",
    "toronto": "YYZ", "vancouver": "YVR", "london": "LHR", "paris": "CDG", "rome": "FCO",
    "madrid": "MAD", "barcelona": "BCN", "frankfurt": "FRA", "munich": "MUC", "berlin": "BER",
    "amsterdam": "AMS", "zurich": "ZRH", "vienna": "VIE", "athens": "ATH", "lisbon": "LIS",
    "sydney": "SYD", "melbourne": "MEL", "auckland": "AKL", "johannesburg": "JNB",
    "nairobi": "NBO", "lagos": "LOS",
}


# ---------------------------------------------------------------------------
# Place resolution: country / city / airport code -> list of IATA codes
# ---------------------------------------------------------------------------
def _norm(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def _country_exact(text):
    key = _norm(text)
    for k in (key, re.sub(r"[^a-z0-9 ]", "", key)):  # second form turns "u.s.a" into "usa"
        if k in COUNTRY_ALIASES:
            return COUNTRY_ALIASES[k]
    try:
        return pycountry.countries.lookup(text.strip()).alpha_2
    except LookupError:
        return None


def _country_fuzzy(text):
    if len(text.strip()) < 4:  # fuzzy matching on 2-3 letters gives junk
        return None
    try:
        return pycountry.countries.search_fuzzy(text.strip())[0].alpha_2
    except LookupError:
        return None


def _country_name(iso2):
    c = pycountry.countries.get(alpha_2=iso2)
    return c.name if c else iso2


def _rank(codes):
    """Put 'International' airports first, then alphabetical."""
    return sorted(codes, key=lambda c: ("International" not in AIRPORTS[c]["name"], AIRPORTS[c]["name"]))


def _hub_airports(iso2, n):
    codes = [c for c in HUBS.get(iso2, []) if c in AIRPORTS]
    if len(codes) < n:
        rest = [c for c, a in AIRPORTS.items() if a.get("country") == iso2 and c not in codes]
        codes += _rank(rest)
    return codes[:n]


def resolve_place(place):
    """Return (iata_codes, label), or (None, None) if the place isn't recognised."""
    p = (place or "").strip()
    if not p:
        return None, None
    key = _norm(p)

    # 1. explicit airport code (e.g. "LHE"), unless it's really a country alias like "usa"
    if re.fullmatch(r"[A-Za-z]{3}", p) and p.upper() in AIRPORTS and key not in COUNTRY_ALIASES:
        return [p.upper()], f"{AIRPORTS[p.upper()]['name']} ({p.upper()})"

    # 2. well-known city -> preferred airport
    if key in CITY_MAIN_AIRPORT:
        code = CITY_MAIN_AIRPORT[key]
        return [code], f"{p.title()} ({code})"

    # 3. country, exact match
    iso2 = _country_exact(p)

    # 4. any other city in the airport database, then fuzzy country as the last resort
    if not iso2:
        city = [c for c, a in AIRPORTS.items() if a["city"].lower() == key]
        if city:
            return _rank(city)[:AIRPORTS_PER_SIDE], p.title()
        iso2 = _country_fuzzy(p)

    if iso2:
        codes = _hub_airports(iso2, AIRPORTS_PER_SIDE)
        if codes:
            return codes, _country_name(iso2)
    return None, None


# ---------------------------------------------------------------------------
# AviationStack API
# ---------------------------------------------------------------------------
def _api_get(dep, arr, date):
    cache_key = (dep, arr, date)
    hit = _cache.get(cache_key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]

    params = {"access_key": API_KEY, "dep_iata": dep, "arr_iata": arr, "limit": 100}
    if date:
        params["flight_date"] = date  # historical dates need a paid plan

    r = requests.get(BASE_URL, params=params, timeout=30)
    try:
        data = r.json()
    except ValueError:
        raise RuntimeError(f"AviationStack returned a non-JSON response (HTTP {r.status_code}).")

    if "error" in data:
        err = data["error"]
        raise RuntimeError(err.get("message") or err.get("info") or str(err))

    flights = data.get("data", [])
    _cache[cache_key] = (time.time(), flights)
    return flights


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def _sched(block):
    return ((block or {}).get("scheduled") or "")[:16].replace("T", " ")


def _format_short(f):
    fl, al = f.get("flight") or {}, f.get("airline") or {}
    dep, arr = f.get("departure") or {}, f.get("arrival") or {}
    return (
        f"{fl.get('iata') or fl.get('number') or '?'} ({al.get('name') or 'Unknown airline'}) | "
        f"{dep.get('iata')} -> {arr.get('iata')} | "
        f"departs {_sched(dep) or '?'} | arrives {_sched(arr) or '?'} | "
        f"status: {f.get('flight_status') or 'unknown'}"
    )


def _format_detailed(f):
    airline = (f.get("airline") or {}).get("name") or "Unknown airline"
    number = (f.get("flight") or {}).get("iata") or "Unknown flight number"
    dep, arr = f.get("departure") or {}, f.get("arrival") or {}

    def delay(block):
        d = block.get("delay")
        return f"{d} minutes" if d is not None else "N/A"

    def side(block, label):
        return (
            f"{label}:\n"
            f"- Airport: {block.get('airport') or 'Unknown airport'}\n"
            f"- IATA: {block.get('iata') or 'Unknown'}\n"
            f"- Terminal: {block.get('terminal') or 'N/A'}\n"
            f"- Gate: {block.get('gate') or 'N/A'}\n"
            f"- Scheduled: {block.get('scheduled') or 'Unknown'}\n"
            f"- Delay: {delay(block)}"
        )

    return (
        f"Airline: {airline}\nFlight: {number}\nStatus: {f.get('flight_status') or 'Unknown'}\n\n"
        f"{side(dep, 'Departure')}\n\n{side(arr, 'Arrival')}"
    )


# ---------------------------------------------------------------------------
# Main function (plain Python; wrap it with @tool later if you want an agent to call it)
# ---------------------------------------------------------------------------
def search_flights(origin: str = "", destination: str = "", max_results: int = 10,
                   flight_date: str = "", detailed: bool = False) -> str:
    """Find flights between two places using live AviationStack schedule data.

    origin / destination: a country ('Bangladesh', 'UAE'), a city ('Tokyo'), or an
    airport code ('DAC'). Leave origin empty to use DEFAULT_ORIGIN_IATA from .env.
    detailed=True prints terminal, gate and delay for every flight.
    flight_date (YYYY-MM-DD) only works on paid AviationStack plans.
    Returns schedules and status. It does NOT return ticket prices."""

    if not API_KEY:
        return (
            "Flight API error: AVIATIONSTACK_API_KEY is missing.\n"
            "Add this to your .env file:\nAVIATIONSTACK_API_KEY=your_api_key_here"
        )

    origin = (origin or "").strip() or DEFAULT_ORIGIN
    if not origin:
        return "I need to know where the trip starts. Which country, city or airport are you flying from?"
    if not (destination or "").strip():
        return "I need a destination. Which country, city or airport do you want to fly to?"
    if flight_date and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", flight_date):
        return "flight_date must be in YYYY-MM-DD format."

    dep_codes, dep_label = resolve_place(origin)
    arr_codes, arr_label = resolve_place(destination)
    if not dep_codes:
        return f"I couldn't recognise the origin '{origin}'. Try a country name, city, or airport code."
    if not arr_codes:
        return f"I couldn't recognise the destination '{destination}'. Try a country name, city, or airport code."

    pairs = [(d, a) for d in dep_codes for a in arr_codes if d != a][:MAX_API_CALLS]
    if not pairs:
        return "The origin and destination resolve to the same airport."

    found, error = {}, None
    for d, a in pairs:
        try:
            for f in _api_get(d, a, flight_date):
                if (f.get("flight") or {}).get("codeshared"):  # skip duplicate codeshare listings
                    continue
                key = ((f.get("flight") or {}).get("iata"), _sched(f.get("departure")))
                found.setdefault(key, f)
        except Exception as e:
            error = str(e)
            break  # the same error (bad key, quota, plan limit) would repeat and waste calls

    if not found:
        if error:
            return f"Flight API error: {error}"
        return (
            f"No live flight data found from {dep_label} ({', '.join(dep_codes)}) to {arr_label} "
            f"({', '.join(arr_codes)}).\n\nNote: AviationStack provides live/status flight data, not "
            "ticket prices. For fares, use a flight-pricing API such as Amadeus."
        )

    max_results = max(1, min(int(max_results or 10), 25))
    flights = sorted(found.values(), key=lambda f: _sched(f.get("departure")))[:max_results]

    head = (
        f"Flights from {dep_label} to {arr_label} "
        f"(searched {', '.join(dep_codes)} -> {', '.join(arr_codes)}), showing {len(flights)} of {len(found)}:"
    )
    note = "Source: AviationStack schedules and status. No prices or booking availability."
    if error:
        note += f" Some searches failed: {error}"

    if detailed:
        body = "\n\n---\n\n".join(_format_detailed(f) for f in flights)
        return f"{head}\n\n{body}\n\n{note}"
    return "\n".join([head, *[f"{i}. {_format_short(f)}" for i, f in enumerate(flights, 1)], note])


if __name__ == "__main__":
    print(search_flights("Bangladesh", "Japan"))
    print("\n" + "=" * 80 + "\n")
    print(search_flights("DAC", "Tokyo", max_results=2, detailed=True))