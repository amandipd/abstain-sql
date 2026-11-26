"""Made-up names for the filled copies (plan step 1.3).

Table lists give each name column a `kind` hint, such as clinic, hotel, or tail
number. This module turns a kind into a plausible, fictional name: places
("Riverside Clinic"), organizations ("Northwind Energy"), titles and products
("Velvet Lantern"), and a few special formats (aircraft tail numbers). Unknown
kinds fall back to a place-style name built from the kind itself, so a new
kind in a table list never breaks the filler.

All names are invented from fixed word lists; none name a real organization.
"""

from __future__ import annotations

import random

FIRST_NAMES = [
    "Ava",
    "Liam",
    "Sofia",
    "Noah",
    "Mei",
    "Mateo",
    "Amara",
    "Ethan",
    "Priya",
    "Lucas",
    "Hana",
    "Omar",
    "Chloe",
    "Diego",
    "Zara",
    "Samuel",
    "Leila",
    "Kenji",
    "Grace",
    "Ibrahim",
    "Elena",
    "Malik",
    "Nora",
    "Arjun",
    "Isabel",
    "Tomas",
    "Aisha",
    "Jonah",
    "Yuki",
    "Felix",
    "Maya",
    "Andre",
    "Ruth",
    "Kofi",
    "Lena",
    "Rafael",
    "Ingrid",
    "Dev",
    "Clara",
    "Hassan",
    "Mira",
    "Owen",
    "Fatima",
    "Leo",
    "Anya",
    "Marcus",
    "Iris",
    "Tariq",
    "Julia",
    "Wen",
]
LAST_NAMES = [
    "Garcia",
    "Chen",
    "Okafor",
    "Patel",
    "Kim",
    "Nguyen",
    "Silva",
    "Johnson",
    "Haddad",
    "Novak",
    "Rossi",
    "Tanaka",
    "Mensah",
    "Kowalski",
    "Ahmed",
    "Larsen",
    "Moreno",
    "Singh",
    "Dubois",
    "Park",
    "Ivanova",
    "Brooks",
    "Castillo",
    "Yilmaz",
    "Osei",
    "Fischer",
    "Reyes",
    "Sato",
    "Murphy",
    "Abbasi",
    "Lindqvist",
    "Ortiz",
    "Wright",
    "Mwangi",
    "Costa",
    "Bauer",
    "Hughes",
    "Ali",
    "Petrov",
    "Santos",
    "Walker",
    "Nakamura",
    "Adeyemi",
    "Klein",
    "Rahman",
    "Bianchi",
    "Foster",
    "Lopez",
    "Wu",
    "Dlamini",
]

PLACES = [
    "Riverside",
    "Oakwood",
    "Summit",
    "Harbor",
    "Pinecrest",
    "Maplewood",
    "Cedar Hill",
    "Lakeview",
    "Northgate",
    "Willow Creek",
    "Stonebridge",
    "Fairview",
    "Brookside",
    "Redwood",
    "Silver Lake",
    "Eastfield",
    "Westbrook",
    "Highland",
    "Granite Peak",
    "Bayside",
    "Clearwater",
    "Elmwood",
    "Foxridge",
    "Glenwood",
    "Hillcrest",
    "Ironwood",
    "Juniper Hills",
    "Kingsport",
    "Linden",
    "Meadowbrook",
    "Northfield",
    "Oakridge",
    "Parkside",
    "Quarry Hill",
    "Rosewood",
    "Sandy Point",
    "Thornbury",
    "Valley View",
    "Westwood",
    "Ashford",
    "Blue Ridge",
    "Copper Canyon",
]
COINED = [
    "Northwind",
    "Bluepeak",
    "Brightline",
    "Keystone",
    "Trailhead",
    "Meridian",
    "Crescent",
    "Evergreen",
    "Sterling",
    "Pioneer",
    "Atlas",
    "Beacon",
    "Horizon",
    "Vantage",
    "Cobalt",
    "Juniper",
    "Lumen",
    "Halcyon",
    "Ridgeway",
    "Kestrel",
    "Solstice",
    "Tidewater",
    "Ember",
    "Vireo",
    "Larkspur",
    "Osprey",
    "Polaris",
    "Sablewood",
    "Tanager",
    "Wren",
    "Zephyr",
    "Aurora",
    "Bramble",
    "Calder",
    "Driftwood",
    "Fenwick",
    "Galeway",
    "Hollis",
    "Marlowe",
    "Quillon",
]
ADJECTIVES = [
    "Classic",
    "Bold",
    "Silver",
    "Golden",
    "Quiet",
    "Bright",
    "Wild",
    "Hidden",
    "Urban",
    "Coastal",
    "Midnight",
    "Crimson",
    "Velvet",
    "Northern",
    "Summer",
    "Winter",
    "Electric",
    "Gentle",
    "Rustic",
    "Modern",
    "Lucky",
    "Swift",
    "Iron",
    "Amber",
    "Emerald",
    "Paper",
    "Glass",
    "Hollow",
    "Lunar",
    "Solar",
]
NOUNS = [
    "Lantern",
    "River",
    "Signal",
    "Harbor",
    "Echo",
    "Garden",
    "Compass",
    "Mirror",
    "Orchard",
    "Tide",
    "Meadow",
    "Atlas",
    "Ember",
    "Canyon",
    "Harvest",
    "Voyage",
    "Window",
    "Thread",
    "Summit",
    "Circuit",
]
SYLLABLES = ["za", "lo", "vi", "ra", "ne", "to", "ka", "mi", "sor", "quin", "lu", "tel", "vor", "ben", "dex", "fi"]
ENDINGS = ["in", "ix", "ol", "a", "on", "ia", "ex", "ara"]

# Kind -> suffixes for place-style names: "<place> <suffix>".
PLACE_SUFFIXES = {
    "ambulance station": ["Ambulance Station", "EMS Station"],
    "bank branch": ["Branch", "Banking Center"],
    "bookstore": ["Books", "Bookshop"],
    "campground": ["Campground", "Camp"],
    "car dealership": ["Motors", "Auto Group"],
    "clinic": ["Clinic", "Health Center", "Medical Clinic"],
    "destination": ["Bay", "Coast", "Valley", "Islands", "Highlands"],
    "district": ["District"],
    "farm": ["Farm", "Farms", "Ranch"],
    "farm field": ["Field", "Paddock", "Pasture"],
    "festival": ["Music Festival", "Fest", "Jazz Festival", "Folk Festival"],
    "fish farm": ["Fish Farm", "Aquaculture Site"],
    "forest": ["Forest", "State Forest", "Woods"],
    "gas processing plant": ["Gas Plant", "Processing Plant"],
    "gym": ["Fitness", "Athletic Club", "Gym"],
    "hospital": ["Hospital", "Medical Center"],
    "hotel": ["Hotel", "Inn", "Suites", "Lodge"],
    "lab": ["Lab", "Research Lab", "Laboratory"],
    "mine": ["Mine", "Project", "Pit"],
    "museum": ["Museum", "Art Museum", "History Museum"],
    "national park": ["National Park"],
    "nursing home": ["Care Home", "Nursing Home", "Senior Living"],
    "parcel depot": ["Depot", "Parcel Hub"],
    "parcel locker": ["Locker", "Pickup Point"],
    "pet shop": ["Pet Supply", "Pets"],
    "pipeline": ["Pipeline", "Gas Line", "Crude Line"],
    "power plant": ["Power Station", "Solar Farm", "Wind Farm", "Generating Station"],
    "rail yard": ["Yard", "Rail Yard"],
    "rental branch": ["Car Rental", "Rentals"],
    "restaurant": ["Kitchen", "Bistro", "Grill", "Cafe", "Trattoria", "Diner"],
    "sawmill": ["Sawmill", "Lumber Mill"],
    "store": ["Market", "Store", "Supercenter"],
    "substation": ["Substation"],
    "trail": ["Trail", "Loop", "Ridge Trail"],
    "transit depot": ["Depot", "Bus Garage"],
    "transit station": ["Station"],
    "treatment plant": ["Treatment Plant", "Water Works"],
    "venue": ["Arena", "Hall", "Amphitheater", "Theater"],
    "warehouse": ["Distribution Center", "Warehouse", "Logistics Hub"],
    "water source": ["Reservoir", "Aquifer", "River Intake", "Wellfield"],
    "weather station": ["Weather Station", "Observatory"],
    "wildlife reserve": ["Wildlife Reserve", "Nature Reserve"],
    "tour": ["Walking Tour", "Food Tour", "Day Trip", "Boat Tour"],
    "sports team": ["Hawks", "Rovers", "Comets", "Titans", "Wolves", "Falcons", "Mariners"],
}
# Kind -> suffixes for organization names: "<coined word> <suffix>".
ORG_SUFFIXES = {
    "company": ["Inc", "Group", "Co.", "Holdings"],
    "energy company": ["Energy", "Power", "Resources"],
    "mining company": ["Mining", "Resources", "Minerals"],
    "freight carrier": ["Freight", "Logistics", "Transport"],
    "venture firm": ["Ventures", "Capital", "Partners"],
    "institutional investor": ["Pension Fund", "Endowment", "Capital"],
    "asset manager": ["Asset Management", "Investors"],
    "real estate brokerage": ["Realty", "Properties"],
    "record label": ["Records", "Music"],
    "game studio": ["Games", "Studios", "Interactive"],
    "studio": ["Pictures", "Studios", "Films"],
    "film distributor": ["Releasing", "Distribution"],
    "publisher": ["Press", "Books", "Publishing"],
    "news outlet": ["Times", "Herald", "News", "Post", "Gazette"],
    "streaming platform": ["Stream", "Play", "TV"],
    "contractor": ["Plumbing", "Electric", "Home Services", "Roofing"],
    "tour operator": ["Tours", "Travel", "Expeditions"],
    "feed supplier": ["Feeds", "Nutrition"],
    "food supplier": ["Foods", "Provisions", "Farms"],
    "merchant": ["Market", "Coffee", "Outfitters", "Electronics"],
    "garment factory": ["Garments", "Apparel", "Textiles"],
    "fashion brand": ["Studio", "Atelier", "& Co."],
    "beauty brand": ["Beauty", "Skin", "Cosmetics"],
    "water utility": ["Water", "Water Authority", "Utilities"],
    "league": ["League", "Premier League", "Conference"],
    "esports team": ["Gaming", "Esports"],
    "tournament": ["Open", "Cup", "Championship", "Invitational"],
    "medical device": ["Monitor", "Pump", "Stent", "Scanner", "Implant"],
    "game": ["Legends", "Tactics", "Odyssey", "Arena", "Frontier"],
}
# Kind -> nouns for title and product names: "<adjective> <noun>".
TITLE_NOUNS = {
    "product": ["Blender", "Backpack", "Desk Lamp", "Headphones", "Kettle", "Water Bottle", "Throw Pillow"],
    "beauty product": ["Serum", "Cleanser", "Lipstick", "Moisturizer", "Face Mask", "Eye Cream"],
    "garment": ["Linen Shirt", "Denim Jacket", "Wool Coat", "Midi Dress", "Cargo Pants", "Knit Sweater"],
    "dish": ["Ramen", "Risotto", "Tacos", "Salmon", "Curry", "Burger", "Flatbread", "Noodle Bowl"],
    "pet product": ["Dog Bed", "Cat Tree", "Chew Toy", "Kibble", "Leash", "Litter Box"],
    "virtual item": ["Sword", "Skin", "Helmet", "Emote", "Mount", "Banner"],
    "achievement": ["Explorer", "Champion", "Collector", "Survivor", "Strategist"],
    "album": NOUNS,
    "song": NOUNS,
    "film": NOUNS,
    "book": NOUNS,
    "artwork": ["Study", "Portrait", "Landscape", "Still Life", "Composition"],
    "exhibition": ["Visions", "Echoes", "Horizons", "Traces", "Currents"],
    "ad campaign": ["Campaign", "Launch", "Promo"],
    "newsletter": ["Weekly", "Digest", "Briefing", "Dispatch"],
    "project": ["Initiative", "Program", "Study", "Platform"],
    "music artist": ["Comets", "Lanterns", "Rivers", "Echoes", "Sparrows", "Satellites"],
    "vessel": ["Star", "Pioneer", "Voyager", "Spirit", "Navigator"],
}
HEADLINE_SUBJECTS = ["council", "school board", "local team", "city hall", "transit agency", "hospital", "startup"]
HEADLINE_VERBS = [
    "approves new budget",
    "faces budget shortfall",
    "wins regional title",
    "opens new site",
    "announces expansion",
    "delays vote",
    "reports record year",
    "cuts ribbon on project",
]
PROPOSAL_ACTIONS = [
    "Increase staking rewards",
    "Fund developer grants",
    "Lower trading fees",
    "Add new collateral",
    "Upgrade the oracle",
    "Expand the treasury",
    "Pause token emissions",
    "Adopt a new fee switch",
]


def _coined(rng: random.Random) -> str:
    word = "".join(rng.choice(SYLLABLES) for _ in range(rng.randint(2, 3))) + rng.choice(ENDINGS)
    return word.capitalize()


def _make(kind: str, rng: random.Random) -> str:
    k = kind.lower()
    if k in PLACE_SUFFIXES:
        return f"{rng.choice(PLACES)} {rng.choice(PLACE_SUFFIXES[k])}"
    if k in ORG_SUFFIXES:
        return f"{rng.choice(COINED)} {rng.choice(ORG_SUFFIXES[k])}"
    if k in TITLE_NOUNS:
        name = f"{rng.choice(ADJECTIVES)} {rng.choice(TITLE_NOUNS[k])}"
        return f"The {name}" if k in ("film", "book") and rng.random() < 0.5 else name
    if k == "tail number":
        letters = "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(2))
        return f"N{rng.randint(100, 999)}{letters}"
    if k == "aircraft model":
        return f"{rng.choice('ABEGKMSX')}{rng.choice('ACRTX')}-{rng.randint(1, 9)}{rng.randint(0, 9)}0"
    if k in ("vehicle model", "drug", "startup"):
        return _coined(rng)
    if k == "drilling rig":
        return f"{rng.choice(COINED)} {rng.randint(1, 40)}"
    if k == "fund":
        return f"{rng.choice(COINED)} Fund {rng.choice(['I', 'II', 'III', 'IV', 'V'])}"
    if k in ("username", "gamer tag"):
        return f"{_coined(rng).lower()}{rng.randint(1, 999)}"
    if k == "hashtag":
        return f"#{rng.choice(ADJECTIVES).lower()}{rng.choice(NOUNS).lower()}"
    if k == "headline":
        return f"{rng.choice(PLACES)} {rng.choice(HEADLINE_SUBJECTS)} {rng.choice(HEADLINE_VERBS)}"
    if k == "governance proposal":
        return rng.choice(PROPOSAL_ACTIONS)
    return f"{rng.choice(PLACES)} {kind.title()}"  # unknown kind: still a readable, fictional name


def unique_names(kind: str, n: int, rng: random.Random) -> list[str]:
    """n distinct names of one kind; when the word lists run short, a number keeps them distinct."""
    names: list[str] = []
    seen: set[str] = set()
    while len(names) < n:
        for _ in range(30):
            name = _make(kind, rng)
            if name not in seen:
                break
        else:
            name = f"{name} {len(names) + 1}"
        seen.add(name)
        names.append(name)
    return names
