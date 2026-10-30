"""Heuristic "no personal names" gate for public-facing draft content.

Used twice:
  * before rendering (verticals.draft): fail fast, with one automatic script
    regeneration that is told which phrases were flagged;
  * after rendering (scripts/validate_no_personal_names.py): a safety net over
    the final metadata and burned-in captions.

Design: a capitalised run is a personal name only when a known given name is
followed by a capitalised word that is not an organisation/product/place word.
Known multi-word organisations, media outlets and products are masked first,
organisation/place suffixes ("... Index", "... Bank", "... River") turn a run
into an organisation, and given names that are also everyday words ("Will",
"Mark", "Grace", "Austin") only count when a common surname follows. Explicit
public-figure hints always block.
"""
from __future__ import annotations

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

ALLOWED = {
    "AI", "API", "CEO", "NASA", "YouTube", "TikTok", "Instagram", "Google",
    "Gemini", "OpenAI", "Microsoft", "Apple", "Meta", "Amazon", "Nvidia",
    "Anthropic", "ChatGPT", "GitHub", "Reddit", "Pexels", "Python", "Android",
    "Linux", "Windows", "Cloudflare", "DYPOL", "Signal", "Shorts", "Reels",
    "The", "This", "That", "These", "But", "Here", "What", "Why", "How",
    "When", "Now", "Today", "More", "Traditional", "Quality", "Adapt", "Most",
    "First", "New", "Live", "Video", "Technology", "Tech", "Artificial",
    "Intelligence", "Machine", "Learning", "Generative", "Autonomous",
    "Cinematic", "Minimalist", "Photorealistic", "Electric", "Blue", "Dark",
    "Moody", "High", "Detail", "Shallow", "Future", "Work", "Replacing",
    "Editors", "Agents", "Explained", "Breaking", "Watch", "Discover", "Changing",
    "Editing", "Forever", "Are", "Is", "By", "For", "And", "Of", "With", "From",
    "On", "In", "To", "At", "As", "Over",
}
# Organizations, markets and places that appear in tech-news scripts (not personal names).
ALLOWED |= {
    "Associated", "Press", "Wall", "Street", "Nasdaq", "Claude", "Broadcom", "Oracle",
    "Intel", "AMD", "Tesla", "SpaceX", "Samsung", "Reuters", "Bloomberg", "Silicon",
    "Valley", "Securities", "Exchange", "Commission", "Stock", "Market", "Federal",
    "Reserve", "United", "States", "Big", "IPO", "History", "Biggest", "Coming", "Public",
    "Offering", "Initial", "Blackstone",
}
# Companies, products, AI models/assistants and outlets whose names collide with given names.
ALLOWED |= {
    "Alexa", "Siri", "Cortana", "Bard", "Grok", "Llama", "Mistral", "Devin", "Jules",
    "Copilot", "Gemma", "Phi", "Ernie", "Qwen", "Kimi", "Jasper", "Sora", "Veo", "Nova",
    "Atlas", "Optimus", "Neo", "Figure", "Spot", "Ada", "Julia", "Rust", "Swift", "Kotlin",
    "Pascal", "Ampere", "Hopper", "Blackwell", "Rubin", "Vera", "Lovelace", "Turing",
    "Maxwell", "Kepler", "Volta", "Ryzen", "Xeon", "Snapdragon", "Tensor", "Pixel",
    "Galaxy", "Edison", "Watson", "Dell", "Ford", "Lincoln", "Mercedes", "Porsche",
    "Chevy", "Chevrolet", "Toyota", "Honda", "Forbes", "Fortune", "Hilton", "Marriott",
    "Wells", "Fargo", "Morgan", "Chase", "Schwab", "Goldman", "Sachs", "Barclays", "Citi",
    "Harley", "Davidson", "Boeing", "Airbus", "Lockheed", "Martin", "Raytheon", "Siemens",
    "Bosch", "Philips", "Sony", "Nintendo", "Xbox", "PlayStation", "Netflix", "Disney",
    "Spotify", "Uber", "Lyft", "Airbnb", "Stripe", "Shopify", "Salesforce", "Adobe",
    "Qualcomm", "Arm", "TSMC", "Micron", "Huawei", "Xiaomi", "Alibaba", "Tencent",
    "Baidu", "ByteDance", "DeepSeek", "Perplexity", "Cohere", "Midjourney", "Runway",
    "Hugging", "Face", "Databricks", "Snowflake", "Palantir", "IBM", "Cisco", "Oracle",
    "Reliance", "Jio", "Infosys", "Wipro", "Tata", "Flipkart", "Paytm", "Zomato",
    "Swiggy", "Nike", "Perl", "Norton", "McAfee", "Avast", "Kaspersky", "Lenovo", "Asus", "Acer", "CNBC", "CNN", "BBC", "Axios", "TechCrunch", "Wired", "Engadget", "Verge",
    "Gizmodo", "Mashable", "Politico", "Semafor", "Guardian", "Telegraph", "Economist",
    "Nikkei", "Xinhua", "Yonhap", "Jazeera", "Fox", "Sky", "Variety", "Reuters",
    "Harvard", "Yale", "Stanford", "Berkeley", "Princeton", "Cambridge", "Oxford", "MIT",
    "Caltech", "Wharton", "Nobel", "Pulitzer", "Grammy", "Oscar", "Emmy", "Olympics",
    "Christmas", "Easter", "Diwali", "Ramadan", "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday", "January", "February", "March", "April",
    "May", "June", "July", "August", "September", "October", "November", "December",
}
ALLOWED_LOWER = {word.casefold() for word in ALLOWED}

# Multi-word organisations, outlets and products; masked before the scan.
KNOWN_PHRASES = [
    "Associated Press", "Wall Street", "Wall Street Journal", "Morgan Stanley", "JPMorgan Chase",
    "Charles Schwab", "Goldman Sachs", "Wells Fargo", "Dow Jones", "Lockheed Martin",
    "Rolls Royce", "Rolls-Royce", "Mercedes Benz", "Mercedes-Benz", "Harley Davidson",
    "Harley-Davidson", "Hewlett Packard", "Johnson & Johnson", "Procter & Gamble",
    "Ernst & Young", "Ben & Jerry's", "Jack Daniel's", "Dolby Atmos", "Tom's Hardware",
    "Tom's Guide", "New York Times", "Washington Post", "Financial Times", "Los Angeles Times",
    "Business Insider", "Ars Technica", "Fox News", "Sky News", "Al Jazeera", "Bloomberg News",
    "Daily Mail", "The Verge", "Artificial Analysis", "Grace Hopper", "Grace Blackwell",
    "Vera Rubin", "Ada Lovelace", "Alan Turing Institute", "Hugging Face", "Merrill Lynch",
    "Bank of America", "Arthur Andersen", "Booz Allen", "Kaiser Permanente", "Johns Hopkins",
    "Carnegie Mellon", "Georgia Tech", "Virginia Tech", "Texas Instruments", "Paramount Pictures",
    "Warner Bros", "Walt Disney", "Charles River",
]
_PHRASE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in sorted(KNOWN_PHRASES, key=len, reverse=True)) + r")\b"
)

# A run containing one of these right after the name is an organisation, place or thing.
ORG_SUFFIXES = {
    "inc", "corp", "corporation", "co", "company", "group", "holdings", "index", "labs", "lab",
    "bank", "capital", "ventures", "partners", "technologies", "technology", "tech", "systems",
    "motors", "motor", "press", "times", "post", "journal", "news", "herald", "tribune",
    "gazette", "magazine", "media", "network", "networks", "university", "college", "institute",
    "foundation", "association", "society", "agency", "council", "commission", "committee",
    "department", "ministry", "bureau", "authority", "board", "court", "congress", "senate",
    "parliament", "street", "avenue", "road", "square", "valley", "bay", "city", "county",
    "river", "lake", "falls", "mountain", "mountains", "island", "islands", "beach", "park",
    "airport", "station", "stadium", "center", "centre", "hospital", "clinic", "school",
    "academy", "act", "agreement", "accord", "treaty", "protocol", "summit", "conference",
    "expo", "awards", "award", "prize", "cup", "league", "fed", "reserve", "exchange", "market",
    "markets", "fund", "trust", "securities", "insurance", "airlines", "airways", "energy",
    "electric", "power", "pharmaceuticals", "pharma", "health", "healthcare", "robotics",
    "dynamics", "semiconductor", "semiconductors", "software", "studios", "studio", "games",
    "entertainment", "pictures", "records", "foods", "brands", "industries", "enterprises",
    "enterprise", "solutions", "services", "cloud", "platform", "platforms", "model", "models",
    "benchmark", "leaderboard", "report", "survey", "arena", "bench", "chip", "chips", "phone",
    "pro", "max", "ultra", "plus", "mini", "air", "lite", "edition", "series", "version",
    "update", "mode", "engine", "kit", "os", "hub", "app", "apps", "store", "wallet", "pay",
    "superchip", "gpu", "gpus", "cpu", "processor", "architecture", "effect", "law", "prize",
}

PLACES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut",
    "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas",
    "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada", "ohio", "oklahoma", "oregon",
    "pennsylvania", "tennessee", "texas", "utah", "vermont", "virginia", "washington",
    "wisconsin", "wyoming", "america", "canada", "mexico", "brazil", "argentina", "chile",
    "britain", "england", "scotland", "ireland", "wales", "france", "germany", "spain",
    "italy", "portugal", "netherlands", "belgium", "switzerland", "austria", "sweden",
    "norway", "denmark", "finland", "poland", "ukraine", "russia", "turkey", "israel", "iran",
    "iraq", "egypt", "nigeria", "kenya", "africa", "europe", "asia", "india", "pakistan",
    "bangladesh", "china", "japan", "korea", "taiwan", "singapore", "malaysia", "indonesia",
    "vietnam", "thailand", "philippines", "australia", "zealand", "london", "paris", "berlin",
    "tokyo", "beijing", "shanghai", "shenzhen", "seoul", "mumbai", "delhi", "bengaluru",
    "bangalore", "pune", "hyderabad", "chennai", "kolkata", "dubai", "york", "jersey",
    "angeles", "francisco", "chicago", "boston", "seattle", "austin", "dallas", "houston",
    "denver", "atlanta", "miami", "phoenix", "brussels", "geneva", "zurich", "amsterdam",
    "stockholm", "toronto", "vancouver", "sydney", "melbourne", "europe", "arctic", "pacific",
    "atlantic", "beltway", "brooklyn", "manhattan", "hollywood", "redmond", "cupertino",
}

# Given names that are also everyday words, places or brands: they only count as
# a name when a common surname follows ("Will Smith"), not "Will Robots Take Jobs".
AMBIGUOUS_FIRST = {
    "will", "mark", "bill", "ray", "max", "chase", "hunter", "grant", "art", "rich", "page",
    "park", "price", "prince", "king", "major", "mayor", "duke", "earl", "guy", "case",
    "chance", "way", "west", "wood", "hill", "hall", "rock", "sky", "shadow", "shell", "town",
    "trip", "urban", "yard", "wait", "wake", "wash", "web", "win", "wit", "saw", "say", "see",
    "son", "tan", "temp", "temple", "trace", "tray", "job", "lay", "er", "ev", "ez", "geo",
    "mic", "pen", "sig", "sim", "si", "tab", "ave", "barn", "bay", "bear", "cat", "lion",
    "wolf", "salmon", "ram", "red", "rice", "ware", "worth", "worthy", "noble", "royal",
    "sterling", "sunny", "slim", "skip", "chip", "buck", "buddy", "ace", "angel", "apollo",
    "christ", "jesus", "allah", "cleveland", "orlando", "madison", "richmond", "raleigh",
    "rochester", "salem", "kingston", "sheffield", "somerset", "jackson", "houston", "denver",
    "phoenix", "chelsea", "florence", "victoria", "paris", "london", "mac", "french", "welsh",
    "gay", "merry", "oral", "pace", "forest", "grove", "haven", "heath", "glen", "brook",
    "marsh", "moss", "cliff", "dale", "lane", "penn", "pierce", "porter", "ward", "walker",
    "turner", "grace", "hope", "faith", "joy", "rose", "ruby", "amber", "crystal", "jade",
    "pearl", "iris", "lily", "holly", "ivy", "olive", "violet", "dawn", "summer", "autumn",
    "april", "june", "may", "august", "carol", "christian", "harmony", "honor", "justice",
    "liberty", "mercy", "patience", "precious", "trinity", "storm", "stone", "river", "sage",
    "sol", "frank", "drew", "gene", "lance", "miles", "rod", "wade", "dean", "pat", "sandy",
    "rusty", "penny", "sue", "victor", "rocky", "marshall", "tucker", "dallas", "austin",
    "georgia", "virginia", "carolina", "dakota", "india", "asia", "china", "kenya", "jordan",
    "israel", "sydney", "nevada", "ohio", "lincoln", "dell", "ford", "harley", "jack", "nick",
    "bob", "rob", "don", "ken", "kit", "lou", "tom", "billy", "bobby", "freddie", "winston",
    "darwin", "newton", "edison", "tesla", "hilton", "forbes", "chevy", "mercedes", "devon",
    "kent", "cameron", "logan", "mason", "parker", "taylor", "tyler", "spencer", "carter",
    "harper", "quinn", "reese", "riley", "rowan", "skyler", "morgan", "kelly", "lee",
    "mag", "de", "fan", "pet", "di", "del", "carry", "em", "con", "row", "bird", "les", "dot",
    "lib", "cam", "costa", "von", "van", "ag", "vi", "roman", "valentine", "reg", "ebony", "glad",
    "charity", "hunt", "clay", "eve", "glory", "cal", "doll", "cookie", "val", "rica", "camel",
    "bishop", "jan", "al", "gates", "else", "charlotte", "alberta", "nova", "luna", "aurora", "hazel", "ginger", "candy", "honey", "cherry", "misty", "stormy",
}

# Common surnames: turn an ambiguous first word back into a name ("Mark Smith").
SURNAMES = {
    "smith", "johnson", "williams", "brown", "jones", "garcia", "miller", "davis", "rodriguez",
    "martinez", "hernandez", "lopez", "gonzalez", "wilson", "anderson", "thomas", "taylor",
    "moore", "jackson", "martin", "lee", "perez", "thompson", "white", "harris", "sanchez",
    "clark", "ramirez", "lewis", "robinson", "walker", "young", "allen", "king", "wright",
    "scott", "torres", "nguyen", "hill", "flores", "green", "adams", "nelson", "baker", "hall",
    "rivera", "campbell", "mitchell", "carter", "roberts", "gomez", "phillips", "evans",
    "turner", "diaz", "parker", "cruz", "edwards", "collins", "reyes", "stewart", "morris",
    "morales", "murphy", "cook", "rogers", "gutierrez", "ortiz", "morgan", "cooper", "peterson",
    "bailey", "reed", "kelly", "howard", "ramos", "kim", "cox", "ward", "richardson", "watson",
    "brooks", "chavez", "wood", "james", "bennett", "gray", "mendoza", "ruiz", "hughes", "price",
    "alvarez", "castillo", "sanders", "patel", "myers", "long", "ross", "foster", "jimenez",
    "powell", "jenkins", "perry", "russell", "sullivan", "bell", "coleman", "butler",
    "henderson", "barnes", "gonzales", "fisher", "vasquez", "simmons", "romero", "jordan",
    "patterson", "alexander", "hamilton", "graham", "reynolds", "griffin", "wallace", "moreno",
    "west", "cole", "hayes", "bryant", "herrera", "gibson", "ellis", "tran", "medina", "aguilar",
    "stevens", "murray", "ford", "castro", "marshall", "owens", "harrison", "fernandez",
    "mcdonald", "woods", "washington", "kennedy", "wells", "vargas", "henry", "chen", "freeman",
    "webb", "tucker", "guzman", "burns", "crawford", "olson", "simpson", "porter", "hunter",
    "gordon", "mendez", "silva", "shaw", "snyder", "mason", "dixon", "munoz", "hunt", "hicks",
    "holmes", "palmer", "wagner", "black", "robertson", "boyd", "rose", "stone", "salazar",
    "fox", "warren", "mills", "meyer", "rice", "schmidt", "garza", "daniels", "ferguson",
    "nichols", "stephens", "soto", "weaver", "ryan", "gardner", "payne", "grant", "dunn",
    "kelley", "spencer", "hawkins", "arnold", "pierce", "vazquez", "hansen", "peters", "santos",
    "hart", "bradley", "knight", "elliott", "cunningham", "duncan", "armstrong", "hudson",
    "carroll", "lane", "riley", "andrews", "alvarado", "ray", "delgado", "berry", "perkins",
    "hoffman", "johnston", "matthews", "pena", "richards", "contreras", "willis", "carpenter",
    "lawrence", "sandoval", "guerrero", "george", "chapman", "rios", "estrada", "ortega",
    "watkins", "greene", "nunez", "wheeler", "valdez", "harper", "burke", "larson", "santiago",
    "maldonado", "morrison", "franklin", "carlson", "austin", "dominguez", "carr", "lawson",
    "jacobs", "obrien", "lynch", "singh", "vega", "bishop", "montgomery", "oliver", "jensen",
    "harvey", "williamson", "gilbert", "dean", "sims", "espinoza", "howell", "li", "wong",
    "reid", "hanson", "le", "mccoy", "garrett", "burton", "fuller", "wang", "weber", "welch",
    "rojas", "lucas", "marquez", "fields", "park", "yang", "little", "banks", "padilla", "day",
    "walsh", "bowman", "schultz", "luna", "fowler", "mejia", "davidson", "acosta", "brewer",
    "may", "holland", "juarez", "newman", "pearson", "curtis", "cortez", "douglas", "schneider",
    "joseph", "barrett", "navarro", "figueroa", "keller", "avila", "wade", "molina", "stanley",
    "hopkins", "campos", "barnett", "bates", "chambers", "caldwell", "beck", "lambert",
    "miranda", "byrd", "craig", "ayala", "lowe", "frazier", "powers", "neal", "leonard",
    "gregory", "carrillo", "sutton", "fleming", "rhodes", "shelton", "schwartz", "norris",
    "jennings", "watts", "duran", "walters", "cohen", "mcdaniel", "moran", "parks", "steele",
    "vaughn", "becker", "holt", "deleon", "barker", "terry", "hale", "leon", "hail", "benson",
    "haynes", "horton", "miles", "lyons", "pham", "graves", "bush", "thornton", "wolfe",
    "warner", "cabrera", "mckinney", "mann", "zimmerman", "dawson", "lara", "fletcher", "page",
    "mccarthy", "love", "robles", "cervantes", "solis", "erickson", "reeves", "chang", "klein",
    "salinas", "fuentes", "baldwin", "daniel", "simon", "velasquez", "hardy", "higgins",
    "aguirre", "lin", "cummings", "chandler", "sharp", "barber", "bowen", "ochoa", "dennis",
    "robbins", "liu", "ramsey", "francis", "griffith", "paul", "blair", "oconnor", "cardenas",
    "pacheco", "cross", "calderon", "quinn", "moss", "swanson", "chan", "rivas", "khan",
    "rodgers", "serrano", "fitzgerald", "rosales", "stevenson", "christensen", "manning",
    "gill", "curry", "mclaughlin", "harmon", "mcgee", "gross", "doyle", "garner", "newton",
    "burgess", "reese", "walton", "blake", "trujillo", "adkins", "brady", "goodman", "roman",
    "webster", "goodwin", "fischer", "huang", "potter", "delacruz", "montoya", "todd", "wu",
    "hines", "mullins", "castaneda", "malone", "cannon", "tate", "mack", "sherman", "hubbard",
    "hodges", "zhang", "guerra", "wolf", "valencia", "saunders", "franco", "rowe", "gallagher",
    "farmer", "hammond", "hampton", "townsend", "ingram", "wise", "gallegos", "clarke",
    "barton", "schroeder", "maxwell", "waters", "logan", "camacho", "strickland", "norman",
    "person", "colon", "parsons", "frank", "harrington", "glover", "osborne", "buchanan",
    "casey", "floyd", "patton", "ibarra", "ball", "tyler", "suarez", "bowers", "orozco",
    "salas", "cobb", "gibbs", "andrade", "bauer", "conner", "moody", "escobar", "mcguire",
    "lloyd", "mueller", "hartman", "french", "kramer", "mcbride", "pope", "lindsey", "velazquez",
    "norton", "mccormick", "sparks", "flynn", "yates", "hogan", "marsh", "macias", "villanueva",
    "zamora", "pratt", "stokes", "owen", "ballard", "lang", "brock", "villarreal", "charles",
    "drake", "barrera", "cain", "patrick", "pineda", "burnett", "mercado", "santana", "shepherd",
    "bautista", "ali", "shaffer", "lamb", "trevino", "mckenzie", "hess", "beil", "olsen",
    "cochran", "morton", "nash", "wilkins", "petersen", "briggs", "shah", "roth", "nicholson",
    "holloway", "lozano", "rangel", "flowers", "hoover", "short", "arias", "mora", "valenzuela",
    "bryan", "meyers", "weiss", "underwood", "bass", "greer", "summers", "houston", "carson",
    "morrow", "clayton", "whitaker", "decker", "yoder", "collier", "zuniga", "carey", "wilcox",
    "melendez", "poole", "roberson", "larsen", "conley", "davenport", "copeland", "massey",
    "lam", "huff", "rocha", "cameron", "jefferson", "hood", "monroe", "anthony", "pittman",
    "huynh", "randall", "singleton", "kirk", "combs", "mathis", "christian", "skinner",
    "bradford", "richard", "galvan", "wall", "boone", "kirby", "wilkinson", "bridges", "bruce",
    "atkinson", "velez", "meza", "roy", "vincent", "york", "hodge", "villa", "abbott", "allison",
    "tapia", "gates", "chase", "sosa", "sweeney", "farrell", "wyatt", "dalton", "horn", "barron",
    "phelps", "yu", "dickerson", "heath", "foley", "atkins", "mathews", "bonilla", "acevedo",
    "benitez", "zavala", "hensley", "glenn", "cisneros", "harrell", "shields", "rubio", "choi",
    "huffman", "boyer", "garrison", "arroyo", "bond", "kane", "hancock", "callahan", "dillon",
    "cline", "wiggins", "grimes", "arellano", "melton", "oneill", "savage", "ho", "beltran",
    "pitts", "parrish", "ponce", "rich", "booth", "koch", "golden", "ware", "brennan", "mcdowell",
    "marks", "cantu", "humphrey", "baxter", "sawyer", "clay", "tanner", "hutchinson", "kaur",
    "berg", "wiley", "gilmore", "russo", "villegas", "hobbs", "keith", "wilkerson", "ahmed",
    "beard", "mcclain", "montes", "mata", "rosario", "vang", "walter", "henson", "oneal",
    "mosley", "mcclure", "beasley", "stephenson", "snow", "huerta", "preston", "vance", "barry",
    "johns", "eaton", "blackwell", "dyer", "prince", "macdonald", "solomon", "guevara",
    "stafford", "english", "hurst", "woodard", "cortes", "shannon", "kemp", "nolan", "mccullough",
    "merritt", "murillo", "moon", "salgado", "strong", "kline", "cordova", "barajas", "roach",
    "rosas", "winters", "jacobson", "lester", "knox", "bullock", "kerr", "leach", "meadows",
    "orr", "davila", "whitehead", "pruitt", "kent", "conway", "mckee", "barr", "david", "dejesus",
    "marin", "berger", "mcintyre", "blankenship", "gaines", "palacios", "cuevas", "bartlett",
    "durham", "dorsey", "mccall", "odonnell", "stein", "browning", "stout", "lowery", "sloan",
    "mclean", "hendricks", "calhoun", "sexton", "chung", "gentry", "hull", "duarte", "ellison",
    "nielsen", "gillespie", "buck", "middleton", "sellers", "leblanc", "esparza", "hardin",
    "bradshaw", "mcintosh", "howe", "livingston", "frost", "glass", "morse", "knapp", "herman",
    "stark", "bravo", "noble", "spears", "weeks", "corona", "frederick", "buckley", "mcfarland",
    "hebert", "enriquez", "hickman", "quintero", "randolph", "schaefer", "walls", "trejo",
    "house", "reilly", "pennington", "michael", "conrad", "giles", "benjamin", "crosby",
    "fitzpatrick", "donovan", "mays", "mahoney", "valentine", "raymond", "medrano", "hahn",
    "mcmillan", "small", "bentley", "felix", "peck", "lucero", "boyle", "hanna", "pace", "rush",
    "hurley", "harding", "mcconnell", "bernal", "nava", "ayers", "everett", "ventura", "avery",
    "pugh", "mayer", "bender", "shepard", "mcmahon", "landry", "case", "sampson", "moses",
    "magana", "blackburn", "dunlap", "gould", "duffy", "vaughan", "herring", "mckay", "espinosa",
    "rivers", "farley", "bernard", "ashley", "friedman", "potts", "truong", "costa", "correa",
    "blevins", "nixon", "clements", "fry", "delarosa", "best", "benton", "lugo", "portillo",
    "dougherty", "crane", "haley", "phan", "villalobos", "blanchard", "horne", "finley",
    "quintana", "lynn", "esquivel", "bean", "dodson", "mullen", "xiong", "hayden", "cano",
    "levy", "huber", "richmond", "moyer", "lim", "frye", "sheppard", "mccarty", "avalos",
    "booker", "waller", "parra", "woodward", "jaramillo", "krueger", "rasmussen", "brandt",
    "peralta", "donaldson", "stuart", "faulkner", "maynard", "galindo", "coffey", "estes",
    "sanford", "burch", "maddox", "vo", "oconnell", "vu", "andersen", "spence", "mcpherson",
    "church", "schmitt", "stanton", "leal", "cherry", "compton", "dudley", "sierra", "pollard",
    "alfaro", "hester", "proctor", "lu", "hinton", "novak", "good", "madden", "mccann", "terrell",
    "jarvis", "dickson", "reyna", "cantrell", "mayo", "branch", "hendrix", "rollins", "rowland",
    "whitney", "duke", "odom", "daugherty", "travis", "tang", "kumar", "sharma", "gupta",
    "verma", "reddy", "rao", "iyer", "nair", "mehta", "joshi", "agarwal", "chaturvedi",
    "mishra", "pandey", "tiwari", "yadav", "das", "bose", "banerjee", "chatterjee", "mukherjee",
    "ghosh", "sen", "pillai", "menon", "desai", "jain", "kapoor", "malhotra", "chopra", "bhatt",
    "nadella", "pichai", "musk", "altman", "bezos", "zuckerberg", "amodei", "hassabis",
}

NAME_RUN = re.compile(
    r"\b[A-Z][a-z]+(?:[-'\u2019][A-Z]?[a-z]+)?(?:\s+[A-Z][a-z]+(?:[-'\u2019][A-Z]?[a-z]+)?){1,3}\b"
)
_HINT_NAMES = [
    "Elon", "Sundar", "Satya", "Jensen", "Demis", "Dario", "Sam Altman", "Mark Zuckerberg",
    "Bill Gates", "Steve Jobs", "Tim Cook", "Jeff Bezos", "Andy Jassy", "Lisa Su",
    "Pat Gelsinger", "Lip-Bu Tan", "Mark Gurman", "Marques Brownlee", "Greg Brockman",
    "Ilya Sutskever", "Yann LeCun", "Fei-Fei Li", "Mustafa Suleyman", "Mira Murati",
    "Andrej Karpathy", "Geoffrey Hinton", "Linus Torvalds", "Carl Pei", "Masayoshi Son",
    "Arvind Krishna", "Shantanu Narayen", "Brian Chesky", "Reed Hastings", "Jack Dorsey",
    "Evan Spiegel", "Larry Page", "Larry Ellison", "Sergey Brin", "Donald Trump", "Joe Biden",
    "Kamala Harris", "Narendra Modi", "Xi Jinping", "Vladimir Putin", "Mukesh Ambani",
    "Gautam Adani", "Musk", "Zuckerberg", "Altman", "Bezos", "Pichai", "Nadella", "Amodei",
    "Hassabis", "Karpathy", "Sutskever", "LeCun", "Murati", "Jassy", "Biden", "Trump", "Obama",
]
NAME_HINT = re.compile(r"\b(?:" + "|".join(re.escape(n) for n in _HINT_NAMES) + r")\b", re.I)

_NAMES_FILE = Path(__file__).resolve().parent.parent / "scripts" / "given_names.txt"
GIVEN_NAMES = (
    {n.strip().casefold() for n in _NAMES_FILE.read_text(encoding="utf-8").splitlines() if n.strip()}
    if _NAMES_FILE.exists() else set()
)

DRAFT_FIELDS = (
    "script", "youtube_title", "youtube_description", "youtube_tags", "instagram_caption",
    "tiktok_caption", "thumbnail_prompt", "thumbnail_text", "title", "description", "tags",
    "captions",
)


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

def _bare(word: str) -> str:
    return word[:-2] if word.endswith(("'s", "\u2019s")) else word


def _is_allowed(word: str) -> bool:
    return word in ALLOWED or word.casefold() in ALLOWED_LOWER


def _is_org_or_place(word: str) -> bool:
    low = word.casefold()
    return low in ORG_SUFFIXES or low in PLACES


def _pair_is_name(words, index) -> bool:
    first, second = words[index], words[index + 1]
    low_first, low_second = first.casefold(), second.casefold()
    if _is_allowed(first) or low_first not in GIVEN_NAMES or low_first in PLACES and low_first not in AMBIGUOUS_FIRST:
        return False
    if _is_allowed(second) or _is_org_or_place(second):
        return False
    # "Florence Nightingale Hospital", "Jordan Peterson Index": named organisations/things.
    if index + 2 < len(words) and words[index + 2].casefold() in ORG_SUFFIXES:
        return False
    if low_first in AMBIGUOUS_FIRST and low_second not in SURNAMES:
        return False
    return True


def scan_text(text) -> list[str]:
    """Return the sorted, de-duplicated phrases that look like personal names."""
    text = str(text)
    hits = [match.group(0) for match in NAME_HINT.finditer(text)]
    masked = _PHRASE_RE.sub(lambda m: " org ", text)
    for match in NAME_RUN.finditer(masked):
        words = [_bare(w) for w in match.group(0).split()]
        for index in range(len(words) - 1):
            if _pair_is_name(words, index):
                hits.append(match.group(0))
                break
    return sorted(set(hits), key=str.casefold)


def draft_fields(draft: dict) -> dict:
    fields = {}
    for key in DRAFT_FIELDS:
        if key in draft:
            value = draft[key]
            fields[key] = ", ".join(map(str, value)) if isinstance(value, list) else str(value)
    for index, value in enumerate(draft.get("broll_prompts", []) or []):
        fields["broll_prompts[" + str(index) + "]"] = str(value)
    for index, intent in enumerate(draft.get("visual_intents", []) or []):
        if isinstance(intent, dict) and intent.get("entities"):
            fields["visual_intents[" + str(index) + "].entities"] = ", ".join(map(str, intent["entities"]))
    metadata = draft.get("upload_metadata") or {}
    for key in ("title", "description", "tags"):
        if key in metadata:
            value = metadata[key]
            fields["upload_metadata." + key] = ", ".join(map(str, value)) if isinstance(value, list) else str(value)
    return fields


def find_names(draft: dict) -> dict:
    """{field: [flagged phrases]} for every field with likely personal names."""
    found = {}
    for field, value in draft_fields(draft).items():
        hits = scan_text(value)
        if hits:
            found[field] = hits
    return found


def flagged_phrases(found: dict) -> list[str]:
    return sorted({phrase for phrases in found.values() for phrase in phrases}, key=str.casefold)


def validate_draft(draft: dict, caption_paths=()) -> list[str]:
    issues = [field + ": likely personal-name text detected: " + ", ".join(hits)
              for field, hits in find_names(draft).items()]
    for path in caption_paths:
        path = Path(path)
        if not path.exists():
            issues.append(str(path) + ": caption file missing")
            continue
        found = scan_text(path.read_text(encoding="utf-8", errors="replace"))
        if found:
            issues.append(path.name + ": likely personal-name text detected: " + ", ".join(found))
    return issues
