"""Paths, fixed settings and the outside facts used only to check the model, never to train it."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
FRONTEND = ROOT / "frontend"

SEED = 7
POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]
MIN_READINGS = 4        # a day needs at least this many pollutants to get a score

# The model learns from everything before FIT_END. Lockdown happens after it.
FIT_END = "2020-01-01"
SELECT_TRAIN_END = "2019-01-01"     # to choose factor count: fit before 2019, score on 2019
K_RANGE = range(1, 9)
ELBOW_RULE = 0.20                   # stop when the next factor adds < 20% of the previous gain
MIN_CITY_DAYS = 300                 # a city needs this many pre-2020 days for rankings and the tree

# Names are our reading of the evidence. The model only ever produces factor numbers.
# Each factor takes the name of the group holding its largest loading.
FACTOR_NAMES = [
    ({"PM2.5", "PM10"}, "Dust and smoke", "Particles: fine and coarse dust, soot, smoke"),
    ({"SO2"}, "Coal and heavy industry", "Sulphur dioxide, with some carbon monoxide"),
    ({"O3"}, "Sunlit ozone", "Ozone: not emitted directly, formed by sunlight on other gases"),
    ({"Benzene", "Toluene", "Xylene"}, "Solvents and petrol vapour", "Benzene, toluene and xylene"),
    ({"NO", "NO2", "NOx", "NH3"}, "Traffic and combustion", "Nitrogen oxides from burning fuel, with ammonia"),
]

# Outside facts for the checks
LOCKDOWN_START = "2020-03-25"       # national lockdown announced for 25 March 2020
LOCKDOWN_END = "2020-04-30"
BEFORE_START = "02-01"              # comparison window in the same year, before lockdown
BEFORE_END = "03-20"
DIWALI = ["2015-11-11", "2016-10-30", "2017-10-19", "2018-11-07", "2019-10-27"]
COAL_BELT = ["Talcher", "Jorapokhar", "Brajrajnagar"]
NORTH = ["Delhi", "Gurugram", "Lucknow", "Patna", "Jaipur", "Chandigarh", "Amritsar"]
SOUTH = ["Chennai", "Bengaluru", "Hyderabad", "Kochi", "Thiruvananthapuram", "Coimbatore", "Visakhapatnam", "Amaravati"]

SOURCES = {
    "data": "https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india",
    "receptor": "https://www.pubmed.gov/28455568/",
    "talcher": "https://www.downtoearth.org.in/news/talcher-thermal-power-plant-shuts-down-ntpc-says-modern-unit-will-replace-it-76269",
    "talcher_belt": "https://www.asianage.com/india/why-odishas-talcher-figured-very-often-in-indias-pollution-charts-this-month-1931252",
    "jorapokhar": "https://www.cerns.io/in/jharia",
    "brajrajnagar": "https://en.wikipedia.org/wiki/Brajarajnagar",
    "variability": "https://indiaspend.com/how-to-pinpoint-delhis-air-pollution-sources-45985",
}
