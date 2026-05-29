"""
Keyword taxonomy for patent categorization.

Each entry maps a human-readable category label to a list of keywords.
Matching is case-insensitive and checks for whole-word or substring presence
(configurable in analyzer.py). Add or remove keywords freely.
"""

CATEGORIES: dict[str, list[str]] = {
    "Aircraft (Fixed-Wing)": [
        # Base terms
        "aircraft", "airplane", "aeroplane", "plane", "jet", "airliner",
        # Military variants
        "fighter", "bomber", "interceptor", "reconnaissance", "transport",
        "cargo aircraft", "tanker", "military aircraft", "attack aircraft",
        # General types
        "biplane", "monoplane", "triplane", "glider", "sailplane",
        "seaplane", "floatplane", "amphibian aircraft", "bush plane",
        # Engine/power types
        "turboprop", "turbofan", "turbojet", "piston aircraft", "propeller",
        "jet engine", "turbine", "electric aircraft", "hybrid aircraft",
        # Speed-related
        "supersonic", "subsonic", "hypersonic", "transonic",
        # Airframe components
        "fuselage", "wing", "airfoil", "aileron", "elevator", "rudder",
        "empennage", "tail section", "landing gear", "cockpit",
        "nacelle", "engine pod", "air intake", "pitot",
        # Specific designs
        "canard", "delta wing", "swept wing", "variable geometry",
        "short takeoff", "stol aircraft", "vtol aircraft",
        # Trainer & specialized
        "trainer aircraft", "aerobatic", "experimental aircraft",
    ],
    "Drone / UAV": [
        # Base terms
        "drone", "uav", "uas", "unmanned aerial", "unmanned aircraft",
        "rpv", "remotely piloted", "pilotless",
        # Multirotor types
        "quadcopter", "hexacopter", "octocopter", "multicopter", "multirotor",
        "quadrotor", "hexarotor", "octorotor",
        # Fixed-wing UAV
        "fixed-wing drone", "fixed-wing uav", "uav aircraft",
        # Applications & variants
        "delivery drone", "camera drone", "surveillance drone",
        "inspection drone", "mapping drone", "agricultural drone",
        "racing drone", "aerial platform", "aerial system",
        # Components
        "gimbal", "payload", "propeller", "battery-electric",
        "fpv", "autonomous drone", "autonomous flight",
        # Size variants
        "nano drone", "micro drone", "mini drone", "small uav",
        "tactical uav", "strategic uav",
    ],
    "Helicopter / Rotorcraft": [
        # Base terms
        "helicopter", "rotorcraft", "chopper", "heli",
        "autogyro", "gyroplane", "gyrocopter", "gyromotor",
        # Helicopter types
        "light helicopter", "heavy helicopter", "utility helicopter",
        "attack helicopter", "transport helicopter", "cargo helicopter",
        "ambulance helicopter", "medevac helicopter", "rescue helicopter",
        # Military variants
        "attack chopper", "gunship", "armed helicopter", "military helicopter",
        # Components
        "rotor blade", "tail rotor", "main rotor", "rotor head",
        "swashplate", "cyclic", "collective", "yaw control",
        "rotor system", "blade root", "blade tip",
        # Specific designs
        "tiltrotor", "convertiplane", "compound helicopter",
        "coaxial", "tandem rotor", "side-by-side rotor",
        # Technical
        "vertical takeoff", "hover", "hovering aircraft",
    ],
    "eVTOL / Advanced Air Mobility": [
        # Base terms
        "evtol", "vtol", "vertical takeoff", "vertical landing",
        "electric vertical", "air taxi", "flying vehicle",
        # Urban air mobility
        "urban air mobility", "uam", "urban aviation",
        "city air mobility", "metropolitan air transport",
        # Personal/advanced
        "flying car", "personal air vehicle", "pav", "personal air transport",
        "advanced air mobility", "aam", "next-gen air transport",
        # Electric/alternative
        "electric aircraft", "battery-electric", "hybrid-electric",
        "electric propulsion", "electric motor", "fuel cell aircraft",
        # Specific concepts
        "air taxi service", "aerial taxi", "autonomous air vehicle",
        "passenger drone", "electric copter", "electric helicopter",
        "tiltwing", "tiltwing aircraft", "tilt-rotor",
        "distributed electric", "distributed propulsion",
        # Emerging terms
        "advanced vehicle", "next generation aircraft", "future aircraft",
        "innovative transport", "sustainable aviation", "zero-emission aircraft",
    ],
    "Spacecraft / Rocket": [
        # Base terms
        "spacecraft", "satellite", "rocket", "launch vehicle",
        "space vehicle", "orbital vehicle", "orbital platform",
        # Specific spacecraft types
        "space shuttle", "space plane", "spaceplane", "shuttle",
        "capsule", "orbiter", "space station", "space module",
        "lander", "lunar lander", "moon lander", "mars lander",
        "probe", "explorer probe", "deep space probe",
        # Rocket types
        "launch vehicle", "expendable launch", "reusable launch",
        "heavy-lift launch", "medium-lift launch", "small-launch vehicle",
        "sounding rocket", "suborbital rocket",
        # Components & systems
        "propulsion", "thruster", "nozzle", "fairing", "payload bay",
        "heat shield", "reentry vehicle", "reentry module",
        "solar panel", "radiator", "antenna", "docking mechanism",
        "staging", "booster", "stage separation",
        # Technical
        "reentry", "ablation", "cryogenic", "hypergolic propellant",
        "solid rocket", "liquid rocket", "hybrid rocket",
    ],
    "Ground Vehicle": [
        # Base terms
        "vehicle", "automobile", "car", "automotive",
        # Vehicle types
        "truck", "bus", "van", "pickup", "sedan", "coupe", "hatchback",
        "wagon", "suv", "sport utility", "crossover", "minivan",
        # Specialist vehicles
        "ambulance", "fire truck", "fire engine", "police car", "patrol car",
        "tow truck", "dump truck", "semi truck", "semi-trailer",
        "cement truck", "tanker truck", "flatbed",
        # Recreational & specialty
        "motorcycle", "motorbike", "scooter", "atv", "quad bike",
        "bicycle", "e-bike", "electric bicycle", "trike", "tricycle",
        "trailer", "travel trailer", "camper", "rv", "recreational vehicle",
        # Farm & industrial
        "tractor", "farm vehicle", "construction vehicle", "excavator",
        "bulldozer", "loader", "grader", "forklift", "industrial vehicle",
        # Electric & autonomous
        "electric vehicle", "ev", "plug-in hybrid", "phev", "hybrid vehicle",
        "autonomous vehicle", "self-driving", "driverless car", "robot car",
        # Powered variants
        "electric car", "electric truck", "electric motorcycle",
        "fuel cell vehicle", "hydrogen vehicle",
    ],
    "Marine / Watercraft": [
        # Base terms
        "boat", "ship", "vessel", "watercraft", "marine vessel",
        # Ship types
        "yacht", "sailboat", "motorboat", "speedboat", "runabout",
        "sailship", "container ship", "tanker", "cargo ship",
        "passenger ship", "cruise ship", "liner", "freighter",
        "fishing vessel", "trawler", "seiner", "whaling ship",
        # Small craft
        "dinghy", "raft", "canoe", "kayak", "skiff", "catamaran",
        "trimaran", "pontoon", "houseboat", "barge", "tugboat",
        "ferry", "ferryboat", "water taxi",
        # Military/specialist
        "submarine", "battleship", "destroyer", "frigate", "corvette",
        "mine sweeper", "patrol boat", "torpedo boat",
        "amphibious", "hovercraft", "air cushion vehicle",
        # Components
        "hull", "hull design", "propeller", "screw propeller", "water jet",
        "rudder", "keel", "sail", "mast", "rigging",
        "anchor", "deck", "cabin", "hold",
        # Technical
        "marine propulsion", "marine engine", "diesel engine",
        "electric motor", "marine battery", "fuel cell",
    ],
}
