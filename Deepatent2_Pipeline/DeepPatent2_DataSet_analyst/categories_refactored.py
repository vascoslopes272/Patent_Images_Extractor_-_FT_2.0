"""
Refactored taxonomy with strict separation of:
- PLATFORM_ARCHITECTURES: Aircraft configuration types
- TECHNICAL_SUBSYSTEMS: Components/subsystems (can apply to any platform)

Each patent can match one platform + multiple subsystems.
"""

PLATFORM_ARCHITECTURES = {
    "Fixed-Wing Aircraft": [
        # Base terms
        "aircraft", "airplane", "aeroplane", "plane", "jet", "airliner",
        "fixed wing", "fixed-wing",
        # Military variants
        "fighter", "bomber", "interceptor", "reconnaissance", "transport aircraft",
        "cargo aircraft", "tanker", "military aircraft", "attack aircraft",
        # General types
        "biplane", "monoplane", "triplane", "glider", "sailplane",
        "seaplane", "floatplane", "amphibian aircraft", "bush plane",
        "commuter aircraft", "regional aircraft",
        # Engine/power types
        "turboprop", "turbofan", "turbojet", "piston aircraft",
        "jet engine", "turbine engine", "electric aircraft", "hybrid aircraft",
        # Speed-related
        "supersonic", "subsonic", "hypersonic", "transonic",
        # Specific designs
        "canard", "tailless aircraft", "flying wing",
        "blended wing body", "variable geometry",
        "short takeoff", "stol aircraft",
        # Trainer & specialized
        "trainer aircraft", "aerobatic", "experimental aircraft",
        "microlight", "ultralight", "general aviation",
    ],

    "Rotary-Wing / Helicopter": [
        # Base terms
        "helicopter", "rotorcraft", "chopper", "heli",
        "autogyro", "gyroplane", "gyrocopter", "gyromotor",
        # Helicopter types
        "light helicopter", "heavy helicopter", "utility helicopter",
        "attack helicopter", "transport helicopter", "cargo helicopter",
        "ambulance helicopter", "medevac helicopter", "rescue helicopter",
        "observation helicopter", "training helicopter",
        # Military variants
        "attack chopper", "gunship", "armed helicopter", "military helicopter",
        "anti-tank helicopter",
        # Specific designs
        "tiltrotor", "convertiplane", "compound helicopter",
        "coaxial helicopter", "tandem rotor", "single rotor",
        # Technical features
        "vertical takeoff", "hover", "hovering aircraft",
        "air taxi capability", "autonomous helicopter",
    ],

    "UAV / Drone": [
        # Base terms
        "drone", "uav", "uas", "unmanned aerial", "unmanned aircraft",
        "rpv", "remotely piloted", "pilotless", "autonomous aircraft",
        # Multirotor types
        "quadcopter", "hexacopter", "octocopter", "multicopter", "multirotor",
        "quadrotor", "hexarotor", "octorotor", "tri-rotor", "coaxial quad",
        "counter-rotating rotor", "coaxial rotor",
        # Fixed-wing UAV
        "fixed-wing drone", "fixed-wing uav", "fixed wing drone",
        "flying wing drone", "blended wing drone", "pusher drone",
        # Hybrid & VTOL drones
        "hybrid drone", "vertical takeoff drone", "vtol drone",
        "hybrid vtol", "fixed-wing vtol",
        # Applications & variants
        "delivery drone", "cargo drone", "logistics drone",
        "camera drone", "surveillance drone", "monitoring drone",
        "inspection drone", "mapping drone", "surveying drone",
        "agricultural drone", "crop spraying", "precision agriculture",
        "racing drone", "acrobatic drone", "training drone",
        "search and rescue", "rescue drone", "emergency response",
        # Size variants
        "nano drone", "micro drone", "mini drone", "small uav",
        "medium uav", "large uav", "tactical uav", "strategic uav",
    ],

    "VTOL / Advanced Air Mobility": [
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
        "electric propulsion", "fuel cell aircraft",
        # Specific concepts
        "air taxi service", "aerial taxi", "autonomous air vehicle",
        "passenger drone", "electric copter", "electric helicopter",
        "tiltwing", "tiltwing aircraft", "tilt-rotor",
        "distributed electric", "distributed propulsion",
        # Emerging terms
        "advanced vehicle", "next generation aircraft", "future aircraft",
        "innovative transport", "sustainable aviation", "zero-emission aircraft",
    ],

    "Hybrid / Special Configuration": [
        # Combined/unique architectures
        "hybrid architecture", "hybrid platform", "convertible aircraft",
        "morphing aircraft", "variable geometry aircraft",
        "amphibious aircraft", "amphibian",
    ],
}

TECHNICAL_SUBSYSTEMS = {
    "Fuselage & Structure": [
        # Primary fuselage
        "fuselage", "cabin", "pressurized cabin", "cargo hold",
        "cockpit", "flight deck", "fuselage frame", "fuselage skin",
        "structural composite", "aluminum fuselage", "carbon fiber fuselage",
        "composite fuselage", "monocoque structure",
        # Structural elements
        "frame", "stringers", "stiffeners", "bulkhead",
        "floor beam", "floor panel", "deck",
        "skin panel", "skin thickness",
        # Materials
        "aluminum", "carbon fiber", "composite material",
        "titanium", "kevlar", "aramid fiber",
        "fiberglass", "epoxy resin", "structural material",
    ],

    "Propulsion System": [
        # Engine types
        "engine", "motor", "electric motor", "brushless motor", "dc motor", "ac motor",
        "piston engine", "reciprocating engine", "turbine engine",
        "gas turbine", "turbo shaft", "turbofan", "turbojet", "turboprop",
        "jet engine", "induction motor", "permanent magnet motor",
        # Engine components
        "engine mount", "engine pylon", "nacelle", "engine pod",
        "engine housing", "engine cowling", "engine intake",
        # Fuel system
        "fuel system", "fuel tank", "fuel pump", "fuel line",
        "fuel flow control", "fuel injection", "carburetor",
        "fuel filter", "fuel selector valve",
        # Alternative propulsion
        "fuel cell", "hydrogen fuel", "electric propulsion",
        "hybrid propulsion", "solar powered", "hydrogen powered",
        # Thrust/power output
        "thrust vector", "vectored thrust", "thrust control",
        "power transmission", "drive system", "power delivery",
    ],

    "Landing Gear & Undercarriage": [
        # Gear types
        "landing gear", "undercarriage", "main landing gear", "nose gear",
        "front gear", "rear gear", "wheeled landing gear",
        "skid landing gear", "fixed landing gear", "retractable landing gear",
        # Wheels & tires
        "wheel", "tire", "pneumatic tire", "tire assembly",
        "wheel assembly", "brake assembly", "wheel bearing",
        # Shock absorption
        "shock absorber", "oleo strut", "damper", "spring",
        "hydraulic strut", "pneumatic strut", "suspension system",
        # Alternative gear
        "skid", "pontoon", "float", "ski", "cross-country landing gear",
        # Control & features
        "gear extension", "gear retraction", "gear position indicator",
        "anti-skid system", "brake system",
    ],

    "Rotor & Blade Systems": [
        # Rotor assembly
        "rotor", "rotor system", "rotor assembly", "main rotor", "tail rotor",
        "rotor head", "rotor hub", "rotor mast", "rotor disc",
        # Rotor blades
        "rotor blade", "main rotor blade", "tail rotor blade", "blade root",
        "blade tip", "blade section", "blade design", "airfoil blade",
        # Blade control
        "blade pitch", "collective pitch", "cyclic pitch", "pitch control",
        "variable pitch rotor", "fixed pitch rotor",
        "pitch horn", "pitch link", "pitch actuator",
        # Rotor types
        "pusher rotor", "lift rotor", "ducted rotor", "open rotor",
        "shrouded rotor", "counter-rotating rotor", "coaxial rotor",
        # Technical
        "rotor speed", "rotor rpm", "rotor tracking", "rotor balance",
        "rotor vibration", "rotor noise", "rotor efficiency",
    ],

    "Wing Systems": [
        # Primary wing
        "wing", "wing assembly", "wing section", "main wing",
        # Wing types
        "swept wing", "delta wing", "strut wing", "cantilever wing",
        "folding wing", "variable geometry wing", "morphing wing",
        # Wing components
        "airfoil", "wing surface", "wing box", "wing spar",
        "wing rib", "wing skin", "wing joint",
        # Wing extremities
        "wing tip", "wing root", "wing leading edge", "wing trailing edge",
        "winglet", "sharklet", "split scimitar",
        # Control surfaces on wing
        "aileron", "flap", "slat", "leading edge device",
        "trailing edge device", "spoiler",
    ],

    "Control Surfaces & Flight Control": [
        # Primary control surfaces
        "control surface", "flight surface", "flight control surface",
        "aileron", "elevator", "rudder", "flap", "slat",
        "trim tab", "trim surface", "elevator trim",
        "rudder trim", "aileron trim",
        # Advanced control
        "flight control system", "autopilot", "stability augmentation",
        "electronic control", "fly-by-wire", "mechanical control",
        # Control actuators
        "control actuator", "flight actuator", "servo actuator",
        "hydraulic actuator", "electric actuator", "pneumatic actuator",
        "actuator linkage", "actuator cable",
        # Tail section
        "empennage", "tail section", "tail assembly", "tail structure",
        "vertical stabilizer", "vertical tail", "fin",
        "horizontal stabilizer", "horizontal tail", "stabilator",
    ],

    "Avionics & Control Systems": [
        # Navigation
        "navigation system", "navigation instrument", "gps", "gnss",
        "inertial navigation", "ins", "inertial measurement", "imu",
        "barometer", "altimeter", "airspeed indicator",
        # Flight management
        "autopilot", "automatic flight control", "autonomous flight",
        "flight management system", "flight control system",
        "stability control", "attitude control", "gyroscopic control",
        # Sensors
        "sensor", "lidar", "radar", "infrared", "thermal camera",
        "rgb camera", "depth sensor", "obstacle detection",
        "proximity sensor", "pressure sensor", "temperature sensor",
        # Communication & control
        "communication system", "data link", "telemetry",
        "wireless communication", "remote control", "rc receiver", "transmitter",
        "fpv", "first person view", "video transmission",
        # Cockpit systems
        "instrument panel", "flight instrument", "cockpit display",
        "head-up display", "hud", "glass cockpit", "avionics suite",
    ],

    "Power & Energy Systems": [
        # Battery & energy storage
        "battery", "lithium battery", "li-ion battery", "lithium-ion",
        "battery pack", "battery cell", "battery module",
        "battery management", "battery cooling", "thermal management",
        "energy storage", "power distribution", "power electronics",
        "dc-dc converter", "power conditioning",
        # Alternative energy
        "fuel cell", "hydrogen fuel", "supercapacitor", "ultracapacitor",
        "solar panel", "solar cell", "photovoltaic",
        # Electrical system
        "electrical system", "electrical wiring", "electrical harness",
        "circuit breaker", "electrical distribution",
        "generator", "alternator", "power generation",
        # Power delivery
        "motor controller", "motor drive", "inverter",
        "power inverter", "charging system", "charging circuit",
    ],

    "Air Intake & Exhaust System": [
        # Intake
        "air intake", "intake duct", "intake system", "air inlet",
        "air scoop", "ram air", "ram intake",
        "pitot tube", "static port", "probe",
        # Inlet types
        "ramjet", "scramjet", "supersonic inlet",
        "subsonic inlet", "variable inlet",
        # Exhaust
        "exhaust", "exhaust nozzle", "nozzle", "exhaust system",
        "exhaust duct", "exhaust manifold", "thrust nozzle",
        "afterburner", "nozzle closing mechanism",
    ],

    "Thermal Management & Cooling": [
        # Cooling system
        "cooling system", "thermal management", "heat dissipation",
        "heat sink", "radiator", "cooling radiator",
        "cooling fan", "cooling duct", "air cooling",
        # Thermal types
        "liquid cooling", "oil cooling", "fuel cooling",
        "passive cooling", "active cooling",
        # Thermal components
        "coolant", "coolant pump", "coolant line",
        "thermostat", "temperature control", "thermal valve",
    ],

    "Payload & Mission Systems": [
        # Payload types
        "payload", "mission payload", "payload capacity",
        "cargo system", "cargo hold", "cargo handling",
        # Sensor payloads
        "camera", "gimbal", "camera gimbal", "optical gimbal",
        "imaging system", "reconnaissance payload",
        "surveillance system", "monitoring system",
        # Delivery/mission
        "delivery system", "cargo delivery", "supply delivery",
        "spray system", "spraying mechanism", "agricultural payload",
        # Sensor integration
        "payload integration", "payload pod", "pod assembly",
        "external pod", "equipment pod",
    ],

    "Acoustic & Noise Reduction": [
        # Acoustic treatment
        "acoustic", "acoustic treatment", "noise reduction",
        "sound damping", "sound absorption", "sound insulation",
        "acoustic lining", "acoustic foam", "acoustic panel",
        # Vibration control
        "vibration isolation", "vibration damper", "vibration control",
        "damping system", "isolation mount", "resilient mount",
        # Operational noise
        "rotor noise", "engine noise", "propeller noise",
        "quiet rotor", "low noise", "noise level",
        # Acoustic design
        "acoustic enclosure", "acoustic shroud", "noise suppression",
    ],

    "Safety & Redundancy Systems": [
        # Safety equipment
        "parachute", "ballistic parachute", "emergency parachute",
        "safety system", "emergency system", "fail-safe system",
        # Redundancy
        "redundancy", "redundant system", "backup system",
        "backup power", "emergency power", "power redundancy",
        # Emergency systems
        "emergency landing", "emergency descent", "ditching system",
        "emergency exit", "emergency egress",
        # Safety features
        "crash protection", "impact protection", "structural safety",
        "fire protection", "fire suppression",
    ],
}


def get_all_keywords_flat():
    """Return all keywords from both architectures and subsystems."""
    all_keywords = {}
    for category, keywords in PLATFORM_ARCHITECTURES.items():
        all_keywords[category] = keywords
    for category, keywords in TECHNICAL_SUBSYSTEMS.items():
        all_keywords[category] = keywords
    return all_keywords

# Combined for compatibility
CATEGORIES = {**PLATFORM_ARCHITECTURES, **TECHNICAL_SUBSYSTEMS}
