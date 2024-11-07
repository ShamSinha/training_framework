"""Update this from qct_data constants"""
char2label = {
    "Texture": {
        "non-solid/ggo": 0,
        "non-solid/mixed": 0,
        "part solid/mixed": 1,
        "solid/mixed": 1,
        "solid": 2,
    },
    "Calcification": {
        "popcorn": 0,
        "fat": 0,
        "solid": 0,
        "non-central": 0,
        "central": 0,
        "absent": 1,
    },
    "Spiculation": {
        "no spiculation": 0,
        "no lobulation": 0,
        "nearly no spiculation": 0,
        "near marked spiculation": 1,
        "medium spiculation": 1,
        "marked spiculation": 1,
    },
    "Margin": {
        "poorly defined": 0,
        "near poorly defined": 0,
        "medium margin": 1,
        "near sharp": 1,
        "sharp": 1,
    },
    "Malignancy": {
        "highly unlikely": 0,
        "moderately unlikely": 0,
        "indeterminate": 0,
        "moderately suspicious": 1,
        "highly suspicious": 1,
    },
}
label2char = {
    "Texture": {
        "0": "Non-Solid",
        "1": "Part-Solid",
        "2": "Solid",
    },
    "Calcification": {"0": "Present", "1": "Absent"},
    "Spiculation": {"0": "No-Spiculation", "1": "Spiculation"},
    "Margin": {"0": "Poorly-Defined", "1": "Sharp/Medium-Margin"},
    "Malignancy": {"0": "Unlikely", "1": "Suspicious"},
}

training_fields = [
    "Texture",
    "Spiculation",
    "Calcification",
    "Margin",
    "Malignancy",
]
training_fields_classes = [f"{f}_cls" for f in training_fields]
