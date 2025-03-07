import re
import json
from typing import Dict

def parse_json(text: str) -> Dict:

    pattern = r"<output>\n(.*?)</output>"

    match = re.search(pattern, text, re.DOTALL)

    if match:
        json_text = match.group(1)
        json_dict = json.loads(json_text)
        return json_dict
    else:
        return json.loads(text)