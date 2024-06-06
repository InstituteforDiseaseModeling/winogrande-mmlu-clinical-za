import requests
import json
import os

with open(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../config.json')), 'r') as fp:
    config = json.load(fp)
    gt_api_key = config['google_translate_api_key']


def translate_text(target: str, text: str, backtranslate: bool = False) -> str:
    """Translates text into the target language.

    :param target: Target language code string
    :param text: Text to translate
    :param backtranslate: If False, translates text from English into target language; otherwise, translated text to English from target language
    :return: Google Translated text

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """
    source = 'en'
    if backtranslate:
        source = target
        target = 'en'
    # URL for the Google Translate API
    target_url = f"https://translation.googleapis.com/language/translate/v2?key={gt_api_key}"

    # The data to be sent in the POST request
    translation_data = {
        "q": text,
        "source": source,
        "target": target,
        "format": "text"
    }
    # Send the POST request
    response = requests.post(target_url, data=json.dumps(translation_data),
                             headers={'Content-Type': 'application/json'}).json()
    return response['data']['translations'][0]['translatedText']


if __name__ == '__main__':
    print(translate_text('de', "Hello, world!"))
    print(translate_text('de', "Hallo Welt!", True))
