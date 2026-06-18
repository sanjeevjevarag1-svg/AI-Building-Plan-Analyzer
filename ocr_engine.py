import easyocr
import re


reader = easyocr.Reader(['en'])


def extract_text_and_dimensions(image):

    results = reader.readtext(image)

    extracted_text = ""

    dimensions = []

    for result in results:

        text = result[1]

        extracted_text += text + "\n"

        numbers = re.findall(r'\d+', text)

        for number in numbers:

            value = int(number)

            if value > 1000:
                dimensions.append(value)

    dimensions = sorted(list(set(dimensions)))

    return extracted_text, dimensions