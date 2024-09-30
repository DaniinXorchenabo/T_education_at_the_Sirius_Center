import base64
import traceback

import openai

from src.config import DescriptionTypes


def description_generator(image_filename: str, openai_client: openai.Client, gen_pattern:DescriptionTypes = DescriptionTypes.simple):
    with open(image_filename, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional marketer with experience writing high—conversion ads. To generate a product description, you study the potential target audience and optimize the advertising text so that it appeals specifically to this target audience. Create an ad text with an attention-grabbing title and a compelling call to action that encourages users to take targeted action."},
                {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": "generate a text description of the product in the picture. Don't add anything extra."
                    },
                        {
                            "type": "image_url",
                            "image_url":
                                {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                        }
                    ]
                }
            ]
        )
        result_msg = completion.choices[0].message
    except Exception as e:
        print(e, traceback.format_exc())
        result_msg = str(e)
    return result_msg