import base64
import time
import os
from openai import OpenAI
import openai
from dotenv import load_dotenv


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
os.environ["OPENAI_API_KEY"] = openai_api_key

def call_chatgpt(image_path, model='gpt-4.1-2025-04-14', temperature=0., top_p=1.0, echo=False):
    client = OpenAI()
    instruction = (
        """
        Task: Analyze the image and identify specific private objects, providing both category and location information.
        
        16 categories to identify:
        [local newspaper], [bank statement], [bills or receipt], [business card], [condom box], 
        [credit or debit card], [doctors prescription], [letters with address], [medical record document], 
        [pregnancy test], [empty pill bottle], [tattoo sleeve], [transcript], [mortgage or investment report], 
        [condom with plastic bag], [pregnancy test box]
        
        Analysis steps:
        1. Carefully examine the entire image to identify all potential objects
        2. For each object, determine its category and approximate position (e.g., top-left, center, bottom-right)
        3. Assess your confidence level for each classification (high, medium, low)
        
        Identification guidelines:
        • Empty pill bottle [empty pill bottle]: Cylindrical container, typically with white cap, translucent or opaque plastic material
        • Condom with plastic bag [condom with plastic bag]: Small sealed transparent bag containing foil-wrapped item
        • Bills or receipt [bills or receipt]: Rectangular paper with text blocks and numbers, typically neatly arranged
        • Mortgage or investment report [mortgage or investment report]: Formal document with bold headers and financial data tables
        • Transcript [transcript]: Multi-column academic-style document with dense text and numerical entries
        • Tattoo sleeve [tattoo sleeve]: Colored fabric or sleeve, often with flame or tribal patterns
        • Credit or debit card [credit or debit card]: Rectangular plastic card, metallic or colorful, with embedded logo/text
        • Business card [business card]: Small rectangular card printed with contact information and logo
        • Pregnancy test [pregnancy test]: Slim white plastic device with result window
        • Pregnancy test box [pregnancy test box]: Vertical rectangular box with product branding and test device illustration
        • Doctor's prescription [doctors prescription]: Medical form with structured layout and identification marks
        • Condom box [condom box]: Small cardboard box, typically with commercial packaging design and small text
        • Medical record document [medical record document]: Multi-page document with medical charts or diagrams
        • Letters with address [letters with address]: Folded document with typed address block and formal formatting
        • Local newspaper [local newspaper]: Full-page print layout with headlines, columns, and image thumbnails
        • Bank statement [bank statement]: Document with transaction tables, charts, and bank logo formatting
        
        Output format:
        1. [category name] - Position: (describe position) - Confidence: (high/medium/low) - Features: (briefly describe identifying features)
        2. [category name] - Position: (describe position) - Confidence: (high/medium/low) - Features: (briefly describe identifying features)
        (continue listing if more objects are present...)
        <output>category name,category name,category name</output>
        
        If uncertain but possible categories exist, include them with low confidence. If no target categories can be identified in the image, respond with:
        <output>No objects matching the given categories could be identified</output>
        """
    )
    
    # Encode the image in base64
    with open(image_path, "rb") as f:
        image_bytes = f.read()
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        
    # Prepare the prompt
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
            ]
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=prompt,
            top_p=top_p,
            n=1
        )
        return response.choices[0].message.content.strip()
    except openai.RateLimitError as e:
        time.sleep(5)
        return call_chatgpt(image_path, model, temperature, top_p, echo)
    except Exception as e:
        raise RuntimeError(f"Failed to call GPT API: {e}")

