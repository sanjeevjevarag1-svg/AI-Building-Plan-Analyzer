from groq import Groq


def ask_ai(text, dimensions, question):

    api_key = "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"

    client = Groq(api_key=api_key)

    prompt = f"""
You are an expert architect, NBC consultant, civil engineer,
space planner, and AI building advisor.

Floor plan OCR text:
{text}

Detected dimensions:
{dimensions}

User Question:
{question}

Instructions:
- Answer professionally
- Mention NBC standards where relevant
- Give practical civil engineering suggestions
- Explain risks if any
- Suggest improvements for ventilation, circulation, accessibility, and safety
"""

    try:

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.3,
            max_tokens=1000
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"AI Error: {str(e)}"