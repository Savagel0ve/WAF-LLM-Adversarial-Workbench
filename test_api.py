from openai import OpenAI
import os

# Use the key provided by the user in previous context
api_key = "sk-uqsfyW4IpWwOKhziBHPmoXgLoC9iul4J9cGRoPfzvZCUUUn3"
base_url = "https://hiapi.online/v1"

print(f"Testing connectivity to {base_url} with model gemini-2.5-pro...")

try:
    client = OpenAI(api_key=api_key, base_url=base_url)

    response = client.chat.completions.create(
        model="gemini-2.5-pro",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello! Just testing the connection."},
        ],
        stream=False
    )

    print("\nResponse received:")
    print(f"Type: {type(response)}")
    print(f"Content: {response}")
    print(response.choices[0].message.content)
    print("\nSUCCESS: Connection verification passed.")

except Exception as e:
    print(f"\nERROR: Connection failed.\n{e}")
