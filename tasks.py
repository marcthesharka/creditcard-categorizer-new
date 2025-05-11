import os
import pickle
import openai
import json
import re

def categorize_transactions(transactions, log_file, output_file):
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    for t in transactions:
        try:
            # Your categorization logic here
            # Example:
            prompt = f"Given this credit card transaction description: '{t['description']}', categorize and summarize."
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )
            content = response.choices[0].message.content.strip()
            data = json.loads(content)
            t['category'] = data.get("category", "Uncategorized")
            t['enhanced_description'] = data.get("enhanced_description", t['description'])
            with open(log_file, 'a') as logf:
                logf.write(f"{t['description']}:\n{content}\n\n")
        except Exception as e:
            with open(log_file, 'a') as logf:
                logf.write(f"{t['description']}:\nOpenAI error: {e}\n\n")
            t['category'] = "Uncategorized"
            t['enhanced_description'] = t['description']
    with open(output_file, 'wb') as tf:
        pickle.dump(transactions, tf)
    return output_file  # Path to the results file
