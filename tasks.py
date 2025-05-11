import os
import pickle
import openai
import json
import re
from redis import Redis

def categorize_transactions(transactions, output_file, job_id):
    redis_url = (
        os.environ.get("STACKHERO_REDIS_URL_TLS") or
        os.environ.get("STACKHERO_REDIS_URL_CLEAR") or
        os.environ.get("REDISGREEN_URL") or
        os.environ.get("REDISCLOUD_URL") or
        os.environ.get("MEMETRIA_REDIS_URL")
    )
    redis_conn = Redis.from_url(redis_url)
    progress_key = f"progress:{job_id}"
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    total = len(transactions)
    for idx, t in enumerate(transactions, 1):
        try:
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
            redis_conn.append(progress_key, f"{t['description']}:\n{content}\n")
            redis_conn.append(progress_key, f"Processed {idx}/{total} transactions.\n\n")
        except Exception as e:
            redis_conn.append(progress_key, f"{t['description']}:\nOpenAI error: {e}\n")
            redis_conn.append(progress_key, f"Processed {idx}/{total} transactions.\n\n")
            t['category'] = "Uncategorized"
            t['enhanced_description'] = t['description']
    redis_conn.append(progress_key, "All transactions categorized. You can now view results.\n")
    with open(output_file, 'wb') as tf:
        pickle.dump(transactions, tf)
    return output_file  # Path to the results file
