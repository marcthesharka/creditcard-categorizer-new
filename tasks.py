import os
import pickle
import openai
import json
import re
from redis import Redis

def get_redis_connection():
    redis_url = (
        os.environ.get("STACKHERO_REDIS_URL_TLS") or
        os.environ.get("STACKHERO_REDIS_URL_CLEAR") or
        os.environ.get("REDISGREEN_URL") or
        os.environ.get("REDISCLOUD_URL") or
        os.environ.get("MEMETRIA_REDIS_URL")
    )
    if not redis_url:
        raise ValueError("No Redis URL found in environment variables")
    return Redis.from_url(redis_url)

def update_progress(job_id, message):
    try:
        redis_conn = get_redis_connection()
        redis_conn.set(f"progress:{job_id}", message)
        print(f"Progress updated for job {job_id}: {message}")  # Add logging
    except Exception as e:
        print(f"Error updating progress: {str(e)}")  # Add logging

def categorize_transactions(transactions, output_file, job_id):
    redis_conn = get_redis_connection()
    progress_key = f"progress:{job_id}"
    redis_conn.set(progress_key, "Starting categorization...\n")
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    total = len(transactions)
    for idx, t in enumerate(transactions, 1):
        try:
            prompt = (
                f"Given this credit card transaction description: '{t['description']}',\n"
                "1. Categorize it with one of the following, do not create new categories: 'Food & Beverage', 'Health & Wellness', 'Travel (Taxi / Uber / Lyft / Revel)', 'Travel (Subway / MTA)', 'Gas & Fuel','Travel (Flights / Trains)', 'Hotel', 'Groceries', 'Entertainment', 'Shopping', 'Income / Refunds', 'Utilities (Electricity, Telecom, Internet)', 'Other (Miscellaneous)'.\n"
                "2. Write a short, human-perceivable summary of the expense, including the merchant type and location if available. Follow the format: 'Merchant Name, Location, brief description of expense purpose (no more than 10 words)'\n"
                "Return your answer as JSON in the following format (no markdown, no explanation, just JSON):\n"
                '{"category": "...", "enhanced_description": "..."}'
            )
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )
            content = response.choices[0].message.content.strip()
            # Try to extract JSON from the response
            try:
                data = json.loads(content)
            except Exception as e:
                import re
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    try:
                        data = json.loads(match.group(0))
                    except Exception as e2:
                        data = {}
                else:
                    data = {}
            t['category'] = data.get("category", "Uncategorized")
            t['enhanced_description'] = data.get("enhanced_description", t['description'])
            progress_msg = f"Processed {t['description']}:\n{content}\nProcessed {idx}/{total} transactions.\n\n"
            redis_conn.append(progress_key, progress_msg)
        except Exception as e:
            error_msg = f"Error processing {t['description']}:\n{str(e)}\nProcessed {idx}/{total} transactions.\n\n"
            redis_conn.append(progress_key, error_msg)
            t['category'] = "Uncategorized"
            t['enhanced_description'] = t['description']
    redis_conn.append(progress_key, "All transactions categorized. You can now view results.\n")
    with open(output_file, 'wb') as tf:
        pickle.dump(transactions, tf)
    redis_conn.set(f"results:{job_id}", pickle.dumps(transactions))
    return output_file