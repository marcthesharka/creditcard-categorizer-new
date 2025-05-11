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
    try:
        print(f"Starting categorization for job {job_id}")  # Add logging
        update_progress(job_id, "Starting categorization...")
        
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Process transactions in batches
        batch_size = 5
        total_transactions = len(transactions)
        categorized_transactions = []
        
        for i in range(0, total_transactions, batch_size):
            batch = transactions[i:i + batch_size]
            batch_text = "\n".join([f"{t['date'].strftime('%Y-%m-%d')} - {t['description']} - ${t['amount']:.2f}" for t in batch])
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that categorizes credit card transactions. For each transaction, provide a category that best describes the type of expense. Common categories include: Food & Dining, Shopping, Travel, Entertainment, Transportation, Bills & Utilities, Health & Medical, Education, Personal Care, Home & Garden, etc."},
                        {"role": "user", "content": f"Please categorize these transactions:\n{batch_text}"}
                    ]
                )
                
                # Process response and add categories
                categories = response.choices[0].message.content.strip().split('\n')
                for j, category in enumerate(categories):
                    if j < len(batch):
                        batch[j]['category'] = category.strip()
                        categorized_transactions.append(batch[j])
                
                # Update progress
                progress = f"Processed {min(i + batch_size, total_transactions)} of {total_transactions} transactions..."
                update_progress(job_id, progress)
                print(progress)  # Add logging
                
            except Exception as e:
                error_msg = f"Error processing batch: {str(e)}"
                print(error_msg)  # Add logging
                update_progress(job_id, error_msg)
                raise
        
        # Final progress update
        redis_conn = get_redis_connection()
        redis_conn.append(f"progress:{job_id}", "All transactions categorized. You can now view results.\n")
        
        # Save results to file (for local dev)
        with open(output_file, 'wb') as tf:
            pickle.dump(categorized_transactions, tf)
        
        # Save results to Redis (for Heroku)
        redis_conn.set(f"results:{job_id}", pickle.dumps(categorized_transactions))
        
        update_progress(job_id, "Categorization complete!")
        print("Categorization complete!")  # Add logging
        return categorized_transactions
        
    except Exception as e:
        error_msg = f"Error in categorize_transactions: {str(e)}"
        print(error_msg)  # Add logging
        update_progress(job_id, error_msg)
        raise
