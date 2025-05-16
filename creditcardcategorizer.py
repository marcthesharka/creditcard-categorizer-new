import os
import tempfile
from flask import Flask, render_template, request, redirect, url_for, send_file, session, flash, jsonify
import pdfplumber
import pandas as pd
from datetime import datetime, date, timedelta
import openai
import pickle
import json
import re
from rq import Queue
from redis import Redis
from tasks import categorize_transactions
import stripe

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key in production

api_key = os.getenv("OPENAI_API_KEY")
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

# Use the correct environment variable for your Redis add-on!
redis_url = (
    os.environ.get("STACKHERO_REDIS_URL_TLS") or
    os.environ.get("STACKHERO_REDIS_URL_CLEAR") or
    os.environ.get("REDISGREEN_URL") or
    os.environ.get("REDISCLOUD_URL") or
    os.environ.get("MEMETRIA_REDIS_URL")
)
if not redis_url:
    raise RuntimeError("No Redis URL found in environment variables! Please check your Heroku config vars.")
conn = Redis.from_url(redis_url)

def parse_pdf_transactions(pdf_path):
    import re
    from datetime import datetime
    with pdfplumber.open(pdf_path) as pdf:
        first_page_text = pdf.pages[0].extract_text() or ""
        if "Chase" in first_page_text:
            return parse_chase_pdf_transactions(pdf_path)
        elif "Apple Card" in first_page_text:
            return parse_apple_pdf_transactions(pdf_path)
        elif "Capital One" in first_page_text:
            return parse_capitalone_pdf_transactions(pdf_path)
        elif "American Express" in first_page_text or "americanexpress.com" in first_page_text:
            return parse_amex_pdf_transactions(pdf_path)
        else:
            raise ValueError("Unknown statement format")

def parse_chase_pdf_transactions(pdf_path):
    import re
    from datetime import datetime, date, timedelta
    transactions = []
    in_transactions_section = False
    today = date.today()
    with pdfplumber.open(pdf_path) as pdf:
        print(f"PDF has {len(pdf.pages)} pages (Chase)")
        for i, page in enumerate(pdf.pages[2:], start=3):  # skip first two pages
            text = page.extract_text()
            if not text:
                continue
            lines = text.splitlines()
            for line in lines:
                if "payments and other credits" in line.lower() or "purchase" in line.lower():
                    in_transactions_section = True
                    continue
                if "account activity" in line.lower():
                    continue
                if not in_transactions_section:
                    continue
                if "totals year-to-date" in line.lower() or "interest charges" in line.lower():
                    in_transactions_section = False
                    break
                if line.strip() == "" or "date of" in line.lower() or "merchant name" in line.lower() or "description" in line.lower() or "amount" in line.lower():
                    continue
                match = re.match(r"^(\d{2}/\d{2})\s+(.+?)\s+(-?\$?[\d,]*\.\d{2})$", line)
                if match:
                    date_str, desc, amount_str = match.groups()
                    try:
                        year = today.year
                        parsed_date = datetime.strptime(f"{date_str}/{year}", "%m/%d/%Y").date()
                        if parsed_date > today:
                            parsed_date = parsed_date.replace(year=year-1)
                        date_obj = datetime.combine(parsed_date, datetime.min.time())
                        amount = float(amount_str.replace('$', '').replace(',', ''))
                        transactions.append({
                            'date': date_obj,
                            'description': desc.strip(),
                            'amount': amount,
                            'category': '',
                            'card': 'Chase'
                        })
                    except Exception as e:
                        print(f"Error parsing Chase line: {line} -- {e}")
                        continue
    print(f"Total Chase transactions found: {len(transactions)}")
    return transactions

def parse_apple_pdf_transactions(pdf_path):
    import re
    from datetime import datetime, date, timedelta
    transactions = []
    today = date.today()
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[1:]:  # Skip first page
            text = page.extract_text()
            if not text:
                continue
            lines = text.splitlines()
            in_transactions = False
            for line in lines:
                if "Transactions" in line:
                    in_transactions = True
                    continue
                if not in_transactions:
                    continue
                if line.strip() == "" or "Date" in line or "Description" in line or "Amount" in line or "Daily Cash" in line:
                    continue
                match = re.match(r"^(\d{2}/\d{2}/\d{4})\s+(.+?)\s+\d+%\s+\$[\d,.]+\s+(-?\$[\d,.]+)$", line)
                if match:
                    date_str, desc, amount_str = match.groups()
                    try:
                        parsed_date = datetime.strptime(date_str, "%m/%d/%Y").date()
                        if parsed_date > today:
                            parsed_date = parsed_date.replace(year=parsed_date.year-1)
                        date_obj = datetime.combine(parsed_date, datetime.min.time())
                        amount = float(amount_str.replace('$', '').replace(',', ''))
                        transactions.append({
                            'date': date_obj,
                            'description': desc.strip(),
                            'amount': amount,
                            'category': '',
                            'card': 'Apple Card'
                        })
                    except Exception as e:
                        print(f"Error parsing Apple Card line: {line} -- {e}")
                        continue
    print(f"Total Apple Card transactions found: {len(transactions)}")
    return transactions

def parse_capitalone_pdf_transactions(pdf_path):
    import re
    from datetime import datetime, date
    transactions = []
    today = date.today()
    in_transactions_section = False
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages[2:], start=3):  # skip first two pages
            text = page.extract_text()
            if not text:
                continue
            lines = text.splitlines()
            for line in lines:
                # Start parsing when we hit the Transactions section
                if "Transactions" in line and not in_transactions_section:
                    in_transactions_section = True
                    continue
                if not in_transactions_section:
                    continue
                # Stop if we hit Fees or Interest or end of transactions
                if "Fees" in line or "Interest Charged" in line or "Total Transactions for This Period" in line:
                    in_transactions_section = False
                    break
                # Skip headers and empty lines
                if line.strip() == "" or "Trans Date" in line or "Description" in line or "Amount" in line or "Post Date" in line:
                    continue
                # Match lines like: Mar 14   H MARTNEW YORKNY   $8.33
                match = re.match(r"^([A-Za-z]{3} \d{1,2})\s+(.+?)\s+\$?(-?[\d,]+\.\d{2})$", line)
                if match:
                    date_str, desc, amount_str = match.groups()
                    try:
                        # Parse date (assume current year, adjust if in future)
                        year = today.year
                        parsed_date = datetime.strptime(f"{date_str} {year}", "%b %d %Y").date()
                        if parsed_date > today:
                            parsed_date = parsed_date.replace(year=year-1)
                        date_obj = datetime.combine(parsed_date, datetime.min.time())
                        amount = float(amount_str.replace(',', ''))
                        # Exclude repayments: description contains 'CAPITAL ONE AUTOPAY'
                        if 'CAPITAL ONE AUTOPAY' in desc.upper():
                            continue
                        transactions.append({
                            'date': date_obj,
                            'description': desc.strip(),
                            'amount': amount,
                            'category': '',
                            'card': 'Capital One'
                        })
                    except Exception as e:
                        print(f"Error parsing Capital One line: {line} -- {e}")
                        continue
    print(f"Total Capital One transactions found: {len(transactions)}")
    return transactions

def parse_amex_pdf_transactions(pdf_path):
    import re
    from datetime import datetime, date
    transactions = []
    today = date.today()
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages[2:], start=3):  # skip first two pages
            text = page.extract_text()
            if not text:
                continue
            lines = text.splitlines()
            in_charges_section = False
            detail_count = 0
            header_found = False
            for idx, line in enumerate(lines):
                # Count "Detail" sections
                if "Detail" in line:
                    detail_count += 1
                    if detail_count == 2:
                        in_charges_section = True
                        header_found = False  # Reset for new section
                    else:
                        in_charges_section = False
                    continue

                # Only parse after second "Detail"
                if not in_charges_section:
                    continue

                # Look for the header row (user name + "Card Ending ..."), then "Amount"
                if not header_found:
                    if "Card Ending" in line:
                        # Next line should be the header with "Amount"
                        if idx + 1 < len(lines) and "Amount" in lines[idx + 1]:
                            header_found = True
                        continue
                    else:
                        continue

                # Stop parsing if we hit "Fees"
                if "Fees" in line:
                    break

                # Match transaction rows: MM/DD/YY  MERCHANT  CITY  STATE  $AMOUNT
                match = re.match(
                    r"^(\d{2}/\d{2}/\d{2,4})\s+([A-Z0-9 .&'/-]+)\s+([A-Z .&'/-]+)\s+([A-Z]{2})\s+\$?(-?[\d,]+\.\d{2})$",
                    line
                )
                if match:
                    date_str, merchant, city, state, amount_str = match.groups()
                    try:
                        # Try both 2-digit and 4-digit year
                        try:
                            parsed_date = datetime.strptime(date_str, "%m/%d/%y").date()
                        except ValueError:
                            parsed_date = datetime.strptime(date_str, "%m/%d/%Y").date()
                        if parsed_date > today:
                            parsed_date = parsed_date.replace(year=parsed_date.year - 1)
                        date_obj = datetime.combine(parsed_date, datetime.min.time())
                        amount = float(amount_str.replace('$', '').replace(',', ''))
                        transactions.append({
                            'date': date_obj,
                            'description': f"{merchant.strip()} {city.strip()} {state.strip()}",
                            'amount': amount,
                            'category': '',
                            'card': 'American Express'
                        })
                    except Exception as e:
                        print(f"Error parsing Amex line: {line} -- {e}")
                        continue
    print(f"Total American Express transactions found: {len(transactions)}")
    return transactions

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check payment
        if request.form.get('payment_complete') != '1':
            flash('There was an error processing your payment. We did not charge you. Please try again or use a different card.', 'danger')
            return render_template('index.html')
        files = request.files.getlist('pdf')
        if not files or not any(f.filename for f in files):
            flash('Please upload at least one PDF.', 'danger')
            return render_template('index.html')
        all_transactions = []
        try:
            for file in files:
                if file and file.filename.endswith('.pdf'):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                        file.save(tmp.name)
                        transactions = parse_pdf_transactions(tmp.name)
                        os.unlink(tmp.name)
                    all_transactions.extend(transactions)
            job_id = os.urandom(8).hex()
            output_file = f"/tmp/results_{job_id}.pkl"
            job = Queue(name='default', connection=conn).enqueue(
                categorize_transactions,
                all_transactions,
                output_file,
                job_id,
                job_id=job_id
            )
            return render_template('progress.html', job_id=job.get_id())
        except Exception as e:
            flash('There was an error processing your file(s). Your payment was not captured. If you see a pending charge, it will be automatically voided by your bank.', 'danger')
            return render_template('index.html')
    return render_template('index.html')

@app.route('/progress')
def progress_no_id():
    return redirect(url_for('index'))

@app.route('/progress/<job_id>')
def progress(job_id=None):
    if not job_id:
        return "No job ID provided"
    try:
        redis_url = (
            os.environ.get("STACKHERO_REDIS_URL_TLS") or
            os.environ.get("STACKHERO_REDIS_URL_CLEAR") or
            os.environ.get("REDISGREEN_URL") or
            os.environ.get("REDISCLOUD_URL") or
            os.environ.get("MEMETRIA_REDIS_URL")
        )
        if not redis_url:
            return "Error: No Redis URL found in environment variables"
        redis_conn = Redis.from_url(redis_url)
        progress = redis_conn.get(f"progress:{job_id}")
        if progress:
            return progress.decode()
        # Check if job exists in queue
        job = Queue(name='default', connection=redis_conn).fetch_job(job_id)
        if job:
            if job.is_finished:
                return "Job completed. Processing results..."
            elif job.is_failed:
                return f"Job failed: {job.exc_info}"
            else:
                return "Job is running..."
        return "Starting..."
    except Exception as e:
        print(f"Error in progress endpoint: {str(e)}")
        return f"Error checking progress: {str(e)}"

@app.route('/categorize/<job_id>', methods=['GET', 'POST'])
def categorize(job_id):
    output_file = f"/tmp/results_{job_id}.pkl"
    redis_url = (
        os.environ.get("STACKHERO_REDIS_URL_TLS") or
        os.environ.get("STACKHERO_REDIS_URL_CLEAR") or
        os.environ.get("REDISGREEN_URL") or
        os.environ.get("REDISCLOUD_URL") or
        os.environ.get("MEMETRIA_REDIS_URL")
    )
    redis_conn = Redis.from_url(redis_url) if redis_url else None
    transactions = None
    # Try to load from Redis first
    if redis_conn:
        results = redis_conn.get(f"results:{job_id}")
        if results:
            print(f"DEBUG: Results found in Redis for job_id={job_id}")
            transactions = pickle.loads(results)
        else:
            print(f"DEBUG: No results found in Redis for job_id={job_id}")
    # Fallback to file (for local dev)
    if transactions is None:
        if not os.path.exists(output_file):
            print(f"DEBUG: No results file found for job_id={job_id}")
            return redirect(url_for('progress', job_id=job_id))
        with open(output_file, 'rb') as tf:
            print(f"DEBUG: Results loaded from file for job_id={job_id}")
            transactions = pickle.load(tf)
    # Set session variable for export and summary
    session['transactions_file'] = output_file
    session['job_id'] = job_id
    # Sort transactions by date descending
    transactions.sort(key=lambda t: t['date'], reverse=True)
    if request.method == 'POST':
        for i, t in enumerate(transactions):
            t['category'] = request.form.get(f'category_{i}', '')
        with open(output_file, 'wb') as tf:
            pickle.dump(transactions, tf)
        if redis_conn:
            redis_conn.set(f"results:{job_id}", pickle.dumps(transactions))
        flash('Changes saved!')
        return redirect(url_for('categorize', job_id=job_id))
    # Filter out repayment transactions for total spend calculation
    def is_repayment(txn):
        desc = txn['description'].strip().upper()
        return (
            desc == 'AUTOMATIC PAYMENT - THANK YOU' or
            desc.startswith('ACH DEPOSIT INTERNET TRANSFER')
        )
    filtered_transactions = [t for t in transactions if not is_repayment(t)]
    return render_template(
        'categorize.html',
        transactions=transactions,
        filtered_transactions=filtered_transactions
    )

@app.route('/export')
def export():
    transactions_file = session.get('transactions_file')
    job_id = session.get('job_id')
    transactions = None
    if transactions_file and os.path.exists(transactions_file):
        with open(transactions_file, 'rb') as tf:
            transactions = pickle.load(tf)
    elif job_id:
        redis_url = (
            os.environ.get("STACKHERO_REDIS_URL_TLS") or
            os.environ.get("STACKHERO_REDIS_URL_CLEAR") or
            os.environ.get("REDISGREEN_URL") or
            os.environ.get("REDISCLOUD_URL") or
            os.environ.get("MEMETRIA_REDIS_URL")
        )
        redis_conn = Redis.from_url(redis_url) if redis_url else None
        if redis_conn:
            results = redis_conn.get(f"results:{job_id}")
            if results:
                transactions = pickle.loads(results)
    if not transactions:
        return redirect(url_for('index'))
    # Sort transactions by date descending
    transactions.sort(key=lambda t: t['date'], reverse=True)
    df = pd.DataFrame(transactions)
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    df['Month'] = df['date'].dt.strftime('%b-%y')
    df['amount'] = df['amount'].apply(lambda x: round(float(x), 2))
    df['Amount'] = df['amount']
    df = df[
        ~(
            df['description'].str.strip().str.upper().eq('AUTOMATIC PAYMENT - THANK YOU') |
            df['description'].str.strip().str.upper().str.startswith('ACH DEPOSIT INTERNET TRANSFER')
        )
    ]
    sum_amount = df['amount'].sum()
    sum_row = {
        'Month': '',
        'description': 'TOTAL (excluding payments)',
        'amount': sum_amount,
        'Amount': round(float(sum_amount), 2),
        'category': '',
        'card': ''
    }
    df = pd.concat([df, pd.DataFrame([sum_row])], ignore_index=True)
    df.rename(columns={'description': 'Raw Txn Description', 'enhanced_description': 'Enhanced Txn Description', 'category': 'Category', 'card': 'Card'}, inplace=True)
    export_cols = ['Month', 'Card', 'Raw Txn Description', 'Enhanced Txn Description', 'Amount', 'Category']
    output = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
    df.to_excel(output.name, index=False, columns=export_cols)
    return send_file(output.name, as_attachment=True, download_name='transactions.xlsx')

@app.route('/summary')
def summary():
    transactions_file = session.get('transactions_file')
    job_id = session.get('job_id')
    transactions = None
    if transactions_file and os.path.exists(transactions_file):
        with open(transactions_file, 'rb') as tf:
            transactions = pickle.load(tf)
    elif job_id:
        redis_url = (
            os.environ.get("STACKHERO_REDIS_URL_TLS") or
            os.environ.get("STACKHERO_REDIS_URL_CLEAR") or
            os.environ.get("REDISGREEN_URL") or
            os.environ.get("REDISCLOUD_URL") or
            os.environ.get("MEMETRIA_REDIS_URL")
        )
        redis_conn = Redis.from_url(redis_url) if redis_url else None
        if redis_conn:
            results = redis_conn.get(f"results:{job_id}")
            if results:
                transactions = pickle.loads(results)
    if not transactions:
        return redirect(url_for('index'))
    def is_repayment(txn):
        desc = txn['description'].strip().upper()
        return (
            desc == 'AUTOMATIC PAYMENT - THANK YOU' or
            desc.startswith('ACH DEPOSIT INTERNET TRANSFER')
        )
    filtered_transactions = [t for t in transactions if not is_repayment(t) and t.get('category') not in ['Income / Refunds']]
    df = pd.DataFrame(filtered_transactions)
    summary = df.groupby('category')['amount'].sum().reset_index()
    summary = summary.sort_values(by='amount', ascending=False)
    labels = summary['category'].tolist()
    values = summary['amount'].tolist()
    total_spend = df['amount'].sum()
    if not df.empty:
        min_date = df['date'].min().strftime('%Y-%m-%d')
        max_date = df['date'].max().strftime('%Y-%m-%d')
    else:
        min_date = max_date = ''
    if not df.empty:
        df['Month'] = pd.to_datetime(df['date']).dt.strftime('%b-%y')
        df['Month_dt'] = pd.to_datetime(df['date']).dt.to_period('M')
        pivot = df.pivot_table(index=['Month', 'Month_dt'], columns='category', values='amount', aggfunc='sum', fill_value=0)
        pivot = pivot.sort_index(level='Month_dt')
        bar_labels = pivot.index.get_level_values('Month').tolist()
        bar_datasets = []
        colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc949', '#af7aa1', '#ff9da7', '#9c755f', '#bab0ab']
        for i, cat in enumerate(pivot.columns):
            bar_datasets.append({
                'label': cat,
                'data': pivot[cat].tolist(),
                'backgroundColor': colors[i % len(colors)]
            })
    else:
        bar_labels = []
        bar_datasets = []
    return render_template(
        'summary.html',
        summary=summary,
        labels=labels,
        values=values,
        total_spend=total_spend,
        min_date=min_date,
        max_date=max_date,
        bar_labels=bar_labels,
        bar_datasets=bar_datasets,
        job_id=job_id
    )

def categorize_and_enhance_transaction(description):
    # Special case for card repayment
    if description.strip().upper() == 'AUTOMATIC PAYMENT - THANK YOU':
        return 'Card Repayment', 'Credit card bill payment'
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = (
        f"Given this credit card transaction description: '{description}',\n"
        "1. Categorize it with one of the following, do not create new categories: 'Food & Beverage', 'Health & Wellness', 'Travel (Taxi / Uber / Lyft / Revel)', 'Travel (Subway / MTA)', 'Gas & Fuel','Travel (Flights / Trains)', 'Hotel', 'Groceries', 'Entertainment / Leisure Activities', 'Shopping', 'Income / Refunds', 'Utilities (Electricity, Telecom, Internet)', 'Other (Miscellaneous)'.\n"
        "2. Write a short, human-perceivable summary of the expense, including the merchant type and location if available. Follow the format: 'Merchant Name, Location, brief description of expense purpose (no more than 10 words)'\n"
        "Return your answer as JSON in the following format (no markdown, no explanation, just JSON):\n"
        '{"category": "...", "enhanced_description": "..."}'
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3
        )
        content = response.choices[0].message.content.strip()
        print("OpenAI raw response:", content)  # Log for debugging
        # Try to extract JSON from the response
        try:
            data = json.loads(content)
        except Exception as e:
            print("JSON decode error:", e)
            # Try to extract JSON substring if extra text is present
            import re
            match = re.search(r'\\{.*\\}', content, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except Exception as e2:
                    print("Still failed to parse JSON:", e2)
                    data = {}
            else:
                data = {}
        return data.get("category", "Uncategorized"), data.get("enhanced_description", description)
    except Exception as e:
        print(f"OpenAI error (combined): {e}")
        return "Uncategorized", description

@app.route('/create-payment-intent', methods=['POST'])
def create_payment_intent():
    data = request.get_json()
    num_pdfs = data.get('num_pdfs', 1)
    amount = num_pdfs * 200  # $2 per PDF, in cents
    try:
        intent = stripe.PaymentIntent.create(
            amount=amount,
            currency='usd',
            automatic_payment_methods={'enabled': True},
        )
        return jsonify({'clientSecret': intent.client_secret})
    except Exception as e:
        print(f'Stripe error: {e}')
        return jsonify({'error': 'Payment setup failed. Please try again or use a different card.'}), 400

if __name__ == '__main__':
    app.run(debug=True)