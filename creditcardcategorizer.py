print("Starting app.py")

import os
import tempfile
from flask import Flask, render_template, request, redirect, url_for, send_file, session
import pdfplumber
import pandas as pd
from datetime import datetime, date, timedelta
import openai
import pickle
import json
import re

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key in production

api_key = os.getenv("OPENAI_API_KEY")

LOG_FILE = os.path.join(tempfile.gettempdir(), 'openai_progress.log')

def parse_pdf_transactions(pdf_path):
    import re
    from datetime import datetime
    with pdfplumber.open(pdf_path) as pdf:
        first_page_text = pdf.pages[0].extract_text() or ""
        if "Chase" in first_page_text:
            return parse_chase_pdf_transactions(pdf_path)
        elif "Apple Card" in first_page_text:
            return parse_apple_pdf_transactions(pdf_path)
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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Clear the log file at the start
        open(LOG_FILE, 'w').close()
        files = request.files.getlist('pdf')
        all_transactions = []
        for file in files:
            if file and file.filename.endswith('.pdf'):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                    file.save(tmp.name)
                    transactions = parse_pdf_transactions(tmp.name)
                    os.unlink(tmp.name)
                all_transactions.extend(transactions)
        # Auto-categorize using OpenAI
        for t in all_transactions:
            t['category'], t['enhanced_description'] = categorize_and_enhance_transaction(t['description'])
        # Save to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tf:
            pickle.dump(all_transactions, tf)
            temp_filename = tf.name
        session['transactions_file'] = temp_filename
        print(f"Total transactions parsed: {len(all_transactions)}")
        return redirect(url_for('categorize'))
    return render_template('index.html')

@app.route('/categorize', methods=['GET', 'POST'])
def categorize():
    transactions_file = session.get('transactions_file')
    if not transactions_file or not os.path.exists(transactions_file):
        return redirect(url_for('index'))
    with open(transactions_file, 'rb') as tf:
        transactions = pickle.load(tf)
    # Sort transactions by date descending
    transactions.sort(key=lambda t: t['date'], reverse=True)
    if request.method == 'POST':
        for i, t in enumerate(transactions):
            t['category'] = request.form.get(f'category_{i}', '')
        with open(transactions_file, 'wb') as tf:
            pickle.dump(transactions, tf)
        return redirect(url_for('summary'))

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
    if not transactions_file or not os.path.exists(transactions_file):
        return redirect(url_for('index'))
    with open(transactions_file, 'rb') as tf:
        transactions = pickle.load(tf)
    if not transactions:
        return redirect(url_for('index'))
    # Sort transactions by date descending
    transactions.sort(key=lambda t: t['date'], reverse=True)
    df = pd.DataFrame(transactions)
    # Ensure all dates are timezone-unaware and format as 'Mon-YY'
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    df['Month'] = df['date'].dt.strftime('%b-%y')  # e.g., 'Mar-25'
    # Format amount as a number rounded to 2 decimals
    df['amount'] = df['amount'].apply(lambda x: round(float(x), 2))
    df['Amount'] = df['amount']
    # Exclude repayment transactions from summary totals
    df = df[
        ~(
            df['description'].str.strip().str.upper().eq('AUTOMATIC PAYMENT - THANK YOU') |
            df['description'].str.strip().str.upper().str.startswith('ACH DEPOSIT INTERNET TRANSFER')
        )
    ]
    sum_amount = df['amount'].sum()
    # Append sum row
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
    if not transactions_file or not os.path.exists(transactions_file):
        return redirect(url_for('index'))
    with open(transactions_file, 'rb') as tf:
        transactions = pickle.load(tf)
    # Use the same repayment exclusion logic as categorize
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
    # Bar chart data: monthly spend by category
    if not df.empty:
        df['Month'] = pd.to_datetime(df['date']).dt.strftime('%b-%y')
        # Sort months chronologically
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
        bar_datasets=bar_datasets
    )

def categorize_and_enhance_transaction(description):
    # Special case for card repayment
    if description.strip().upper() == 'AUTOMATIC PAYMENT - THANK YOU':
        return 'Card Repayment', 'Credit card bill payment'
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = (
        f"Given this credit card transaction description: '{description}',\n"
        "1. Categorize it with one of the following, do not create new categories: 'Food & Beverage', 'Health & Wellness', 'Travel (Taxi / Uber / Lyft / Revel)', 'Travel (Subway / MTA)', 'Gas & Fuel','Travel (Flights / Trains)', 'Hotel', 'Groceries', 'Entertainment', 'Shopping', 'Income / Refunds', 'Utilities (Electricity, Telecom, Internet)', 'Other (Miscellaneous)'.\n"
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
        print("OpenAI raw response:", content)
        with open(LOG_FILE, 'a') as logf:
            logf.write(f"{description}:\n{content}\n\n")
        # Remove code block markers if present
        if content.startswith("```"):
            content = re.sub(r"^```[a-zA-Z]*\\n?", "", content)
            content = content.rstrip("`").strip()
        data = json.loads(content)
        return data.get("category", "Uncategorized"), data.get("enhanced_description", description)
    except Exception as e:
        with open(LOG_FILE, 'a') as logf:
            logf.write(f"{description}:\nOpenAI error: {e}\n\n")
        print(f"OpenAI error (combined): {e}")
        return "Uncategorized", description

@app.route('/progress')
def progress():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            return f.read()
    return ""

if __name__ == '__main__':
    app.run(debug=True)