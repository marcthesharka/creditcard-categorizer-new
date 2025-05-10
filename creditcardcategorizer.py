print("Starting app.py")

import os
import tempfile
from flask import Flask, render_template, request, redirect, url_for, send_file, session, jsonify
import pdfplumber
import pandas as pd
from datetime import datetime, date, timedelta
import openai
import pickle
import json
import re
import httpx

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
        # Initialize progress tracking
        session['progress'] = {
            'current': 0,
            'total': 0,
            'status': 'parsing'
        }
        # Parse PDFs first
        for file in files:
            if file and file.filename.endswith('.pdf'):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                    file.save(tmp.name)
                    transactions = parse_pdf_transactions(tmp.name)
                    os.unlink(tmp.name)
                all_transactions.extend(transactions)
        # Update total count
        session['progress']['total'] = len(all_transactions)
        session['progress']['status'] = 'categorizing'
        # Process transactions in batches
        categorized_transactions = []
        batch_size = 5  # Process 5 transactions at a time
        for i in range(0, len(all_transactions), batch_size):
            batch = all_transactions[i:i + batch_size]
            for transaction in batch:
                result = categorize_and_enhance_transaction(transaction)
                categorized_transactions.append({
                    'date': transaction['date'],
                    'description': transaction['description'],
                    'amount': transaction['amount'],
                    'category': result['category'],
                    'enhanced_description': result['enhanced_description'],
                    'card': transaction['card']
                })
                session['progress']['current'] += 1
                session.modified = True
        # Store the categorized data in session
        session['categorized_data'] = categorized_transactions
        session['progress']['status'] = 'complete'
        session.modified = True
        return redirect(url_for('categorize'))
    return render_template('index.html')

@app.route('/categorize', methods=['GET'])
def categorize():
    try:
        if 'categorized_data' not in session:
            return redirect(url_for('index'))
            
        categorized_data = session['categorized_data']
        # Sort transactions by date descending
        categorized_data.sort(key=lambda x: x['date'], reverse=True)
        
        # Filter out repayment transactions for total spend calculation
        filtered_transactions = [
            t for t in categorized_data 
            if not (t['description'].strip().upper() == 'AUTOMATIC PAYMENT - THANK YOU' or 
                   t['description'].strip().upper().startswith('ACH DEPOSIT INTERNET TRANSFER'))
        ]
        
        return render_template(
            'categorize.html',
            transactions=categorized_data,
            filtered_transactions=filtered_transactions
        )
    except Exception as e:
        app.logger.error(f"Error in categorize endpoint: {str(e)}")
        return render_template('error.html', error=str(e))

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

@app.route('/progress')
def progress():
    try:
        if 'progress' not in session:
            return jsonify({'error': 'No progress found'}), 404
        progress_data = session['progress']
        current = progress_data.get('current', 0)
        total = progress_data.get('total', 0)
        status = progress_data.get('status', '')
        if current >= total and status == 'complete':
            return jsonify({
                'current': current,
                'total': total,
                'status': 'complete',
                'redirect': url_for('categorize')
            })
        return jsonify({
            'current': current,
            'total': total,
            'status': status
        })
    except Exception as e:
        app.logger.error(f"Error in progress endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

def categorize_and_enhance_transaction(transaction):
    try:
        # Special case for card repayment
        if "APPLE CARD PAYMENT" in transaction['description'].upper():
            return {
                'category': 'Card Payment',
                'enhanced_description': 'Apple Card payment'
            }
        
        # Configure OpenAI client with explicit HTTP client settings
        client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
            http_client=httpx.Client(
                timeout=httpx.Timeout(30.0, connect=10.0),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            )
        )
        
        # Create a more concise prompt
        prompt = f"""Categorize this credit card transaction and provide an enhanced description:
        Description: {transaction['description']}
        Amount: {transaction['amount']}
        Date: {transaction['date']}
        
        Return ONLY a JSON object with two fields:
        1. "category": The transaction category (e.g., "Travel", "Food & Dining", "Shopping")
        2. "enhanced_description": A clear, detailed description of the transaction
        
        Example response format:
        {{"category": "Food & Dining", "enhanced_description": "Restaurant name, location, meal type"}}"""
        
        # Make the API call with a shorter timeout
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150,
            timeout=25  # 25 second timeout for the API call
        )
        
        # Parse the response
        try:
            result = json.loads(response.choices[0].message.content)
            return result
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract category and description from the text
            content = response.choices[0].message.content
            category = content.split('"category": "')[1].split('"')[0] if '"category": "' in content else "Uncategorized"
            enhanced_description = content.split('"enhanced_description": "')[1].split('"')[0] if '"enhanced_description": "' in content else transaction['description']
            return {
                'category': category,
                'enhanced_description': enhanced_description
            }
            
    except Exception as e:
        app.logger.error(f"Error categorizing transaction: {str(e)}")
        return {
            'category': 'Uncategorized',
            'enhanced_description': transaction['description']
        }

if __name__ == '__main__':
    app.run(debug=True)