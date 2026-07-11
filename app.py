# ================================================================
#   StockIQ — Complete Flask Backend (FINAL VERSION)
#   File     : app.py
#   Location : StockProject/app.py
#   Run      : python app.py
#   Open     : http://localhost:5000
# ================================================================

from flask import (
    Flask, render_template, request,
    redirect, url_for, session, jsonify
)
import sqlite3, hashlib, random, os

app = Flask(__name__)
app.secret_key = 'stockiq_manoj_2024_secret'

DATABASE = 'users.db'

# ── DB helpers ─────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
        id       INTEGER PRIMARY KEY AUTOINCREMENT,
        name     TEXT    NOT NULL,
        email    TEXT    NOT NULL UNIQUE,
        password TEXT    NOT NULL,
        balance  REAL    DEFAULT 100000.0
    )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS portfolio (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id   INTEGER NOT NULL,
        symbol    TEXT    NOT NULL,
        qty       INTEGER NOT NULL,
        avg_price REAL    NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS transactions (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id   INTEGER NOT NULL,
        symbol    TEXT    NOT NULL,
        action    TEXT    NOT NULL,
        qty       INTEGER NOT NULL,
        price     REAL    NOT NULL,
        total     REAL    NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()
    print("✅ Database ready!")

def hash_password(p):
    return hashlib.sha256(p.encode()).hexdigest()

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


# ================================================================
#   AUTH ROUTES
# ================================================================

@app.route('/')
def home():
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email    = request.form.get('email','').strip()
        password = request.form.get('password','').strip()
        if not email or not password:
            return render_template('login.html', error='Please enter email and password!')
        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE email=?',(email,)).fetchone()
        conn.close()
        if user and user['password'] == hash_password(password):
            session['user_id']    = user['id']
            session['user_name']  = user['name']
            session['user_email'] = user['email']
            return redirect(url_for('dashboard'))
        return render_template('login.html', error='Wrong email or password. Try again!')
    return render_template('login.html')


@app.route('/register', methods=['POST'])
def register():
    name     = request.form.get('name','').strip()
    email    = request.form.get('email','').strip()
    password = request.form.get('password','').strip()
    if not name or not email or not password:
        return render_template('login.html', error='Please fill all fields!')
    if len(password) < 6:
        return render_template('login.html', error='Password must be at least 6 characters!')
    try:
        conn = get_db()
        conn.execute('INSERT INTO users (name,email,password,balance) VALUES (?,?,?,?)',
                     (name, email, hash_password(password), 100000.0))
        conn.commit()
        conn.close()
        return render_template('login.html',
                               success='Account created! Login with your email and password.')
    except sqlite3.IntegrityError:
        return render_template('login.html', error='This email is already registered!')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# ================================================================
#   PAGE ROUTES
# ================================================================

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', user_name=session['user_name'])

@app.route('/markets')
@login_required
def markets():
    return render_template('markets.html', user_name=session['user_name'])

@app.route('/charts')
@login_required
def charts():
    return render_template('charts.html', user_name=session['user_name'])

@app.route('/predict')
@login_required
def predict():
    return render_template('ai_prediction.html', user_name=session['user_name'])

@app.route('/portfolio')
@login_required
def portfolio():
    return render_template('portfolio.html', user_name=session['user_name'])

@app.route('/watchlist')
@login_required
def watchlist():
    return render_template('watchlist.html', user_name=session['user_name'])

@app.route('/news')
@login_required
def news():
    return render_template('news.html', user_name=session['user_name'])

@app.route('/screener')
@login_required
def screener():
    return render_template('screener.html', user_name=session['user_name'])

@app.route('/analysis')
@login_required
def analysis():
    return render_template('analysis.html', user_name=session['user_name'])

@app.route('/settings')
@login_required
def settings():
    return render_template('settings.html', user_name=session['user_name'])


# ================================================================
#   API ROUTES
# ================================================================

@app.route('/api/balance')
@login_required
def get_balance():
    conn = get_db()
    user = conn.execute('SELECT balance FROM users WHERE id=?',
                        (session['user_id'],)).fetchone()
    conn.close()
    return jsonify({'balance': round(user['balance'], 2)})


@app.route('/api/add_money', methods=['POST'])
@login_required
def add_money():
    data   = request.get_json()
    amount = float(data.get('amount', 0))
    if amount <= 0:
        return jsonify({'success': False, 'error': 'Enter a valid amount!'})
    if amount > 1000000:
        return jsonify({'success': False, 'error': 'Maximum add limit is ₹10,00,000!'})
    conn = get_db()
    conn.execute('UPDATE users SET balance = balance + ? WHERE id=?',
                 (amount, session['user_id']))
    conn.commit()
    new_bal = conn.execute('SELECT balance FROM users WHERE id=?',
                           (session['user_id'],)).fetchone()['balance']
    conn.close()
    return jsonify({'success': True, 'new_balance': round(new_bal, 2),
                    'message': f'₹{amount:,.0f} added to your portfolio!'})


@app.route('/api/buy', methods=['POST'])
@login_required
def buy_stock():
    data   = request.get_json()
    symbol = data.get('symbol','').upper()
    qty    = int(data.get('qty', 0))
    price  = float(data.get('price', 0))
    total  = qty * price
    if qty <= 0 or price <= 0:
        return jsonify({'success': False, 'error': 'Invalid quantity or price!'})
    conn = get_db()
    user = conn.execute('SELECT balance FROM users WHERE id=?',
                        (session['user_id'],)).fetchone()
    if user['balance'] < total:
        conn.close()
        return jsonify({'success': False,
                        'error': f'Not enough balance! Need ₹{total:,.0f}, have ₹{user["balance"]:,.0f}'})
    existing = conn.execute(
        'SELECT * FROM portfolio WHERE user_id=? AND symbol=?',
        (session['user_id'], symbol)).fetchone()
    if existing:
        new_qty = existing['qty'] + qty
        new_avg = round((existing['avg_price']*existing['qty'] + price*qty) / new_qty, 2)
        conn.execute('UPDATE portfolio SET qty=?, avg_price=? WHERE id=?',
                     (new_qty, new_avg, existing['id']))
    else:
        conn.execute('INSERT INTO portfolio (user_id,symbol,qty,avg_price) VALUES (?,?,?,?)',
                     (session['user_id'], symbol, qty, price))
    conn.execute('UPDATE users SET balance=balance-? WHERE id=?',
                 (total, session['user_id']))
    conn.execute('INSERT INTO transactions (user_id,symbol,action,qty,price,total) VALUES (?,?,?,?,?,?)',
                 (session['user_id'], symbol, 'BUY', qty, price, total))
    conn.commit()
    conn.close()
    return jsonify({'success': True,
                    'message': f'✅ Bought {qty} shares of {symbol} for ₹{total:,.0f}!'})


@app.route('/api/sell', methods=['POST'])
@login_required
def sell_stock():
    data   = request.get_json()
    symbol = data.get('symbol','').upper()
    qty    = int(data.get('qty', 0))
    price  = float(data.get('price', 0))
    total  = qty * price
    if qty <= 0 or price <= 0:
        return jsonify({'success': False, 'error': 'Invalid quantity or price!'})
    conn = get_db()
    existing = conn.execute(
        'SELECT * FROM portfolio WHERE user_id=? AND symbol=?',
        (session['user_id'], symbol)).fetchone()
    if not existing or existing['qty'] < qty:
        conn.close()
        return jsonify({'success': False,
                        'error': f'Not enough shares! You have {existing["qty"] if existing else 0} shares.'})
    new_qty = existing['qty'] - qty
    if new_qty == 0:
        conn.execute('DELETE FROM portfolio WHERE id=?', (existing['id'],))
    else:
        conn.execute('UPDATE portfolio SET qty=? WHERE id=?', (new_qty, existing['id']))
    conn.execute('UPDATE users SET balance=balance+? WHERE id=?',
                 (total, session['user_id']))
    conn.execute('INSERT INTO transactions (user_id,symbol,action,qty,price,total) VALUES (?,?,?,?,?,?)',
                 (session['user_id'], symbol, 'SELL', qty, price, total))
    conn.commit()
    conn.close()
    return jsonify({'success': True,
                    'message': f'🔴 Sold {qty} shares of {symbol} for ₹{total:,.0f}!'})


@app.route('/api/portfolio')
@login_required
def get_portfolio():
    conn     = get_db()
    holdings = conn.execute(
        'SELECT * FROM portfolio WHERE user_id=?', (session['user_id'],)).fetchall()
    balance  = conn.execute(
        'SELECT balance FROM users WHERE id=?', (session['user_id'],)).fetchone()['balance']
    conn.close()
    result = [{'symbol':h['symbol'], 'qty':h['qty'], 'avg_price':h['avg_price']}
              for h in holdings]
    return jsonify({'holdings': result, 'balance': round(balance, 2)})


@app.route('/api/transactions')
@login_required
def get_transactions():
    conn = get_db()
    txns = conn.execute(
        'SELECT * FROM transactions WHERE user_id=? ORDER BY timestamp DESC LIMIT 20',
        (session['user_id'],)).fetchall()
    conn.close()
    result = [{'symbol':t['symbol'], 'action':t['action'], 'qty':t['qty'],
               'price':t['price'], 'total':t['total'], 'time':t['timestamp']}
              for t in txns]
    return jsonify({'transactions': result})


@app.route('/api/stock/<symbol>')
def stock_data(symbol):
    bases = {'RELIANCE':2700,'TCS':3650,'INFOSYS':1580,'HDFC':1650,
             'WIPRO':490,'AAPL':178,'TSLA':220,'ADANIENT':2600,
             'SUNPHARMA':1180,'BAJFINANCE':6800}
    price = bases.get(symbol.upper(), 1000)
    prices = []
    for _ in range(60):
        price *= (1 + random.uniform(-0.015, 0.018))
        prices.append(round(price, 2))
    return jsonify({'symbol':symbol.upper(),'prices':prices,
                    'current':prices[-1],
                    'change':round(prices[-1]-prices[-2],2),
                    'pct':round((prices[-1]-prices[-2])/prices[-2]*100,2)})


@app.route('/api/market')
def market_data():
    return jsonify({
        'nifty50'  :{'value':22147,'change':+162.4,'pct':+0.74},
        'sensex'   :{'value':73088,'change':+442.1,'pct':+0.61},
        'banknifty':{'value':47312,'change':-218.5,'pct':-0.46},
        'indiavix' :{'value':13.42,'change': -0.32,'pct':-2.33},
    })


# ================================================================
#   START
# ================================================================
if __name__ == '__main__':
    init_db()
    print()
    print("=" * 55)
    print("   🚀  StockIQ Server is RUNNING!")
    print("=" * 55)
    print("   🌐  Open Chrome → http://localhost:5000")
    print("=" * 55)
    app.run(debug=True, host='0.0.0.0', port=5000)