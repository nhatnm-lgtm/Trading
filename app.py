import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import io
import warnings
warnings.filterwarnings('ignore')

# VectorBT import wrapper for graceful fallback
try:
    import vectorbt as vbt
    HAS_VECTORBT = True
except Exception:
    HAS_VECTORBT = False

# Try hyperopt import
try:
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
    HAS_HYPEROPT = True
except Exception:
    HAS_HYPEROPT = False


# ==============================================================================
# PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="ACB Trading Strategy - EMA & OBV Backtest",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .status-box-buy {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #86EFAC;
        padding: 1rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.2rem;
        text-align: center;
    }
    .status-box-sell {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
        padding: 1rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.2rem;
        text-align: center;
    }
    .status-box-hold {
        background-color: #FEF9C3;
        color: #854D0E;
        border: 1px solid #FDE047;
        padding: 1rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.2rem;
        text-align: center;
    }
    .status-box-aside {
        background-color: #F1F5F9;
        color: #475569;
        border: 1px solid #CBD5E1;
        padding: 1rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.2rem;
        text-align: center;
    }
</style>
""", unsafe_allow_allowed_html=True)


# ==============================================================================
# HELPER FUNCTIONS: DATA PROCESSING & INDICATORS
# ==============================================================================
@st.cache_data
def load_default_data():
    """Tải dữ liệu ACB.csv mặc định nếu có trong thư mục."""
    try:
        df = pd.read_csv('ACB.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        return df
    except Exception:
        return None

def process_uploaded_data(file):
    """Xử lý file CSV do người dùng tải lên."""
    try:
        df = pd.read_csv(file)
        # Chuẩn hóa tên cột
        col_map = {c: c.strip().capitalize() for c in df.columns}
        df.rename(columns=col_map, inplace=True)
        
        # Tìm cột Date
        date_col = None
        for c in df.columns:
            if c.lower() in ['date', 'time', 'ngày', 'datetime']:
                date_col = c
                break
        
        if date_col:
            df['Date'] = pd.to_datetime(df[date_col])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
        else:
            st.error("Không tìm thấy cột Ngày (Date) trong file CSV.")
            return None

        # Kiểm tra cột Close
        if 'Close' not in df.columns:
            # Thử lấy cột giá khác nếu không có Close
            price_cols = [c for c in df.columns if 'close' in c.lower() or 'giá' in c.lower()]
            if price_cols:
                df.rename(columns={price_cols[0]: 'Close'}, inplace=True)
            else:
                st.error("File CSV cần có cột giá đóng cửa 'Close'.")
                return None
                
        if 'Volume' not in df.columns:
            vol_cols = [c for c in df.columns if 'vol' in c.lower() or 'khối lượng' in c.lower()]
            if vol_cols:
                df.rename(columns={vol_cols[0]: 'Volume'}, inplace=True)
            else:
                df['Volume'] = 1_000_000

        return df
    except Exception as e:
        st.error(f"Lỗi đọc file CSV: {e}")
        return None

def calculate_signals(data, strategy_type, ema_period, obv_slope_period, price_col='Close'):
    """
    Tính toán chỉ báo và tín hiệu Mua/Bán theo chiến lược.
    Áp dụng shift(1) để tránh Look-ahead Bias.
    """
    df = data.copy()
    close = df[price_col]
    volume = df['Volume'] if 'Volume' in df.columns else pd.Series(1, index=df.index)

    # 1. EMA
    ema = ta.trend.ema_indicator(close, window=int(ema_period))
    df['EMA'] = ema

    # 2. OBV & OBV Slope
    obv = ta.volume.on_balance_volume(close, volume)
    obv_slope = obv.diff(int(obv_slope_period))
    df['OBV'] = obv
    df['OBV_Slope'] = obv_slope

    # 3. Tín hiệu chưa shift (Raw Signals)
    if strategy_type == 'EMA Riêng lẻ':
        raw_entries = (close > ema)
        raw_exits = (close < ema)
    elif strategy_type == 'OBV Riêng lẻ':
        raw_entries = (obv_slope > 0)
        raw_exits = (obv_slope < 0)
    else:  # EMA + OBV Kết hợp
        raw_entries = (close > ema) & (obv_slope > 0)
        raw_exits = (close < ema)

    # Áp dụng shift(1) để triệt tiêu Look-ahead Bias (dùng tín hiệu T-1 cho quyết định T)
    entries = raw_entries.shift(1, fill_value=False)
    exits = raw_exits.shift(1, fill_value=False)

    df['Raw_Entry'] = raw_entries
    df['Raw_Exit'] = raw_exits
    df['Entry'] = entries
    df['Exit'] = exits

    return df, entries, exits


# ==============================================================================
# BACKTEST ENGINE (PANDAS SIMULATION & VECTORBT)
# ==============================================================================
def run_backtest_pandas(df, entries, exits, initial_capital=100_000_000, fee_pct=0.002, slippage_pct=0.001, stop_loss_pct=0.07, price_col='Close'):
    """
    Engine kiểm định thuần Pandas với hạch toán vị thế (Position Accounting)
    và quản lý rủi ro Stop Loss.
    """
    prices = df[price_col].values
    dates = df.index
    n = len(df)

    position = np.zeros(n, dtype=int)
    buy_orders = np.zeros(n, dtype=bool)
    sell_orders = np.zeros(n, dtype=bool)

    cash = initial_capital
    holdings = 0.0
    equity = np.zeros(n)
    
    trades = []
    entry_price = 0.0
    entry_date = None
    entry_idx = 0
    curr_pos = 0

    for i in range(n):
        price = prices[i]
        
        # Kiểm tra điều kiện Stop Loss nếu đang giữ vị thế
        stop_loss_triggered = False
        if curr_pos == 1 and stop_loss_pct > 0:
            if price <= entry_price * (1.0 - stop_loss_pct):
                stop_loss_triggered = True

        # Quyết định giao dịch dựa trên vị thế hiện tại và tín hiệu
        if curr_pos == 0 and entries.iloc[i]: # Tín hiệu MUA khi đang cầm tiền
            exec_price = price * (1.0 + slippage_pct)
            shares = (cash * (1.0 - fee_pct)) / exec_price
            if shares > 0:
                holdings = shares
                cash = 0.0
                curr_pos = 1
                entry_price = exec_price
                entry_date = dates[i]
                entry_idx = i
                buy_orders[i] = True

        elif curr_pos == 1 and (exits.iloc[i] or stop_loss_triggered): # Tín hiệu BÁN hoặc Stop Loss khi đang giữ cổ phiếu
            exec_price = price * (1.0 - slippage_pct)
            sale_proceeds = holdings * exec_price * (1.0 - fee_pct)
            pnl = sale_proceeds - (holdings * entry_price)
            ret_pct = (sale_proceeds / (holdings * entry_price) - 1.0) * 100.0
            duration = (dates[i] - entry_date).days
            
            exit_reason = "Stop Loss" if stop_loss_triggered else "Exit Signal (EMA/OBV)"
            
            trades.append({
                'Trade ID': len(trades) + 1,
                'Entry Date': entry_date,
                'Exit Date': dates[i],
                'Entry Price': entry_price,
                'Exit Price': exec_price,
                'PnL (VND)': pnl,
                'Return (%)': ret_pct,
                'Duration (Days)': duration,
                'Exit Reason': exit_reason
            })
            
            cash = sale_proceeds
            holdings = 0.0
            curr_pos = 0
            sell_orders[i] = True

        position[i] = curr_pos
        equity[i] = cash + (holdings * price)

    # Benchmark: Buy & Hold
    buy_hold_shares = (initial_capital * (1.0 - fee_pct)) / (prices[0] * (1.0 + slippage_pct))
    benchmark_equity = buy_hold_shares * prices

    # Thống kê hiệu suất
    total_return_pct = (equity[-1] / initial_capital - 1.0) * 100.0
    benchmark_return_pct = (benchmark_equity[-1] / initial_capital - 1.0) * 100.0

    # Tính Sharpe Ratio
    equity_series = pd.Series(equity, index=dates)
    daily_returns = equity_series.pct_change().dropna()
    if daily_returns.std() > 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = np.nan

    # Tính Max Drawdown
    cummax = equity_series.cummax()
    drawdown = (equity_series - cummax) / cummax
    max_drawdown_pct = drawdown.min() * 100.0

    # Phân tích các giao dịch
    trades_df = pd.DataFrame(trades)
    closed_trades = len(trades_df)
    
    if closed_trades > 0:
        win_trades = len(trades_df[trades_df['Return (%)'] > 0])
        win_rate_pct = (win_trades / closed_trades) * 100.0
        gains = trades_df[trades_df['PnL (VND)'] > 0]['PnL (VND)'].sum()
        losses = abs(trades_df[trades_df['PnL (VND)'] < 0]['PnL (VND)'].sum())
        profit_factor = (gains / losses) if losses > 0 else (np.inf if gains > 0 else np.nan)
    else:
        win_rate_pct = 0.0
        profit_factor = np.nan

    metrics = {
        'Total Return (%)': total_return_pct,
        'Benchmark Return (%)': benchmark_return_pct,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown (%)': max_drawdown_pct,
        'Closed Trades': closed_trades,
        'Win Rate (%)': win_rate_pct,
        'Profit Factor': profit_factor,
        'Final Equity (VND)': equity[-1]
    }

    results = {
        'equity': equity_series,
        'benchmark': pd.Series(benchmark_equity, index=dates),
        'drawdown': drawdown * 100.0,
        'position': pd.Series(position, index=dates),
        'buy_orders': pd.Series(buy_orders, index=dates),
        'sell_orders': pd.Series(sell_orders, index=dates),
        'trades': trades_df,
        'metrics': metrics
    }

    return results

def run_backtest_vbt(df, entries, exits, fee_pct=0.002, slippage_pct=0.001, stop_loss_pct=0.07, price_col='Close'):
    """Engine VectorBT backtest nếu hệ thống cài sẵn vectorbt."""
    if not HAS_VECTORBT:
        return None
    try:
        pf = vbt.Portfolio.from_signals(
            close=df[price_col],
            entries=entries.to_numpy(dtype=bool),
            exits=exits.to_numpy(dtype=bool),
            direction='longonly',
            accumulate=False,
            fees=fee_pct,
            slippage=slippage_pct,
            sl_stop=stop_loss_pct if stop_loss_pct > 0 else None,
            freq='D'
        )
        return pf
    except Exception:
        return None


# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
st.sidebar.title("⚙️ Cấu Hình Hệ Thống")

# 1. Nguồn Dữ Liệu
st.sidebar.subheader("1. Nguồn Dữ Liệu Giá")
data_source = st.sidebar.radio("Chọn nguồn dữ liệu:", ["ACB.csv Mặc Định", "Tải File CSV Mới"])

df_raw = None
if data_source == "ACB.csv Mặc Định":
    df_raw = load_default_data()
    if df_raw is None:
        st.sidebar.error("Không tìm thấy file ACB.csv trong thư mục gốc. Vui lòng tải file CSV lên.")
else:
    uploaded_file = st.sidebar.file_uploader("Tải file CSV cổ phiếu (có cột Date, Close, Volume):", type=['csv'])
    if uploaded_file is not None:
        df_raw = process_uploaded_data(uploaded_file)

if df_raw is None:
    st.info("👋 **Chào mừng bạn đến với App Kiểm Định Chiến Lược Trading EMA & OBV!**")
    st.warning("Vui lòng tải file `ACB.csv` hoặc tải file CSV dữ liệu lịch sử cổ phiếu ở thanh Sidebar bên trái để bắt đầu.")
    st.stop()

# Đảm bảo phạm vi ngày
min_date = df_raw.index.min().date()
max_date = df_raw.index.max().date()

st.sidebar.subheader("2. Phạm Vi Thời Gian Kiểm Định")
date_range = st.sidebar.date_input(
    "Chọn khoảng thời gian:",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    df = df_raw.loc[str(start_date):str(end_date)].copy()
else:
    df = df_raw.copy()

if len(df) < 30:
    st.error("Dữ liệu trong khoảng thời gian đã chọn quá ít (cần ít nhất 30 phiên).")
    st.stop()

# 3. Chọn Chiến Lược & Tham Số
st.sidebar.subheader("3. Thiết Lập Chiến Lược")
strategy_type = st.sidebar.selectbox(
    "Loại Chiến Lược:",
    ["EMA + OBV Kết hợp", "EMA Riêng lẻ", "OBV Riêng lẻ"]
)

col_p1, col_p2 = st.sidebar.columns(2)
with col_p1:
    ema_period = st.number_input("Chu kỳ EMA:", min_value=5, max_value=200, value=20, step=1)
with col_p2:
    obv_slope_period = st.number_input("Chu kỳ OBV Slope:", min_value=1, max_value=50, value=3, step=1)

# 4. Quản Lý Vốn & Rủi Ro
st.sidebar.subheader("4. Vốn & Quản Lý Rủi Ro")
initial_capital = st.sidebar.number_input("Vốn ban đầu (VND):", min_value=1_000_000, value=100_000_000, step=10_000_000)
col_r1, col_r2 = st.sidebar.columns(2)
with col_r1:
    fee_pct = st.number_input("Phí giao dịch (%):", min_value=0.0, max_value=2.0, value=0.2, step=0.05) / 100.0
    slippage_pct = st.number_input("Trượt giá (%):", min_value=0.0, max_value=2.0, value=0.1, step=0.05) / 100.0
with col_r2:
    stop_loss_pct = st.number_input("Cắt lỗ Stop-loss (%):", min_value=0.0, max_value=30.0, value=7.0, step=0.5) / 100.0

# Tính toán tín hiệu & Backtest
df_processed, entries, exits = calculate_signals(df, strategy_type, ema_period, obv_slope_period)
bt_results = run_backtest_pandas(df_processed, entries, exits, initial_capital, fee_pct, slippage_pct, stop_loss_pct)


# ==============================================================================
# MAIN LAYOUT & TABS
# ==============================================================================
st.markdown('<div class="main-header">📈 Kiểm Định Chiến Lược Giao Dịch EMA & OBV</div>', unsafe_allow_allowed_html=True)
st.markdown(f'<div class="sub-header">Dữ liệu kiểm định từ <b>{df.index.min().strftime("%d/%m/%Y")}</b> đến <b>{df.index.max().strftime("%d/%m/%Y")}</b> ({len(df)} phiên giao dịch)</div>', unsafe_allow_allowed_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📌 Khuyến Nghị & Tín Hiệu", 
    "📊 Kết Quả Backtest", 
    "⚙️ Tối Ưu Hóa Tham Số", 
    "📖 Phương Pháp & Tài Liệu"
])


# ==============================================================================
# TAB 1: KHUYẾN NGHỊ & TÍN HIỆU HIỆN TẠI
# ==============================================================================
with tab1:
    st.subheader("1. Khuyến Nghị Trạng Thái Giao Dịch Mới Nhất")
    
    # Lấy trạng thái phiên cuối cùng
    last_date = df_processed.index[-1].strftime("%d/%m/%Y")
    last_close = df_processed['Close'].iloc[-1]
    curr_position = bt_results['position'].iloc[-1] # 1: Holding stock, 0: Cash
    last_entry_sig = entries.iloc[-1]
    last_exit_sig = exits.iloc[-1]

    # Logic khuyến nghị dựa trên Vị thế + Tín hiệu
    col_status, col_desc = st.columns([1, 2])
    
    with col_status:
        if curr_position == 1:
            if last_exit_sig:
                st.markdown('<div class="status-box-sell">🔴 KHUYẾN NGHỊ: BÁN (SELL)<br><small>Đang giữ cổ phiếu & Giá đóng cửa cắt xuống EMA</small></div>', unsafe_allow_allowed_html=True)
            else:
                st.markdown('<div class="status-box-hold">🟢 KHUYẾN NGHỊ: NẮM GIỮ (HOLD)<br><small>Đang giữ cổ phiếu & Xu hướng tăng tiếp diễn</small></div>', unsafe_allow_allowed_html=True)
        else: # curr_position == 0
            if last_entry_sig:
                st.markdown('<div class="status-box-buy">🔵 KHUYẾN NGHỊ: MUA (BUY)<br><small>Đang cầm tiền mặt & Xuất hiện tín hiệu Mua xác nhận</small></div>', unsafe_allow_allowed_html=True)
            else:
                st.markdown('<div class="status-box-aside">⚪ KHUYẾN NGHỊ: ĐỨNG NGOÀI (STAND ASIDE)<br><small>Đang cầm tiền mặt & Chưa đạt điều kiện gia nhập</small></div>', unsafe_allow_allowed_html=True)

    with col_desc:
        st.markdown(f"""
        **Chi tiết dữ liệu phiên gần nhất ({last_date}):**
        - **Giá đóng cửa:** `{last_close:,.0f} VND`
        - **Đường EMA ({ema_period}):** `{df_processed['EMA'].iloc[-1]:,.2f}`
        - **Độ dốc OBV ({obv_slope_period} phiên):** `{df_processed['OBV_Slope'].iloc[-1]:,.0f}`
        - **Trạng thái tài khoản hiện tại:** `{"Nắm giữ Cổ phiếu 📈" if curr_position == 1 else "Cầm Tiền mặt 💵"}`
        """)

    st.markdown("---")
    st.subheader("2. Biểu Đồ Giá, Chỉ Báo & Tín Hiệu Giao Dịch")

    # Vẽ biểu đồ nến / giá + EMA + OBV + Điểm Mua Bán bằng Plotly
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3],
                        subplot_titles=('Biểu đồ Giá & Tín hiệu Mua/Bán', f'Chỉ báo OBV & Độ dốc ({obv_slope_period} phiên)'))

    # Đường Giá & EMA
    fig.add_trace(go.Scatter(x=df_processed.index, y=df_processed['Close'], mode='lines', name='Giá Đóng Cửa', line=dict(color='#1E3A8A', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_processed.index, y=df_processed['EMA'], mode='lines', name=f'EMA ({ema_period})', line=dict(color='#F59E0B', width=1.5)), row=1, col=1)

    # Điểm Lệnh Mua
    buy_dates = df_processed.index[bt_results['buy_orders']]
    buy_prices = df_processed['Close'][bt_results['buy_orders']]
    fig.add_trace(go.Scatter(
        x=buy_dates, y=buy_prices, mode='markers', name='Lệnh MUA',
        marker=dict(symbol='triangle-up', size=11, color='#16A34A', line=dict(width=1, color='black'))
    ), row=1, col=1)

    # Điểm Lệnh Bán
    sell_dates = df_processed.index[bt_results['sell_orders']]
    sell_prices = df_processed['Close'][bt_results['sell_orders']]
    fig.add_trace(go.Scatter(
        x=sell_dates, y=sell_prices, mode='markers', name='Lệnh BÁN',
        marker=dict(symbol='triangle-down', size=11, color='#DC2626', line=dict(width=1, color='black'))
    ), row=1, col=1)

    # Biểu đồ OBV Slope
    colors_obv = ['#16A34A' if val > 0 else '#DC2626' for val in df_processed['OBV_Slope']]
    fig.add_trace(go.Bar(x=df_processed.index, y=df_processed['OBV_Slope'], name='Độ Dốc OBV', marker_color=colors_obv), row=2, col=1)

    fig.update_layout(height=650, margin=dict(l=20, r=20, t=40, b=20), hovermode='x unified', template='plotly_white')
    fig.update_yaxes(title_text="Giá (VND)", row=1, col=1)
    fig.update_yaxes(title_text="Độ Dốc OBV", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# TAB 2: KẾT QUẢ BACKTEST CHI TIẾT
# ==============================================================================
with tab2:
    st.subheader("1. Tổng Quan Hiệu Suất Đầu Tư")
    m = bt_results['metrics']

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Tổng Lợi Nhuận", f"{m['Total Return (%)']:.2f}%", delta=f"{m['Total Return (%)'] - m['Benchmark Return (%)']:.2f}% vs Buy & Hold")
    kpi2.metric("Buy & Hold Return", f"{m['Benchmark Return (%)']:.2f}%")
    kpi3.metric("Sharpe Ratio", f"{m['Sharpe Ratio']:.2f}" if np.isfinite(m['Sharpe Ratio']) else "N/A")
    kpi4.metric("Max Drawdown", f"{m['Max Drawdown (%)']:.2f}%")
    kpi5.metric("Tỷ Lệ Thắng (Win Rate)", f"{m['Win Rate (%)']:.1f}%", f"{m['Closed Trades']} giao dịch")

    st.markdown("---")
    st.subheader("2. Biểu Đồ Tăng Trưởng Tài Sản (Equity Curve) & Drawdown")

    fig_equity = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3],
                               subplot_titles=('Đường Vốn Chiến Lược vs Buy & Hold (VND)', 'Mức Sụt Giảm Tài Sản (Drawdown %)'))

    fig_equity.add_trace(go.Scatter(x=bt_results['equity'].index, y=bt_results['equity'], mode='lines', name='Chiến lược EMA+OBV', line=dict(color='#2563EB', width=2)), row=1, col=1)
    fig_equity.add_trace(go.Scatter(x=bt_results['benchmark'].index, y=bt_results['benchmark'], mode='lines', name='Buy & Hold (Benchmark)', line=dict(color='#9CA3AF', width=1.5, dash='dash')), row=1, col=1)

    fig_equity.add_trace(go.Scatter(x=bt_results['drawdown'].index, y=bt_results['drawdown'], mode='lines', name='Drawdown (%)', fill='tozeroy', line=dict(color='#EF4444', width=1)), row=2, col=1)

    fig_equity.update_layout(height=600, margin=dict(l=20, r=20, t=40, b=20), hovermode='x unified', template='plotly_white')
    fig_equity.update_yaxes(title_text="Giá Trị Danh Mục (VND)", row=1, col=1)
    fig_equity.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    st.plotly_chart(fig_equity, use_container_width=True)

    st.markdown("---")
    st.subheader("3. Nhật Ký Chi Tiết Các Giao Dịch (Trade Log)")

    trades_df = bt_results['trades']
    if len(trades_df) > 0:
        col_download, col_space = st.columns([1, 4])
        with col_download:
            csv_buffer = io.StringIO()
            trades_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Tải Lịch Sử Giao Dịch (CSV)",
                data=csv_buffer.getvalue(),
                file_name="trading_log_ema_obv.csv",
                mime="text/csv"
            )

        # Định dạng hiển thị bảng lệnh đẹp mắt
        trades_display = trades_df.copy()
        trades_display['Entry Date'] = pd.to_datetime(trades_display['Entry Date']).dt.strftime('%d/%m/%Y')
        trades_display['Exit Date'] = pd.to_datetime(trades_display['Exit Date']).dt.strftime('%d/%m/%Y')
        trades_display['Entry Price'] = trades_display['Entry Price'].map('{:,.0f} VND'.format)
        trades_display['Exit Price'] = trades_display['Exit Price'].map('{:,.0f} VND'.format)
        trades_display['PnL (VND)'] = trades_display['PnL (VND)'].map('{:,.0f} VND'.format)
        trades_display['Return (%)'] = trades_display['Return (%)'].map('{:+.2f}%'.format)

        st.dataframe(trades_display, use_container_width=True, hide_index=True)
    else:
        st.info("Không có giao dịch nào được thực hiện trong khoảng thời gian đã chọn.")


# ==============================================================================
# TAB 3: TỐI ƯU HÓA THAM SỐ (TRAIN VS TEST)
# ==============================================================================
with tab3:
    st.subheader("⚙️ Tối Ưu Hóa Tham Số Chiến Lược (Train vs Test)")
    st.markdown("""
    Tính năng này giúp tìm kiếm chu kỳ **EMA** và **OBV Slope** tối ưu nhất dựa trên chỉ số **Sharpe Ratio** trên tập Train, 
    sau đó kiểm định lại độc lập trên tập Test (Hold-out) để đánh giá hiện tượng Overfitting.
    
    *Lưu ý:* Để đảm bảo ý nghĩa thống kê, chỉ các bộ tham số tạo ra **tối thiểu 5 giao dịch (MIN_TRADES >= 5)** mới được xem xét.
    """)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        train_year_end = st.slider("Năm kết thúc tập TRAIN:", min_value=df_raw.index.min().year, max_value=df_raw.index.max().year - 1, value=2020)
    with col_t2:
        min_trades_threshold = st.number_input("Số lượng giao dịch tối thiểu (MIN_TRADES):", min_value=1, max_value=20, value=5)

    train_df_opt = df_raw.loc[:f'{train_year_end}'].copy()
    test_df_opt = df_raw.loc[f'{train_year_end + 1}':].copy()

    st.write(f"📌 **Tập TRAIN ({train_df_opt.index.min().year} - {train_df_opt.index.max().year}):** {len(train_df_opt)} phiên | 📌 **Tập TEST ({test_df_opt.index.min().year} - {test_df_opt.index.max().year}):** {len(test_df_opt)} phiên")

    if st.button("🚀 Bắt Đầu Tối Ưu Hóa (Grid Search / Hyperopt)", type="primary"):
        with st.spinner("Đang quét không gian tham số và chạy Backtest..."):
            best_sharpe = -999.0
            best_ema = 20
            best_obv = 3
            results_grid = []

            # Thử nghiệm lưới tham số EMA (10..50) & OBV Slope (2..20)
            for ema_p in range(10, 52, 2):
                for obv_p in range(2, 22, 1):
                    df_tr, ent_tr, ext_tr = calculate_signals(train_df_opt, strategy_type, ema_p, obv_p)
                    res_tr = run_backtest_pandas(df_tr, ent_tr, ext_tr, initial_capital, fee_pct, slippage_pct, stop_loss_pct)
                    
                    n_trades = res_tr['metrics']['Closed Trades']
                    sharpe = res_tr['metrics']['Sharpe Ratio']
                    
                    valid = np.isfinite(sharpe) and n_trades >= min_trades_threshold
                    
                    if valid:
                        results_grid.append({
                            'EMA Period': ema_p,
                            'OBV Slope Period': obv_p,
                            'Sharpe Ratio': sharpe,
                            'Total Return (%)': res_tr['metrics']['Total Return (%)'],
                            'Max Drawdown (%)': res_tr['metrics']['Max Drawdown (%)'],
                            'Trades': n_trades
                        })
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_ema = ema_p
                            best_obv = obv_p

            if len(results_grid) == 0:
                st.error("Không tìm thấy bộ tham số nào thỏa mãn điều kiện số giao dịch tối thiểu.")
            else:
                st.success(f"🎉 **Đã tìm thấy Tham Số Tối Ưu:** EMA = `{best_ema}`, OBV Slope = `{best_obv}` (Sharpe Tập Train: `{best_sharpe:.2f}`)")

                # Backtest lại bộ Mặc định và Bộ Tối ưu trên cả Train và Test
                # 1. Train Default vs Optimized
                df_tr_def, ent_tr_def, ext_tr_def = calculate_signals(train_df_opt, strategy_type, 20, 3)
                res_tr_def = run_backtest_pandas(df_tr_def, ent_tr_def, ext_tr_def, initial_capital, fee_pct, slippage_pct, stop_loss_pct)

                df_tr_opt, ent_tr_opt, ext_tr_opt = calculate_signals(train_df_opt, strategy_type, best_ema, best_obv)
                res_tr_opt = run_backtest_pandas(df_tr_opt, ent_tr_opt, ext_tr_opt, initial_capital, fee_pct, slippage_pct, stop_loss_pct)

                # 2. Test Default vs Optimized
                df_te_def, ent_te_def, ext_te_def = calculate_signals(test_df_opt, strategy_type, 20, 3)
                res_te_def = run_backtest_pandas(df_te_def, ent_te_def, ext_te_def, initial_capital, fee_pct, slippage_pct, stop_loss_pct)

                df_te_opt, ent_te_opt, ext_te_opt = calculate_signals(test_df_opt, strategy_type, best_ema, best_obv)
                res_te_opt = run_backtest_pandas(df_te_opt, ent_te_opt, ext_te_opt, initial_capital, fee_pct, slippage_pct, stop_loss_pct)

                # Bảng so sánh kết quả
                cmp_data = [
                    {"Tập Dữ Liệu": "TRAIN (Chạy Tối ưu)", "Bộ Tham Số": "Mặc định (EMA=20, OBV=3)", "Sharpe Ratio": res_tr_def['metrics']['Sharpe Ratio'], "Tổng Lợi Nhuận (%)": res_tr_def['metrics']['Total Return (%)'], "Max Drawdown (%)": res_tr_def['metrics']['Max Drawdown (%)'], "Số Giao Dịch": res_tr_def['metrics']['Closed Trades']},
                    {"Tập Dữ Liệu": "TRAIN (Chạy Tối ưu)", "Bộ Tham Số": f"Tối ưu (EMA={best_ema}, OBV={best_obv})", "Sharpe Ratio": res_tr_opt['metrics']['Sharpe Ratio'], "Tổng Lợi Nhuận (%)": res_tr_opt['metrics']['Total Return (%)'], "Max Drawdown (%)": res_tr_opt['metrics']['Max Drawdown (%)'], "Số Giao Dịch": res_tr_opt['metrics']['Closed Trades']},
                    {"Tập Dữ Liệu": "TEST (Kiểm định độc lập)", "Bộ Tham Số": "Mặc định (EMA=20, OBV=3)", "Sharpe Ratio": res_te_def['metrics']['Sharpe Ratio'], "Tổng Lợi Nhuận (%)": res_te_def['metrics']['Total Return (%)'], "Max Drawdown (%)": res_te_def['metrics']['Max Drawdown (%)'], "Số Giao Dịch": res_te_def['metrics']['Closed Trades']},
                    {"Tập Dữ Liệu": "TEST (Kiểm định độc lập)", "Bộ Tham Số": f"Tối ưu (EMA={best_ema}, OBV={best_obv})", "Sharpe Ratio": res_te_opt['metrics']['Sharpe Ratio'], "Tổng Lợi Nhuận (%)": res_te_opt['metrics']['Total Return (%)'], "Max Drawdown (%)": res_te_opt['metrics']['Max Drawdown (%)'], "Số Giao Dịch": res_te_opt['metrics']['Closed Trades']}
                ]

                cmp_df = pd.DataFrame(cmp_data)
                st.subheader("📊 Bảng So Sánh Hiệu Suất Tập Train vs Tập Test")
                st.dataframe(cmp_df.style.format({
                    'Sharpe Ratio': '{:.2f}',
                    'Tổng Lợi Nhuận (%)': '{:+.2f}%',
                    'Max Drawdown (%)': '{:.2f}%'
                }), use_container_width=True, hide_index=True)


# ==============================================================================
# TAB 4: PHƯƠNG PHÁP & TÀI LIỆU
# ==============================================================================
with tab4:
    st.subheader("📖 Phương Pháp Luận & Quy Trình Hạch Toán Vị Thế")

    st.markdown(r"""
    ### 1. Nguyên Lý Tín Hiệu EMA & OBV
    - **Exponential Moving Average (EMA):** Chỉ báo xu hướng làm mượt giá đóng cửa, gán trọng số cao hơn cho các phiên gần nhất.
      - **Điều kiện Mua:** Giá đóng cửa cắt lên trên đường EMA ($Close > EMA$).
      - **Điều kiện Bán:** Giá đóng cửa cắt xuống dưới đường EMA ($Close < EMA$).
    - **On-Balance Volume (OBV):** Chỉ báo khối lượng tích lũy đo lường áp lực Mua/Bán dòng.
      - **Độ dốc OBV ($OBV\_Slope$):** Sự thay đổi của OBV trong $N$ phiên gần nhất ($OBV_t - OBV_{t-N}$).
      - **Điều kiện Xác nhận Mua:** Độ dốc OBV dương ($OBV\_Slope > 0$) cho thấy khối lượng ủng hộ đà tăng giá.

    ---

    ### 2. Loại Bỏ Look-Ahead Bias (Độ trễ tín hiệu 1 phiên)
    Trong mô phỏng giao dịch thực tế:
    - Giá đóng cửa phiên $T$ chỉ được biết **khi kết thúc phiên**.
    - Do đó, tín hiệu tạo ra từ phiên $T$ chỉ được dùng để mở/đóng vị thế vào **phiên $T+1$**.
    - Trong thuật toán: Tín hiệu `entries` và `exits` đều được thực hiện `shift(1, fill_value=False)` trước khi đưa vào hạch toán lệnh.

    ---

    ### 3. Hạch Toán Vị Thế Giao Dịch (Position Accounting)
    Ứng dụng áp dụng mô hình trạng thái vị thế nghiêm ngặt (**Long-only / Cash**):
    - Khi tài khoản đang cầm tiền mặt ($Position = 0$): Chỉ chấp nhận lệnh **MUA** khi gặp tín hiệu Mua.
    - Khi tài khoản đang giữ cổ phiếu ($Position = 1$): Bỏ qua các tín hiệu Mua tiếp theo (không nhồi lệnh) và chỉ chấp nhận lệnh **BÁN** khi gặp tín hiệu Bán hoặc chạm ngưỡng **Stop Loss**.

    ---

    ### 4. Quản Lý Rủi Ro Stop Loss
    - Ngưỡng cắt lỗ cố định được thiết lập từ giá gia nhập vị thế ($Entry\_Price$).
    - Nếu giá trong phiên sụt giảm quá $\%$ cắt lỗ quy định (ví dụ 7%), vị thế sẽ tự động đóng vào phiên tiếp theo bất chấp tín hiệu EMA/OBV.
    """)

st.markdown("---")
st.caption("Developed with Streamlit & VectorBT / Pandas Trading Framework • Deploy Ready for Streamlit Cloud")
