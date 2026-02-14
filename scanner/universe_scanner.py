# scanner/universe_scanner.py
"""
Universe Scanner with Minervini Trend Template

This module scans all US stocks and filters them through:
1. Basic filters (price, volume, market cap)
2. Minervini's Trend Template (Stage 2 criteria)

The final universe = Scanner Results ∪ Core Universe

LOGGING:
- Detailed criterion-level pass/fail for each symbol
- Summary statistics for troubleshooting
- Failure reason tracking
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
import os
import json

# Type hint for logger without circular import
if TYPE_CHECKING:
    from core.backtest_logger import BacktestLogger


class TrendTemplate:
    """
    Minervini's Trend Template - Stage 2 Qualification Criteria

    A stock must meet ALL of these criteria to be considered in a Stage 2 uptrend:
    1. Price > 50-day SMA
    2. Price > 150-day SMA
    3. Price > 200-day SMA
    4. 50-day SMA > 150-day SMA > 200-day SMA
    5. 200-day SMA rising for at least 1 month (20 trading days)
    6. Price within 25% of 52-week high
    7. Price at least 30% above 52-week low
    8. RS Rating > 70 (relative to market)
    """

    def __init__(self, logger: 'BacktestLogger' = None):
        self.min_rs_rating = 70
        self.max_pct_from_high = 25.0
        self.min_pct_above_low = 30.0
        self.ma_rising_days = 20
        self.logger = logger

    def check_template(self, df: pd.DataFrame, market_df: pd.DataFrame = None, symbol: str = None) -> Dict:
        """
        Check if stock passes Trend Template criteria

        Args:
            df: Stock OHLCV dataframe with at least 252 days of data
            market_df: Market (SPY) dataframe for RS calculation
            symbol: Symbol name for logging

        Returns:
            Dict with 'passes' bool and individual criteria results
        """
        if len(df) < 252:
            if self.logger and symbol:
                self.logger.log_data_insufficient(symbol, len(df), 252, "Below 252 days for template")
            return {'passes': False, 'reason': 'Insufficient data', 'available_days': len(df)}

        # Calculate required indicators
        df = self._calculate_indicators(df)

        current = df.iloc[-1]

        results = {
            'price': current['close'],
            'criteria': {},
            'criteria_values': {}  # Store actual values for logging
        }

        # Criterion 1: Price > 50-day SMA
        c1 = current['close'] > current['sma_50']
        results['criteria']['price_above_50sma'] = c1
        results['criteria_values']['price'] = round(current['close'], 2)
        results['criteria_values']['sma_50'] = round(current['sma_50'], 2)
        if self.logger and symbol:
            self.logger.log_scanner_criterion_check(
                symbol, 'price_above_50sma', c1,
                value=current['close'], threshold=current['sma_50']
            )

        # Criterion 2: Price > 150-day SMA
        c2 = current['close'] > current['sma_150']
        results['criteria']['price_above_150sma'] = c2
        results['criteria_values']['sma_150'] = round(current['sma_150'], 2)
        if self.logger and symbol:
            self.logger.log_scanner_criterion_check(
                symbol, 'price_above_150sma', c2,
                value=current['close'], threshold=current['sma_150']
            )

        # Criterion 3: Price > 200-day SMA
        c3 = current['close'] > current['sma_200']
        results['criteria']['price_above_200sma'] = c3
        results['criteria_values']['sma_200'] = round(current['sma_200'], 2)
        if self.logger and symbol:
            self.logger.log_scanner_criterion_check(
                symbol, 'price_above_200sma', c3,
                value=current['close'], threshold=current['sma_200']
            )

        # Criterion 4: 50 SMA > 150 SMA > 200 SMA
        c4 = (current['sma_50'] > current['sma_150']) and (current['sma_150'] > current['sma_200'])
        results['criteria']['sma_alignment'] = c4
        if self.logger and symbol:
            self.logger.log_scanner_criterion_check(
                symbol, 'sma_alignment', c4,
                value=current['sma_50'], threshold=current['sma_150']
            )

        # Criterion 5: 200 SMA rising for 20+ days
        if len(df) >= self.ma_rising_days + 200:
            sma_200_now = current['sma_200']
            sma_200_past = df['sma_200'].iloc[-self.ma_rising_days]
            c5 = sma_200_now > sma_200_past
            results['criteria_values']['sma_200_change'] = round(sma_200_now - sma_200_past, 2)
        else:
            c5 = False
            results['criteria_values']['sma_200_change'] = None
        results['criteria']['200sma_rising'] = c5
        if self.logger and symbol:
            self.logger.log_scanner_criterion_check(symbol, '200sma_rising', c5)

        # Criterion 6: Price within 25% of 52-week high
        high_52w = df['high'].iloc[-252:].max()
        pct_from_high = ((high_52w - current['close']) / high_52w) * 100
        c6 = pct_from_high <= self.max_pct_from_high
        results['criteria']['near_52w_high'] = c6
        results['pct_from_52w_high'] = round(pct_from_high, 2)
        results['criteria_values']['high_52w'] = round(high_52w, 2)
        if self.logger and symbol:
            self.logger.log_scanner_criterion_check(
                symbol, 'near_52w_high', c6,
                value=pct_from_high, threshold=self.max_pct_from_high
            )

        # Criterion 7: Price at least 30% above 52-week low
        low_52w = df['low'].iloc[-252:].min()
        pct_above_low = ((current['close'] - low_52w) / low_52w) * 100
        c7 = pct_above_low >= self.min_pct_above_low
        results['criteria']['above_52w_low'] = c7
        results['pct_above_52w_low'] = round(pct_above_low, 2)
        results['criteria_values']['low_52w'] = round(low_52w, 2)
        if self.logger and symbol:
            self.logger.log_scanner_criterion_check(
                symbol, 'above_52w_low', c7,
                value=pct_above_low, threshold=self.min_pct_above_low
            )

        # Criterion 8: RS Rating > 70
        if market_df is not None and len(market_df) >= 252:
            rs_rating = self._calculate_rs_rating(df, market_df)
            c8 = rs_rating >= self.min_rs_rating
            results['rs_rating'] = round(rs_rating, 1)
        else:
            c8 = True  # Skip RS check if no market data
            rs_rating = 0
            results['rs_rating'] = None
        results['criteria']['rs_rating_pass'] = c8
        if self.logger and symbol:
            self.logger.log_scanner_criterion_check(
                symbol, 'rs_rating_pass', c8,
                value=rs_rating if rs_rating else 0, threshold=self.min_rs_rating
            )

        # Overall pass
        results['passes'] = all([c1, c2, c3, c4, c5, c6, c7, c8])
        results['criteria_met'] = sum([c1, c2, c3, c4, c5, c6, c7, c8])

        # Determine first failure reason for logging
        if not results['passes']:
            failure_reasons = []
            if not c1: failure_reasons.append('price_below_50sma')
            if not c2: failure_reasons.append('price_below_150sma')
            if not c3: failure_reasons.append('price_below_200sma')
            if not c4: failure_reasons.append('sma_not_aligned')
            if not c5: failure_reasons.append('200sma_not_rising')
            if not c6: failure_reasons.append('too_far_from_52w_high')
            if not c7: failure_reasons.append('not_enough_above_52w_low')
            if not c8: failure_reasons.append('rs_rating_too_low')
            results['failure_reasons'] = failure_reasons
            results['first_failure'] = failure_reasons[0] if failure_reasons else 'Unknown'

        return results

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate required moving averages"""
        df = df.copy()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_150'] = df['close'].rolling(window=150).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        return df

    def _calculate_rs_rating(self, stock_df: pd.DataFrame, market_df: pd.DataFrame) -> float:
        """
        Calculate Relative Strength Rating (0-99)

        Compares stock's performance vs market over multiple timeframes
        """
        periods = [63, 126, 189, 252]  # ~3, 6, 9, 12 months
        weights = [0.4, 0.2, 0.2, 0.2]

        stock_returns = []
        market_returns = []

        for period in periods:
            if len(stock_df) > period and len(market_df) > period:
                stock_ret = (stock_df['close'].iloc[-1] / stock_df['close'].iloc[-period] - 1) * 100
                market_ret = (market_df['close'].iloc[-1] / market_df['close'].iloc[-period] - 1) * 100
                stock_returns.append(stock_ret)
                market_returns.append(market_ret)
            else:
                stock_returns.append(0)
                market_returns.append(0)

        # Calculate weighted relative performance
        weighted_stock = sum(r * w for r, w in zip(stock_returns, weights))
        weighted_market = sum(r * w for r, w in zip(market_returns, weights))

        # Convert to RS Rating (simplified - in production you'd rank against all stocks)
        relative_perf = weighted_stock - weighted_market

        # Map to 0-99 scale (rough approximation)
        rs_rating = 50 + (relative_perf * 2)
        rs_rating = max(1, min(99, rs_rating))

        return rs_rating


class UniverseScanner:
    """
    Scans all US stocks and filters to qualified universe

    Pipeline:
    1. Get all US stocks from exchange
    2. Apply basic filters (price, volume, market cap)
    3. Apply Trend Template
    4. Union with core universe

    LOGGING:
    - Accepts optional BacktestLogger for detailed diagnostic logging
    - Logs each symbol's template check results
    - Tracks failure reasons for troubleshooting
    """

    def __init__(
        self,
        core_universe_path: str = 'live_universe.txt',
        min_price: float = 3.0,
        max_price: float = 10000.0,
        min_avg_volume: int = 200000,
        min_market_cap: float = 500_000_000,  # $500M
        exclude_sectors: List[str] = None,
        logger: 'BacktestLogger' = None
    ):
        """
        Initialize scanner

        Args:
            core_universe_path: Path to core universe file
            min_price: Minimum stock price
            max_price: Maximum stock price
            min_avg_volume: Minimum 50-day average volume
            min_market_cap: Minimum market cap in dollars
            exclude_sectors: Sectors to exclude (e.g., ['Biotechnology'])
            logger: Optional BacktestLogger for diagnostic logging
        """
        self.core_universe_path = core_universe_path
        self.min_price = min_price
        self.max_price = max_price
        self.min_avg_volume = min_avg_volume
        self.min_market_cap = min_market_cap
        self.exclude_sectors = exclude_sectors or ['Biotechnology']
        self.logger = logger

        self.trend_template = TrendTemplate(logger=logger)
        self.core_universe = self._load_core_universe()

        # Cache for market data
        self._market_df = None

    def _load_core_universe(self) -> set:
        """Load core universe from file"""
        if os.path.exists(self.core_universe_path):
            with open(self.core_universe_path, 'r') as f:
                symbols = {line.strip() for line in f if line.strip()}
            return symbols
        return set()

    def get_all_us_stocks(self) -> List[str]:
        """
        Get list of all US stocks dynamically.

        Sources:
        1. S&P 500 from Wikipedia
        2. NASDAQ-100 from Wikipedia
        3. Additional growth/momentum stocks
        4. Core universe

        Results are cached to avoid repeated API calls.
        Falls back to cached/hardcoded list if fetch fails.
        """
        cache_file = 'data/scanner_universe_cache.json'
        cache_max_age_days = 7  # Refresh weekly

        # Try to load from cache first
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
                cache_date = datetime.strptime(cache['date'], '%Y-%m-%d')
                if (datetime.now() - cache_date).days < cache_max_age_days:
                    print(f"  Using cached scanner universe ({len(cache['symbols'])} stocks)")
                    return list(set(cache['symbols']) | self.core_universe)
            except:
                pass

        all_symbols = set()

        # Helper function to fetch with proper headers
        def fetch_wikipedia(url):
            import ssl
            import urllib.request
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
            )
            with urllib.request.urlopen(req, context=ssl_context, timeout=30) as response:
                return response.read()

        # 1. Fetch S&P 500 from Wikipedia
        try:
            print("  Fetching S&P 500 constituents...")
            sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            html = fetch_wikipedia(sp500_url)
            tables = pd.read_html(html)
            sp500 = tables[0]['Symbol'].str.replace('.', '-', regex=False).tolist()
            all_symbols.update(sp500)
            print(f"    Found {len(sp500)} S&P 500 stocks")
        except Exception as e:
            print(f"    Failed to fetch S&P 500: {e}")

        # 2. Fetch NASDAQ-100 from Wikipedia
        try:
            print("  Fetching NASDAQ-100 constituents...")
            nasdaq_url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
            html = fetch_wikipedia(nasdaq_url)
            tables = pd.read_html(html)
            # Find the table with ticker symbols
            for table in tables:
                if 'Ticker' in table.columns:
                    nasdaq100 = table['Ticker'].tolist()
                    all_symbols.update(nasdaq100)
                    print(f"    Found {len(nasdaq100)} NASDAQ-100 stocks")
                    break
        except Exception as e:
            print(f"    Failed to fetch NASDAQ-100: {e}")

        # 3. Add growth/momentum stocks (curated list of high-momentum names)
        growth = [
            'CRWD', 'DDOG', 'NET', 'ZS', 'SHOP', 'SQ', 'ROKU', 'SNOW', 'PLTR',
            'MELI', 'SE', 'PINS', 'SNAP', 'TTD', 'U', 'HOOD', 'COIN', 'RBLX',
            'ABNB', 'DASH', 'UBER', 'LYFT', 'RIVN', 'LCID', 'NIO', 'XPEV',
            'ARM', 'SMCI', 'IONQ', 'ANET', 'MSTR', 'APP', 'TOST', 'NU',
            'GRAB', 'SOFI', 'CPNG', 'MNDY', 'DOCN', 'PATH', 'MDB', 'VEEV',
            'RKLB', 'ASTS', 'LUNR', 'RDW', 'ALAB', 'APLD', 'ONDS', 'EOSE',
            'RCAT', 'ACHR', 'JOBY', 'OKLO', 'SMR', 'VST', 'CEG', 'GEV'
        ]
        all_symbols.update(growth)

        # 4. Add ETFs
        etfs = [
            'SPY', 'QQQ', 'IWM', 'DIA', 'XLK', 'XLF', 'XLE', 'XLV', 'XLI',
            'ARKK', 'ARKQ', 'ARKG', 'ARKW', 'IGV', 'SOXX', 'SMH', 'XBI',
            'SOXL', 'TQQQ', 'UPRO', 'TNA', 'LABU', 'FNGU'
        ]
        all_symbols.update(etfs)

        # 5. Add core universe
        all_symbols.update(self.core_universe)

        # Fallback if we got nothing from dynamic sources
        if len(all_symbols) < 200:
            print("  Warning: Dynamic fetch returned few stocks, using fallback list")
            fallback = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
                'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'ABBV',
                'MRK', 'PFE', 'KO', 'PEP', 'COST', 'TMO', 'WMT', 'DIS', 'CSCO',
                'ABT', 'VZ', 'ADBE', 'CRM', 'NKE', 'CMCSA', 'ACN', 'DHR', 'TXN'
            ]
            all_symbols.update(fallback)

        # Cache the results
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump({
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'symbols': list(all_symbols)
                }, f)
            print(f"  Cached {len(all_symbols)} stocks for future use")
        except Exception as e:
            print(f"  Warning: Could not cache results: {e}")

        print(f"  Total scanner universe: {len(all_symbols)} stocks")
        return list(all_symbols)

    def apply_basic_filters(
        self,
        symbols: List[str],
        progress_callback=None
    ) -> Tuple[List[str], Dict]:
        """
        Apply basic filters: price, volume, market cap

        Args:
            symbols: List of symbols to filter
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (filtered_symbols, filter_stats)
        """
        passed = []
        stats = {
            'total': len(symbols),
            'failed_price': 0,
            'failed_volume': 0,
            'failed_mcap': 0,
            'failed_sector': 0,
            'failed_data': 0,
            'passed': 0
        }

        for i, symbol in enumerate(symbols):
            if progress_callback:
                progress_callback(i, len(symbols), symbol)

            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info

                # Get current price
                price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
                if not (self.min_price <= price <= self.max_price):
                    stats['failed_price'] += 1
                    continue

                # Get average volume
                avg_volume = info.get('averageVolume', 0)
                if avg_volume < self.min_avg_volume:
                    stats['failed_volume'] += 1
                    continue

                # Get market cap
                market_cap = info.get('marketCap', 0)
                if market_cap < self.min_market_cap:
                    stats['failed_mcap'] += 1
                    continue

                # Check sector
                sector = info.get('industry', '')
                if any(exc.lower() in sector.lower() for exc in self.exclude_sectors):
                    stats['failed_sector'] += 1
                    continue

                passed.append(symbol)
                stats['passed'] += 1

            except Exception as e:
                stats['failed_data'] += 1
                continue

        return passed, stats

    def apply_trend_template(
        self,
        symbols: List[str],
        data_dir: str = 'data',
        progress_callback=None
    ) -> Tuple[List[str], Dict[str, Dict]]:
        """
        Apply Trend Template filter to symbols

        Args:
            symbols: Symbols that passed basic filters
            data_dir: Directory containing stock CSV files
            progress_callback: Optional callback for progress

        Returns:
            Tuple of (qualified_symbols, detailed_results)
        """
        qualified = []
        results = {}

        # Load market data for RS calculation
        market_df = self._load_market_data(data_dir)

        for i, symbol in enumerate(symbols):
            if progress_callback:
                progress_callback(i, len(symbols), symbol)

            try:
                # Load stock data
                df = self._load_stock_data(symbol, data_dir)
                if df is None or len(df) < 252:
                    results[symbol] = {
                        'passes': False,
                        'reason': 'Insufficient data',
                        'available_days': len(df) if df is not None else 0
                    }
                    if self.logger:
                        self.logger.log_data_insufficient(
                            symbol,
                            len(df) if df is not None else 0,
                            252,
                            "Below 252 days for trend template"
                        )
                    continue

                # Check trend template - pass symbol for logging
                template_result = self.trend_template.check_template(df, market_df, symbol=symbol)
                results[symbol] = template_result

                # Log the result to the logger
                if self.logger:
                    from core.backtest_logger import ScannerSymbolResult
                    scanner_result = ScannerSymbolResult(
                        symbol=symbol,
                        passed=template_result['passes'],
                        criteria_results=template_result.get('criteria', {}),
                        criteria_values=template_result.get('criteria_values', {}),
                        failure_reason=template_result.get('first_failure'),
                        rs_rating=template_result.get('rs_rating'),
                        price=template_result.get('price'),
                        pct_from_52w_high=template_result.get('pct_from_52w_high'),
                        pct_above_52w_low=template_result.get('pct_above_52w_low')
                    )
                    self.logger.log_scanner_symbol_result(scanner_result)

                if template_result['passes']:
                    qualified.append(symbol)

            except Exception as e:
                results[symbol] = {'passes': False, 'reason': str(e)}
                if self.logger:
                    from core.backtest_logger import ScannerSymbolResult
                    scanner_result = ScannerSymbolResult(
                        symbol=symbol,
                        passed=False,
                        failure_reason=f"Error: {str(e)}"
                    )
                    self.logger.log_scanner_symbol_result(scanner_result)
                continue

        return qualified, results

    def _load_stock_data(self, symbol: str, data_dir: str) -> Optional[pd.DataFrame]:
        """Load stock data from CSV or download"""
        filepath = os.path.join(data_dir, f"{symbol}.csv")

        if os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            return df

        # Try to download
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="2y")
            if len(df) > 0:
                df = df.rename(columns={
                    'Open': 'open', 'High': 'high',
                    'Low': 'low', 'Close': 'close', 'Volume': 'volume'
                })
                df = df[['open', 'high', 'low', 'close', 'volume']]
                return df
        except:
            pass

        return None

    def _load_market_data(self, data_dir: str) -> Optional[pd.DataFrame]:
        """Load SPY data for RS calculation"""
        if self._market_df is not None:
            return self._market_df

        filepath = os.path.join(data_dir, "SPY.csv")
        if os.path.exists(filepath):
            self._market_df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            return self._market_df

        # Try to download
        try:
            ticker = yf.Ticker("SPY")
            df = ticker.history(period="2y")
            if len(df) > 0:
                df = df.rename(columns={'Close': 'close'})
                self._market_df = df
                return self._market_df
        except:
            pass

        return None

    def scan(
        self,
        data_dir: str = 'data',
        use_cache: bool = True,
        cache_file: str = 'scanner_cache.json',
        verbose: bool = True
    ) -> Tuple[List[str], Dict]:
        """
        Run full scan pipeline

        Args:
            data_dir: Directory with stock data
            use_cache: Whether to use cached results
            cache_file: Path to cache file
            verbose: Print progress

        Returns:
            Tuple of (final_universe, scan_results)
        """
        scan_results = {
            'scan_date': datetime.now().isoformat(),
            'basic_filter_stats': {},
            'template_results': {},
            'scanner_qualified': [],
            'core_universe': list(self.core_universe),
            'final_universe': []
        }

        if verbose:
            print(f"\n{'='*60}")
            print("UNIVERSE SCANNER")
            print(f"{'='*60}")
            print(f"Core Universe: {len(self.core_universe)} stocks")
            print(f"Min Price: ${self.min_price}")
            print(f"Min Avg Volume: {self.min_avg_volume:,}")
            print(f"Min Market Cap: ${self.min_market_cap/1e9:.1f}B")

        # Step 1: Get all symbols
        all_symbols = self.get_all_us_stocks()
        if verbose:
            print(f"\nStep 1: Starting universe: {len(all_symbols)} symbols")

        # Step 2: Basic filters
        if verbose:
            print(f"Step 2: Applying basic filters...")

        def basic_progress(i, total, symbol):
            if verbose and i % 50 == 0:
                print(f"  Progress: {i}/{total} ({symbol})")

        filtered, basic_stats = self.apply_basic_filters(all_symbols, basic_progress)
        scan_results['basic_filter_stats'] = basic_stats

        if verbose:
            print(f"  Passed basic filters: {len(filtered)}")

        # Step 3: Trend Template
        if verbose:
            print(f"Step 3: Applying Trend Template...")

        def template_progress(i, total, symbol):
            if verbose and i % 20 == 0:
                print(f"  Progress: {i}/{total} ({symbol})")

        qualified, template_results = self.apply_trend_template(filtered, data_dir, template_progress)
        scan_results['template_results'] = {k: v for k, v in template_results.items() if v.get('passes', False)}
        scan_results['scanner_qualified'] = qualified

        if verbose:
            print(f"  Passed Trend Template: {len(qualified)}")

        # Step 4: Union with core universe
        final_universe = list(set(qualified) | self.core_universe)
        scan_results['final_universe'] = sorted(final_universe)

        if verbose:
            core_in_scanner = len(set(qualified) & self.core_universe)
            core_added = len(self.core_universe - set(qualified))
            print(f"\nStep 4: Final Universe")
            print(f"  Scanner qualified: {len(qualified)}")
            print(f"  Core in scanner: {core_in_scanner}")
            print(f"  Core added (didn't qualify): {core_added}")
            print(f"  Final universe: {len(final_universe)}")
            print(f"{'='*60}\n")

        # Cache results
        if use_cache:
            with open(cache_file, 'w') as f:
                json.dump(scan_results, f, indent=2, default=str)

        return final_universe, scan_results

    def quick_scan(self, data_dir: str = 'data') -> List[str]:
        """
        Quick scan using only data we already have downloaded

        Skips basic filters (assumes data directory has valid stocks)
        Only applies Trend Template to existing CSV files

        Returns:
            Sorted list of symbols in final universe
        """
        # Get symbols from data directory
        symbols = []
        if os.path.exists(data_dir):
            for f in os.listdir(data_dir):
                if f.endswith('.csv') and not f.startswith('^'):
                    symbols.append(f.replace('.csv', ''))

        # Add core universe
        symbols = list(set(symbols) | self.core_universe)

        # Log scanner start
        if self.logger:
            self.logger.log_scanner_start(
                total_symbols=len(symbols),
                core_universe_size=len(self.core_universe),
                scanner_config={
                    'min_price': self.min_price,
                    'max_price': self.max_price,
                    'min_avg_volume': self.min_avg_volume,
                    'min_market_cap': self.min_market_cap,
                    'exclude_sectors': self.exclude_sectors,
                    'trend_template': {
                        'min_rs_rating': self.trend_template.min_rs_rating,
                        'max_pct_from_high': self.trend_template.max_pct_from_high,
                        'min_pct_above_low': self.trend_template.min_pct_above_low,
                        'ma_rising_days': self.trend_template.ma_rising_days
                    }
                }
            )

        # Apply Trend Template
        qualified, _ = self.apply_trend_template(symbols, data_dir)

        # Union with core
        final = list(set(qualified) | self.core_universe)

        # Log scanner completion
        if self.logger:
            self.logger.log_scanner_complete(final)

        return sorted(final)


# Convenience function
def get_daily_universe(
    core_universe_path: str = 'live_universe.txt',
    data_dir: str = 'data',
    quick: bool = True
) -> List[str]:
    """
    Get today's trading universe

    Args:
        core_universe_path: Path to core universe file
        data_dir: Directory with stock data
        quick: If True, only scan existing data (faster)

    Returns:
        List of symbols in today's universe
    """
    scanner = UniverseScanner(core_universe_path=core_universe_path)

    if quick:
        return scanner.quick_scan(data_dir)
    else:
        universe, _ = scanner.scan(data_dir)
        return universe


if __name__ == "__main__":
    # Test scanner
    scanner = UniverseScanner()

    print("Running quick scan on existing data...")
    universe = scanner.quick_scan('data')

    print(f"\nFinal Universe: {len(universe)} stocks")
    print(f"First 20: {universe[:20]}")
