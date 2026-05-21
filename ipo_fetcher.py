# ipo_fetcher.py (Corrected with Real URLs)

import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from typing import Dict


@st.cache_data(ttl=3600)
def fetch_all_ipo_data_separated() -> Dict[str, pd.DataFrame]:
    """
    Fetches Current, Past, and Upcoming IPOs and returns them as a
    dictionary of separate, cleaned, and crash-proof DataFrames.
    """
    print("Fetching new separated IPO data from NSE...")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json",
        "Referer": "https://www.nseindia.com/"
    }

    # --- THIS IS THE CORRECTED PART ---
    endpoints = {
        "Current": "https://www.nseindia.com/api/ipo-current-issue",
        "Upcoming": "https://www.nseindia.com/api/all-upcoming-issues?category=ipo",
        "Past": "https://www.nseindia.com/api/public-past-issues"
    }
    # ------------------------------------

    session = requests.Session()
    try:
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
    except requests.exceptions.RequestException as e:
        print(f"NSE Session Error: {e}")
        return {}

    ipo_data_dict = {}
    for status, url in endpoints.items():
        try:
            resp = session.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if data:
                ipo_data_dict[status] = pd.DataFrame(data)
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Failed to fetch {status} IPOs: {e}")
            ipo_data_dict[status] = pd.DataFrame()

    # --- Defensive Cleaning ---

    def safe_select(df, desired_cols):
        existing_cols = [col for col in desired_cols if col in df.columns]
        return df[existing_cols]

    # Clean CURRENT IPOs
    if 'Current' in ipo_data_dict and not ipo_data_dict['Current'].empty:
        df = ipo_data_dict['Current']
        df.rename(columns={
            "symbol": "Symbol", "companyName": "Company Name", "issueStartDate": "Start Date",
            "issueEndDate": "End Date", "status": "Bidding Status", "issuePrice": "Price Band",
            "noOfSharesOffered": "Shares Offered", "noOfsharesBid": "Shares Bid", "noOfTime": "Subscription (x)"
        }, inplace=True)
        numeric_cols = ['Shares Offered', 'Shares Bid', 'Subscription (x)']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').round(2)

        desired_cols = ["Company Name", "Symbol", "Bidding Status", "Start Date", "End Date",
                        "Price Band", "Shares Offered", "Shares Bid", "Subscription (x)"]
        ipo_data_dict['Current'] = safe_select(df, desired_cols)

    # Clean UPCOMING IPOs
    if 'Upcoming' in ipo_data_dict and not ipo_data_dict['Upcoming'].empty:
        df = ipo_data_dict['Upcoming']
        df.rename(columns={
            "smSymbol": "Symbol", "companyName": "Company Name", "issueStartDate": "Start Date",
            "issueEndDate": "End Date", "issueSize": "Issue Size (in Cr)", "issuePrice": "Price Band"
        }, inplace=True)

        desired_cols = ["Company Name", "Symbol", "Start Date", "End Date", "Price Band", "Issue Size (in Cr)"]
        df = safe_select(df, desired_cols)
        if "Start Date" in df.columns:
            df = df.sort_values(by="Start Date", ascending=True)
        ipo_data_dict['Upcoming'] = df

    # Clean PAST IPOs
    if 'Past' in ipo_data_dict and not ipo_data_dict['Past'].empty:
        df = ipo_data_dict['Past']
        if 'companyName' in df.columns and 'company' in df.columns:
            df['Company Name'] = df['companyName'].fillna(df['company'])
        elif 'company' in df.columns:
            df['Company Name'] = df['company']
        elif 'companyName' in df.columns:
            df['Company Name'] = df['companyName']

        df.rename(columns={
            "symbol": "Symbol", "ipoStartDate": "Start Date", "ipoEndDate": "End Date",
            "priceRange": "Price Band", "listingDate": "Listing Date", "issuePrice": "Final Price",
            "securityType": "Security Type"
        }, inplace=True)

        desired_cols = ["Company Name", "Symbol", "Security Type", "Start Date", "End Date", "Price Band",
                        "Final Price", "Listing Date"]
        df = safe_select(df, desired_cols)
        if "Start Date" in df.columns:
            df = df.sort_values(by="Start Date", ascending=False)
        ipo_data_dict['Past'] = df

    return ipo_data_dict