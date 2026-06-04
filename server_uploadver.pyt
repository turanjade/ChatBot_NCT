import os, ssl, urllib3

# ── Must be first, before importing anthropic or httpx ───────────────────────
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings()

import re, io, base64, sqlite3, tempfile, threading
import numpy as np
import pandas as pd
import openmatrix as omx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import httpx
import anthropic
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor
import webbrowser
from threading import Timer

import math as _math
from flask.json.provider import DefaultJSONProvider as _DefaultJSONProvider

def _sanitize(obj):
    """Recursively replace NaN/Inf (float and numpy scalar) with None."""
    if isinstance(obj, float):
        return None if (_math.isnan(obj) or _math.isinf(obj)) else obj
    # numpy scalar types (np.float64, np.float32, etc.)
    if hasattr(obj, "item"):
        return _sanitize(obj.item())
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj

class _SafeJSONProvider(_DefaultJSONProvider):
    def dumps(self, obj, **kwargs):
        return super().dumps(_sanitize(obj), **kwargs)

app = Flask(__name__)
app.json_provider_class = _SafeJSONProvider
app.json = _SafeJSONProvider(app)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024

DB = {"conn": None, "schema": None}
PERF = {"data": None}
MODEL_RUNS_ROOT = r"L:\NCTModel\ModelRuns"

DB_LOCK = threading.Lock()
WRITE_LOCK = threading.Lock()

# ── Anthropic client with SSL verification disabled ───────────────────────────
http_client = httpx.Client(verify=False, timeout=60.0)
client = anthropic.Anthropic(
    api_key="YOUR_API_KEY",
    http_client=http_client,
)
MODEL = "claude-sonnet-4-20250514"

# ── Mode choice field metadata (field name → human description, unit) ─────────
# Source: ModelRunReports_SummaryFields (ModeChoice_Summary_Fields)
# The first column in every modechoice table is the geography key (COUNTY/CITY/MPA12).
# All remaining columns follow the pattern <PURPOSE>_<DIRECTION>_<INTRA>_<MODE>_PTRIPS
# or <PURPOSE>_<DIRECTION>_<INTRA>_VOCC.
MODECHOICE_META = {
    # ── Home-Based Work (HBW) ─────────────────────────────────────────────────
    "HBW_PRD_NOINTRA_WLKTR_PTRIPS": ("HBW Person Trips Produced (No Intrazonals) - Transit Walk",         "trip"),
    "HBW_PRD_NOINTRA_DRVTR_PTRIPS": ("HBW Person Trips Produced (No Intrazonals) - Transit Drive",        "trip"),
    "HBW_PRD_NOINTRA_TR_PTRIPS":    ("HBW Person Trips Produced (No Intrazonals) - Transit Total",        "trip"),
    "HBW_PRD_NOINTRA_DAAUTO_PTRIPS":("HBW Person Trips Produced (No Intrazonals) - Auto Drive Alone",     "trip"),
    "HBW_PRD_NOINTRA_SR2AUTO_PTRIPS":("HBW Person Trips Produced (No Intrazonals) - Auto Shared Ride 2",  "trip"),
    "HBW_PRD_NOINTRA_SR3AUTO_PTRIPS":("HBW Person Trips Produced (No Intrazonals) - Auto Shared Ride 3+", "trip"),
    "HBW_PRD_NOINTRA_AUTO_PTRIPS":  ("HBW Person Trips Produced (No Intrazonals) - Auto Total",           "trip"),
    "HBW_PRD_NOINTRA_TOT_PTRIPS":   ("HBW Person Trips Produced (No Intrazonals) - All Modes Total",      "trip"),
    "HBW_ATT_NOINTRA_WLKTR_PTRIPS": ("HBW Person Trips Attracted (No Intrazonals) - Transit Walk",        "trip"),
    "HBW_ATT_NOINTRA_DRVTR_PTRIPS": ("HBW Person Trips Attracted (No Intrazonals) - Transit Drive",       "trip"),
    "HBW_ATT_NOINTRA_TR_PTRIPS":    ("HBW Person Trips Attracted (No Intrazonals) - Transit Total",       "trip"),
    "HBW_ATT_NOINTRA_DAAUTO_PTRIPS":("HBW Person Trips Attracted (No Intrazonals) - Auto Drive Alone",    "trip"),
    "HBW_ATT_NOINTRA_SR2AUTO_PTRIPS":("HBW Person Trips Attracted (No Intrazonals) - Auto Shared Ride 2", "trip"),
    "HBW_ATT_NOINTRA_SR3AUTO_PTRIPS":("HBW Person Trips Attracted (No Intrazonals) - Auto Shared Ride 3+","trip"),
    "HBW_ATT_NOINTRA_AUTO_PTRIPS":  ("HBW Person Trips Attracted (No Intrazonals) - Auto Total",          "trip"),
    "HBW_ATT_NOINTRA_TOT_PTRIPS":   ("HBW Person Trips Attracted (No Intrazonals) - All Modes Total",     "trip"),
    "HBW_INTRA_WLKTR_PTRIPS":       ("HBW Person Trips Intrazonal - Transit Walk",                        "trip"),
    "HBW_INTRA_DRVTR_PTRIPS":       ("HBW Person Trips Intrazonal - Transit Drive",                       "trip"),
    "HBW_INTRA_TR_PTRIPS":          ("HBW Person Trips Intrazonal - Transit Total",                       "trip"),
    "HBW_INTRA_DAAUTO_PTRIPS":      ("HBW Person Trips Intrazonal - Auto Drive Alone",                    "trip"),
    "HBW_INTRA_SR2AUTO_PTRIPS":     ("HBW Person Trips Intrazonal - Auto Shared Ride 2",                  "trip"),
    "HBW_INTRA_SR3AUTO_PTRIPS":     ("HBW Person Trips Intrazonal - Auto Shared Ride 3+",                 "trip"),
    "HBW_INTRA_AUTO_PTRIPS":        ("HBW Person Trips Intrazonal - Auto Total",                          "trip"),
    "HBW_INTRA_TOT_PTRIPS":         ("HBW Person Trips Intrazonal - All Modes Total",                     "trip"),
    # ── Home-Based Non-Work (HNW) ─────────────────────────────────────────────
    "HNW_PRD_NOINTRA_WLKTR_PTRIPS": ("HNW Person Trips Produced (No Intrazonals) - Transit Walk",         "trip"),
    "HNW_PRD_NOINTRA_DRVTR_PTRIPS": ("HNW Person Trips Produced (No Intrazonals) - Transit Drive",        "trip"),
    "HNW_PRD_NOINTRA_TR_PTRIPS":    ("HNW Person Trips Produced (No Intrazonals) - Transit Total",        "trip"),
    "HNW_PRD_NOINTRA_DAAUTO_PTRIPS":("HNW Person Trips Produced (No Intrazonals) - Auto Drive Alone",     "trip"),
    "HNW_PRD_NOINTRA_SR2AUTO_PTRIPS":("HNW Person Trips Produced (No Intrazonals) - Auto Shared Ride 2",  "trip"),
    "HNW_PRD_NOINTRA_SR3AUTO_PTRIPS":("HNW Person Trips Produced (No Intrazonals) - Auto Shared Ride 3+", "trip"),
    "HNW_PRD_NOINTRA_AUTO_PTRIPS":  ("HNW Person Trips Produced (No Intrazonals) - Auto Total",           "trip"),
    "HNW_PRD_NOINTRA_TOT_PTRIPS":   ("HNW Person Trips Produced (No Intrazonals) - All Modes Total",      "trip"),
    "HNW_ATT_NOINTRA_WLKTR_PTRIPS": ("HNW Person Trips Attracted (No Intrazonals) - Transit Walk",        "trip"),
    "HNW_ATT_NOINTRA_DRVTR_PTRIPS": ("HNW Person Trips Attracted (No Intrazonals) - Transit Drive",       "trip"),
    "HNW_ATT_NOINTRA_TR_PTRIPS":    ("HNW Person Trips Attracted (No Intrazonals) - Transit Total",       "trip"),
    "HNW_ATT_NOINTRA_DAAUTO_PTRIPS":("HNW Person Trips Attracted (No Intrazonals) - Auto Drive Alone",    "trip"),
    "HNW_ATT_NOINTRA_SR2AUTO_PTRIPS":("HNW Person Trips Attracted (No Intrazonals) - Auto Shared Ride 2", "trip"),
    "HNW_ATT_NOINTRA_SR3AUTO_PTRIPS":("HNW Person Trips Attracted (No Intrazonals) - Auto Shared Ride 3+","trip"),
    "HNW_ATT_NOINTRA_AUTO_PTRIPS":  ("HNW Person Trips Attracted (No Intrazonals) - Auto Total",          "trip"),
    "HNW_ATT_NOINTRA_TOT_PTRIPS":   ("HNW Person Trips Attracted (No Intrazonals) - All Modes Total",     "trip"),
    "HNW_INTRA_WLKTR_PTRIPS":       ("HNW Person Trips Intrazonal - Transit Walk",                        "trip"),
    "HNW_INTRA_DRVTR_PTRIPS":       ("HNW Person Trips Intrazonal - Transit Drive",                       "trip"),
    "HNW_INTRA_TR_PTRIPS":          ("HNW Person Trips Intrazonal - Transit Total",                       "trip"),
    "HNW_INTRA_DAAUTO_PTRIPS":      ("HNW Person Trips Intrazonal - Auto Drive Alone",                    "trip"),
    "HNW_INTRA_SR2AUTO_PTRIPS":     ("HNW Person Trips Intrazonal - Auto Shared Ride 2",                  "trip"),
    "HNW_INTRA_SR3AUTO_PTRIPS":     ("HNW Person Trips Intrazonal - Auto Shared Ride 3+",                 "trip"),
    "HNW_INTRA_AUTO_PTRIPS":        ("HNW Person Trips Intrazonal - Auto Total",                          "trip"),
    "HNW_INTRA_TOT_PTRIPS":         ("HNW Person Trips Intrazonal - All Modes Total",                     "trip"),
    # ── Non-Home-Based (NHB) ──────────────────────────────────────────────────
    "NHB_PRD_NOINTRA_WLKTR_PTRIPS": ("NHB Person Trips Produced (No Intrazonals) - Transit Walk",         "trip"),
    "NHB_PRD_NOINTRA_DRVTR_PTRIPS": ("NHB Person Trips Produced (No Intrazonals) - Transit Drive",        "trip"),
    "NHB_PRD_NOINTRA_TR_PTRIPS":    ("NHB Person Trips Produced (No Intrazonals) - Transit Total",        "trip"),
    "NHB_PRD_NOINTRA_DAAUTO_PTRIPS":("NHB Person Trips Produced (No Intrazonals) - Auto Drive Alone",     "trip"),
    "NHB_PRD_NOINTRA_SR2AUTO_PTRIPS":("NHB Person Trips Produced (No Intrazonals) - Auto Shared Ride 2",  "trip"),
    "NHB_PRD_NOINTRA_SR3AUTO_PTRIPS":("NHB Person Trips Produced (No Intrazonals) - Auto Shared Ride 3+", "trip"),
    "NHB_PRD_NOINTRA_AUTO_PTRIPS":  ("NHB Person Trips Produced (No Intrazonals) - Auto Total",           "trip"),
    "NHB_PRD_NOINTRA_TOT_PTRIPS":   ("NHB Person Trips Produced (No Intrazonals) - All Modes Total",      "trip"),
    "NHB_ATT_NOINTRA_WLKTR_PTRIPS": ("NHB Person Trips Attracted (No Intrazonals) - Transit Walk",        "trip"),
    "NHB_ATT_NOINTRA_DRVTR_PTRIPS": ("NHB Person Trips Attracted (No Intrazonals) - Transit Drive",       "trip"),
    "NHB_ATT_NOINTRA_TR_PTRIPS":    ("NHB Person Trips Attracted (No Intrazonals) - Transit Total",       "trip"),
    "NHB_ATT_NOINTRA_DAAUTO_PTRIPS":("NHB Person Trips Attracted (No Intrazonals) - Auto Drive Alone",    "trip"),
    "NHB_ATT_NOINTRA_SR2AUTO_PTRIPS":("NHB Person Trips Attracted (No Intrazonals) - Auto Shared Ride 2", "trip"),
    "NHB_ATT_NOINTRA_SR3AUTO_PTRIPS":("NHB Person Trips Attracted (No Intrazonals) - Auto Shared Ride 3+","trip"),
    "NHB_ATT_NOINTRA_AUTO_PTRIPS":  ("NHB Person Trips Attracted (No Intrazonals) - Auto Total",          "trip"),
    "NHB_ATT_NOINTRA_TOT_PTRIPS":   ("NHB Person Trips Attracted (No Intrazonals) - All Modes Total",     "trip"),
    "NHB_INTRA_WLKTR_PTRIPS":       ("NHB Person Trips Intrazonal - Transit Walk",                        "trip"),
    "NHB_INTRA_DRVTR_PTRIPS":       ("NHB Person Trips Intrazonal - Transit Drive",                       "trip"),
    "NHB_INTRA_TR_PTRIPS":          ("NHB Person Trips Intrazonal - Transit Total",                       "trip"),
    "NHB_INTRA_DAAUTO_PTRIPS":      ("NHB Person Trips Intrazonal - Auto Drive Alone",                    "trip"),
    "NHB_INTRA_SR2AUTO_PTRIPS":     ("NHB Person Trips Intrazonal - Auto Shared Ride 2",                  "trip"),
    "NHB_INTRA_SR3AUTO_PTRIPS":     ("NHB Person Trips Intrazonal - Auto Shared Ride 3+",                 "trip"),
    "NHB_INTRA_AUTO_PTRIPS":        ("NHB Person Trips Intrazonal - Auto Total",                          "trip"),
    "NHB_INTRA_TOT_PTRIPS":         ("NHB Person Trips Intrazonal - All Modes Total",                     "trip"),
    # ── All Purposes (TOTAL) ─────────────────────────────────────────────────
    "TOTAL_PRD_NOINTRA_WLKTR_PTRIPS": ("All Purposes Person Trips Produced (No Intrazonals) - Transit Walk",         "trip"),
    "TOTAL_PRD_NOINTRA_DRVTR_PTRIPS": ("All Purposes Person Trips Produced (No Intrazonals) - Transit Drive",        "trip"),
    "TOTAL_PRD_NOINTRA_TR_PTRIPS":    ("All Purposes Person Trips Produced (No Intrazonals) - Transit Total",        "trip"),
    "TOTAL_PRD_NOINTRA_DAAUTO_PTRIPS":("All Purposes Person Trips Produced (No Intrazonals) - Auto Drive Alone",     "trip"),
    "TOTAL_PRD_NOINTRA_SR2AUTO_PTRIPS":("All Purposes Person Trips Produced (No Intrazonals) - Auto Shared Ride 2",  "trip"),
    "TOTAL_PRD_NOINTRA_SR3AUTO_PTRIPS":("All Purposes Person Trips Produced (No Intrazonals) - Auto Shared Ride 3+", "trip"),
    "TOTAL_PRD_NOINTRA_AUTO_PTRIPS":  ("All Purposes Person Trips Produced (No Intrazonals) - Auto Total",           "trip"),
    "TOTAL_PRD_NOINTRA_TOT_PTRIPS":   ("All Purposes Person Trips Produced (No Intrazonals) - All Modes Total",      "trip"),
    "TOTAL_ATT_NOINTRA_WLKTR_PTRIPS": ("All Purposes Person Trips Attracted (No Intrazonals) - Transit Walk",        "trip"),
    "TOTAL_ATT_NOINTRA_DRVTR_PTRIPS": ("All Purposes Person Trips Attracted (No Intrazonals) - Transit Drive",       "trip"),
    "TOTAL_ATT_NOINTRA_TR_PTRIPS":    ("All Purposes Person Trips Attracted (No Intrazonals) - Transit Total",       "trip"),
    "TOTAL_ATT_NOINTRA_DAAUTO_PTRIPS":("All Purposes Person Trips Attracted (No Intrazonals) - Auto Drive Alone",    "trip"),
    "TOTAL_ATT_NOINTRA_SR2AUTO_PTRIPS":("All Purposes Person Trips Attracted (No Intrazonals) - Auto Shared Ride 2", "trip"),
    "TOTAL_ATT_NOINTRA_SR3AUTO_PTRIPS":("All Purposes Person Trips Attracted (No Intrazonals) - Auto Shared Ride 3+","trip"),
    "TOTAL_ATT_NOINTRA_AUTO_PTRIPS":  ("All Purposes Person Trips Attracted (No Intrazonals) - Auto Total",          "trip"),
    "TOTAL_ATT_NOINTRA_TOT_PTRIPS":   ("All Purposes Person Trips Attracted (No Intrazonals) - All Modes Total",     "trip"),
    "TOTAL_INTRA_WLKTR_PTRIPS":       ("All Purposes Person Trips Intrazonal - Transit Walk",                        "trip"),
    "TOTAL_INTRA_DRVTR_PTRIPS":       ("All Purposes Person Trips Intrazonal - Transit Drive",                       "trip"),
    "TOTAL_INTRA_TR_PTRIPS":          ("All Purposes Person Trips Intrazonal - Transit Total",                       "trip"),
    "TOTAL_INTRA_DAAUTO_PTRIPS":      ("All Purposes Person Trips Intrazonal - Auto Drive Alone",                    "trip"),
    "TOTAL_INTRA_SR2AUTO_PTRIPS":     ("All Purposes Person Trips Intrazonal - Auto Shared Ride 2",                  "trip"),
    "TOTAL_INTRA_SR3AUTO_PTRIPS":     ("All Purposes Person Trips Intrazonal - Auto Shared Ride 3+",                 "trip"),
    "TOTAL_INTRA_AUTO_PTRIPS":        ("All Purposes Person Trips Intrazonal - Auto Total",                          "trip"),
    "TOTAL_INTRA_TOT_PTRIPS":         ("All Purposes Person Trips Intrazonal - All Modes Total",                     "trip"),
    # ── Vehicle Occupancy (VOCC) ─────────────────────────────────────────────
    "HBW_PRD_NOINTRA_VOCC":   ("HBW Auto Vehicle Occupancy - Produced (No Intrazonals)",      "persons/vehicle"),
    "HBW_ATT_NOINTRA_VOCC":   ("HBW Auto Vehicle Occupancy - Attracted (No Intrazonals)",     "persons/vehicle"),
    "HBW_INTRA_VOCC":         ("HBW Auto Vehicle Occupancy - Intrazonal",                     "persons/vehicle"),
    "HNW_PRD_NOINTRA_VOCC":   ("HNW Auto Vehicle Occupancy - Produced (No Intrazonals)",      "persons/vehicle"),
    "HNW_ATT_NOINTRA_VOCC":   ("HNW Auto Vehicle Occupancy - Attracted (No Intrazonals)",     "persons/vehicle"),
    "HNW_INTRA_VOCC":         ("HNW Auto Vehicle Occupancy - Intrazonal",                     "persons/vehicle"),
    "NHB_PRD_NOINTRA_VOCC":   ("NHB Auto Vehicle Occupancy - Produced (No Intrazonals)",      "persons/vehicle"),
    "NHB_ATT_NOINTRA_VOCC":   ("NHB Auto Vehicle Occupancy - Attracted (No Intrazonals)",     "persons/vehicle"),
    "NHB_INTRA_VOCC":         ("NHB Auto Vehicle Occupancy - Intrazonal",                     "persons/vehicle"),
    "TOTAL_PRD_NOINTRA_VOCC": ("All Purposes Auto Vehicle Occupancy - Produced (No Intrazonals)", "persons/vehicle"),
    "TOTAL_ATT_NOINTRA_VOCC": ("All Purposes Auto Vehicle Occupancy - Attracted (No Intrazonals)","persons/vehicle"),
    "TOTAL_INTRA_VOCC":       ("All Purposes Auto Vehicle Occupancy - Intrazonal",               "persons/vehicle"),
}

def _build_modechoice_system_section():
    """Build the MODE CHOICE COLUMNS section of the system prompt from MODECHOICE_META."""
    lines = [
        "MODE CHOICE COLUMNS (county_modechoice, city_modechoice, mpa_modechoice)",
        "  The geography key column (first column) is COUNTY, CITY, or MPA12 depending on the table.",
        "  All other columns are numeric. Column naming pattern:",
        "    <PURPOSE>_<DIRECTION>_<INTRA>_<MODE>_PTRIPS  — person trips",
        "    <PURPOSE>_<DIRECTION>_<INTRA>_VOCC            — vehicle occupancy (persons/vehicle)",
        "",
        "  Segment codes:",
        "    PURPOSE   : HBW=Home-Based Work, HNW=Home-Based Non-Work, NHB=Non-Home-Based, TOTAL=All Purposes",
        "    DIRECTION : PRD=Produced, ATT=Attracted",
        "    INTRA     : NOINTRA=No Intrazonals (external), INTRA=Intrazonal only",
        "    MODE      : WLKTR=Transit Walk, DRVTR=Transit Drive, TR=Transit Total,",
        "                DAAUTO=Auto Drive Alone, SR2AUTO=Auto Shared Ride 2,",
        "                SR3AUTO=Auto Shared Ride 3+, AUTO=Auto Total, TOT=All Modes Total",
        "",
        "  Complete field reference (field_name → description [unit]):",
    ]
    for field, (desc, unit) in MODECHOICE_META.items():
        lines.append(f"    {field:<42} → {desc} [{unit}]")
    lines += [
        "",
        "  COMMON QUERY PATTERNS:",
        "    Transit mode share (HBW produced, no intrazonals):",
        "      HBW_PRD_NOINTRA_TR_PTRIPS * 1.0 / NULLIF(HBW_PRD_NOINTRA_TOT_PTRIPS, 0)",
        "    Auto drive-alone share (all purposes, produced+attracted combined):",
        "      (TOTAL_PRD_NOINTRA_DAAUTO_PTRIPS + TOTAL_ATT_NOINTRA_DAAUTO_PTRIPS)",
        "      / NULLIF(TOTAL_PRD_NOINTRA_TOT_PTRIPS + TOTAL_ATT_NOINTRA_TOT_PTRIPS, 0)",
        "    Total person trips including intrazonals (all purposes):",
        "      TOTAL_PRD_NOINTRA_TOT_PTRIPS + TOTAL_INTRA_TOT_PTRIPS",
        "    Note: PRD and ATT count the same trip from each end — do NOT sum both",
        "          unless explicitly comparing productions vs attractions.",
        "          For a simple trip total, use PRD_NOINTRA + INTRA.",
    ]
    return "\n".join(lines)


SYSTEM = """You are a travel demand modeling assistant for the North Central Texas region.
You help analysts query model run data stored in SQLite tables covering three geographic levels:
COUNTY, CITY, and MPA12 (Metropolitan Planning Area — Inside/Outside/Total).

TABLE OVERVIEW
==============
Each level has up to 3 table types. All table names follow the pattern <level>_<type>:

COUNTY-level tables  (rows = individual counties; TOTAL row already excluded at load time)
  county_demographics  — socioeconomic data per county
  county_modechoice    — person trips by mode/purpose per county
  county_perfmeasures  — roadway performance measures per county

CITY-level tables  (rows = individual cities; TOTAL row already excluded at load time)
  city_demographics    — socioeconomic data per city
  city_modechoice      — person trips by mode/purpose per city
  city_perfmeasures    — roadway performance measures per city

MPA12-level tables  (rows = Inside / Outside / Total — all three rows are valid)
  mpa_demographics     — socioeconomic data for MPA12 zones
  mpa_modechoice       — person trips by mode/purpose for MPA12 zones
  mpa_perfmeasures     — roadway performance measures for MPA12 zones

GEOGRAPHY KEY COLUMNS
=====================
  county_*  tables: first column is COUNTY
  city_*    tables: first column is CITY
  mpa_*     tables: first column is MPA12

DEMOGRAPHICS COLUMNS (county_demographics, city_demographics, mpa_demographics)
  HHOLD       — households
  POP         — population
  BASIC       — basic employment
  RETAIL      — retail employment
  SERVICE     — service employment
  EMP         — total employment
  LAND_AREA   — land area (sq mi)
  TOT_AREA    — total area (sq mi)
  HHSIZE      — average household size
  PE_RATIO    — population-to-employment ratio
  POP_DENSITY — population density (persons/sq mi)
  EMP_DENSITY — employment density (jobs/sq mi)
  ACT_DENSITY — activity density (persons+jobs/sq mi)

""" + _build_modechoice_system_section() + """

PERFORMANCE MEASURES TABLE COLUMNS (county_perfmeasures, city_perfmeasures, mpa_perfmeasures)
  <geo_col>   — geography key (COUNTY / CITY / MPA12)
  section     — metric group (e.g. 'Lane Miles', 'VMT', 'VHT', 'Average Loaded Speed (MPH)',
                 'Lane Miles at Level of Service F', etc.)
  funcl       — functional class: 1=Freeways, 2=Principal Arterials, 3=Minor Arterials,
                 4=Collectors, 6=Freeway Ramps, 7=Frontage Roads
  description — road description
  am, pm, op, daily — time-of-day values

OMX TRIP/SKIM TABLES (loaded separately via file upload — may not be present)
  trips  — columns: origin, destination, purpose (HBW/HNW/NHB), trips
  skim   — columns: origin, destination, walk_time (minutes)

QUERY RULES
===========
- Always use SUM() for trip totals, never COUNT(*)
- TOTAL rows are already excluded from county_* and city_* tables — no need to filter them
- For MPA12 queries, Inside/Outside/Total rows are all present — include/exclude per user intent
- For performance measure queries: filter by section to select the metric,
  filter by the geo column for geography
- For mode share calculations always guard against divide-by-zero with NULLIF(..., 0)
- Do NOT sum PRD and ATT for the same trip purpose — they represent the same trips from
  each end of the trip; use PRD only (or ATT only) for directional totals
- Intrazonal trips: origin = destination (trips table); or INTRA columns (modechoice tables)
- The modechoice_metadata table (field, description, unit) can be queried to look up
  any column definition
- For mode share queries, prefer a WIDE single-row SELECT (one column per mode) rather than
  UNION ALL with one row per mode. Example for Dallas county mode share:
    SELECT COUNTY,
           ROUND(TOTAL_PRD_NOINTRA_WLKTR_PTRIPS * 100.0 / NULLIF(TOTAL_PRD_NOINTRA_TOT_PTRIPS,0), 2) AS transit_walk_pct,
           ROUND(TOTAL_PRD_NOINTRA_DRVTR_PTRIPS * 100.0 / NULLIF(TOTAL_PRD_NOINTRA_TOT_PTRIPS,0), 2) AS transit_drive_pct,
           ROUND(TOTAL_PRD_NOINTRA_TR_PTRIPS    * 100.0 / NULLIF(TOTAL_PRD_NOINTRA_TOT_PTRIPS,0), 2) AS transit_total_pct,
           ROUND(TOTAL_PRD_NOINTRA_DAAUTO_PTRIPS* 100.0 / NULLIF(TOTAL_PRD_NOINTRA_TOT_PTRIPS,0), 2) AS drive_alone_pct,
           ROUND(TOTAL_PRD_NOINTRA_SR2AUTO_PTRIPS*100.0 / NULLIF(TOTAL_PRD_NOINTRA_TOT_PTRIPS,0), 2) AS sr2_pct,
           ROUND(TOTAL_PRD_NOINTRA_SR3AUTO_PTRIPS*100.0 / NULLIF(TOTAL_PRD_NOINTRA_TOT_PTRIPS,0), 2) AS sr3_pct,
           ROUND(TOTAL_PRD_NOINTRA_AUTO_PTRIPS  * 100.0 / NULLIF(TOTAL_PRD_NOINTRA_TOT_PTRIPS,0), 2) AS auto_total_pct
    FROM county_modechoice WHERE COUNTY = 'Dallas'
  This produces one compact row instead of 7+ UNION ALL blocks.
"""


# ── SQLite performance settings ───────────────────────────────────────────────
def fast_sqlite_settings(conn):
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA cache_size = 1000000")
    conn.execute("PRAGMA locking_mode = EXCLUSIVE")
    conn.execute("PRAGMA temp_store = MEMORY")


# ── Trip OMX loader ───────────────────────────────────────────────────────────
def load_one_core(args):
    core, matrix, zone_ids, conn = args
    n = matrix.shape[0]
    origins      = np.repeat(zone_ids, n)
    destinations = np.tile(zone_ids, n)
    values       = matrix.flatten()
    mask         = values > 0
    df = pd.DataFrame({
        "origin":      origins[mask],
        "destination": destinations[mask],
        "purpose":     core.replace("new_", "").upper(),
        "trips":       values[mask],
    })
    with WRITE_LOCK:
        df.to_sql("trips", conn, if_exists="append", index=False,
                  method="multi", chunksize=50_000)
    return core, len(df)


def load_omx_trips(path, conn):
    fast_sqlite_settings(conn)
    with omx.open_file(path, "r") as f:
        try:
            zone_ids = list(f.mapping("zone_id"))
        except Exception:
            zone_ids = list(range(1, f.shape()[0] + 1))
        cores    = [core for core in f.list_matrices() if core != "new_all"]
        matrices = {core: np.array(f[core]) for core in cores}

    conn.execute("DROP TABLE IF EXISTS trips")
    conn.execute("""
        CREATE TABLE trips (
            origin      INTEGER,
            destination INTEGER,
            purpose     TEXT,
            trips       REAL
        )
    """)
    args = [(core, matrices[core], zone_ids, conn) for core in cores]
    with ThreadPoolExecutor(max_workers=len(cores)) as ex:
        results = list(ex.map(load_one_core, args))

    conn.execute("CREATE INDEX idx_trips_purpose ON trips(purpose)")
    conn.execute("CREATE INDEX idx_trips_od      ON trips(origin, destination)")
    conn.commit()
    total_rows = sum(r[1] for r in results)
    print(f"Trips loaded: {total_rows:,} non-zero rows across {len(cores)} cores, {len(zone_ids)} zones")
    return cores, len(zone_ids)


def load_omx_skim(path, conn):
    with omx.open_file(path, "r") as f:
        try:
            zone_ids = list(f.mapping("zone_id"))
        except Exception:
            zone_ids = list(range(1, f.shape()[0] + 1))
        core_name = list(f.list_matrices())[0]
        matrix    = np.array(f[core_name])

    n    = matrix.shape[0]
    vals = matrix.flatten()
    mask = vals > 0
    df = pd.DataFrame({
        "origin":      np.repeat(zone_ids, n)[mask],
        "destination": np.tile(zone_ids, n)[mask],
        "walk_time":   vals[mask],
    })
    df.to_sql("skim", conn, if_exists="replace", index=False,
              method="multi", chunksize=50_000)
    conn.execute("CREATE INDEX idx_skim_od ON skim(origin, destination)")
    conn.commit()
    print(f"Skim loaded: {len(df):,} non-zero rows, core: {core_name}")
    return core_name


# ── Schema extractor ──────────────────────────────────────────────────────────
def get_schema(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [r[0] for r in cursor.fetchall()]
    lines  = ["DATABASE SCHEMA:"]
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        cols = cursor.fetchall()

        # For wide modechoice tables, only show the geo key + column count to avoid
        # flooding the context — the full field reference is already in the system prompt.
        if table.endswith("_modechoice"):
            geo_key = cols[0][1] if cols else "?"
            lines.append(
                f"\nTable: {table}\n"
                f"  Columns: {geo_key} (TEXT) + {len(cols)-1} numeric mode-choice columns"
                f" — see MODE CHOICE COLUMNS in system prompt for full field reference\n"
                f"  Rows: {pd.read_sql(f'SELECT COUNT(*) AS n FROM {table}', conn).iloc[0,0]}"
            )
            continue

        col_str = ", ".join(f"{c[1]} ({c[2]})" for c in cols)
        sample  = pd.read_sql(f"SELECT * FROM {table} LIMIT 2", conn)
        lines.append(f"\nTable: {table}\n  Columns: {col_str}\n  Sample:\n{sample.to_string(index=False)}")

    # extra context for perf measure tables
    for tbl in [t for t in tables if t.endswith("_perfmeasures")]:
        try:
            sections = pd.read_sql(f"SELECT DISTINCT section FROM {tbl} ORDER BY section", conn)
            lines.append(f"\n{tbl} sections: {sections['section'].tolist()}")
        except Exception:
            pass

    lines += ["", "NOTES:", "  Intrazonal trips: origin=destination (trips table)",
              "  TLD: join trips+skim, bin walk_time"]
    return "\n".join(lines)


# ── Performance Measures CSV parser ──────────────────────────────────────────
def parse_perf_csv(csv_path, geo_col):
    """
    Parse a TAFT performance measures CSV.
    geo_col: the name to use for the geography column (COUNTY, CITY, or MPA12).
    Returns a list of dicts with {geo_col, section, funcl, description, am, pm, op, daily}.
    """
    area_name = os.path.splitext(os.path.basename(csv_path))[0].split("_")[-1]

    rows = []
    current_section = None

    with open(csv_path, newline="", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line  = line.rstrip("\n")
            parts = [p.strip() for p in line.split(",")]

            if not parts[0] or parts[0] in ("FUNCL", "TOTAL"):
                continue

            if not _is_numeric(parts[0]) and (len(parts) < 2 or parts[1] in ("", "AM", "PM", "OP", "DAILY")):
                if len(parts) > 1 and parts[1] == "DESCRIPTION":
                    continue
                current_section = parts[0]
                continue

            if len(parts) >= 4 and _is_numeric(parts[0]) and parts[1]:
                try:
                    rows.append({
                        geo_col:       area_name,
                        "section":     current_section,
                        "funcl":       int(float(parts[0])),
                        "description": parts[1],
                        "am":          _to_float(parts[2]) if len(parts) > 2 else None,
                        "pm":          _to_float(parts[3]) if len(parts) > 3 else None,
                        "op":          _to_float(parts[4]) if len(parts) > 4 else None,
                        "daily":       _to_float(parts[5]) if len(parts) > 5 else None,
                    })
                except Exception:
                    pass

    sections_found = list({r["section"] for r in rows if r["section"]})
    print(f"  [{area_name}] {len(rows)} rows, sections: {sorted(sections_found)}")
    return rows


def _is_numeric(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _to_float(s):
    try:
        return float(s) if s.strip() != "" else None
    except (ValueError, TypeError):
        return None


# ── Scenario loader helpers ───────────────────────────────────────────────────
# Maps level folder name → (geo column name, table prefix, rows to exclude)
LEVEL_CONFIG = {
    "COUNTY": {
        "geo_col":  "COUNTY",
        "prefix":   "county",
        "exclude":  {"total", "external"},
    },
    "CITY": {
        "geo_col":  "CITY",
        "prefix":   "city",
        "exclude":  {"total"},
    },
    "MPA12": {
        "geo_col":  "MPA12",
        "prefix":   "mpa",
        "exclude":  set(),          # Inside / Outside / Total are all valid
    },
}

# CSV filename patterns to table suffix
CSV_PATTERNS = [
    ("Demographics_Summary", "demographics"),
    ("ModeChoice_Summary",   "modechoice"),
]


def _load_flat_csv(csv_path, geo_col, exclude_set):
    """Load a wide-format CSV (demographics or modechoice), drop excluded geo rows.

    These CSVs have a trailing comma on every data row (one more comma than header),
    which causes pandas to silently promote the header into a row index and shift all
    column names one position to the right.  index_col=False suppresses that behavior;
    dropna(axis=1, how='all') then removes the phantom empty column the trailing comma creates.
    skipinitialspace=True strips the leading space from every numeric value.
    """
    df = pd.read_csv(csv_path, encoding="utf-8", skipinitialspace=True,
                     index_col=False, on_bad_lines="skip")
    df = df.dropna(axis=1, how="all")          # remove phantom trailing empty column
    df.columns = df.columns.str.strip()
    if geo_col in df.columns:
        df = df[~df[geo_col].astype(str).str.strip().str.lower().isin(exclude_set)]
    return df


def load_scenario_data(folder, conn):
    """
    Load all CSV data for a model run into SQLite.
    Creates tables: county_demographics, county_modechoice, county_perfmeasures,
                    city_demographics,   city_modechoice,   city_perfmeasures,
                    mpa_demographics,    mpa_modechoice,    mpa_perfmeasures
    Returns a summary dict.
    """
    base = os.path.join(MODEL_RUNS_ROOT, folder, "Reports", "ModelRunReports")
    summary = {}

    for level_folder, cfg in LEVEL_CONFIG.items():
        level_dir = os.path.join(base, level_folder)
        if not os.path.isdir(level_dir):
            print(f"  Skipping {level_folder}: folder not found")
            continue

        geo_col    = cfg["geo_col"]
        prefix     = cfg["prefix"]
        exclude    = cfg["exclude"]

        # ── Flat CSVs: demographics and modechoice ────────────────────────────
        for fname in os.listdir(level_dir):
            if not fname.lower().endswith(".csv"):
                continue
            for pattern, suffix in CSV_PATTERNS:
                if pattern in fname:
                    table_name = f"{prefix}_{suffix}"
                    path = os.path.join(level_dir, fname)
                    try:
                        df = _load_flat_csv(path, geo_col, exclude)
                        df.to_sql(table_name, conn, if_exists="replace", index=False)
                        conn.commit()
                        summary[table_name] = len(df)
                        print(f"  Loaded {table_name}: {len(df)} rows from {fname}")
                    except Exception as e:
                        print(f"  Error loading {fname}: {e}")
                    break

        # ── Performance measures CSV per area ─────────────────────────────────
        perf_dir = os.path.join(level_dir, "PerformanceMeasures")
        if os.path.isdir(perf_dir):
            table_name = f"{prefix}_perfmeasures"
            all_rows   = []
            for fname in os.listdir(perf_dir):
                if not fname.lower().endswith(".csv"):
                    continue
                area_name = os.path.splitext(fname)[0].split("_")[-1]
                if area_name.lower() in exclude or area_name.lower() == "total":
                    print(f"  Skipping {fname} ({area_name} excluded)")
                    continue
                path = os.path.join(perf_dir, fname)
                try:
                    rows = parse_perf_csv(path, geo_col)
                    all_rows.extend(rows)
                except Exception as e:
                    print(f"  Skipping {fname}: {e}")

            if all_rows:
                df = pd.DataFrame(all_rows)
                df.to_sql(table_name, conn, if_exists="replace", index=False)
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_geo "
                             f"ON {table_name}({geo_col})")
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_section "
                             f"ON {table_name}(section)")
                conn.commit()
                summary[table_name] = len(df)
                print(f"  Loaded {table_name}: {len(df)} rows from {len(all_rows)} perf records")

    # ── Mode choice metadata lookup table ─────────────────────────────────────
    meta_rows = [
        {"field": field, "description": desc, "unit": unit}
        for field, (desc, unit) in MODECHOICE_META.items()
    ]
    pd.DataFrame(meta_rows).to_sql(
        "modechoice_metadata", conn, if_exists="replace", index=False
    )
    conn.commit()
    print(f"  Loaded modechoice_metadata: {len(meta_rows)} field definitions")

    return summary


# ── SQL generation ────────────────────────────────────────────────────────────
def generate_sql(question, schema, error=None):
    if error:
        prompt = (f"{schema}\n\nUser asked: \"{question}\"\n"
                  f"Previous SQL failed: {error}\n"
                  f"Write corrected SQLite SQL. Return ONLY raw SQL.")
    else:
        prompt = (f"{schema}\n\nUser asked: \"{question}\"\n"
                  f"Write a single SQLite SQL query. Return ONLY raw SQL, no markdown, no backticks.")
    r   = client.messages.create(model=MODEL, max_tokens=1200, system=SYSTEM,
                                  messages=[{"role": "user", "content": prompt}])
    sql = r.content[0].text.strip()
    sql = re.sub(r"^```(?:sql)?|```$", "", sql, flags=re.MULTILINE).strip()
    return sql


def run_query(sql, conn):
    if not sql.strip().upper().startswith("SELECT"):
        return None, "Only SELECT queries allowed."
    try:
        return pd.read_sql(sql, conn), None
    except Exception as e:
        return None, str(e)


def run_with_retry(question, schema, conn, max_retries=3):
    error, sql = None, None
    for attempt in range(1, max_retries + 1):
        sql      = generate_sql(question, schema, error)
        df, error = run_query(sql, conn)
        if error is None:
            return df, sql, attempt, None
    return None, sql, max_retries, error


def summarize(question, sql, df):
    result_text = (df.to_string(index=False) if len(df) <= 50
                   else df.head(20).to_string(index=False) + f"\n...({len(df)} rows)")
    r = client.messages.create(
        model=MODEL, max_tokens=512, system=SYSTEM,
        messages=[{"role": "user", "content":
            f"User asked: \"{question}\"\nSQL: {sql}\nResult:\n{result_text}\n"
            f"Summarize in 2-3 sentences for a travel demand modeler or traffic planner. "
            f"Mention specific numbers."}])
    return r.content[0].text.strip()


# ── Chart type detector ───────────────────────────────────────────────────────
def detect_chart_type(question):
    """Return 'bar', 'line', 'pie', or 'scatter' based on keywords in the question."""
    q = question.lower()
    if any(k in q for k in ["pie chart", "pie graph", "pie plot", "donut", "doughnut"]):
        return "pie"
    if any(k in q for k in ["line chart", "line graph", "line plot", "trend", "over time",
                             "time series", "timeseries"]):
        return "line"
    if any(k in q for k in ["scatter", "scatter plot", "scatter chart", "scatterplot"]):
        return "scatter"
    # bar is the default — also explicit keywords
    return "bar"


# ── Chart builder ─────────────────────────────────────────────────────────────
FUNCL_LABELS = {
    1: "Freeways",
    2: "Principal Arterials",
    3: "Minor Arterials",
    4: "Collectors",
    6: "Freeway Ramps",
    7: "Frontage Roads",
}
TOD_COLS   = ["am", "pm", "op", "daily"]
TOD_COLORS = {"am": "#39d353", "pm": "#58a6ff", "op": "#f78166", "daily": "#d2a8ff"}


def _fmt_val(v):
    try:
        f = float(v)
        return f"{f:,.1f}" if f % 1 else f"{f:,.0f}"
    except Exception:
        return str(v)


def _apply_dark(fig, ax):
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.tick_params(colors="#8b949e", labelsize=9)
    ax.yaxis.label.set_color("#8b949e")
    ax.xaxis.label.set_color("#8b949e")
    ax.title.set_color("#8b949e")


def make_chart(df, question="", chart_type="bar"):
    if df is None or df.empty:
        return None

    numeric = df.select_dtypes(include="number").columns.tolist()
    text    = df.select_dtypes(exclude="number").columns.tolist()
    if not numeric:
        return None

    # ── Wide single-row result: column names are categories, values are the data.
    # e.g. SELECT COUNTY, transit_pct, drive_alone_pct, sr2_pct ... WHERE COUNTY='Dallas'
    # Unpivot so each numeric column becomes a row: (mode, value).
    # This covers any result with 1 row and 2+ numeric columns.
    if len(df) == 1 and len(numeric) >= 2:
        id_cols  = text   # e.g. ['COUNTY'] — keep as title context
        title_prefix = (", ".join(df[c].iloc[0] for c in id_cols) + " — ") if id_cols else ""
        df = df[numeric].T.reset_index()
        df.columns = ["mode", "value"]
        df["mode"]  = df["mode"].astype(str)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        df.attrs["title"] = title_prefix.rstrip(" — ") if title_prefix else ""
        numeric = ["value"]
        text    = ["mode"]

    if len(df) < 2:
        return None

    q = question.lower()
    percent_keywords = ["percent", "percentage", "pct", "share", "proportion"]
    wants_percent = any(k in q for k in percent_keywords)

    funcl_col = next((c for c in df.columns if c.lower() == "funcl"), None)
    if funcl_col:
        df = df.copy()
        df[funcl_col] = df[funcl_col].apply(
            lambda v: FUNCL_LABELS.get(int(v), str(v)) if pd.notna(v) else v
        )

    county_col  = next((c for c in df.columns if c.lower() in ("county", "city", "mpa12")), None)
    present_tod = [c for c in df.columns if c.lower() in TOD_COLS]

    period_col = None
    for c in text:
        try:
            vals = df[c].dropna().astype(str).str.lower().unique()
            if len(vals) >= 1 and all(v in TOD_COLS for v in vals):
                period_col = c
                break
        except Exception:
            pass

    COUNTY_COLORS = ["#39d353", "#58a6ff", "#f78166", "#d2a8ff", "#ffa657", "#79c0ff"]

    # ── Alternative renderers (line / pie / scatter) ──────────────────────────
    def _legend(ax):
        leg = ax.legend(fontsize=8, facecolor="#161b22", edgecolor="#30363d")
        for t in leg.get_texts():
            t.set_color("#8b949e")
        return leg

    def _titled_legend(ax, title):
        leg = ax.legend(title=title, title_fontsize=8,
                        fontsize=8, facecolor="#161b22", edgecolor="#30363d")
        leg.get_title().set_color("#8b949e")
        for t in leg.get_texts():
            t.set_color("#8b949e")

    def _render_line(ax, plot_df, x_col, series_list, color_map):
        """Draw a line chart: x = x_col categories, one line per series."""
        cats = plot_df[x_col].astype(str).tolist()
        x    = np.arange(len(cats))
        for i, series in enumerate(series_list):
            color = (color_map[i] if isinstance(color_map, list)
                     else color_map.get(series, color_map.get(series.lower(), "#39d353")))
            ax.plot(x, plot_df[series], marker="o", markersize=4,
                    linewidth=1.8, color=color, label=series)
            for xi, v in zip(x, plot_df[series]):
                if pd.notna(v):
                    ax.annotate(_fmt_val(v), (xi, v),
                                textcoords="offset points", xytext=(0, 6),
                                ha="center", fontsize=6, color="#8b949e")
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=8, color="#8b949e")

    def _render_pie(ax, labels, values, colors):
        """Draw a pie chart with percentage labels inside and value labels outside."""
        clean = [(l, v) for l, v in zip(labels, values)
                 if pd.notna(v) and float(v) > 0]
        if not clean:
            return
        lbls, vals = zip(*clean)
        pie_colors = [colors[i % len(colors)] for i in range(len(lbls))]
        _, _, autotexts = ax.pie(
            vals, labels=lbls, autopct="%1.1f%%", startangle=90,
            colors=pie_colors,
            wedgeprops={"edgecolor": "#0d1117", "linewidth": 1},
            textprops={"color": "#8b949e", "fontsize": 8},
        )
        for at in autotexts:
            at.set_color("#0d1117")
            at.set_fontsize(7)
        ax.set_facecolor("#0d1117")

    def _render_scatter(ax, plot_df, x_col, y_col, label_col, color):
        """Draw a scatter plot: x = x_col (numeric), y = y_col, points labelled by label_col."""
        xs = pd.to_numeric(plot_df[x_col], errors="coerce")
        ys = pd.to_numeric(plot_df[y_col], errors="coerce")
        ax.scatter(xs, ys, color=color, edgecolors="#0d1117", s=60, zorder=3)
        if label_col:
            for xi, yi, lbl in zip(xs, ys, plot_df[label_col].astype(str)):
                if pd.notna(xi) and pd.notna(yi):
                    ax.annotate(lbl, (xi, yi),
                                textcoords="offset points", xytext=(4, 4),
                                fontsize=7, color="#8b949e")
        ax.set_xlabel(x_col, fontsize=9)
        ax.set_ylabel(y_col, fontsize=9)

    # ── Bar helper (unchanged logic, extracted so line/pie can reuse prep) ────
    def _grouped_bar(ax, plot_df, _group_col, series_list, color_map, x_col):
        cats  = plot_df[x_col].astype(str).tolist()
        n     = len(cats)
        n_s   = len(series_list)
        width = 0.7 / n_s
        x     = np.arange(n)
        for i, series in enumerate(series_list):
            color  = (color_map[i] if isinstance(color_map, list)
                      else color_map.get(series, color_map.get(series.lower(), "#39d353")))
            offset = (i - (n_s - 1) / 2) * width
            bars   = ax.bar(x + offset, plot_df[series], width=width,
                            color=color, edgecolor="#0d1117", label=series)
            for bar in bars:
                h = bar.get_height()
                if h and not np.isnan(h):
                    ax.text(bar.get_x() + bar.get_width() / 2, h * 1.02,
                            _fmt_val(h), ha="center", va="bottom",
                            fontsize=6, color="#8b949e")
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=8, color="#8b949e")

    mentioned_tod = [t for t in TOD_COLS if re.search(rf"\b{t}\b", q)]
    if mentioned_tod:
        present_tod = [c for c in present_tod if c.lower() in mentioned_tod]

    tod_alias_map = {}
    if not present_tod and numeric:
        search_periods = mentioned_tod if mentioned_tod else TOD_COLS
        for col in numeric:
            col_lower = col.lower()
            for t in search_periods:
                if re.search(rf"\b{t}\b|_{t}_|_{t}$|^{t}_", col_lower):
                    tod_alias_map[col] = t
                    break
        if tod_alias_map:
            present_tod = list(tod_alias_map.keys())

    funcl_mentioned = funcl_col and any(
        re.search(rf"\b{k}\b", q)
        for k in ["freeway", "freeways", "arterial", "arterials", "collector", "collectors",
                  "ramp", "ramps", "frontage"] + [str(v) for v in FUNCL_LABELS.keys()]
    )

    _has_wide_tod = len(present_tod) >= 2
    _has_long_tod = county_col and period_col and numeric

    print(f"[make_chart] type={chart_type} cols={list(df.columns)} county={county_col} "
          f"present_tod={present_tod} period_col={period_col} "
          f"mentioned={mentioned_tod} wide={_has_wide_tod} long={_has_long_tod}")

    # ── Scatter: needs two numeric columns; bypass all other logic ────────────
    if chart_type == "scatter" and len(numeric) >= 2:
        x_col_s = numeric[0]
        y_col_s = numeric[1]
        lbl_col = text[0] if text else None
        fig, ax = plt.subplots(figsize=(7, 4))
        _apply_dark(fig, ax)
        _render_scatter(ax, df.head(50), x_col_s, y_col_s, lbl_col, "#39d353")

    # ── Pie: single x-label + single numeric value ────────────────────────────
    elif chart_type == "pie":
        # determine labels and values from the df
        if text and len(numeric) == 1:
            pie_labels = df[text[0]].astype(str).tolist()
            pie_values = df[numeric[0]].tolist()
        elif county_col and len(numeric) >= 1:
            val_col    = numeric[0]
            agg        = df.groupby(county_col, sort=False)[val_col].sum().reset_index()
            pie_labels = agg[county_col].astype(str).tolist()
            pie_values = agg[val_col].tolist()
        elif text and numeric:
            pie_labels = df[text[0]].astype(str).tolist()
            pie_values = df[numeric[0]].tolist()
        else:
            pie_labels = [str(i) for i in range(len(df))]
            pie_values = df[numeric[0]].tolist()

        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")
        _render_pie(ax, pie_labels, pie_values, COUNTY_COLORS)

    # ── Line / Bar — shared data-prep, branch only at draw time ──────────────
    elif county_col and (_has_wide_tod or _has_long_tod):
        if _has_long_tod and not _has_wide_tod:
            val_col = numeric[0]
            df = df.copy()
            df[period_col] = df[period_col].astype(str).str.lower()
            if mentioned_tod:
                df = df[df[period_col].isin(mentioned_tod)]
            agg_fn  = "sum" if funcl_mentioned else "mean"
            pivot   = (df.groupby([county_col, period_col], sort=False)[val_col]
                         .agg(agg_fn).unstack(level=1).reset_index())
            pivot.columns.name = None
            present_tod = [c for c in pivot.columns if c != county_col]
            plot_df = pivot
        else:
            cols_needed = [county_col] + present_tod
            plot_df = df[[c for c in cols_needed if c in df.columns]].dropna(subset=[county_col]).copy()
            agg_fn  = "sum" if funcl_mentioned else "mean"
            plot_df = plot_df.groupby(county_col, sort=False)[present_tod].agg(agg_fn).reset_index()

        tod_color_map = {
            col: TOD_COLORS.get(tod_alias_map.get(col, col.lower()), "#39d353")
            for col in present_tod
        }
        fig, ax = plt.subplots(figsize=(max(7, len(plot_df) * 1.4), 4))
        _apply_dark(fig, ax)
        if chart_type == "line":
            _render_line(ax, plot_df, county_col, present_tod, tod_color_map)
        else:
            _grouped_bar(ax, plot_df, county_col, present_tod, tod_color_map, county_col)
        ax.set_ylabel("Value", fontsize=9)
        _titled_legend(ax, "Time of Day")

    elif funcl_col and county_col:
        val_col = next((c for c in numeric if c.lower() not in TOD_COLS and c != funcl_col),
                       present_tod[0] if present_tod else (numeric[0] if numeric else None))
        if val_col is not None:
            pivot = (df[[county_col, funcl_col, val_col]]
                     .dropna()
                     .groupby([county_col, funcl_col], sort=False)[val_col]
                     .sum().unstack(level=1).reset_index())
            pivot.columns.name = None
            funcl_cols = [c for c in pivot.columns if c != county_col]
            color_map  = {c: COUNTY_COLORS[i % len(COUNTY_COLORS)] for i, c in enumerate(funcl_cols)}
            fig, ax = plt.subplots(figsize=(max(7, len(pivot) * 1.4), 4))
            _apply_dark(fig, ax)
            if chart_type == "line":
                _render_line(ax, pivot, county_col, funcl_cols, color_map)
            else:
                _grouped_bar(ax, pivot, funcl_col, funcl_cols, color_map, county_col)
            ax.set_ylabel(val_col, fontsize=9)
            _titled_legend(ax, "Funcl")

    elif funcl_col and len(present_tod) >= 2:
        plot_df = df[[funcl_col] + present_tod].dropna(subset=[funcl_col]).copy()
        plot_df = plot_df.groupby(funcl_col, sort=False)[present_tod].sum().reset_index()
        tod_cm  = {t.lower(): TOD_COLORS.get(t.lower(), "#39d353") for t in present_tod}
        fig, ax = plt.subplots(figsize=(max(7, len(plot_df) * 1.2), 4))
        _apply_dark(fig, ax)
        if chart_type == "line":
            _render_line(ax, plot_df, funcl_col, present_tod, tod_cm)
        else:
            _grouped_bar(ax, plot_df, funcl_col, present_tod, tod_cm, funcl_col)
        ax.set_ylabel("Value", fontsize=9)
        _titled_legend(ax, "Time of Day")

    elif funcl_col and numeric:
        val_col = next((c for c in numeric if c.lower() not in TOD_COLS and c != funcl_col),
                       numeric[0])
        plot_df = df[[funcl_col, val_col]].dropna().copy()
        plot_df = plot_df.groupby(funcl_col, sort=False)[val_col].sum().reset_index()
        y_vals  = plot_df[val_col].astype(float)
        if wants_percent:
            total  = y_vals.sum()
            y_vals = y_vals / total * 100 if total > 0 else y_vals * 0
            y_label = "Percent"
        else:
            y_label = val_col

        fig, ax = plt.subplots(figsize=(max(7, len(plot_df) * 1.2), 4))
        _apply_dark(fig, ax)
        cats = plot_df[funcl_col].astype(str).tolist()
        x    = np.arange(len(cats))
        if chart_type == "line":
            ax.plot(x, y_vals, marker="o", markersize=5, linewidth=1.8, color="#39d353")
            for xi, v in zip(x, y_vals):
                if pd.notna(v):
                    ax.annotate(f"{v:.1f}%" if wants_percent else _fmt_val(v), (xi, v),
                                textcoords="offset points", xytext=(0, 6),
                                ha="center", fontsize=7, color="#8b949e")
        elif chart_type == "pie":
            _render_pie(ax, cats, y_vals.tolist(), COUNTY_COLORS)
        else:
            bars = ax.bar(x, y_vals, color="#39d353", edgecolor="#0d1117", width=0.6)
            for bar, v in zip(bars, y_vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                        f"{v:.1f}%" if wants_percent else _fmt_val(v),
                        ha="center", va="bottom", fontsize=7, color="#8b949e")
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=8, color="#8b949e")
        ax.set_ylabel(y_label, fontsize=9)

    elif funcl_col and period_col and numeric:
        val_col    = numeric[0]
        plot_df    = df[[funcl_col, period_col, val_col]].dropna().copy()
        pivot      = (plot_df.groupby([funcl_col, period_col], sort=False)[val_col]
                      .sum().unstack(level=1).reset_index())
        pivot.columns.name = None
        tod_series = [c for c in pivot.columns if c != funcl_col]
        tod_cm     = {t.lower(): TOD_COLORS.get(t.lower(), "#39d353") for t in tod_series}
        fig, ax    = plt.subplots(figsize=(max(7, len(pivot) * 1.2), 4))
        _apply_dark(fig, ax)
        if chart_type == "line":
            _render_line(ax, pivot, funcl_col, tod_series, tod_cm)
        else:
            _grouped_bar(ax, pivot, funcl_col, tod_series, tod_cm, funcl_col)
        ax.set_ylabel(val_col, fontsize=9)
        _legend(ax)

    elif period_col and numeric:
        val_col = numeric[0]
        plot_df = df[[period_col, val_col]].dropna().copy()
        plot_df = plot_df.groupby(period_col, sort=False)[val_col].sum().reset_index()
        cats    = plot_df[period_col].astype(str).tolist()
        x       = np.arange(len(cats))
        colors_ = [TOD_COLORS.get(p.lower(), "#39d353") for p in cats]
        fig, ax = plt.subplots(figsize=(max(5, len(plot_df) * 1.2), 3.6))
        _apply_dark(fig, ax)
        if chart_type == "line":
            ax.plot(x, plot_df[val_col], marker="o", markersize=5,
                    linewidth=1.8, color="#39d353")
            for xi, v in zip(x, plot_df[val_col]):
                if pd.notna(v):
                    ax.annotate(_fmt_val(v), (xi, v),
                                textcoords="offset points", xytext=(0, 6),
                                ha="center", fontsize=7, color="#8b949e")
        elif chart_type == "pie":
            _render_pie(ax, cats, plot_df[val_col].tolist(), COUNTY_COLORS)
        else:
            bars = ax.bar(x, plot_df[val_col], color=colors_, edgecolor="#0d1117", width=0.6)
            for bar, v in zip(bars, plot_df[val_col]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                        _fmt_val(v), ha="center", va="bottom", fontsize=7, color="#8b949e")
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=0, ha="center", fontsize=9, color="#8b949e")
        ax.set_ylabel(val_col, fontsize=9)

    elif text and numeric:
        x_col, y_col = text[0], numeric[0]
        plot_df = df[[x_col, y_col]].dropna().head(20).copy()
        y_name  = y_col.lower()
        y_is_pct_col = any(k in y_name for k in ["percent", "percentage", "pct", "share"])
        if y_is_pct_col:
            y_vals  = plot_df[y_col].astype(float)
            y_vals  = y_vals * 100 if y_vals.max() <= 1 else y_vals
            y_label = "Percent"
            labels  = [f"{v:.1f}%" for v in y_vals]
        elif wants_percent:
            total   = plot_df[y_col].sum()
            y_vals  = plot_df[y_col] / total * 100 if total > 0 else plot_df[y_col] * 0
            y_label = "Percent"
            labels  = [f"{v:.1f}%" for v in y_vals]
        else:
            y_vals  = plot_df[y_col]
            y_label = y_col
            labels  = [_fmt_val(v) for v in y_vals]

        cats = plot_df[x_col].astype(str).tolist()
        x    = np.arange(len(cats))
        x_vals_lower = plot_df[x_col].astype(str).str.lower()
        bar_colors   = ([TOD_COLORS.get(v, "#39d353") for v in x_vals_lower]
                        if x_vals_lower.isin(TOD_COLS).all() else "#39d353")

        if chart_type == "pie":
            fig, ax = plt.subplots(figsize=(6, 5))
            fig.patch.set_facecolor("#0d1117")
            ax.set_facecolor("#0d1117")
            _render_pie(ax, cats, y_vals.tolist(), COUNTY_COLORS)
        else:
            fig, ax = plt.subplots(figsize=(7, 3.6))
            _apply_dark(fig, ax)
            if chart_type == "line":
                ax.plot(x, y_vals, marker="o", markersize=5, linewidth=1.8, color="#39d353")
                for xi, v, lbl in zip(x, y_vals, labels):
                    if pd.notna(v):
                        ax.annotate(lbl, (xi, v),
                                    textcoords="offset points", xytext=(0, 6),
                                    ha="center", fontsize=7, color="#8b949e")
            else:
                bars = ax.bar(x, y_vals, color=bar_colors, edgecolor="#0d1117", width=0.7)
                for bar, lbl in zip(bars, labels):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                            lbl, ha="center", va="bottom", fontsize=7, color="#8b949e")
            ax.set_xticks(x)
            ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=8, color="#8b949e")
            ax.set_ylabel(y_label, fontsize=9)

    elif len(numeric) == 1 and not text:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        _apply_dark(fig, ax)
        ax.hist(df[numeric[0]].dropna(), bins=20, color="#39d353", edgecolor="#0d1117")
        ax.set_xlabel(numeric[0], fontsize=9)
        ax.set_ylabel("Count", fontsize=9)

    else:
        # fallback: line plot of all numeric columns
        fig, ax = plt.subplots(figsize=(7, 3.6))
        _apply_dark(fig, ax)
        colors_ = ["#39d353", "#58a6ff", "#f78166"]
        for i, col in enumerate(numeric[:3]):
            ax.plot(df.index, df[col], marker="o", markersize=3,
                    label=col, linewidth=1.5, color=colors_[i % len(colors_)])
        _legend(ax)

    # Show geography context as chart title when unpivoted from wide single-row
    chart_title = df.attrs.get("title", "")
    if chart_title:
        try:
            ax.set_title(chart_title, fontsize=10, color="#8b949e", pad=8)
        except Exception:
            pass

    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/upload", methods=["POST"])
def upload():
    """Upload OMX trip/skim files (existing functionality)."""
    try:
        trip_file = request.files.get("trip")
        skim_file = request.files.get("skim")
        if not trip_file:
            return jsonify({"error": "Trip OMX required"}), 400

        with DB_LOCK:
            conn = sqlite3.connect(":memory:", check_same_thread=False)
            info = {}

            with tempfile.NamedTemporaryFile(suffix=".omx", delete=False) as t:
                trip_file.save(t.name)
                cores, zones = load_omx_trips(t.name, conn)
                info["trip_cores"] = cores
                info["zones"]      = zones
            os.unlink(t.name)

            if skim_file:
                with tempfile.NamedTemporaryFile(suffix=".omx", delete=False) as s:
                    skim_file.save(s.name)
                    info["skim_core"] = load_omx_skim(s.name, conn)
                os.unlink(s.name)

            DB["conn"]   = conn
            DB["schema"] = get_schema(conn)
            purposes     = pd.read_sql(
                "SELECT purpose, SUM(trips) as total FROM trips GROUP BY purpose", conn)
            info["purposes"] = purposes.to_dict(orient="records")

        return jsonify({"ok": True, "info": info})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/list_model_runs", methods=["GET"])
def list_model_runs():
    """Return subdirectory names under MODEL_RUNS_ROOT."""
    try:
        if not os.path.isdir(MODEL_RUNS_ROOT):
            return jsonify({"error": f"Folder not found: {MODEL_RUNS_ROOT}"}), 404
        folders = sorted(
            d for d in os.listdir(MODEL_RUNS_ROOT)
            if os.path.isdir(os.path.join(MODEL_RUNS_ROOT, d))
        )
        return jsonify({"folders": folders})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/load_scenario", methods=["POST"])
def load_scenario():
    """
    Body: { "folder": "M50_APR26_YR50_Y50FM_260417RUN" }
    Loads all demographics, mode choice, and performance measure CSVs
    for CITY, COUNTY, and MPA12 levels into SQLite.
    """
    try:
        data   = request.json
        folder = data.get("folder", "").strip()
        if not folder:
            return jsonify({"error": "No folder specified"}), 400

        base = os.path.join(MODEL_RUNS_ROOT, folder, "Reports", "ModelRunReports")
        if not os.path.isdir(base):
            return jsonify({"error": f"ModelRunReports folder not found: {base}"}), 404

        with DB_LOCK:
            conn = DB["conn"]
            if conn is None:
                conn = sqlite3.connect(":memory:", check_same_thread=False)
                DB["conn"] = conn

            summary = load_scenario_data(folder, conn)
            DB["schema"] = get_schema(conn)

        tables_loaded = list(summary.keys())
        total_rows    = sum(summary.values())

        # Build a friendly summary for the UI
        counties = []
        try:
            df_c = pd.read_sql("SELECT DISTINCT COUNTY FROM county_demographics ORDER BY COUNTY", conn)
            counties = df_c["COUNTY"].tolist()
        except Exception:
            pass

        cities_count = 0
        try:
            df_ci = pd.read_sql("SELECT COUNT(DISTINCT CITY) as n FROM city_demographics", conn)
            cities_count = int(df_ci["n"].iloc[0])
        except Exception:
            pass

        sections = []
        for tbl in ["county_perfmeasures", "city_perfmeasures", "mpa_perfmeasures"]:
            try:
                df_s = pd.read_sql(f"SELECT DISTINCT section FROM {tbl} ORDER BY section", conn)
                sections = df_s["section"].tolist()
                break
            except Exception:
                pass

        print(f"Scenario loaded: {folder}, {total_rows:,} total rows across {len(tables_loaded)} tables")
        return jsonify({
            "ok":           True,
            "folder":       folder,
            "tables":       tables_loaded,
            "total_rows":   total_rows,
            "counties":     sorted(counties),
            "cities_count": cities_count,
            "sections":     sorted(sections),
            "summary":      summary,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# Keep the old /load_perf route alive so any existing bookmarks still work
@app.route("/load_perf", methods=["POST"])
def load_perf():
    return load_scenario()


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data     = request.json
        question = data.get("question", "").strip()
        if not question:
            return jsonify({"error": "No question"}), 400

        with DB_LOCK:
            conn   = DB["conn"]
            schema = DB["schema"]
        if conn is None:
            return jsonify({"error": "No data loaded. Select and load a scenario first."}), 400

        df, sql, attempts, error = run_with_retry(question, schema, conn)
        if error:
            return jsonify({"error": f"Query failed after 3 attempts: {error}", "sql": sql})

        answer = summarize(question, sql, df)
        chart_keywords = ["chart", "plot", "graph", "visuali", "bar", "histogram",
                          "distribution", "show me", "compare"]
        wants_chart = any(k in question.lower() for k in chart_keywords)
        chart_type  = detect_chart_type(question) if wants_chart else "bar"
        chart  = make_chart(df, question, chart_type) if wants_chart else None

        # Replace NaN/Inf with None so the response is valid JSON
        df_clean = df.head(50).where(pd.notnull(df.head(50)), other=None)
        rows = df_clean.to_dict(orient="records")
        cols = list(df.columns)
        return jsonify({"answer": answer, "sql": sql, "attempts": attempts,
                        "chart": chart, "rows": rows, "cols": cols, "total_rows": len(df)})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return send_file(os.path.join(base_dir, "chatbot_perfmeasure.html"))


@app.route("/health")
def health():
    return jsonify({"ok": True, "loaded": DB["conn"] is not None})


def open_browser():
    webbrowser.open("http://127.0.0.1:5050")


if __name__ == "__main__":
    Timer(1.5, open_browser).start()
    app.run(host="127.0.0.1", port=5050, debug=True, threaded=True,
            extra_files=[], exclude_patterns=["*.venv*", "*site-packages*"])
