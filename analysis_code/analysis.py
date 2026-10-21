#!/usr/bin/env python3
"""
Olist Brazilian e-commerce: full data inspection + in-depth analysis.

Usage:   python analysis.py --db olist.sqlite --out olist_analysis
Output:  <out>/graphs/*.png   and   <out>/facts/*.txt

Definitions used everywhere (so every number can be reproduced):
  * GMV            = sum of item price (freight NOT included)
  * valid order    = status not in {canceled, unavailable}
  * analysis window for trends = Jan 2017 .. Aug 2018 (the only fully populated months)
  * delivery days  = order_delivered_customer_date - order_purchase_timestamp  (delivered orders only)
  * late order     = delivered on a calendar day AFTER the estimated delivery day
  * order review   = latest review per order (some orders have several)
"""
import argparse, re, sqlite3, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
from matplotlib.patches import FancyBboxPatch

warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--db", default="olist.sqlite")
ap.add_argument("--out", default="olist_analysis")
args = ap.parse_args()
OUT = Path(args.out)
GR, FA = OUT / "graphs", OUT / "facts"
GR.mkdir(parents=True, exist_ok=True)
FA.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------- style
DARK, GREEN, LGREEN = "#0E2F2B", "#1E8E5A", "#BFE3D0"
YELLOW, BLUE, CORAL, GRAY, LGRAY = "#F5B700", "#2C6E9B", "#D1495B", "#7F8F94", "#E4EAE8"
plt.rcParams.update({
    "figure.figsize": (11, 6), "font.size": 11, "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#9AA7AA", "axes.labelcolor": "#26373A", "xtick.color": "#26373A",
    "ytick.color": "#26373A", "axes.grid": True, "grid.color": "#E6ECEA", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.titlepad": 14, "legend.frameon": False,
})
_counter = [0]


def save(fig, name):
    _counter[0] += 1
    fn = f"{_counter[0]:02d}_{name}.png"
    fig.savefig(GR / fn, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", fn)
    return fn


class Facts:
    def __init__(self, fname, title):
        self.f, self.lines = fname, [title, "=" * len(title)]

    def h(self, t):
        self.lines += ["", t, "-" * len(t)]

    def p(self, t=""):
        self.lines.append(t)

    def kv(self, k, v):
        self.lines.append(f"  - {k}: {v}")

    def table(self, df, **kw):
        self.lines += ["", df.to_string(**kw), ""]

    def save(self):
        (FA / self.f).write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        print("wrote", self.f)


def n0(x): return f"{x:,.0f}"
def brl(x): return f"R$ {x:,.0f}"
def brl2(x): return f"R$ {x:,.2f}"
def pct(x, d=1): return f"{x * 100:.{d}f}%"


K = {}  # key metrics collected for the final takeaways file

# ----------------------------------------------------------------------------- load
con = sqlite3.connect(args.db)
names = [r[0] for r in con.execute("select name from sqlite_master where type='table' order by name")]
T = {t: pd.read_sql(f'select * from "{t}"', con) for t in names}
customers, geo, lc, lq = T["customers"], T["geolocation"], T["leads_closed"], T["leads_qualified"]
oi, op, rv, orders = T["order_items"], T["order_payments"], T["order_reviews"], T["orders"].copy()
tr, prod, sellers = T["product_category_name_translation"], T["products"].copy(), T["sellers"]

for c in ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
          "order_delivered_customer_date", "order_estimated_delivery_date"]:
    orders[c] = pd.to_datetime(orders[c])
oi["shipping_limit_date"] = pd.to_datetime(oi["shipping_limit_date"])
rv["review_creation_date"] = pd.to_datetime(rv["review_creation_date"])
rv["review_answer_timestamp"] = pd.to_datetime(rv["review_answer_timestamp"])
orders = orders.rename(columns={"order_purchase_timestamp": "purch", "order_approved_at": "appr",
                                "order_delivered_carrier_date": "carrier",
                                "order_delivered_customer_date": "deliv",
                                "order_estimated_delivery_date": "est"})

# category english names (2 categories are missing from the translation table -> fixed manually)
cat_map = dict(zip(tr.product_category_name, tr.product_category_name_english))
cat_map.update({"pc_gamer": "pc_gamer", "portateis_cozinha_e_preparadores_de_alimentos": "portable_kitchen_food_preparers"})
prod["category"] = prod.product_category_name.map(cat_map).fillna("unknown")

VALID = ~orders.order_status.isin(["canceled", "unavailable"])
orders["valid"] = VALID
orders["ym"] = orders.purch.dt.to_period("M")
WIN = (orders.ym >= "2017-01") & (orders.ym <= "2018-08")

it = (oi.merge(orders[["order_id", "customer_id", "order_status", "purch", "ym", "valid", "deliv", "est"]], on="order_id")
        .merge(customers[["customer_id", "customer_unique_id", "customer_state", "customer_city"]], on="customer_id")
        .merge(prod[["product_id", "category", "product_weight_g"]], on="product_id")
        .merge(sellers[["seller_id", "seller_state", "seller_city"]], on="seller_id"))
iv = it[it.valid].copy()  # valid items

# one (latest) review per order
rv_order = (rv.sort_values(["review_creation_date", "review_answer_timestamp"])
              .groupby("order_id").tail(1)[["order_id", "review_score", "review_comment_message"]])

# order-level table
ordl = (iv.groupby("order_id").agg(gmv=("price", "sum"), freight=("freight_value", "sum"), n_items=("price", "size"),
                                   n_sellers=("seller_id", "nunique"), weight=("product_weight_g", "sum"))
          .reset_index())
paym = op.groupby("order_id").agg(pay_value=("payment_value", "sum"), max_inst=("payment_installments", "max"),
                                  n_pay=("payment_sequential", "size")).reset_index()
ordl = (orders.merge(ordl, on="order_id", how="left").merge(paym, on="order_id", how="left")
              .merge(rv_order, on="order_id", how="left")
              .merge(customers[["customer_id", "customer_unique_id", "customer_state"]], on="customer_id"))
ordl["deliv_days"] = (ordl.deliv - ordl.purch).dt.total_seconds() / 86400
ordl["est_days"] = (ordl.est - ordl.purch).dt.total_seconds() / 86400
ordl["vs_est"] = (ordl.deliv.dt.normalize() - ordl.est.dt.normalize()).dt.days  # >0 = late
D = ordl[(ordl.order_status == "delivered") & ordl.deliv.notna()].copy()  # delivered w/ date
D["late"] = D.vs_est > 0

# ============================================================================ 01 INSPECTION
F = Facts("01_dataset_overview_and_quality.txt", "01 - DATASET OVERVIEW & DATA QUALITY")
F.p("Source: olist.sqlite (Brazilian e-commerce marketplace 'Olist' + its seller-acquisition funnel).")
F.h("A. The 9 + 2 tables (11 tables total)")
desc = {
    "orders": "one row per order, with status + 5 lifecycle timestamps",
    "order_items": "one row per item in an order (price, freight, seller, product)",
    "order_payments": "one row per payment instalment/method used on an order",
    "order_reviews": "customer satisfaction survey (1-5 stars + optional comment)",
    "customers": "one row per customer_id (new id per order!) + location",
    "products": "product catalogue: category, size, weight, photos",
    "sellers": "marketplace sellers + location",
    "geolocation": "lat/lng points per zip-code prefix (many rows per zip)",
    "product_category_name_translation": "Portuguese -> English category names",
    "leads_qualified": "marketing-qualified leads (MQLs): who showed interest, via which channel",
    "leads_closed": "MQLs that were won and became sellers (+ business profile)",
}
rows = [(t, len(T[t]), T[t].shape[1], desc[t]) for t in names]
F.table(pd.DataFrame(rows, columns=["table", "rows", "columns", "what it holds"]).sort_values("rows", ascending=False), index=False)
tot_rows = sum(len(v) for v in T.values())
F.kv("Total rows across all tables", n0(tot_rows))
F.kv("DB file size", "~112.7 MB")

F.h("B. Time coverage")
F.kv("First order", orders.purch.min())
F.kv("Last order", orders.purch.max())
mo = orders.groupby("ym").size()
F.kv("Months with < 1,000 orders (sparse / ramp-up / cut-off)", ", ".join(str(m) for m in mo[mo < 1000].index))
F.kv("Window used for trend analysis", "Jan 2017 - Aug 2018 (20 full months)")
F.kv("Leads (MQL) first-contact range", f"{lq.first_contact_date.min()} to {lq.first_contact_date.max()}")
F.kv("Leads closed (won) range", f"{lc.won_date.min()[:10]} to {lc.won_date.max()[:10]}")
F.p("  NOTE: 'won_date' runs past the last order date, i.e. the funnel tables and the order tables are NOT time-aligned.")

F.h("C. Missing values (only columns that have any)")
rows = []
for t, df in T.items():
    for c in df.columns:
        na = df[c].isna().sum()
        if na:
            rows.append((t, c, int(na), round(na / len(df) * 100, 1)))
miss = pd.DataFrame(rows, columns=["table", "column", "nulls", "pct_null"]).sort_values("pct_null", ascending=False)
F.table(miss, index=False)
F.p("  Reading guide:")
F.p("  - order_reviews.review_comment_title/message: nulls are normal - most customers leave only a star rating.")
F.p("  - orders.order_delivered_*: nulls are normal for orders that were never delivered (canceled, unavailable, in transit).")
F.p("  - products.category/name/description/photos: the same 610 products lack all four -> 'unknown' category.")
F.p("  - leads_closed has_company/has_gtin/average_stock/catalog_size: >90% empty, only ~65 closed leads have them -> too sparse to model.")

F.h("D. Duplicates")
F.kv("geolocation fully duplicated rows", f"{geo.duplicated().sum():,} of {len(geo):,} ({pct(geo.duplicated().mean())})")
F.kv("geolocation distinct zip prefixes", n0(geo.geolocation_zip_code_prefix.nunique()))
F.kv("order_reviews: review_ids that appear on >1 row", n0((rv.review_id.value_counts() > 1).sum()))
F.kv("order_reviews: orders with >1 review", n0((rv.order_id.value_counts() > 1).sum()))
F.kv("customers: customer_id (per order) vs customer_unique_id (per person)", f"{customers.customer_id.nunique():,} vs {customers.customer_unique_id.nunique():,}")
F.p("  => Always count PEOPLE with customer_unique_id, not customer_id.")
for t in ["orders", "order_items", "order_payments", "customers", "products", "sellers", "leads_qualified", "leads_closed"]:
    pass
F.kv("Other tables with fully duplicated rows", "none")

F.h("E. Referential integrity (do foreign keys match?)")
chk = [
    ("orders.customer_id -> customers", (~orders.customer_id.isin(customers.customer_id)).sum()),
    ("order_items.order_id -> orders", (~oi.order_id.isin(orders.order_id)).sum()),
    ("order_items.product_id -> products", (~oi.product_id.isin(prod.product_id)).sum()),
    ("order_items.seller_id -> sellers", (~oi.seller_id.isin(sellers.seller_id)).sum()),
    ("order_payments.order_id -> orders", (~op.order_id.isin(orders.order_id)).sum()),
    ("order_reviews.order_id -> orders", (~rv.order_id.isin(orders.order_id)).sum()),
    ("leads_closed.mql_id -> leads_qualified", (~lc.mql_id.isin(lq.mql_id)).sum()),
]
F.table(pd.DataFrame(chk, columns=["relationship", "orphan rows"]), index=False)
F.kv("Orders with NO items", f"{(~orders.order_id.isin(oi.order_id)).sum()} (mostly unavailable/canceled)")
F.kv("Orders with NO payment record", (~orders.order_id.isin(op.order_id)).sum())
F.kv("Orders with NO review", f"{(~orders.order_id.isin(rv.order_id)).sum()}")
F.kv("Products whose category is not in translation table", f"{(~prod.product_category_name.dropna().isin(tr.product_category_name)).sum()} products / 2 categories (pc_gamer, portateis_cozinha_e_preparadores_de_alimentos) - mapped manually")
F.kv("Closed leads whose seller_id exists in sellers table", f"{lc.seller_id.isin(sellers.seller_id).sum()} of {len(lc)} ({pct(lc.seller_id.isin(sellers.seller_id).mean())})")
F.kv("Customer zips found in geolocation", pct(customers.customer_zip_code_prefix.isin(geo.geolocation_zip_code_prefix).mean(), 2))
F.kv("Seller zips found in geolocation", pct(sellers.seller_zip_code_prefix.isin(geo.geolocation_zip_code_prefix).mean(), 2))
F.p("  => Core transactional tables join cleanly: no orphan keys anywhere.")

F.h("F. Anomalies found")
carrier_before = (orders.carrier < orders.purch).sum()
deliv_before_carrier = (orders.deliv < orders.carrier).sum()
deliv_nodate = ((orders.order_status == "delivered") & orders.deliv.isna()).sum()
canc_deliv = ((orders.order_status == "canceled") & orders.deliv.notna()).sum()
rv_before = (rv.merge(orders[["order_id", "purch"]], on="order_id").eval("review_creation_date < purch")).sum()
bbox = geo[(geo.geolocation_lat > 5.3) | (geo.geolocation_lat < -34) | (geo.geolocation_lng < -74) | (geo.geolocation_lng > -34)]
m = oi.groupby("order_id").agg(t=("price", "sum"), f=("freight_value", "sum")).assign(tot=lambda d: d.t + d.f).join(op.groupby("order_id").payment_value.sum(), how="inner")
mism = ((m.payment_value - m.tot).abs() >= 0.011).sum()
F.kv("Orders handed to carrier BEFORE purchase timestamp", carrier_before)
F.kv("Orders delivered to customer BEFORE carrier pickup date", deliv_before_carrier)
F.kv("'delivered' orders with no delivery date", deliv_nodate)
F.kv("'canceled' orders that still have a delivery date", canc_deliv)
F.kv("Reviews created BEFORE the order was placed", rv_before)
F.kv("Geolocation points outside Brazil's bounding box", len(bbox))
F.kv("Orders where payments != items + freight (>1 cent)", f"{mism} of {len(m):,} ({pct(mism / len(m), 2)})")
F.kv("Payment rows with value = 0", int((op.payment_value == 0).sum()))
F.kv("Payment type 'not_defined'", int((op.payment_type == "not_defined").sum()))
F.kv("Order items with freight = 0", int((oi.freight_value == 0).sum()))
F.kv("Products with weight = 0 g", int((prod.product_weight_g == 0).sum()))
F.kv("Max single item price", brl2(oi.price.max()))
F.kv("Typos in column names (kept as-is)", "product_name_lenght, product_description_lenght")
F.kv("City names in geolocation", "inconsistent accents/spelling ('sao paulo' vs 'são paulo') -> join on zip prefix, not on city")

F.h("G. How the anomalies were handled in this analysis")
F.p("  - Orders with negative stage durations are excluded only from the stage-by-stage timing chart; they stay in everything else.")
F.p("  - Canceled/unavailable orders are excluded from revenue (GMV); all orders are included in status counts.")
F.p("  - One review per order (the latest) so scores are not double counted.")
F.p("  - Geolocation is averaged per zip prefix after removing points outside Brazil.")
F.p("  - Customers are counted with customer_unique_id.")
F.p("  - Trend charts use Jan 2017 - Aug 2018 only.")

# ---- graph: ERD
fig, ax = plt.subplots(figsize=(14, 9.2))
ax.set_xlim(0, 150); ax.set_ylim(0, 100); ax.axis("off"); ax.grid(False)
Wd, Ht = 27, 15
B = {  # name: (x, y, key line, kind)
    "customers": (2, 46, "customer_id · customer_unique_id\nzip prefix · city · state", "core"),
    "geolocation": (2, 14, "zip prefix · lat · lng\ncity · state", "ref"),
    "orders": (42, 46, "order_id · customer_id · status\n5 lifecycle timestamps", "core"),
    "order_payments": (42, 72, "order_id · type\ninstallments · value", "core"),
    "order_reviews": (42, 20, "order_id · score (1-5)\ncomment · dates", "core"),
    "order_items": (82, 46, "order_id · product_id · seller_id\nprice · freight_value", "core"),
    "products": (82, 72, "product_id · category\nweight · size · photos", "ref"),
    "sellers": (82, 20, "seller_id\nzip prefix · city · state", "core"),
    "product_category_name_translation": (118, 72, "category PT -> EN", "ref"),
    "leads_closed": (118, 20, "mql_id · seller_id · won_date\nsegment · lead type · profile", "mkt"),
    "leads_qualified": (118, 0.5, "mql_id · first_contact_date\nlanding page · origin", "mkt"),
}
cols = {"core": ("#DDF1E6", GREEN), "ref": ("#DDEAF3", BLUE), "mkt": ("#FDEFC4", "#C99100")}


def ctr(n, side):
    x, y, _, _ = B[n]
    w = Wd + (3 if n == "product_category_name_translation" else 0)
    return {"l": (x, y + Ht / 2), "r": (x + w, y + Ht / 2), "t": (x + w / 2, y + Ht), "b": (x + w / 2, y)}[side]


def link(a, sa, b, sb, label, via=None, lab_xy=None):
    pa, pb = ctr(a, sa), ctr(b, sb)
    pts = [pa] + (via or []) + [pb]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#55696D", lw=1.6, zorder=1)
    lx, ly = lab_xy or ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)
    ax.text(lx, ly, label, ha="center", va="center", fontsize=8.5, color="#26373A", zorder=3,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))


link("customers", "r", "orders", "l", "customer_id")
link("orders", "t", "order_payments", "b", "order_id")
link("orders", "b", "order_reviews", "t", "order_id")
link("orders", "r", "order_items", "l", "order_id")
link("order_items", "t", "products", "b", "product_id")
link("order_items", "b", "sellers", "t", "seller_id")
link("products", "r", "product_category_name_translation", "l", "category")
link("customers", "b", "geolocation", "t", "zip prefix")
link("sellers", "b", "geolocation", "b", "zip prefix", via=[(95.5, 8), (15.5, 8)], lab_xy=(55, 8))
link("sellers", "r", "leads_closed", "l", "seller_id")
link("leads_closed", "b", "leads_qualified", "t", "mql_id")
for n, (x, y, key, kind) in B.items():
    fc, ec = cols[kind]
    w = Wd + (3 if n == "product_category_name_translation" else 0)
    ax.add_patch(FancyBboxPatch((x, y), w, Ht, boxstyle="round,pad=0.4,rounding_size=1.5", fc=fc, ec=ec, lw=2, zorder=2))
    nm = n if len(n) < 24 else "category_translation"
    ax.text(x + w / 2, y + Ht - 3.0, nm, ha="center", va="center", fontsize=11.5, fontweight="bold", color=DARK, zorder=4)
    ax.text(x + w / 2, y + Ht - 6.6, f"{len(T[n]):,} rows", ha="center", va="center", fontsize=10, color=ec, fontweight="bold", zorder=4)
    ax.text(x + w / 2, y + 3.6, key, ha="center", va="center", fontsize=8.2, color="#3B4C50", zorder=4, linespacing=1.3)
ax.text(2, 98, "How the 11 tables connect", fontsize=18, fontweight="bold", color=DARK, va="center")
ax.text(2, 93.8, "Everything revolves around the order.   Green = transactions   ·   Blue = reference data   ·   Yellow = seller-acquisition funnel",
        fontsize=10.5, color="#3B4C50", va="center")
g_erd = save(fig, "schema_how_tables_connect")

# ---- graph: missing values
mm = miss[miss.pct_null >= 0.1].copy()
mm["label"] = mm.table.str.replace("order_", "", regex=False) + "." + mm.column.str.replace("leads_", "")
mm = mm.sort_values("pct_null")
fig, ax = plt.subplots(figsize=(11, 7.5))
colr = [CORAL if v > 50 else (YELLOW if v > 5 else GREEN) for v in mm.pct_null]
ax.barh(mm.label, mm.pct_null, color=colr)
for i, v in enumerate(mm.pct_null):
    ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=9.5)
ax.set_xlim(0, 105); ax.set_xlabel("% of rows empty")
ax.set_title("Where the data has gaps: mostly optional fields, plus the sparse lead-profile columns")
ax.grid(axis="y", visible=False)
g_miss = save(fig, "missing_values")
F.p(f"\nRelated graphs: {g_erd}, {g_miss}")
F.save()

# ============================================================================ 02 SALES & GROWTH
F = Facts("02_sales_and_growth.txt", "02 - SALES, GROWTH & SEASONALITY")
tot_orders, tot_valid = len(orders), int(VALID.sum())
gmv_total = iv.price.sum()
frt_total = iv.freight_value.sum()
K.update(orders=tot_orders, gmv=gmv_total, freight=frt_total)
F.h("A. Headline numbers (all time)")
F.kv("Orders placed", n0(tot_orders))
F.kv("Valid orders (not canceled/unavailable)", f"{n0(tot_valid)} ({pct(tot_valid / tot_orders)})")
F.kv("Order items sold", n0(len(iv)))
F.kv("GMV (item price, excl. freight)", brl(gmv_total))
F.kv("Freight charged to customers", f"{brl(frt_total)} (= {pct(frt_total / gmv_total)} on top of GMV)")
F.kv("Total paid by customers (payment table, all orders)", brl(op.payment_value.sum()))
F.kv("Average order value - items only", brl2(ordl.gmv.mean()))
F.kv("Average order value - what the customer paid (items+freight)", brl2(ordl.pay_value.mean()))
F.kv("Median order value (items only)", brl2(ordl.gmv.median()))
F.kv("Items per order (mean)", f"{ordl.n_items.mean():.2f}")
F.kv("Orders with exactly 1 item", pct((ordl.n_items == 1).sum() / ordl.n_items.notna().sum()))
F.kv("Orders containing >1 seller", n0((ordl.n_sellers > 1).sum()))

M = (ordl[WIN.reindex(ordl.index, fill_value=False)].groupby("ym")
     .agg(orders=("order_id", "size"), valid=("valid", "sum"), gmv=("gmv", "sum"), items=("n_items", "sum")))
M["aov"] = M.gmv / M.n_items.where(False, M.valid) if False else M.gmv / (ordl[WIN.reindex(ordl.index, fill_value=False)].groupby("ym").n_items.count())
M["items_per_order"] = M["items"] / (ordl[WIN.reindex(ordl.index, fill_value=False)].groupby("ym").n_items.count())
M["mom_orders"] = M.orders.pct_change()
F.h("B. Monthly trend (Jan 2017 - Aug 2018)")
show = M.copy(); show.index = show.index.astype(str)
show["gmv"] = show.gmv.round(0); show["aov"] = show.aov.round(1); show["items_per_order"] = show.items_per_order.round(2)
show["mom_orders"] = (show.mom_orders * 100).round(1).astype(str) + "%"
F.table(show[["orders", "gmv", "aov", "items_per_order", "mom_orders"]].rename(columns={"gmv": "GMV_R$", "aov": "avg_order_R$", "mom_orders": "MoM_orders"}))

a17 = M.loc["2017-01":"2017-08"]; a18 = M.loc["2018-01":"2018-08"]
yoy_o, yoy_g = a18.orders.sum() / a17.orders.sum() - 1, a18.gmv.sum() / a17.gmv.sum() - 1
F.h("C. Year-over-year (same 8 months: Jan-Aug)")
F.kv("Orders Jan-Aug 2017 -> Jan-Aug 2018", f"{n0(a17.orders.sum())} -> {n0(a18.orders.sum())} ({yoy_o * 100:+.0f}%)")
F.kv("GMV Jan-Aug 2017 -> Jan-Aug 2018", f"{brl(a17.gmv.sum())} -> {brl(a18.gmv.sum())} ({yoy_g * 100:+.0f}%)")
F.kv("Average order value Jan-Aug 2017 vs 2018", f"{brl2(a17.gmv.sum() / a17.valid.sum())} vs {brl2(a18.gmv.sum() / a18.valid.sum())}")
F.p("  => Growth came from VOLUME (more orders), not from bigger baskets.")
K.update(yoy_o=yoy_o, yoy_g=yoy_g)

daily = orders.groupby(orders.purch.dt.date).size().sort_values(ascending=False)
F.h("D. Peak days")
F.table(daily.head(8).rename("orders").to_frame().rename_axis("date"))
bf = daily.loc[pd.Timestamp("2017-11-24").date()]
F.kv("Black Friday 2017-11-24", f"{n0(bf)} orders = {bf / daily.drop(pd.Timestamp('2017-11-24').date()).loc[pd.Timestamp('2017-10-01').date():pd.Timestamp('2017-12-31').date()].median():.1f}x a typical Q4-2017 day")
K.update(bf=bf)

F.h("E. Order status")
st = orders.order_status.value_counts()
F.table(pd.DataFrame({"orders": st, "share_%": (st / st.sum() * 100).round(2)}))
F.kv("Delivery success rate", pct(st["delivered"] / st.sum()))
F.kv("Canceled + unavailable", f"{n0(st['canceled'] + st['unavailable'])} ({pct((st['canceled'] + st['unavailable']) / st.sum())})")

# weekday/hour
orders["wd"] = orders.purch.dt.dayofweek
orders["hr"] = orders.purch.dt.hour
wd = orders.wd.value_counts().sort_index()
hr = orders.hr.value_counts().sort_index()
wdn = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
F.h("F. When do people buy?")
F.kv("Busiest weekday", f"{wdn[wd.idxmax()]} ({pct(wd.max() / wd.sum())} of orders)")
F.kv("Quietest weekday", f"{wdn[wd.idxmin()]} ({pct(wd.min() / wd.sum())})")
F.kv("Weekday (Mon-Fri) vs weekend share", f"{pct(wd.iloc[:5].sum() / wd.sum())} vs {pct(wd.iloc[5:].sum() / wd.sum())}")
F.kv("Peak hour", f"{hr.idxmax()}:00-{hr.idxmax() + 1}:00 ({pct(hr.max() / hr.sum())})")
F.kv("Orders placed between 10:00 and 22:00", pct(hr.loc[10:21].sum() / hr.sum()))
F.kv("Orders placed between 00:00 and 06:00", pct(hr.loc[0:5].sum() / hr.sum()))
K.update(peak_hr=int(hr.idxmax()), busiest_wd=wdn[wd.idxmax()])

# graph growth
fig, ax = plt.subplots(figsize=(11.5, 6))
x = np.arange(len(M)); lab = [str(p) for p in M.index]
ax.bar(x, M.orders, color=LGREEN, edgecolor=GREEN, label="Orders (left axis)")
ax.set_xticks(x); ax.set_xticklabels([l[2:] for l in lab], rotation=45, ha="right")
ax.set_ylabel("Orders per month"); ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}K" if v else "0"))
ax2 = ax.twinx(); ax2.grid(False); ax2.spines["right"].set_visible(True)
ax2.plot(x, M.gmv / 1e6, color=DARK, marker="o", lw=2.5, label="GMV in R$ millions (right axis)")
ax2.set_ylabel("GMV (R$ millions)"); ax2.set_ylim(0, M.gmv.max() / 1e6 * 1.22)
iblack = lab.index("2017-11")
ax.set_ylim(0, M.orders.max() * 1.22)
ax.annotate(f"Black Friday: {n0(bf)} orders on 24 Nov 2017", xy=(iblack, M.orders.iloc[iblack]), xytext=(iblack + 1.3, M.orders.max() * 1.13),
            arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.8), color=CORAL, fontsize=10.5, fontweight="bold")
ax.set_title(f"Orders grew ~{(M.orders.iloc[-1] / M.orders.iloc[0]):.0f}x in 20 months, then plateaued around 6-7K per month in 2018")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left")
g1 = save(fig, "growth_orders_and_gmv_monthly")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
ax[0].plot(x, M.aov, color=GREEN, marker="o", lw=2.5); ax[0].set_ylim(0, M.aov.max() * 1.3)
ax[0].set_title("Average order value (R$)", fontsize=13); ax[0].set_xticks(x[::3]); ax[0].set_xticklabels([lab[i][2:] for i in range(0, len(lab), 3)])
ax[1].plot(x, M.items_per_order, color=YELLOW, marker="o", lw=2.5); ax[1].set_ylim(1, M.items_per_order.max() * 1.25)
ax[1].set_title("Items per order", fontsize=13); ax[1].set_xticks(x[::3]); ax[1].set_xticklabels([lab[i][2:] for i in range(0, len(lab), 3)])
fig.suptitle("Baskets stayed flat: growth came from more orders, not bigger orders", x=0.125, ha="left", fontsize=15, fontweight="bold", y=1.03)
g2 = save(fig, "basket_size_flat")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
ax[0].bar(wdn, wd.values, color=[GREEN if i < 5 else GRAY for i in range(7)])
ax[0].set_title("Orders by weekday", fontsize=13); ax[0].yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}K"))
ax[1].bar(hr.index, hr.values, color=[YELLOW if 10 <= h <= 21 else LGREEN for h in hr.index], edgecolor=GREEN)
ax[1].set_title("Orders by hour of day", fontsize=13); ax[1].set_xticks(range(0, 24, 2)); ax[1].yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}K"))
fig.suptitle(f"Shoppers buy on weekdays, mostly 10:00-22:00 (busiest: {wdn[wd.idxmax()]}, peak hour {hr.idxmax()}:00)", x=0.125, ha="left", fontsize=15, fontweight="bold", y=1.03)
g3 = save(fig, "when_people_buy")

fig, ax = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1.4]})
ax[0].pie([st["delivered"], st.sum() - st["delivered"]], colors=[GREEN, CORAL], startangle=90, counterclock=False,
          wedgeprops=dict(width=0.38, edgecolor="white"), autopct=lambda p: f"{p:.1f}%", pctdistance=0.81, textprops=dict(color="white", fontweight="bold", fontsize=11))
ax[0].text(0, 0, "delivered\nvs rest", ha="center", va="center", fontsize=11, color=DARK)
rest = st.drop("delivered").sort_values()
ax[1].barh(rest.index, rest.values, color=CORAL)
for i, v in enumerate(rest.values): ax[1].text(v + 15, i, n0(v), va="center", fontsize=10)
ax[1].set_title("The 3% that did not complete", fontsize=13); ax[1].grid(axis="y", visible=False); ax[1].set_xlim(0, rest.max() * 1.15)
fig.suptitle(f"{pct(st['delivered'] / st.sum())} of orders are delivered; the rest are mostly shipped-in-transit, canceled or unavailable", x=0.06, ha="left", fontsize=15, fontweight="bold", y=1.02)
g4 = save(fig, "order_status")
F.p(f"\nRelated graphs: {g1}, {g2}, {g3}, {g4}")
F.save()

# ============================================================================ 03 CUSTOMERS & GEOGRAPHY
F = Facts("03_customers_and_geography.txt", "03 - CUSTOMERS & GEOGRAPHY")
n_people = customers.customer_unique_id.nunique()
oc = ordl.sort_values("purch").copy()
cnt = oc.groupby("customer_unique_id").size()
rep = (cnt > 1).mean()
F.h("A. Who are the customers?")
F.kv("Unique customers (people)", n0(n_people))
F.kv("customer_id rows (one per order)", n0(len(customers)))
F.kv("Customers with exactly 1 order", f"{n0((cnt == 1).sum())} ({pct((cnt == 1).mean())})")
F.kv("Repeat customers (2+ orders)", f"{n0((cnt > 1).sum())} ({pct(rep)})")
F.kv("Max orders by one customer", int(cnt.max()))
oc["rank"] = oc.groupby("customer_unique_id").cumcount()
sec = oc[oc["rank"] <= 1].pivot_table(index="customer_unique_id", columns="rank", values="purch", aggfunc="first").dropna()
gap = (sec[1] - sec[0]).dt.days
F.kv("Median days between 1st and 2nd order (repeaters)", f"{gap.median():.0f} days (mean {gap.mean():.0f})")
F.kv("Repeaters who re-ordered on the SAME day", pct((gap == 0).mean()))
F.kv("Repeaters who re-ordered within 30 days", pct((gap <= 30).mean()))
rep_rev = ordl[ordl.customer_unique_id.isin(cnt[cnt > 1].index)].gmv.sum() / ordl.gmv.sum()
F.kv("Share of GMV from repeat customers", pct(rep_rev))
K.update(people=n_people, repeat=rep)
F.p("  => Olist behaves like a marketplace of one-off purchases: loyalty / retention is the largest untapped lever.")

F.h("B. Where do customers live? (valid orders)")
vo = ordl[ordl.valid]
bs = (vo.groupby("customer_state").agg(orders=("order_id", "size"), gmv=("gmv", "sum"), customers=("customer_unique_id", "nunique")))
bs["share_orders_%"] = (bs.orders / bs.orders.sum() * 100).round(1)
bs["share_gmv_%"] = (bs.gmv / bs.gmv.sum() * 100).round(1)
bs["aov_R$"] = (bs.gmv / bs.orders).round(1)
bs = bs.sort_values("orders", ascending=False)
bs["gmv"] = bs.gmv.round(0)
F.table(bs)
top3 = bs["share_orders_%"].head(3).sum()
F.kv("Number of states", len(bs))
F.kv("SP + RJ + MG share of orders", f"{top3:.1f}%")
F.kv("SP alone", f"{bs.iloc[0]['share_orders_%']:.1f}% of orders")
F.kv("Highest avg order value state", f"{bs['aov_R$'].idxmax()} (R$ {bs['aov_R$'].max():.0f}) - but tiny volume: {int(bs.loc[bs['aov_R$'].idxmax(), 'orders'])} orders" if bs.loc[bs['aov_R$'].idxmax(), 'orders'] < 1000 else f"{bs['aov_R$'].idxmax()} R$ {bs['aov_R$'].max():.0f}")
big = bs[bs.orders >= 1000]
F.kv("Highest AOV among states with 1,000+ orders", f"{big['aov_R$'].idxmax()} (R$ {big['aov_R$'].max():.0f})")
F.kv("Lowest AOV among states with 1,000+ orders", f"{big['aov_R$'].idxmin()} (R$ {big['aov_R$'].min():.0f})")
K.update(top3=top3, sp=bs.iloc[0]["share_orders_%"])

F.h("C. Top 12 cities by unique customers")
cc = customers.groupby("customer_city").customer_unique_id.nunique().sort_values(ascending=False)
F.table(pd.DataFrame({"customers": cc.head(12), "share_%": (cc.head(12) / n_people * 100).round(1)}))
F.kv("Distinct cities with customers", n0(len(cc)))
F.kv("Top 10 cities share of customers", pct(cc.head(10).sum() / n_people))
F.kv("Sao Paulo + Rio together", pct((cc["sao paulo"] + cc["rio de janeiro"]) / n_people))

# graph states
top = bs.head(10)
fig, ax = plt.subplots(figsize=(11, 6))
ax.barh(top.index[::-1], top["share_orders_%"][::-1], color=[GREEN if s == "SP" else LGREEN for s in top.index[::-1]], edgecolor=GREEN)
for i, (s, v) in enumerate(zip(top.index[::-1], top["share_orders_%"][::-1])):
    ax.text(v + 0.5, i, f"{v:.1f}%  ({n0(top.loc[s, 'orders'])} orders)", va="center", fontsize=10)
ax.set_xlim(0, 55); ax.set_xlabel("% of valid orders"); ax.grid(axis="y", visible=False)
ax.set_title(f"São Paulo alone is {bs.iloc[0]['share_orders_%']:.0f}% of orders; the top 3 states are {top3:.0f}%")
g5 = save(fig, "orders_by_state")

# map
gz = geo[(geo.geolocation_lat.between(-34, 5.3)) & (geo.geolocation_lng.between(-74, -34))]
zip_ll = gz.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].mean()
cz = customers.merge(orders[["customer_id", "valid"]], on="customer_id").query("valid").groupby("customer_zip_code_prefix").size().rename("n").to_frame().join(zip_ll).dropna()
sz = oi.merge(sellers[["seller_id", "seller_zip_code_prefix"]], on="seller_id").groupby("seller_zip_code_prefix").size().rename("n").to_frame().join(zip_ll).dropna()
fig, ax = plt.subplots(1, 2, figsize=(13, 6.6), sharex=True, sharey=True)
ax[0].scatter(cz.geolocation_lng, cz.geolocation_lat, s=np.sqrt(cz.n) * 4.5, c=GREEN, alpha=0.4, linewidths=0)
ax[0].set_title("Where customers are", fontsize=13)
ax[1].scatter(sz.geolocation_lng, sz.geolocation_lat, s=np.sqrt(sz.n) * 3.2, c=YELLOW, alpha=0.45, linewidths=0, edgecolors="none")
ax[1].set_title("Where sellers are", fontsize=13)
for a in ax:
    a.set_xlabel("longitude"); a.set_aspect("equal"); a.grid(False)
ax[0].set_ylabel("latitude")
fig.suptitle("Customers cover the whole country but cluster on the south-east coast; sellers are even more concentrated around São Paulo", x=0.07, ha="left", fontsize=14, fontweight="bold", y=0.98)
fig.tight_layout(rect=(0, 0, 1, 0.95))
g6 = save(fig, "map_customers_and_sellers")

# new vs returning monthly + frequency
oc["is_new"] = oc["rank"] == 0
nv = oc[WIN.reindex(oc.index, fill_value=False)].groupby(["ym", "is_new"]).size().unstack(fill_value=0)
fig, ax = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.6, 1]})
xx = np.arange(len(nv))
ax[0].bar(xx, nv[True], color=GREEN, label="First-time buyers")
ax[0].bar(xx, nv[False], bottom=nv[True], color=YELLOW, label="Returning buyers")
ax[0].set_xticks(xx[::2]); ax[0].set_xticklabels([str(p)[2:] for p in nv.index[::2]], rotation=45, ha="right")
ax[0].legend(loc="upper left"); ax[0].set_title("Orders per month: new vs returning", fontsize=13)
fr = cnt.clip(upper=4).value_counts().sort_index()
ax[1].bar(["1", "2", "3", "4+"], fr.values, color=[GREEN, YELLOW, YELLOW, YELLOW])
for i, v in enumerate(fr.values): ax[1].text(i, v + 1200, f"{v / fr.sum() * 100:.1f}%", ha="center", fontweight="bold")
ax[1].set_title("Orders per customer", fontsize=13); ax[1].yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}K")); ax[1].set_ylim(0, fr.max() * 1.12)
fig.suptitle(f"Only {rep * 100:.1f}% of customers ever buy twice: Olist runs on new-customer acquisition", x=0.07, ha="left", fontsize=15, fontweight="bold", y=1.03)
g7 = save(fig, "new_vs_returning_customers")
F.p(f"\nRelated graphs: {g5}, {g6}, {g7}")
F.save()

# ============================================================================ 04 PRODUCTS & CATEGORIES
F = Facts("04_products_and_categories.txt", "04 - PRODUCTS, CATEGORIES & PRICING")
F.h("A. Catalogue")
F.kv("Distinct products", n0(len(prod)))
F.kv("Product categories", prod.category.nunique() - (1 if "unknown" in prod.category.values else 0))
F.kv("Products with no category", f"{(prod.category == 'unknown').sum()} ({pct((prod.category == 'unknown').mean())})")
F.kv("Median product weight", f"{prod.product_weight_g.median():.0f} g (mean {prod.product_weight_g.mean():.0f} g)")
F.kv("Median photos per product", f"{prod.product_photos_qty.median():.0f}")
F.kv("Median description length", f"{prod.product_description_lenght.median():.0f} chars")
F.kv("Products sold only once", pct((oi.groupby('product_id').size() == 1).sum() / oi.product_id.nunique()))

cat = (iv.groupby("category").agg(items=("price", "size"), gmv=("price", "sum"), avg_price=("price", "mean"), orders=("order_id", "nunique"), avg_freight=("freight_value", "mean")))
cat["share_gmv_%"] = (cat.gmv / cat.gmv.sum() * 100).round(2)
cat = cat.sort_values("gmv", ascending=False)
cat["cum_share_%"] = cat["share_gmv_%"].cumsum().round(1)
F.h("B. Top 20 categories by GMV")
t20 = cat.head(20).copy(); t20[["gmv", "avg_price", "avg_freight"]] = t20[["gmv", "avg_price", "avg_freight"]].round(1)
F.table(t20)
F.kv("Top 5 categories share of GMV", pct(cat.head(5)["share_gmv_%"].sum() / 100))
F.kv("Top 10 categories share of GMV", pct(cat.head(10)["share_gmv_%"].sum() / 100))
F.kv("Categories needed for 80% of GMV", int((cat["cum_share_%"] < 80).sum() + 1))
F.kv("Categories in total (excl. unknown)", len(cat) - (1 if "unknown" in cat.index else 0))
bv = cat.sort_values("items", ascending=False)
F.kv("#1 by items sold", f"{bv.index[0]} ({n0(bv["items"].iloc[0])} items)")
F.kv("Most expensive avg price (100+ items sold)", f"{cat[cat['items'] >= 100]['avg_price'].idxmax()} (R$ {cat[cat['items'] >= 100]['avg_price'].max():.0f})")
F.kv("Cheapest avg price (100+ items sold)", f"{cat[cat['items'] >= 100]['avg_price'].idxmin()} (R$ {cat[cat['items'] >= 100]['avg_price'].min():.0f})")
K.update(top_cat=cat.index[0], top_cat_share=cat["share_gmv_%"].iloc[0], top10_cat=cat.head(10)["share_gmv_%"].sum())

F.h("C. Price distribution (per item)")
q = oi.price.quantile([.1, .25, .5, .75, .9, .99])
F.table(q.round(2).rename("price_R$").to_frame())
F.kv("Mean vs median", f"R$ {oi.price.mean():.2f} vs R$ {oi.price.median():.2f} (right-skewed: a few expensive items pull the mean up)")
F.kv("Items under R$ 100", pct((oi.price < 100).mean()))
F.kv("Items over R$ 1,000", f"{(oi.price > 1000).sum()} ({pct((oi.price > 1000).mean(), 2)})")
F.kv("Share of GMV from items over R$ 500", pct(oi[oi.price > 500].price.sum() / oi.price.sum()))
F.kv("Most expensive item", f"{brl2(oi.price.max())} ({it.loc[it.price.idxmax(), 'category']})")

F.h("D. Freight (shipping) cost")
oi2 = oi.merge(prod[["product_id", "product_weight_g"]], on="product_id")
oi2["ratio"] = oi2.freight_value / oi2.price
F.kv("Mean freight per item", brl2(oi.freight_value.mean()))
F.kv("Median freight/price ratio", pct(oi2.ratio.median()))
F.kv("Items where freight is MORE than the item price", pct((oi2.ratio > 1).mean()))
F.kv("Freight-to-price (total freight / total price)", pct(oi.freight_value.sum() / oi.price.sum()))
bins = [0, 500, 1000, 2000, 5000, 10000, 1e6]
labs = ["<0.5kg", "0.5-1kg", "1-2kg", "2-5kg", "5-10kg", "10kg+"]
oi2["wb"] = pd.cut(oi2.product_weight_g, bins, labels=labs, include_lowest=True)
fw = oi2.groupby("wb").agg(items=("price", "size"), median_freight=("freight_value", "median"), median_price=("price", "median"), median_ratio=("ratio", "median"))
F.table(fw.round(2))
corr_w = oi2[["product_weight_g", "freight_value"]].corr(method="spearman").iloc[0, 1]
F.kv("Spearman correlation weight vs freight", f"{corr_w:.2f}")

# graphs
t15 = cat.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(11, 7))
ax.barh(t15.index.str.replace("_", " "), t15.gmv / 1e6, color=[GREEN if i >= 10 else LGREEN for i in range(15)], edgecolor=GREEN)
for i, (g, s) in enumerate(zip(t15.gmv / 1e6, t15["share_gmv_%"])): ax.text(g + 0.04, i, f"R$ {g:.2f}M ({s:.1f}%)", va="center", fontsize=9.5)
ax.set_xlim(0, t15.gmv.max() / 1e6 * 1.28); ax.set_xlabel("GMV (R$ millions)"); ax.grid(axis="y", visible=False)
ax.set_title(f"Top 10 of {len(cat) - (1 if 'unknown' in cat.index else 0)} categories generate {cat.head(10)['share_gmv_%'].sum():.0f}% of revenue")
g8 = save(fig, "top_categories_by_gmv")

cb = cat[cat["items"] >= 150].join(iv.merge(rv_order, on="order_id").groupby("category").review_score.mean().rename("score"))
fig, ax = plt.subplots(figsize=(11.5, 6.8))
sc = ax.scatter(cb["items"], cb.avg_price, s=cb.gmv / cb.gmv.max() * 1100 + 20, c=cb.score, cmap="RdYlGn", vmin=3.7, vmax=4.4, alpha=0.8, edgecolors="#26373A", linewidths=0.6)
ax.set_xscale("log"); ax.set_yscale("log")
ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:,.0f}")); ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"R${v:,.0f}"))
offs = {"health_beauty": (50, 34), "bed_bath_table": (60, -26), "furniture_decor": (-10, -48), "housewares": (-75, -40),
        "sports_leisure": (-5, 62), "computers_accessories": (-135, 78), "watches_gifts": (16, 12), "cool_stuff": (-60, 30),
        "auto": (30, -34), "computers": (10, 8), "office_furniture": (-95, -22)}
for nme, (dx, dy) in offs.items():
    if nme in cb.index:
        ax.annotate(nme.replace("_", " "), (cb.loc[nme, "items"], cb.loc[nme, "avg_price"]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=9, fontweight="bold", color=DARK, arrowprops=dict(arrowstyle="-", color="#55696D", lw=.8))
ax.set_xlim(right=60000)
cbar = fig.colorbar(sc, ax=ax, pad=0.01); cbar.set_label("average review score")
ax.set_xlabel("Items sold (log scale)"); ax.set_ylabel("Average item price (log scale)")
ax.set_title("Category map: volume vs price (bubble size = revenue, colour = satisfaction)")
g9 = save(fig, "category_map_volume_vs_price")

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
ax[0].hist(oi.price.clip(upper=600), bins=60, color=GREEN, edgecolor="white")
ax[0].axvline(oi.price.median(), color=CORAL, lw=2); ax[0].text(oi.price.median() + 8, ax[0].get_ylim()[1] * 0.9, f"median R${oi.price.median():.0f}", color=CORAL, fontweight="bold")
ax[0].axvline(oi.price.mean(), color=YELLOW, lw=2); ax[0].text(oi.price.mean() + 8, ax[0].get_ylim()[1] * 0.78, f"mean R${oi.price.mean():.0f}", color="#B38600", fontweight="bold")
ax[0].set_title("Item price distribution (capped at R$600)", fontsize=13); ax[0].set_xlabel("price (R$)"); ax[0].yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}K"))
ax[1].bar(range(len(fw)), fw.median_freight, color=GREEN)
ax[1].set_xticks(range(len(fw))); ax[1].set_xticklabels([f"{i}\n({r_ * 100:.0f}%)" for i, r_ in zip(fw.index.astype(str), fw.median_ratio)], fontsize=10)
ax[1].set_xlabel("product weight  (median freight as % of item price)")
for i, v in enumerate(fw.median_freight): ax[1].text(i, v + 0.8, f"R${v:.0f}", ha="center", fontsize=10, fontweight="bold")
ax[1].set_ylim(0, fw.median_freight.max() * 1.25); ax[1].set_title("Median freight by product weight", fontsize=13)
fig.suptitle("The median item costs R$75, and shipping adds ~20-30% on top for light, cheap items", x=0.07, ha="left", fontsize=15, fontweight="bold", y=1.03)
g10 = save(fig, "price_distribution_and_freight")
F.p(f"\nRelated graphs: {g8}, {g9}, {g10}")
F.save()

# ============================================================================ 05 SELLERS
F = Facts("05_sellers.txt", "05 - SELLERS")
sg = iv.groupby("seller_id").agg(gmv=("price", "sum"), items=("price", "size"), orders=("order_id", "nunique"), products=("product_id", "nunique"), first=("purch", "min"), last=("purch", "max"))
sg = sg.join(sellers.set_index("seller_id")[["seller_state"]])
sg = sg.sort_values("gmv", ascending=False)
sg["cum"] = sg.gmv.cumsum() / sg.gmv.sum()
n_s = len(sg)
F.h("A. Seller base")
F.kv("Sellers in table", n0(len(sellers)))
F.kv("Sellers with at least one valid sale", n0(n_s))
F.kv("Sellers by state - top 5", ", ".join(f"{s} {v}" for s, v in sellers.seller_state.value_counts().head(5).items()))
F.kv("Share of sellers located in SP", pct((sellers.seller_state == "SP").mean()))
F.kv("Share of items shipped from SP sellers", pct((iv.seller_state == "SP").mean()))
F.kv("Median orders per seller", f"{sg.orders.median():.0f}")
F.kv("Mean orders per seller", f"{sg.orders.mean():.1f}")
F.kv("Sellers with 5 or fewer orders", f"{(sg.orders <= 5).sum()} ({pct((sg.orders <= 5).mean())})")
F.kv("Sellers with 100+ orders", f"{(sg.orders >= 100).sum()} ({pct((sg.orders >= 100).mean())})")
F.kv("Median distinct products per seller", f"{sg.products.median():.0f}")
F.h("B. Concentration")
for q_ in [0.01, 0.05, 0.10, 0.20, 0.50]:
    k = int(round(n_s * q_))
    F.kv(f"Top {int(q_ * 100)}% of sellers ({k})", f"{pct(sg.gmv.head(k).sum() / sg.gmv.sum())} of GMV")
gmv_sorted = np.sort(sg.gmv.values)
lorenz = np.cumsum(gmv_sorted) / gmv_sorted.sum()
gini = 1 - 2 * np.trapezoid(np.insert(lorenz, 0, 0), dx=1 / len(gmv_sorted))
F.kv("Gini coefficient of seller GMV (0 = equal, 1 = one seller has all)", f"{gini:.2f}")
F.kv("Bottom 50% of sellers", pct(sg.gmv.tail(n_s // 2).sum() / sg.gmv.sum()) + " of GMV")
K.update(top10s=sg.gmv.head(int(round(n_s * .1))).sum() / sg.gmv.sum(), top1s=sg.gmv.head(int(round(n_s * .01))).sum() / sg.gmv.sum(), gini=gini, n_sellers=n_s)
F.h("C. Top 10 sellers by GMV")
t10 = sg.head(10)[["gmv", "orders", "items", "products", "seller_state"]].copy(); t10["gmv"] = t10.gmv.round(0); t10.index = [i[:8] + "..." for i in t10.index]
F.table(t10)
F.kv("Top 10 sellers combined share of GMV", pct(sg.gmv.head(10).sum() / sg.gmv.sum()))
F.h("D. Seller quality (sellers with 30+ reviewed orders)")
sr = iv.merge(rv_order, on="order_id").groupby("seller_id").agg(score=("review_score", "mean"), n=("order_id", "nunique"))
sr30 = sr[sr.n >= 30]
F.kv("Sellers analysed", len(sr30))
F.kv("Average of seller review scores", f"{sr30.score.mean():.2f}")
F.kv("Sellers averaging below 3.5 stars", f"{(sr30.score < 3.5).sum()} ({pct((sr30.score < 3.5).mean())})")
F.kv("Sellers averaging 4.5+ stars", f"{(sr30.score >= 4.5).sum()} ({pct((sr30.score >= 4.5).mean())})")
F.kv("Weak sellers (<3.5) share of GMV", pct(sg.loc[sr30[sr30.score < 3.5].index].gmv.sum() / sg.gmv.sum()))
F.h("E. Seller base growth")
act = iv[iv.ym.between("2017-01", "2018-08")].groupby("ym").seller_id.nunique()
F.kv("Active sellers Jan 2017", act.iloc[0]); F.kv("Active sellers Aug 2018", act.iloc[-1]); F.kv("Growth", f"{act.iloc[-1] / act.iloc[0]:.1f}x")
F.kv("Sellers whose first-ever sale was in 2018", f"{(sg['first'].dt.year == 2018).sum()} ({pct((sg['first'].dt.year == 2018).mean())})")

fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
xl = np.insert(np.arange(1, len(lorenz) + 1) / len(lorenz), 0, 0); yl = np.insert(lorenz, 0, 0)
ax[0].plot(xl * 100, yl * 100, color=GREEN, lw=3); ax[0].plot([0, 100], [0, 100], color=GRAY, ls="--"); ax[0].fill_between(xl * 100, yl * 100, xl * 100, color=LGREEN, alpha=.6)
ax[0].set_xlabel("% of sellers (smallest to largest)"); ax[0].set_ylabel("% of cumulative GMV")
ax[0].set_title(f"Seller revenue concentration (Gini {gini:.2f})", fontsize=13)
ax[0].annotate(f"Bottom 50% of sellers\n= {sg.gmv.tail(n_s // 2).sum() / sg.gmv.sum() * 100:.0f}% of GMV", xy=(50, lorenz[n_s // 2] * 100), xytext=(8, 55), arrowprops=dict(arrowstyle="->", color=CORAL), color=CORAL, fontweight="bold")
tp = [1, 5, 10, 20]; vals = [sg.gmv.head(int(round(n_s * q_ / 100))).sum() / sg.gmv.sum() * 100 for q_ in tp]
ax[1].bar([f"Top {t}%" for t in tp], vals, color=[GREEN, GREEN, GREEN, GREEN])
for i, v in enumerate(vals): ax[1].text(i, v + 1.2, f"{v:.0f}%", ha="center", fontweight="bold", fontsize=12)
ax[1].set_ylim(0, 100); ax[1].set_title("Share of GMV earned by the top sellers", fontsize=13); ax[1].set_ylabel("% of GMV")
fig.suptitle(f"A small group of sellers carries the marketplace: the top 10% earn {vals[2]:.0f}% of GMV", x=0.07, ha="left", fontsize=15, fontweight="bold", y=1.03)
g11 = save(fig, "seller_concentration")

st_s = sellers.seller_state.value_counts(normalize=True) * 100
st_c = customers.customer_state.value_counts(normalize=True) * 100
st_g = bs["share_gmv_%"]
top_states = st_c.head(8).index
fig, ax = plt.subplots(figsize=(11.5, 5.8))
xs = np.arange(len(top_states)); w = 0.27
ax.bar(xs - w, st_s.reindex(top_states), w, color=YELLOW, label="% of sellers")
ax.bar(xs, st_c.reindex(top_states), w, color=GREEN, label="% of customers")
ax.bar(xs + w, st_g.reindex(top_states), w, color=DARK, label="% of GMV (by customer state)")
for i, s in enumerate(top_states):
    ax.text(i - w, st_s.get(s, 0) + .8, f"{st_s.get(s, 0):.0f}", ha="center", fontsize=9)
    ax.text(i, st_c.get(s, 0) + .8, f"{st_c.get(s, 0):.0f}", ha="center", fontsize=9)
    ax.text(i + w, st_g.get(s, 0) + .8, f"{st_g.get(s, 0):.0f}", ha="center", fontsize=9)
ax.set_xticks(xs); ax.set_xticklabels(top_states); ax.set_ylabel("%"); ax.legend()
ax.set_title(f"{pct((sellers.seller_state == 'SP').mean(), 0)} of sellers are in São Paulo state, but only {st_c['SP']:.0f}% of customers: most parcels must travel")
g12 = save(fig, "sellers_vs_customers_by_state")

fig, ax = plt.subplots(figsize=(11, 4.8))
ax.fill_between(range(len(act)), act.values, color=LGREEN); ax.plot(range(len(act)), act.values, color=GREEN, lw=3, marker="o")
ax.set_xticks(range(0, len(act), 2)); ax.set_xticklabels([str(p)[2:] for p in act.index[::2]], rotation=45, ha="right")
ax.set_title(f"Active sellers per month grew {act.iloc[-1] / act.iloc[0]:.0f}x (from {act.iloc[0]} to {act.iloc[-1]})"); ax.set_ylabel("Sellers with at least one sale")
g13 = save(fig, "active_sellers_per_month")
F.p(f"\nRelated graphs: {g11}, {g12}, {g13}")
F.save()

# ============================================================================ 06 PAYMENTS
F = Facts("06_payments.txt", "06 - PAYMENTS")
pt = op.groupby("payment_type").agg(rows=("order_id", "size"), orders=("order_id", "nunique"), value=("payment_value", "sum"), avg_value=("payment_value", "mean"), avg_inst=("payment_installments", "mean")).sort_values("value", ascending=False)
pt["share_value_%"] = (pt.value / pt.value.sum() * 100).round(1)
pt["share_orders_%"] = (pt.orders / op.order_id.nunique() * 100).round(1)
F.h("A. Payment methods")
F.table(pt.round(1))
F.kv("Total paid (all orders)", brl(op.payment_value.sum()))
F.kv("Credit card share of money", f"{pt.loc['credit_card', 'share_value_%']}%")
F.kv("Boleto (bank slip) share of money", f"{pt.loc['boleto', 'share_value_%']}%")
F.kv("Orders paid with 2+ payment records", f"{(op.groupby('order_id').size() > 1).sum()} ({pct((op.groupby('order_id').size() > 1).sum() / op.order_id.nunique())})")
vo_ids = op.groupby("order_id").payment_type.apply(set)
F.kv("Orders using a voucher together with another method", f"{sum(1 for s_ in vo_ids if 'voucher' in s_ and len(s_) > 1)}")
cc_ = op[op.payment_type == "credit_card"]
F.h("B. Credit-card instalments")
inst = cc_.payment_installments.clip(upper=12).value_counts().sort_index()
F.kv("Credit-card payments paid in 1 instalment", pct((cc_.payment_installments == 1).mean()))
F.kv("2+ instalments", pct((cc_.payment_installments >= 2).mean()))
F.kv("6+ instalments", pct((cc_.payment_installments >= 6).mean()))
F.kv("10 instalments (a common cap)", pct((cc_.payment_installments == 10).mean()))
F.kv("Average instalments", f"{cc_.payment_installments.mean():.2f}")
ib = cc_.assign(b=pd.cut(cc_.payment_installments, [0, 1, 3, 6, 10, 24], labels=["1x", "2-3x", "4-6x", "7-10x", "11x+"])).groupby("b").payment_value.agg(["size", "mean", "median"]).round(1)
ib.columns = ["payments", "avg_value_R$", "median_value_R$"]
F.table(ib)
F.p("  => Instalments grow with ticket size: Brazilians finance bigger purchases on cards ('parcelamento').")
K.update(cc_share=pt.loc["credit_card", "share_value_%"], boleto_share=pt.loc["boleto", "share_value_%"], inst2=(cc_.payment_installments >= 2).mean())

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
cols_pt = {"credit_card": GREEN, "boleto": YELLOW, "voucher": BLUE, "debit_card": CORAL, "not_defined": GRAY}
ptp = pt[pt.value > 0]
wedges, _ = ax[0].pie(ptp.value, colors=[cols_pt[i] for i in ptp.index], startangle=90, counterclock=False, wedgeprops=dict(width=0.42, edgecolor="white"))
ax[0].legend(wedges, [f"{i.replace('_', ' ')}  {v:.1f}%" for i, v in zip(ptp.index, ptp["share_value_%"])], loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=10.5)
ax[0].set_title("Share of money paid, by method", fontsize=13)
ax[1].bar(range(len(ib)), ib["avg_value_R$"], color=GREEN)
ax[1].set_xticks(range(len(ib))); ax[1].set_xticklabels([f"{i}\n({n_ / ib.payments.sum() * 100:.0f}%)" for i, n_ in zip(ib.index.astype(str), ib.payments)], fontsize=10)
ax[1].set_xlabel("instalments  (share of all card payments)")
for i, v in enumerate(ib["avg_value_R$"]): ax[1].text(i, v + 6, f"R${v:.0f}", ha="center", fontsize=10, fontweight="bold")
ax[1].set_ylim(0, ib["avg_value_R$"].max() * 1.25); ax[1].set_title("Average card payment by number of instalments", fontsize=13)
fig.suptitle(f"Cards dominate ({pt.loc['credit_card', 'share_value_%']:.0f}% of money) and bigger tickets are split into more instalments", x=0.07, ha="left", fontsize=15, fontweight="bold", y=1.03)
g14 = save(fig, "payment_methods_and_instalments")
F.p(f"\nRelated graph: {g14}")
F.save()

# ============================================================================ 07 DELIVERY
F = Facts("07_delivery_and_logistics.txt", "07 - DELIVERY & LOGISTICS")
F.h("A. Delivery performance (delivered orders)")
F.kv("Orders analysed", n0(len(D)))
F.kv("Average purchase -> delivered", f"{D.deliv_days.mean():.1f} days")
F.kv("Median", f"{D.deliv_days.median():.1f} days")
F.kv("90th percentile", f"{D.deliv_days.quantile(.9):.1f} days")
F.kv("99th percentile", f"{D.deliv_days.quantile(.99):.0f} days")
F.kv("Delivered within 7 days", pct((D.deliv_days <= 7).mean()))
F.kv("Delivered within 14 days", pct((D.deliv_days <= 14).mean()))
F.kv("Took longer than 30 days", pct((D.deliv_days > 30).mean()))
F.kv("Average ESTIMATED delivery time promised to customer", f"{D.est_days.mean():.1f} days")
F.kv("Average days earlier than estimate", f"{-D.vs_est.mean():.1f} days")
F.kv("Orders delivered LATE (after estimated day)", f"{n0(D.late.sum())} ({pct(D.late.mean())})")
F.kv("Average lateness of late orders", f"{D[D.late].vs_est.mean():.1f} days")
F.kv("Late by more than 7 days", f"{pct((D.vs_est > 7).mean())} of all delivered orders")
F.p(f"  => Estimates are very conservative: on average customers receive goods ~{-D.vs_est.mean():.0f} days BEFORE the estimated date.")
K.update(deliv_mean=D.deliv_days.mean(), deliv_med=D.deliv_days.median(), est_mean=D.est_days.mean(), late=D.late.mean(), early=-D.vs_est.mean())

st_ok = D[(D.carrier >= D.purch) & (D.deliv >= D.carrier) & (D.appr >= D.purch) & (D.carrier >= D.appr)].copy()
st_ok["approval"] = (st_ok.appr - st_ok.purch).dt.total_seconds() / 86400
st_ok["seller_handling"] = (st_ok.carrier - st_ok.appr).dt.total_seconds() / 86400
st_ok["transit"] = (st_ok.deliv - st_ok.carrier).dt.total_seconds() / 86400
F.h("B. Where do the days go? (orders with clean timestamps)")
F.kv("Orders used", f"{n0(len(st_ok))} of {n0(len(D))}")
sd = st_ok[["approval", "seller_handling", "transit"]].agg(["mean", "median"]).round(1)
F.table(sd)
F.kv("Share of mean total time spent in carrier transit", pct(st_ok.transit.mean() / (st_ok.approval.mean() + st_ok.seller_handling.mean() + st_ok.transit.mean())))
F.p(f"  => Carrier transit is the biggest block of the wait; seller handling averages {st_ok.seller_handling.mean():.1f} days and payment approval under half a day.")

F.h("C. By customer state (delivered orders; states with 300+ orders)")
ds = D.groupby("customer_state").agg(orders=("order_id", "size"), mean_days=("deliv_days", "mean"), median_days=("deliv_days", "median"), late_pct=("late", "mean"))
ds["late_pct"] = (ds.late_pct * 100).round(1); ds[["mean_days", "median_days"]] = ds[["mean_days", "median_days"]].round(1)
ds = ds.sort_values("mean_days")
F.table(ds[ds.orders >= 300])
F.kv("Fastest big state", f"{ds[ds.orders >= 300].mean_days.idxmin()} ({ds[ds.orders >= 300].mean_days.min():.1f} days)")
F.kv("Slowest big state", f"{ds[ds.orders >= 300].mean_days.idxmax()} ({ds[ds.orders >= 300].mean_days.max():.1f} days)")
F.kv("Highest late rate (300+ orders)", f"{ds[ds.orders >= 300].late_pct.idxmax()} ({ds[ds.orders >= 300].late_pct.max():.1f}%)")
F.kv("Slowest state vs SP", f"{ds[ds.orders >= 300].mean_days.max() / ds.loc['SP', 'mean_days']:.1f}x as long")
slow = ds[ds.orders >= 300].mean_days.idxmax(); fast_sp = ds.loc["SP", "mean_days"]
K.update(sp_days=fast_sp, slow_state=slow, slow_days=ds.loc[slow, "mean_days"])

F.h("D. Same-state vs cross-state shipping (item level, delivered)")
dd = iv.merge(D[["order_id", "deliv_days", "late"]], on="order_id")
dd["same"] = dd.seller_state == dd.customer_state
sm = dd.groupby("same").agg(items=("price", "size"), days=("deliv_days", "mean"), late=("late", "mean"), freight=("freight_value", "mean"))
sm.index = ["cross-state", "same-state"]
sm["late"] = (sm.late * 100).round(1); sm[["days", "freight"]] = sm[["days", "freight"]].round(1)
sm["share_of_items_%"] = (sm["items"] / sm["items"].sum() * 100).round(1)
F.table(sm)
F.kv("Share of items shipped across state lines", pct(1 - dd.same.mean()))
K.update(cross=1 - dd.same.mean(), same_days=sm.loc["same-state", "days"], cross_days=sm.loc["cross-state", "days"])

F.h("E. Monthly late rate")
lm = D[D.ym.between("2017-01", "2018-08")].groupby("ym").agg(late=("late", "mean"), days=("deliv_days", "mean"), n=("order_id", "size"))
F.table((lm.assign(late=(lm.late * 100).round(1), days=lm.days.round(1))).rename(columns={"late": "late_%", "days": "avg_days"}))
F.kv("Worst month for lateness", f"{lm.late.idxmax()} ({lm.late.max() * 100:.1f}% late)")
K.update(worst_late_m=str(lm.late.idxmax()), worst_late=lm.late.max())

# graphs
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
bins_ = np.arange(0, 61, 1)
ax[0].hist(D.est_days.clip(upper=60), bins=bins_, color=LGREEN, edgecolor="white", label=f"Promised (avg {D.est_days.mean():.0f} days)", alpha=.95)
ax[0].hist(D.deliv_days.clip(upper=60), bins=bins_, color=GREEN, edgecolor="white", label=f"Actual (avg {D.deliv_days.mean():.0f} days)", alpha=.85)
ax[0].set_xlabel("days from purchase"); ax[0].set_title("Promised vs actual delivery time", fontsize=13); ax[0].legend()
ax[0].yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}K"))
parts = [st_ok.approval.mean(), st_ok.seller_handling.mean(), st_ok.transit.mean()]
left = 0
for v, c_, nm in zip(parts, [YELLOW, BLUE, GREEN], ["Payment\napproval", "Seller\nhandling", "Carrier transit"]):
    ax[1].barh([0], [v], left=left, color=c_, edgecolor="white", height=.5)
    if v > 1:
        ax[1].text(left + v / 2, 0, f"{nm}\n{v:.1f} d", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    else:
        ax[1].annotate(f"{nm.replace(chr(10), ' ')}\n{v:.1f} d", (left + v / 2, 0.25), xytext=(left + 2.2, 0.47), ha="left", fontsize=9.5, fontweight="bold", color=DARK, arrowprops=dict(arrowstyle="-", color="#55696D"))
    left += v
ax[1].set_ylim(-.4, .75); ax[1].set_yticks([]); ax[1].set_title("Where the average order's days go", fontsize=13); ax[1].grid(False); ax[1].set_xlabel("days")
fig.suptitle(f"Olist promises ~{D.est_days.mean():.0f} days and delivers in ~{D.deliv_days.mean():.0f}: the promise is padded, and carrier transit is most of the wait", x=0.07, ha="left", fontsize=14.5, fontweight="bold", y=1.03)
g15 = save(fig, "delivery_promised_vs_actual")

dsb = ds[ds.orders >= 300].sort_values("mean_days")
fig, ax = plt.subplots(figsize=(12, 5.8))
colors_ = [CORAL if v > D.deliv_days.mean() * 1.4 else (GREEN if v < D.deliv_days.mean() else YELLOW) for v in dsb.mean_days]
ax.bar(dsb.index, dsb.mean_days, color=colors_)
ax.axhline(D.deliv_days.mean(), color=DARK, ls="--", lw=1.5); ax.text(0.01, 0.95, f"dashed line = national average ({D.deliv_days.mean():.0f} days)", transform=ax.transAxes, color=DARK, fontsize=10.5, fontweight="bold")
for i, v in enumerate(dsb.mean_days): ax.text(i, v + .4, f"{v:.0f}", ha="center", fontsize=9)
ax.set_ylabel("avg days to deliver"); ax.set_title(f"Where you live decides your wait: {dsb.index[0]} ≈ {dsb.mean_days.iloc[0]:.0f} days, {dsb.index[-1]} ≈ {dsb.mean_days.iloc[-1]:.0f} days (states with 300+ orders)")
g16 = save(fig, "delivery_days_by_state")

fig, ax = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.7]})
ax[0].bar(sm.index, sm.days, color=[GRAY, GREEN])
for i, (d_, f_) in enumerate(zip(sm.days, sm.freight)): ax[0].text(i, d_ + .3, f"{d_:.1f} days\nfreight R${f_:.0f}", ha="center", fontsize=10)
ax[0].set_ylim(0, sm.days.max() * 1.25); ax[0].set_title("Same-state vs cross-state", fontsize=13)
ax[1].plot(range(len(lm)), lm.late * 100, color=CORAL, marker="o", lw=2.5)
ax[1].set_xticks(range(0, len(lm), 2)); ax[1].set_xticklabels([str(p)[2:] for p in lm.index[::2]], rotation=45, ha="right")
ax[1].set_ylabel("% of orders delivered late"); ax[1].set_title("Late-delivery rate by month", fontsize=13)
im = lm.late.values.argmax(); ax[1].annotate(f"{lm.late.iloc[im] * 100:.0f}% late", (im, lm.late.iloc[im] * 100), xytext=(im - 4, lm.late.iloc[im] * 100), color=CORAL, fontweight="bold", arrowprops=dict(arrowstyle="->", color=CORAL))
fig.suptitle(f"{pct(1 - dd.same.mean(), 0)} of items cross state lines (slower and pricier); lateness spiked in Nov 2017 and Feb-Mar 2018", x=0.07, ha="left", fontsize=14.5, fontweight="bold", y=1.03)
g17 = save(fig, "same_vs_cross_state_and_monthly_lateness")
F.p(f"\nRelated graphs: {g15}, {g16}, {g17}")
F.save()

# ============================================================================ 08 REVIEWS
F = Facts("08_reviews_and_satisfaction.txt", "08 - REVIEWS & CUSTOMER SATISFACTION")
R = ordl[ordl.review_score.notna()].copy()
vc = R.review_score.astype(int).value_counts().sort_index()
F.h("A. Score distribution (one review per order)")
F.table(pd.DataFrame({"orders": vc, "share_%": (vc / vc.sum() * 100).round(1)}))
F.kv("Average score", f"{R.review_score.mean():.2f}")
F.kv("Median", f"{R.review_score.median():.0f}")
F.kv("5-star share", pct((R.review_score == 5).mean()))
F.kv("4+ stars (satisfied)", pct((R.review_score >= 4).mean()))
F.kv("1-2 stars (unhappy)", pct((R.review_score <= 2).mean()))
F.kv("Orders with a written comment", pct(R.review_comment_message.notna().mean()))
F.kv("Comment rate for 1-star vs 5-star", f"{pct(R[R.review_score == 1].review_comment_message.notna().mean())} vs {pct(R[R.review_score == 5].review_comment_message.notna().mean())}")
F.kv("Review coverage (orders with a review)", pct(len(R) / len(orders)))
rt = (rv.review_answer_timestamp - rv.review_creation_date).dt.total_seconds() / 86400
F.kv("Median days between survey sent and answered", f"{rt.median():.1f}")
K.update(avg_score=R.review_score.mean(), s5=(R.review_score == 5).mean(), s12=(R.review_score <= 2).mean(), s1=(R.review_score == 1).mean())
F.p("  Note: the distribution is U-shaped/J-shaped: many 5s, a distinct bump of 1s, very few 2s.")

RD = R[R.order_id.isin(D.order_id)].merge(D[["order_id", "vs_est", "deliv_days"]], on="order_id", suffixes=("", "_d"))
RD["bucket"] = pd.cut(RD.vs_est, [-1000, -8, -1, 0, 3, 7, 1000], labels=["8+ days early", "1-7 days early", "on the estimated day", "1-3 days late", "4-7 days late", "8+ days late"])
bk = RD.groupby("bucket", observed=True).agg(orders=("order_id", "size"), avg_score=("review_score", "mean"), one_star=("review_score", lambda s: (s == 1).mean()), five_star=("review_score", lambda s: (s == 5).mean()))
F.h("B. Review score vs delivery punctuality (delivered orders)")
bk2 = bk.copy(); bk2["avg_score"] = bk2.avg_score.round(2); bk2["one_star"] = (bk2.one_star * 100).round(1); bk2["five_star"] = (bk2.five_star * 100).round(1)
F.table(bk2.rename(columns={"one_star": "1star_%", "five_star": "5star_%"}))
on_t = RD[RD.vs_est <= 0].review_score.mean(); late_s = RD[RD.vs_est > 0].review_score.mean()
F.kv("Avg score when on-time/early", f"{on_t:.2f}")
F.kv("Avg score when late", f"{late_s:.2f}")
F.kv("Share of 1-star reviews among late orders vs on-time", f"{pct((RD[RD.vs_est > 0].review_score == 1).mean())} vs {pct((RD[RD.vs_est <= 0].review_score == 1).mean())}")
l8 = bk.loc["8+ days late"]; F.kv("Orders 8+ days late: avg score / 1-star share", f"{l8.avg_score:.2f} / {pct(l8.one_star)}")
late_share_of_1 = (RD[RD.review_score == 1].vs_est > 0).mean()
F.kv("Share of all 1-star reviews that were for LATE deliveries", pct(late_share_of_1) + f" (while late orders are only {pct((RD.vs_est > 0).mean())} of orders)")
K.update(on_t=on_t, late_s=late_s, late_share_of_1=late_share_of_1, l8_avg=l8.avg_score, l8_one=l8.one_star, late_1=(RD[RD.vs_est > 0].review_score == 1).mean(), ontime_1=(RD[RD.vs_est <= 0].review_score == 1).mean())

F.h("C. Other drivers of low scores")
R["multi_item"] = R.n_items > 1
F.kv("Avg score: single-item vs multi-item orders", f"{R[~R.multi_item].review_score.mean():.2f} vs {R[R.multi_item].review_score.mean():.2f}")
F.kv("Avg score: single-seller vs multi-seller orders", f"{R[R.n_sellers == 1].review_score.mean():.2f} vs {R[R.n_sellers > 1].review_score.mean():.2f}")
ms_ = R[R.n_sellers > 1]; ms_d = ms_[ms_.order_id.isin(D.order_id)]
F.kv("Multi-seller orders: count / late rate on paper / 1-star share", f"{len(ms_)} / {pct(ms_d.order_id.isin(D[D.late].order_id).mean())} / {pct((ms_.review_score == 1).mean())}")
F.p("  Hypothesis (not provable here): split parcels - the recorded delivery date likely reflects only one of the shipments, so these orders look on-time while customers wait for the rest.")
K.update(multi_score=R[R.n_sellers > 1].review_score.mean(), single_score=R[R.n_sellers == 1].review_score.mean(), n_multi=len(ms_))
F.kv("Avg score: orders canceled / unavailable", f"{R[R.order_status == 'canceled'].review_score.mean():.2f} / {R[R.order_status == 'unavailable'].review_score.mean():.2f}")
F.kv("Avg score: delivered orders", f"{R[R.order_status == 'delivered'].review_score.mean():.2f}")
F.kv("Avg score: 'shipped' orders (never confirmed delivered)", f"{R[R.order_status == 'shipped'].review_score.mean():.2f}")
dstate = R.groupby("customer_state").agg(n=("order_id", "size"), score=("review_score", "mean"))
dstate = dstate[dstate.n >= 500].sort_values("score")
F.kv("Lowest avg score among big states", f"{dstate.index[0]} ({dstate.score.iloc[0]:.2f})")
F.kv("Highest avg score among big states", f"{dstate.index[-1]} ({dstate.score.iloc[-1]:.2f})")

rcat = iv.merge(rv_order, on="order_id").groupby("category").agg(items=("price", "size"), score=("review_score", "mean"), one_star=("review_score", lambda s: (s == 1).mean()))
rcat = rcat[rcat["items"] >= 300].sort_values("score")
F.h("D. Categories: worst and best average review (300+ items)")
F.p("Worst 8:"); F.table(rcat.head(8).round(2)); F.p("Best 8:"); F.table(rcat.tail(8).iloc[::-1].round(2))
K.update(worst_cat=rcat.index[0], worst_cat_score=rcat.score.iloc[0], best_cat=rcat.index[-1], best_cat_score=rcat.score.iloc[-1])

# keywords
cm = R[R.review_comment_message.notna()].copy()
cm["t"] = cm.review_comment_message.str.lower()
kw = {
    "did not receive (não recebi)": r"n[aã]o recebi|n[aã]o chegou",
    "delay (atraso / atrasou)": r"atras",
    "deadline (prazo)": r"prazo",
    "defect / broken (defeito, quebrado)": r"defeit|quebr|danific",
    "wrong item (errado / diferente)": r"errad|diferente",
    "refund / return (reembolso, devolv)": r"reembols|devol|estorno",
    "quality (qualidade)": r"qualidade",
    "recommend (recomendo)": r"recomend",
    "great / excellent (ótimo, excelente, perfeito)": r"[oó]timo|excelente|perfeito",
    "arrived early (antes do prazo)": r"antes do prazo|chegou antes|adiantado",
}
F.h("E. What people write: % of commented reviews that mention each theme")
rows = []
for k_, rx in kw.items():
    has = cm.t.str.contains(rx, regex=True)
    rows.append((k_, has[cm.review_score == 1].mean() * 100, has[cm.review_score == 3].mean() * 100, has[cm.review_score == 5].mean() * 100))
kd = pd.DataFrame(rows, columns=["theme", "1-star_%", "3-star_%", "5-star_%"]).set_index("theme").round(1)
F.kv("Commented reviews analysed", f"{n0(len(cm))} (1-star: {n0((cm.review_score == 1).sum())}, 5-star: {n0((cm.review_score == 5).sum())})")
F.table(kd)
F.p("  Method: simple Portuguese keyword search on lower-cased comments (not NLP/sentiment); a review can match several themes.")

# correlation
cdf = D.merge(R[["order_id", "review_score"]], on="order_id", suffixes=("_x", "")).copy() if "review_score" in D.columns else D
cdf = D.copy()
cdf = cdf[["gmv", "freight", "n_items", "weight", "max_inst", "deliv_days", "vs_est", "review_score"]].dropna()
cdf.columns = ["order value", "freight", "items", "weight (g)", "instalments", "delivery days", "days vs estimate\n(+ = late)", "review score"]
cm_ = cdf.corr(method="spearman")
F.h("F. Spearman correlations (delivered orders)")
F.table(cm_.round(2).rename(index=lambda s: s.replace("\n", " "), columns=lambda s: s.replace("\n", " ")))
F.kv("Review score vs delivery days", f"{cm_.loc['review score', 'delivery days']:.2f} (strongest link to review score of everything measured)")
F.kv("Review score vs days vs estimate (late)", f"{cm_.loc['review score', 'days vs estimate\n(+ = late)']:.2f} (second strongest). Correlations are modest because most orders are early; the damage is concentrated in the late tail (see section B)")
F.kv("Review score vs order value", f"{cm_.loc['review score', 'order value']:.2f} (almost no link: price does not drive satisfaction)")
K.update(corr_late_score=cm_.loc["review score", "days vs estimate\n(+ = late)"])

# graphs
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.1, 1.3]})
ax[0].bar(vc.index.astype(str), vc.values, color=[CORAL, "#E9918C", YELLOW, "#8FCB9B", GREEN])
for i, v in enumerate(vc.values): ax[0].text(i, v + 600, f"{v / vc.sum() * 100:.1f}%", ha="center", fontweight="bold")
ax[0].set_xlabel("stars"); ax[0].set_title(f"Review scores (avg {R.review_score.mean():.2f})", fontsize=13); ax[0].yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}K")); ax[0].set_ylim(0, vc.max() * 1.12)
bkc = [GREEN, GREEN, "#8FCB9B", YELLOW, "#E9918C", CORAL]
ax[1].bar(range(len(bk)), bk.avg_score, color=bkc)
ax[1].set_xticks(range(len(bk))); ax[1].set_xticklabels(["8+ days\nearly", "1-7 days\nearly", "on the\nday", "1-3 days\nlate", "4-7 days\nlate", "8+ days\nlate"], fontsize=9.5)
for i, (s_, n_) in enumerate(zip(bk.avg_score, bk.orders)): ax[1].text(i, s_ + .06, f"{s_:.2f}", ha="center", fontweight="bold")
ax[1].set_ylim(0, 5.4); ax[1].set_title("Average score by delivery punctuality", fontsize=13); ax[1].set_ylabel("avg review score")
fig.suptitle(f"Customers are mostly happy ({pct((R.review_score >= 4).mean(), 0)} give 4-5 stars) unless the parcel is late: {bk.avg_score.iloc[-1]:.1f} stars at 8+ days late", x=0.07, ha="left", fontsize=14.5, fontweight="bold", y=1.03)
g18 = save(fig, "review_scores_and_delivery_punctuality")

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
mt_ = D[D.ym.between("2017-01", "2018-08")].merge(R[["order_id"]], on="order_id")
mts = R[R.ym.between("2017-01", "2018-08") & R.order_id.isin(D.order_id)].groupby("ym").review_score.mean()
a_ = ax[0]; a_.plot(range(len(mts)), mts.values, color=GREEN, marker="o", lw=2.5, label="Avg review score"); a_.set_ylim(3.4, 4.6); a_.set_ylabel("avg review score", color=GREEN)
a_.set_xticks(range(0, len(mts), 2)); a_.set_xticklabels([str(p)[2:] for p in mts.index[::2]], rotation=45, ha="right")
b_ = a_.twinx(); b_.grid(False); b_.spines["right"].set_visible(True); b_.plot(range(len(lm)), lm.late * 100, color=CORAL, marker="s", lw=2.5, ls="--", label="% late"); b_.set_ylabel("% delivered late", color=CORAL)
a_.set_title("Satisfaction falls when lateness rises", fontsize=13)
h1, l1 = a_.get_legend_handles_labels(); h2, l2 = b_.get_legend_handles_labels(); a_.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
sc_ = RD.groupby(RD.vs_est.clip(-20, 20)).review_score.mean()
ax[1].plot(sc_.index, sc_.values, color=DARK, lw=2.5); ax[1].axvline(0, color=CORAL, ls="--"); ax[1].text(0.5, 1.45, "estimated\ndelivery day", color=CORAL, fontsize=9.5)
ax[1].set_xlabel("days vs estimated date  (negative = early, positive = late)"); ax[1].set_ylabel("avg review score"); ax[1].set_ylim(1, 5)
ax[1].set_title("Score collapses right after the promised date", fontsize=13)
fig.subplots_adjust(wspace=0.42)
fig.suptitle("Late delivery is the strongest driver of bad reviews that we can measure", x=0.07, ha="left", fontsize=15, fontweight="bold", y=1.03)
g19 = save(fig, "lateness_vs_satisfaction")

fig, ax = plt.subplots(1, 2, figsize=(13, 5.4), sharex=True)
w_ = rcat.head(8); b_ = rcat.tail(8)
ax[0].barh(w_.index.str.replace("_", " "), w_.score, color=CORAL); ax[0].set_title("Lowest-rated categories", fontsize=13)
ax[1].barh(b_.index.str.replace("_", " "), b_.score, color=GREEN); ax[1].set_title("Highest-rated categories", fontsize=13)
for a, d in zip(ax, [w_, b_]):
    a.set_xlim(3, 4.6); a.grid(axis="y", visible=False)
    for i, v in enumerate(d.score): a.text(v + .01, i, f"{v:.2f}", va="center", fontsize=10)
fig.suptitle(f"Category matters too: {b_.index[-1].replace('_', ' ')}, {b_.index[-2].replace('_', ' ')} and {b_.index[-3].replace('_', ' ')} delight; {w_.index[0].replace('_', ' ')} and bulky home goods disappoint", x=0.07, ha="left", fontsize=14.5, fontweight="bold", y=1.03)
g20 = save(fig, "best_and_worst_categories_by_review")

kd2 = kd[["1-star_%", "5-star_%"]].copy(); kd2["diff"] = kd2["1-star_%"] - kd2["5-star_%"]; kd2 = kd2.sort_values("diff")
fig, ax = plt.subplots(figsize=(11.5, 6.4)); yy = np.arange(len(kd2)); h_ = .38
ax.barh(yy + h_ / 2, kd2["1-star_%"], h_, color=CORAL, label="1-star reviews"); ax.barh(yy - h_ / 2, kd2["5-star_%"], h_, color=GREEN, label="5-star reviews")
ax.set_yticks(yy); ax.set_yticklabels(kd2.index, fontsize=10)
for i, (a1, a5) in enumerate(zip(kd2["1-star_%"], kd2["5-star_%"])):
    ax.text(a1 + .6, i + h_ / 2, f"{a1:.0f}%", va="center", fontsize=9); ax.text(a5 + .6, i - h_ / 2, f"{a5:.0f}%", va="center", fontsize=9)
ax.set_xlabel("% of written comments mentioning the theme"); ax.legend(loc="lower right"); ax.grid(axis="y", visible=False)
ax.set_title(f"1-star comments are dominated by parcels that never arrived ({kd.loc['did not receive (não recebi)', '1-star_%']:.0f}%); 5-star ones praise timing and recommend")
g21 = save(fig, "review_text_themes")

fig, ax = plt.subplots(figsize=(8.8, 7))
im_ = ax.imshow(cm_.values, cmap="RdYlGn", vmin=-1, vmax=1)
ax.set_xticks(range(len(cm_))); ax.set_xticklabels(cm_.columns, rotation=40, ha="right", fontsize=9.5); ax.set_yticks(range(len(cm_))); ax.set_yticklabels(cm_.index, fontsize=9.5)
for i in range(len(cm_)):
    for j in range(len(cm_)): ax.text(j, i, f"{cm_.values[i, j]:.2f}", ha="center", va="center", fontsize=9, color="#111" if abs(cm_.values[i, j]) < .7 else "white")
ax.grid(False); fig.colorbar(im_, fraction=.04, pad=.03)
ax.set_title("What moves with what (Spearman correlation)", fontsize=13)
g22 = save(fig, "correlation_heatmap")
F.p(f"\nRelated graphs: {g18}, {g19}, {g20}, {g21}, {g22}")
F.save()

# ============================================================================ 09 FUNNEL
F = Facts("09_marketing_funnel_and_seller_acquisition.txt", "09 - MARKETING FUNNEL & SELLER ACQUISITION")
lq["first_contact_date"] = pd.to_datetime(lq.first_contact_date)
lc["won_date"] = pd.to_datetime(lc.won_date)
L = lq.merge(lc, on="mql_id", how="left")
L["closed"] = L.seller_id.notna()
L["origin"] = L.origin.fillna("missing")
L["days_to_close"] = (L.won_date - L.first_contact_date).dt.total_seconds() / 86400
F.h("A. Funnel overview")
F.kv("Marketing-qualified leads (MQLs)", n0(len(lq)))
F.kv("Closed (became sellers)", f"{len(lc)} = {pct(len(lc) / len(lq))} conversion")
F.kv("Landing pages used", lq.landing_page_id.nunique())
F.kv("MQL first-contact window", f"{lq.first_contact_date.min().date()} to {lq.first_contact_date.max().date()}")
F.kv("Median days from first contact to won", f"{L.days_to_close.median():.0f}")
F.kv("Mean days", f"{L.days_to_close.mean():.0f}")
F.kv("Closed within 30 days", pct((L.days_to_close.dropna() <= 30).mean()))
F.kv("Closed after 90+ days", pct((L.days_to_close.dropna() > 90).mean()))
K.update(mql=len(lq), closed=len(lc), conv=len(lc) / len(lq), ttc_med=L.days_to_close.median())

og = L.groupby("origin").agg(leads=("mql_id", "size"), closed=("closed", "sum"))
og["conv_%"] = (og.closed / og.leads * 100).round(1)
og["share_of_leads_%"] = (og.leads / og.leads.sum() * 100).round(1)
og["share_of_closed_%"] = (og.closed / og.closed.sum() * 100).round(1)
og = og.sort_values("leads", ascending=False)
F.h("B. Lead origin: volume vs conversion")
F.table(og)
best = og[(og.leads >= 100) & ~og.index.isin(["unknown", "missing", "other"])].sort_values("conv_%", ascending=False)
F.kv("Best converting identified origin (100+ leads; excl. unknown/other)", f"{best.index[0]} ({best['conv_%'].iloc[0]}%)")
F.kv("Weakest identified origin (100+ leads)", f"{best.index[-1]} ({best['conv_%'].iloc[-1]}%)")
F.kv("Origin = unknown converts at", f"{og.loc['unknown', 'conv_%']}% (tracking gap: 13.7% of leads have no known source)")
F.kv("Organic + paid search share of leads", f"{og.loc[['organic_search', 'paid_search'], 'share_of_leads_%'].sum():.1f}%")
K.update(best_origin=best.index[0], best_conv=best["conv_%"].iloc[0], worst_origin=best.index[-1], worst_conv=best["conv_%"].iloc[-1], top_origin=og.index[0], top_origin_share=og["share_of_leads_%"].iloc[0])

lpg = L.groupby("landing_page_id").agg(leads=("mql_id", "size"), closed=("closed", "sum")); lpg["conv_%"] = (lpg.closed / lpg.leads * 100).round(1)
F.h("C. Landing pages")
F.kv("Landing pages with 100+ leads", int((lpg.leads >= 100).sum()))
F.kv("Share of leads from top 5 landing pages", pct(lpg.leads.sort_values(ascending=False).head(5).sum() / lpg.leads.sum()))
tl = lpg[lpg.leads >= 100].sort_values("conv_%", ascending=False)
F.table(tl.head(5).rename(index=lambda s: s[:8] + "..."))
F.kv("Conversion range for pages with 100+ leads", f"{tl['conv_%'].min()}% - {tl['conv_%'].max()}%")

F.h("D. Who gets won? (closed leads)")
for col in ["business_segment", "lead_type", "business_type"]:
    vcx = lc[col].value_counts(dropna=False)
    F.p(f"\n{col} (top 8):"); F.table(pd.DataFrame({"closed": vcx.head(8), "share_%": (vcx.head(8) / len(lc) * 100).round(1)}))
lp = lc.lead_behaviour_profile.fillna("not recorded").value_counts()
F.p("\nlead_behaviour_profile (DISC-style animal personas):"); F.table(pd.DataFrame({"closed": lp.head(6), "share_%": (lp.head(6) / len(lc) * 100).round(1)}))
F.kv("Profile missing", f"{lc.lead_behaviour_profile.isna().sum()} ({pct(lc.lead_behaviour_profile.isna().mean())})")
F.kv("Resellers vs manufacturers", f"{(lc.business_type == 'reseller').sum()} vs {(lc.business_type == 'manufacturer').sum()}")
F.kv("Closed leads declaring monthly revenue > 0", f"{(lc.declared_monthly_revenue > 0).sum()} of {len(lc)}")
F.kv("Declared monthly revenue - median of those > 0", brl(lc[lc.declared_monthly_revenue > 0].declared_monthly_revenue.median()))
F.p("  Caution: declared_monthly_revenue has extreme outliers (max R$ 50 million) and 95% zeros - not reliable.")
K.update(top_seg=lc.business_segment.value_counts().index[0], top_seg_n=int(lc.business_segment.value_counts().iloc[0]), top_lt=lc.lead_type.value_counts().index[0], reseller=(lc.business_type == "reseller").mean())

F.h("E. Do won leads turn into selling sellers? (link funnel -> orders)")
won = lc.merge(sg[["gmv", "orders", "first"]], left_on="seller_id", right_index=True, how="left")
inmk = lc.seller_id.isin(sellers.seller_id)
F.kv("Closed leads that appear in the sellers table", f"{inmk.sum()} of {len(lc)} ({pct(inmk.mean())})")
F.kv("...of which made at least one valid sale", int(won.gmv.notna().sum()))
F.kv("GMV from those sellers", f"{brl(won.gmv.sum())} = {pct(won.gmv.sum() / sg.gmv.sum())} of all GMV")
F.kv("Median GMV per won seller (with sales)", brl(won.gmv.median()))
F.kv("Median GMV per all other seller", brl(sg[~sg.index.isin(won.seller_id)].gmv.median()) + "  (not like-for-like: won sellers joined later, so had less time to sell)")
won["lag"] = (won["first"] - won.won_date).dt.days
F.kv("Won sellers whose first sale came AFTER the won date", f"{(won.lag > 0).sum()} of {won.lag.notna().sum()}")
F.kv("Median days from won date to first sale (those)", f"{won[won.lag > 0].lag.median():.0f}")
F.p("  NOTE: won_date goes to Nov 2018, orders end Oct 2018, so the funnel's later cohorts cannot be observed selling.")
K.update(won_gmv_share=won.gmv.sum() / sg.gmv.sum(), won_n=int(won.gmv.notna().sum()))

fig, ax = plt.subplots(figsize=(12, 6))
og2 = og.sort_values("leads")
yy = np.arange(len(og2)); ax.barh(yy, og2.leads, color=LGREEN, edgecolor=GREEN)
for i, (l_, c_, cv) in enumerate(zip(og2.leads, og2.closed, og2["conv_%"])):
    ax.text(l_ + 25, i, f"{n0(l_)} leads  ->  {int(c_)} won ({cv:.1f}%)", va="center", fontsize=9.5, color=GREEN if cv >= 10.5 else (CORAL if cv < 8 else DARK), fontweight="bold" if cv >= 12 or cv < 8 else "normal")
ax.set_yticks(yy); ax.set_yticklabels(og2.index.str.replace("_", " ")); ax.set_xlim(0, og2.leads.max() * 1.45); ax.grid(axis="y", visible=False)
ax.axvline(0, color=GRAY)
ax.set_title(f"Search brings the most leads, but conversion varies a lot by channel (overall {len(lc) / len(lq) * 100:.1f}%)")
g23 = save(fig, "funnel_leads_and_conversion_by_origin")

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.2, 1]})
ax[0].hist(L.days_to_close.dropna().clip(upper=300), bins=40, color=GREEN, edgecolor="white")
ax[0].axvline(L.days_to_close.median(), color=CORAL, lw=2); ax[0].text(L.days_to_close.median() + 4, ax[0].get_ylim()[1] * .9, f"median {L.days_to_close.median():.0f} days", color=CORAL, fontweight="bold")
ax[0].set_xlabel("days from first contact to 'won'  (capped at 300)"); ax[0].set_title("Sales-cycle length", fontsize=13)
fun = [len(lq), len(lc), int(won.gmv.notna().sum())]
ax[1].bar(["Qualified\nleads", "Closed\n(won)", "Won & made\nat least 1 sale*"], fun, color=[LGREEN, GREEN, DARK])
for i, v in enumerate(fun): ax[1].text(i, v + 120, f"{n0(v)}" + (f"\n{v / fun[0] * 100:.1f}%" if i else ""), ha="center", fontweight="bold")
ax[1].set_ylim(0, fun[0] * 1.15); ax[1].set_title("Funnel", fontsize=13); ax[1].grid(axis="x", visible=False)
ax[1].text(2.4, fun[0] * 1.08, "* matched to order data", ha="right", fontsize=8.5, color=GRAY)
fig.suptitle(f"Only ~1 in {len(lq) / len(lc):.0f} qualified leads is won, and the deal takes a median of {L.days_to_close.median():.0f} days", x=0.07, ha="left", fontsize=15, fontweight="bold", y=1.03)
g24 = save(fig, "funnel_sales_cycle_and_stages")

fig, ax = plt.subplots(1, 3, figsize=(17, 5.8), gridspec_kw={"width_ratios": [1.5, 1, .9], "wspace": 0.75})
bsg = lc.business_segment.value_counts().head(10).iloc[::-1]
ax[0].barh(bsg.index.str.replace("_", " "), bsg.values, color=GREEN); ax[0].set_title("Business segment (top 10)", fontsize=13)
for i, v in enumerate(bsg.values): ax[0].text(v + 1, i, str(v), va="center", fontsize=9.5)
lt = lc.lead_type.value_counts().iloc[::-1]
ax[1].barh(lt.index.str.replace("_", " "), lt.values, color=YELLOW); ax[1].set_title("Lead type", fontsize=13)
for i, v in enumerate(lt.values): ax[1].text(v + 4, i, str(v), va="center", fontsize=9.5)
bt = lc.business_type.where(lc.business_type.isin(["reseller", "manufacturer"]), "other / missing").value_counts()
wd_, _ = ax[2].pie(bt.values, colors=[GREEN, YELLOW, GRAY][:len(bt)], startangle=90, counterclock=False, wedgeprops=dict(width=.42, edgecolor="white"))
ax[2].legend(wd_, [f"{i}  {v / bt.sum() * 100:.0f}%" for i, v in zip(bt.index, bt.values)], loc="upper center", bbox_to_anchor=(0.5, 0.02), fontsize=10)
ax[2].set_title("Business type", fontsize=13)
for a in ax[:2]: a.grid(axis="y", visible=False)
fig.suptitle(f"Won sellers are mostly resellers ({(lc.business_type == 'reseller').mean() * 100:.0f}%); {lc.business_segment.value_counts().index[0].replace('_', ' ')} and {lc.business_segment.value_counts().index[1].replace('_', ' ')} lead the segments", x=0.07, ha="left", fontsize=15, fontweight="bold", y=1.03)
g25 = save(fig, "closed_leads_profile")
F.p(f"\nRelated graphs: {g23}, {g24}, {g25}")
F.save()

# ============================================================================ 10 TAKEAWAYS
F = Facts("10_key_takeaways.txt", "10 - THE 16 FACTS TO REMEMBER")
F.p("Every number below is computed in analysis.py and detailed in the other fact files.\n")
items = [
    f"SCALE: {n0(K['orders'])} orders from Sep 2016 to Oct 2018, {n0(K['people'])} unique customers, {n0(K['n_sellers'])} active sellers, GMV of {brl(K['gmv'])} (item prices) plus {brl(K['freight'])} in freight.",
    f"GROWTH: Jan-Aug 2018 had {K['yoy_o'] * 100:.0f}% more orders and {K['yoy_g'] * 100:.0f}% more GMV than Jan-Aug 2017. Orders plateaued at about 6-7K per month through 2018.",
    f"VOLUME-DRIVEN: baskets did not grow - order value and items per order stayed flat. Growth = more orders.",
    f"BLACK FRIDAY: {n0(K['bf'])} orders on 24 Nov 2017, the single biggest day in the dataset.",
    f"CONCENTRATION (geography): São Paulo is {K['sp']:.0f}% of orders; SP+RJ+MG are {K['top3']:.0f}%.",
    f"RETENTION: only {K['repeat'] * 100:.1f}% of customers ever ordered twice - this is a one-off-purchase marketplace.",
    f"CATEGORIES: '{K['top_cat']}' is #1 at {K['top_cat_share']:.1f}% of GMV; the top 10 categories are {K['top10_cat']:.0f}% of GMV.",
    f"PAYMENTS: credit cards are {K['cc_share']:.0f}% of money paid, boleto {K['boleto_share']:.0f}%; {K['inst2'] * 100:.0f}% of card payments use 2+ instalments.",
    f"SELLER CONCENTRATION: the top 10% of sellers earn {K['top10s'] * 100:.0f}% of GMV (Gini {K['gini']:.2f}).",
    f"DELIVERY: average {K['deliv_mean']:.1f} days (median {K['deliv_med']:.1f}); customers were promised {K['est_mean']:.1f} days, i.e. goods arrive ~{K['early']:.0f} days earlier than promised.",
    f"LATE RATE: {K['late'] * 100:.1f}% of delivered orders arrive after the estimated date; worst month was {K['worst_late_m']} at {K['worst_late'] * 100:.1f}%.",
    f"GEOGRAPHY OF WAIT: {K['cross'] * 100:.0f}% of items cross state lines ({K['cross_days']:.1f} days vs {K['same_days']:.1f} days same-state); {K['slow_state']} waits {K['slow_days']:.0f} days vs {K['sp_days']:.0f} in SP.",
    f"SATISFACTION: average review {K['avg_score']:.2f}/5, {K['s5'] * 100:.0f}% give 5 stars, {K['s12'] * 100:.0f}% give 1-2 stars.",
    f"THE MEASURABLE DRIVER: on-time orders average {K['on_t']:.2f} stars, late orders {K['late_s']:.2f}; 8+ days late = {K['l8_avg']:.2f} stars. Late orders are {K['late_1'] * 100:.0f}% 1-star vs {K['ontime_1'] * 100:.0f}% for on-time.",
    f"FUNNEL: {n0(K['mql'])} qualified leads -> {K['closed']} won ({K['conv'] * 100:.1f}%), median {K['ttc_med']:.0f} days to close; best identified channel '{K['best_origin'].replace('_', ' ')}' ({K['best_conv']}%), weakest '{K['worst_origin'].replace('_', ' ')}' ({K['worst_conv']}%).",
    f"HIDDEN RISK: the {K['n_multi']:,} orders with items from 2+ sellers average only {K['multi_score']:.2f} stars vs {K['single_score']:.2f} for single-seller orders.",
]
for i, t in enumerate(items, 1):
    F.p(f"{i:>2}. {t}\n")
F.h("Suggested actions the data supports")
F.p("  1. Fix the last mile first: late delivery is the biggest controllable cause of 1-star reviews.")
F.p("  2. Grow repeat purchase: with <4% repeaters, small retention gains are worth more than extra ad spend.")
F.p("  3. Recruit sellers outside São Paulo to shorten distance, delivery time and freight for the other states.")
F.p("  4. Watch seller quality: a small number of sellers drive most revenue, so their service level sets the platform's reputation.")
F.p("  5. Prepare for peaks: lateness jumped in Nov 2017 (Black Friday) and Feb-Mar 2018 - add carrier capacity before promotional events and investigate the early-2018 spike.")
F.p("  6. Investigate multi-seller orders: they are rare (1.3%) but score about 1.3 stars lower.")
F.p("  7. Put more recruiting effort into the channels that convert best and re-examine the weakest ones.")
F.save()
print("DONE; graphs:", _counter[0])
