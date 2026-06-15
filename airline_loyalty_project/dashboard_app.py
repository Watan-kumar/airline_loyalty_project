"""
Airline Loyalty Program — Behavioral Intelligence Dashboard
Streamlit App | Consulting & Analytics Club, IIT Guwahati
Run: streamlit run dashboard_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ── Page config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Airline Loyalty Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styles ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .metric-card {
        background: white; border-radius: 10px; padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07); border-left: 4px solid;
        margin-bottom: 8px;
    }
    .metric-val { font-size: 2em; font-weight: 700; line-height: 1.1; }
    .metric-lbl { font-size: 0.82em; color: #666; margin-top: 2px; }
    .section-header {
        font-size: 1.25em; font-weight: 700; color: #1F4E79;
        border-bottom: 2px solid #2E75B6; padding-bottom: 6px;
        margin: 18px 0 10px 0;
    }
    .insight-box {
        background: #EBF3FB; border-left: 4px solid #2E75B6;
        padding: 12px 16px; border-radius: 0 8px 8px 0;
        font-size: 0.9em; margin: 8px 0;
    }
    .alert-box {
        background: #FEF3E2; border-left: 4px solid #E67E22;
        padding: 12px 16px; border-radius: 0 8px 8px 0;
        font-size: 0.9em; margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('final_dataset.csv')
    seg = pd.read_csv('segment_summary.csv')
    return df, seg

df, seg = load_data()

SEG_ORDER  = ['Champions','Passive Loyalists','At-Risk Valuables','New Low-Engagers','Churned Members']
PALETTE    = {'Champions':'#2ecc71','Passive Loyalists':'#3498db',
              'At-Risk Valuables':'#e67e22','New Low-Engagers':'#9b59b6',
              'Churned Members':'#e74c3c'}

# Remap segment names in df if needed
rename = {
    'Champions':'Champions','Loyal Actives':'Churned Members',
    'At-Risk Valuables':'At-Risk Valuables',
    'Occasional Flyers':'New Low-Engagers','Dormant / Lost':'Passive Loyalists'
}
if 'segment_name' not in df.columns:
    st.error("Run 02_churn_model.py first to generate final_dataset.csv with segments.")
    st.stop()
if 'Loyal Actives' in df['segment_name'].values:
    df['segment_name'] = df['segment_name'].map(rename)

# ── Sidebar ───────────────────────────────────────────────────────────────
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3125/3125713.png", width=60)
st.sidebar.title("✈️ Loyalty Intelligence")
st.sidebar.markdown("**Consulting & Analytics Club**  \nIIT Guwahati | Summer 2026")
st.sidebar.divider()

page = st.sidebar.radio("Navigate", [
    "📊 Executive Overview",
    "🎯 Segment Explorer",
    "⚠️ Churn Risk Finder",
    "🗺️ Provincial Analysis",
    "📋 Retention Playbook"
])

st.sidebar.divider()
# Global filters
selected_segs = st.sidebar.multiselect(
    "Filter Segments", SEG_ORDER, default=[s for s in SEG_ORDER if s != 'Churned Members'],
    help="Filter the active member views"
)
card_filter = st.sidebar.multiselect("Loyalty Card", ['Star','Nova','Aurora'], default=['Star','Nova','Aurora'])

filtered = df[df['segment_name'].isin(selected_segs) & df['Loyalty Card'].isin(card_filter)]
active   = df[df['segment_name'] != 'Churned Members']

# ════════════════════════════════════════════════════════════════════════
# PAGE 1 — Executive Overview
# ════════════════════════════════════════════════════════════════════════
if page == "📊 Executive Overview":
    st.title("Airline Loyalty Program — Executive Overview")
    st.caption("Behavioral intelligence analysis of 16,737 Canadian loyalty members | 2017–2018 flight data")

    # Top metrics
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1:
        st.markdown(f'<div class="metric-card" style="border-color:#1F4E79"><div class="metric-val" style="color:#1F4E79">{len(df):,}</div><div class="metric-lbl">Total Members</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card" style="border-color:#e74c3c"><div class="metric-val" style="color:#e74c3c">{df["churned"].mean()*100:.1f}%</div><div class="metric-lbl">Churn Rate</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card" style="border-color:#27ae60"><div class="metric-val" style="color:#27ae60">0.969</div><div class="metric-lbl">Model AUC (GBM)</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card" style="border-color:#e67e22"><div class="metric-val" style="color:#e67e22">${df[df["churned"]==1]["CLV"].sum()/1e6:.1f}M</div><div class="metric-lbl">CLV Lost to Churn</div></div>', unsafe_allow_html=True)
    with c5:
        ar_n = len(df[(df['segment_name']=='At-Risk Valuables')])
        st.markdown(f'<div class="metric-card" style="border-color:#9b59b6"><div class="metric-val" style="color:#9b59b6">{ar_n:,}</div><div class="metric-lbl">At-Risk Valuables</div></div>', unsafe_allow_html=True)

    st.markdown("")
    col_left, col_right = st.columns([1,1])

    with col_left:
        st.markdown('<div class="section-header">Segment Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6,4))
        seg_s = seg.set_index('segment_name').reindex(SEG_ORDER)
        seg_s['n'] = seg_s['n'].fillna(0).astype(int)
        valid = seg_s['n'] > 0
        colors = [PALETTE[s] for s in SEG_ORDER if seg_s.loc[s,'n'] > 0]
        wedges, texts, autotexts = ax.pie(
            seg_s.loc[valid,'n'], 
            labels=seg_s.loc[valid,'n'].index.tolist(),
            colors=colors,
            autopct='%1.1f%%', pctdistance=0.8,
            wedgeprops={'edgecolor':'white','linewidth':2}
        )
        for at in autotexts: at.set_fontsize(8)
        for t in texts: t.set_fontsize(8)
        ax.set_title('Member Distribution by Segment', fontweight='bold', fontsize=11)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_right:
        st.markdown('<div class="section-header">CLV vs Churn Risk by Segment</div>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(6,4))
        for s in [x for x in SEG_ORDER if x != 'Churned Members']:
            sub = df[df['segment_name']==s]
            ax2.scatter(sub['CLV'].clip(upper=25000), sub['churn_probability'],
                       c=PALETTE[s], alpha=0.2, s=6, label=s)
        ax2.axhline(0.3, color='red', linestyle='--', lw=1, label='Risk threshold (30%)')
        ax2.set_xlabel('CLV (CAD)'); ax2.set_ylabel('Churn Probability')
        ax2.legend(fontsize=7, loc='upper right')
        ax2.set_title('Value vs Risk — Active Members', fontweight='bold', fontsize=11)
        ax2.grid(alpha=0.2)
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    # Segment summary table
    st.markdown('<div class="section-header">Segment Performance Summary</div>', unsafe_allow_html=True)
    tbl = seg.set_index('segment_name').reindex(SEG_ORDER)[['n','avg_clv','avg_flights','avg_tenure','churn_rate','churn_prob']].copy()
    tbl.columns = ['Members','Avg CLV','Avg Flights','Tenure (mo)','Actual Churn %','Predicted Churn %']
    tbl['Avg CLV'] = tbl['Avg CLV'].map('${:,.0f}'.format)
    tbl['Actual Churn %'] = (tbl['Actual Churn %']*100).map('{:.1f}%'.format)
    tbl['Predicted Churn %'] = (tbl['Predicted Churn %']*100).map('{:.1f}%'.format)
    tbl['Avg Flights'] = tbl['Avg Flights'].map('{:.0f}'.format)
    tbl['Tenure (mo)'] = tbl['Tenure (mo)'].map('{:.0f}'.format)
    tbl['Members'] = tbl['Members'].map('{:,}'.format)
    st.dataframe(tbl, use_container_width=True)

    st.markdown('<div class="insight-box">💡 <strong>Key Insight:</strong> Passive Loyalists (44.5% of all members) have been with the program for 49 months on average but are stuck at Star tier with the lowest CLV. Tier upgrade activation is the single largest CLV opportunity in the program.</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════
# PAGE 2 — Segment Explorer
# ════════════════════════════════════════════════════════════════════════
elif page == "🎯 Segment Explorer":
    st.title("Segment Explorer")
    seg_choice = st.selectbox("Select a segment to explore", SEG_ORDER)
    sub = df[df['segment_name'] == seg_choice]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Members", f"{len(sub):,}")
    c2.metric("Avg CLV", f"${sub['CLV'].mean():,.0f}")
    c3.metric("Avg Flights", f"{sub['total_flights'].mean():.0f}")
    c4.metric("Churn Rate", f"{sub['churned'].mean()*100:.1f}%")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Flight Activity Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5,3))
        ax.hist(sub['total_flights'].clip(upper=100), bins=30, color=PALETTE[seg_choice], edgecolor='white', alpha=0.85)
        ax.set_xlabel('Total Flights'); ax.set_ylabel('Members')
        ax.set_title(f'{seg_choice} — Flight Distribution', fontsize=10)
        st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        st.markdown('<div class="section-header">Churn Probability Distribution</div>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(5,3))
        ax2.hist(sub['churn_probability'], bins=30, color=PALETTE[seg_choice], edgecolor='white', alpha=0.85)
        ax2.axvline(0.3, color='red', linestyle='--', lw=1.5, label='High Risk (30%)')
        ax2.set_xlabel('Churn Probability'); ax2.set_ylabel('Members')
        ax2.legend(fontsize=8)
        ax2.set_title(f'{seg_choice} — Churn Risk', fontsize=10)
        st.pyplot(fig2, use_container_width=True); plt.close()

    # Demographics breakdown
    st.markdown('<div class="section-header">Demographics Profile</div>', unsafe_allow_html=True)
    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        card_d = sub['Loyalty Card'].value_counts()
        fig3, ax3 = plt.subplots(figsize=(4,3))
        ax3.bar(card_d.index, card_d.values, color=['#e74c3c','#3498db','#2ecc71'])
        ax3.set_title('Card Tier', fontsize=9); st.pyplot(fig3, use_container_width=True); plt.close()
    with dc2:
        prov_d = sub['Province'].value_counts().head(6)
        fig4, ax4 = plt.subplots(figsize=(4,3))
        ax4.barh(prov_d.index[::-1], prov_d.values[::-1], color='#3498db', alpha=0.8)
        ax4.set_title('Top Provinces', fontsize=9); st.pyplot(fig4, use_container_width=True); plt.close()
    with dc3:
        edu_d = sub['Education'].value_counts()
        fig5, ax5 = plt.subplots(figsize=(4,3))
        ax5.pie(edu_d.values, labels=edu_d.index, autopct='%1.0f%%',
               textprops={'fontsize':7}, startangle=90)
        ax5.set_title('Education', fontsize=9); st.pyplot(fig5, use_container_width=True); plt.close()

# ════════════════════════════════════════════════════════════════════════
# PAGE 3 — Churn Risk Finder
# ════════════════════════════════════════════════════════════════════════
elif page == "⚠️ Churn Risk Finder":
    st.title("Churn Risk Finder")
    st.caption("Identify members who need immediate attention")

    risk_thresh = st.slider("Churn probability threshold", 0.0, 1.0, 0.20, 0.05,
                            help="Show members with churn probability above this value")

    at_risk_members = active[active['churn_probability'] >= risk_thresh].copy()
    at_risk_members = at_risk_members.sort_values('churn_probability', ascending=False)

    st.markdown(f"**{len(at_risk_members):,} members** above {risk_thresh*100:.0f}% churn risk threshold  |  Combined CLV: **${at_risk_members['CLV'].sum():,.0f}**")

    # Risk band chart
    bins   = [0, 0.1, 0.2, 0.3, 0.5, 1.0]
    labels_rb = ['<10%','10-20%','20-30%','30-50%','>50%']
    active_copy = active.copy()
    active_copy['risk_band'] = pd.cut(active_copy['churn_probability'], bins=bins, labels=labels_rb)
    rb_counts = active_copy['risk_band'].value_counts().reindex(labels_rb)

    col1, col2 = st.columns([1,2])
    with col1:
        fig, ax = plt.subplots(figsize=(4,3))
        rc = ['#2ecc71','#a8e063','#f39c12','#e67e22','#e74c3c']
        ax.bar(labels_rb, rb_counts.values, color=rc, edgecolor='white')
        ax.set_title('Members by Risk Band', fontsize=10)
        ax.set_ylabel('Members')
        for i,v in enumerate(rb_counts.values):
            ax.text(i, v+10, f'{v:,}', ha='center', fontsize=8, fontweight='bold')
        st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        display_cols = ['Loyalty Number','segment_name','Loyalty Card','Province',
                        'CLV','total_flights','flight_consistency','churn_probability']
        display_df = at_risk_members[display_cols].head(50).copy()
        display_df['CLV'] = display_df['CLV'].map('${:,.0f}'.format)
        display_df['churn_probability'] = (display_df['churn_probability']*100).map('{:.1f}%'.format)
        display_df['flight_consistency'] = display_df['flight_consistency'].map('{:.2f}'.format)
        display_df.columns = ['ID','Segment','Card','Province','CLV','Flights','Consistency','Churn Prob']
        st.dataframe(display_df, use_container_width=True, height=280)

    st.markdown('<div class="alert-box">⚠️ <strong>Action Required:</strong> At-Risk Valuables with churn probability > 10% should receive the 90-day onboarding sprint. Champions above 5% threshold should receive a personal outreach call within 7 days.</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════
# PAGE 4 — Provincial Analysis
# ════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Provincial Analysis":
    st.title("Provincial Analysis")
    prov_df = df.groupby('Province').agg(
        n=('Loyalty Number','count'),
        avg_clv=('CLV','mean'),
        churn_rate=('churned','mean'),
        avg_flights=('total_flights','mean'),
        avg_tenure=('tenure_months','mean')
    ).reset_index().sort_values('churn_rate', ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Churn Rate by Province</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6,5))
        prov_sort = prov_df.sort_values('churn_rate')
        colors = ['#e74c3c' if c > df['churned'].mean() else '#3498db' for c in prov_sort['churn_rate']]
        ax.barh(prov_sort['Province'], prov_sort['churn_rate']*100, color=colors)
        ax.axvline(df['churned'].mean()*100, color='black', linestyle='--', lw=1.2,
                   label=f'National avg ({df["churned"].mean()*100:.1f}%)')
        ax.set_xlabel('Churn Rate (%)'); ax.legend(fontsize=8)
        ax.set_title('Red = Above National Average', fontsize=10)
        st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        st.markdown('<div class="section-header">CLV vs Churn Rate (Bubble = Member Count)</div>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(6,5))
        sc = ax2.scatter(prov_df['avg_clv'], prov_df['churn_rate']*100,
                        s=prov_df['n']/3, alpha=0.7, c=prov_df['churn_rate'],
                        cmap='RdYlGn_r', edgecolors='grey', linewidths=0.5)
        for _, row in prov_df.iterrows():
            ax2.annotate(row['Province'][:3], (row['avg_clv'], row['churn_rate']*100),
                        fontsize=7, ha='center')
        ax2.set_xlabel('Average CLV (CAD)'); ax2.set_ylabel('Churn Rate (%)')
        ax2.set_title('Provincial Performance Matrix', fontsize=10)
        ax2.grid(alpha=0.2)
        st.pyplot(fig2, use_container_width=True); plt.close()

    st.markdown('<div class="section-header">Province Data Table</div>', unsafe_allow_html=True)
    prov_tbl = prov_df.copy()
    prov_tbl['avg_clv'] = prov_tbl['avg_clv'].map('${:,.0f}'.format)
    prov_tbl['churn_rate'] = (prov_tbl['churn_rate']*100).map('{:.1f}%'.format)
    prov_tbl['avg_flights'] = prov_tbl['avg_flights'].map('{:.0f}'.format)
    prov_tbl['avg_tenure'] = prov_tbl['avg_tenure'].map('{:.0f} mo'.format)
    prov_tbl.columns = ['Province','Members','Avg CLV','Churn Rate','Avg Flights','Avg Tenure']
    st.dataframe(prov_tbl.reset_index(drop=True), use_container_width=True)

    st.markdown('<div class="alert-box">⚠️ <strong>Manitoba & Newfoundland</strong> show the highest churn rates (17.1% each), yet their average CLV is near the national median. Before designing retention offers, conduct a route gap analysis — no loyalty intervention can substitute for network competitiveness.</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════
# PAGE 5 — Retention Playbook
# ════════════════════════════════════════════════════════════════════════
elif page == "📋 Retention Playbook":
    st.title("Retention Playbook")
    st.caption("Segment-specific interventions — every recommendation names WHO, WHEN, WHAT, and HOW to measure success")

    playbook = [
        {
            "segment": "At-Risk Valuables", "color": "#e67e22",
            "trigger": "Enrolled < 3 months, > 5 flights, zero redemptions",
            "intervention": "90-Day Onboarding Sprint: push notifications at flight milestones (5th, 10th, 20th flight), in-app rewards tutorial, offer first redemption bonus (2x value on first redeem)",
            "timeline": "Activate immediately upon trigger, run for 90 days post-enrollment",
            "metric": "Redemption ratio > 0.05 within 90 days of trigger",
            "why": "These members are flying the most (avg 54 flights) but have the lowest redemption rate (0.004). They are unaware of or unengaged with rewards — a targeted education sprint converts active behavior into program engagement before dropout.",
            "tradeoff": "Cost of 2x first-redemption bonus (~$15-25 CAD per member). Expected reduction in 10.6% churn rate to <5% justifies the spend given avg CLV of $7,960."
        },
        {
            "segment": "Passive Loyalists", "color": "#3498db",
            "trigger": "Tenure > 24 months, Star tier, CLV < $7,000",
            "intervention": "Tier-Upgrade Campaign: personalised email showing points gap to Nova status, bonus 500 points on next 2 flights, tier progress tracker in app",
            "timeline": "Q1 campaign launch, 6-month window, re-evaluate quarterly",
            "metric": "Nova upgrade rate > 15% of targeted Passive Loyalists within 6 months",
            "why": "7,456 members (44.5% of the program) have been loyal for 49 months but generate only $5,669 avg CLV. Their entire cohort holds Star tier — the program has never given them a reason to spend more. A visible upgrade pathway converts habitual flyers into engaged members.",
            "tradeoff": "Bonus points cost per upgrade campaign: ~$8 CAD per member. If 15% of 7,456 upgrade to Nova (CLV uplift ~$2,400 from $5,669 to $8,046), that is $2.7M in incremental CLV against ~$90k campaign cost."
        },
        {
            "segment": "New Low-Engagers", "color": "#9b59b6",
            "trigger": "Enrolled < 12 months, < 10 total flights, no tier upgrade",
            "intervention": "Double-points offer on next 3 flights, tier roadmap email with visual progress bar, personalised route recommendation based on home province",
            "timeline": "Month 2 post-enrollment trigger (allow 1 month for natural onboarding)",
            "metric": "Flight frequency increase of +20% vs control group in 90-day window",
            "why": "This segment has the second-highest redemption ratio (0.019) among active members, signaling reward-motivation. They are not flying enough to graduate to higher tiers — the double-points offer directly addresses the frequency gap.",
            "tradeoff": "Risk of discount dependency: if double-points becomes the norm, members may not fly without it. Strict one-time offer with clear messaging prevents anchoring."
        },
        {
            "segment": "Champions", "color": "#2ecc71",
            "trigger": "Churn probability rises above 5% (real-time model scoring)",
            "intervention": "Dedicated relationship manager outreach (call or personalised email), exclusive route access or confirmed upgrade voucher, no-questions-asked resolution offer",
            "timeline": "Within 7 days of threshold breach — speed is critical for high-CLV members",
            "metric": "Churn probability returns below 2% within 30 days; no formal cancellation within 90 days",
            "why": "Champions represent 19.3% of members but a disproportionate share of CLV ($13,772 avg). A 1 percentage point improvement in their retention rate recovers ~$445k in CLV. Personalised outreach costs less than one lost Champion.",
            "tradeoff": "Relationship manager capacity constraint — this playbook requires trained staff for personal outreach. Recommend automating triage (email first, call if no response in 48h)."
        },
        {
            "segment": "Churned Members (Win-Back)", "color": "#e74c3c",
            "trigger": "Cancelled within 12 months, CLV > $8,000 (above median)",
            "intervention": "Win-back offer: 1,000 bonus points on reactivation, membership fee waiver for 12 months, personalised message from a named customer success manager",
            "timeline": "Send within 30 days of cancellation; second touchpoint at 90 days if no response",
            "metric": "Reactivation rate > 5% of targeted cohort (high-CLV recent churners)",
            "why": "2,067 formally cancelled members have avg CLV of $8,192 — above the program median. Recent cancellations (within 12 months) are most recoverable. Targeting only high-CLV churners ensures cost efficiency.",
            "tradeoff": "Win-back campaigns risk re-attracting members who will churn again. Cap win-back offers to one attempt per member lifetime."
        }
    ]

    for p in playbook:
        with st.expander(f"**{p['segment']}** — {p['trigger']}", expanded=(p['segment']=='At-Risk Valuables')):
            cols = st.columns([2,1])
            with cols[0]:
                st.markdown(f"**Intervention:** {p['intervention']}")
                st.markdown(f"**Timeline:** {p['timeline']}")
                st.markdown(f"**Success Metric:** {p['metric']}")
            with cols[1]:
                st.markdown(f"**Why This Works:**")
                st.info(p['why'])
                st.warning(f"**Trade-off:** {p['tradeoff']}")

    st.markdown("")
    st.markdown('<div class="section-header">Aurora Tier Churn — Investigation Priority</div>', unsafe_allow_html=True)
    st.markdown('<div class="alert-box">Aurora members churn at 15.0% — the highest of any tier, despite having the highest CLV ($10,673). This is counter-intuitive. Hypothesis: Aurora members are frequent travelers holding status across multiple airlines and cancel when a competitor offers a superior status match. Recommended action: structured exit survey of recently cancelled Aurora members before designing any retention offer for this segment.</div>', unsafe_allow_html=True)

