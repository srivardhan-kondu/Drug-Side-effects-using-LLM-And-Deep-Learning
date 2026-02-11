import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import time
import plotly.express as px

# Import Services
from services.llm_service import explain_side_effect, analyze_drug_risk, get_drug_details_from_llm
from data.drug_database_manager import db

# Configuration
st.set_page_config(
    page_title="PharmaAI - Drug Safety Analysis",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    h1 {
        color: #60a5fa;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h2, h3 {
        color: #93c5fd;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
    }
    .stAlert {
        border-radius: 8px;
    }
    div[data-testid="stMetric"] {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State Initialization ──
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "analysis_drug" not in st.session_state:
    st.session_state.analysis_drug = ""
if "analysis_source" not in st.session_state:
    st.session_state.analysis_source = ""
if "analysis_condition" not in st.session_state:
    st.session_state.analysis_condition = ""

# Helper to manage DB reloading
def load_db():
    db.load_database()
    return db

# Sidebar
with st.sidebar:
    st.title("💊 PharmaAI")
    st.write("Drug Side Effect Prediction System")
    
    menu = st.radio("Navigation", ["🔍 Analyze Drug", "📚 Drug Database", "ℹ️ About"])
    
    st.markdown("---")
    st.caption("Powered by PyTorch & OpenAI GPT")
    st.caption("v2.1.0 | Production Build")

if menu == "🔍 Analyze Drug":
    st.title("Drug Risk Analysis")
    st.markdown("Enter any drug name to analyze its safety profile, predicted side effects, and get AI-powered explanations.")
    
    # Reload DB to get latest
    current_db = load_db()
    
    # ── Input Section ──
    drug_input = st.text_input(
        "Enter Drug Name:",
        placeholder="e.g., Aspirin, Metformin, Ozempic...",
        help="Type any drug name. We'll check our database first. If not found, our AI agent will fetch real drug data automatically."
    )
    condition = st.text_input("Medical Condition (Optional):", placeholder="e.g., Type 2 Diabetes, Hypertension")
    
    analyze_btn = st.button("🔍 Analyze Risk Profile")
    
    # Trim input
    selected_drug = drug_input.strip() if drug_input else ""
        
    # ── Run Analysis (stores result in session_state) ──
    if analyze_btn and selected_drug:
        with st.spinner("Analyzing pharmacological data..."):
            # Step 1: Check database first (transparent to user)
            drug_data = current_db.get_drug(selected_drug)
            source = "database"
            
            # Step 2: If not found, silently fall back to LLM
            if not drug_data:
                try:
                    new_data = get_drug_details_from_llm(selected_drug)
                    if new_data and isinstance(new_data.get("side_effects"), dict) and len(new_data["side_effects"]) > 0:
                        # Add to DB for future use
                        db.add_drug(selected_drug, new_data)
                        drug_data = new_data
                        source = "ai_discovery"
                    else:
                        st.error(f"❌ Could not find data for **\"{selected_drug}\"**. Please check the spelling and try again.")
                        st.info("💡 Try common drug names like: Aspirin, Metformin, Ibuprofen, Amoxicillin, Atorvastatin")
                        st.stop()
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.stop()
            
            # Store in session state so it persists across button clicks
            st.session_state.analysis_result = drug_data
            st.session_state.analysis_drug = selected_drug
            st.session_state.analysis_source = source
            st.session_state.analysis_condition = condition
    
    # ── Display Results (from session state) ──
    if st.session_state.analysis_result:
        drug_data = st.session_state.analysis_result
        selected_drug = st.session_state.analysis_drug
        source = st.session_state.analysis_source
        condition_val = st.session_state.analysis_condition
        
        st.divider()
        
        # Header
        st.subheader(f"📋 Results: {selected_drug}")
        if source == "ai_discovery":
            st.caption("✨ Dynamically retrieved via AI Research Agent")
        else:
            st.caption("✅ Verified Database Entry")
            
        # Info Cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Category", drug_data.get("category", "Unknown"))
        c2.metric("Mol. Weight", f"{drug_data.get('molecular_weight', 0):.1f}")
        c3.metric("LogP", f"{drug_data.get('log_p', 0):.2f}")
        
        # Calculate risk level from severity scores
        effects = drug_data.get("side_effects", {})
        if effects and isinstance(effects, dict):
            avg_severity = sum(effects.values()) / len(effects)
            risk = "🟢 Low" if avg_severity < 0.35 else "🟡 Medium" if avg_severity < 0.6 else "🔴 High"
        else:
            risk = "Unknown"
        c4.metric("Risk Level", risk)
        
        # Side Effects Chart
        if effects and isinstance(effects, dict):
            # Prepare data for plot
            plot_data = []
            for effect, severity in effects.items():
                sev_val = float(severity) if isinstance(severity, (int, float)) else 0.5
                plot_data.append({
                    "Side Effect": effect, 
                    "Severity Score": sev_val,
                })
            
            df_plot = pd.DataFrame(plot_data).sort_values("Severity Score", ascending=True)
            
            fig = px.bar(
                df_plot, 
                x="Severity Score", 
                y="Side Effect", 
                orientation='h',
                color="Severity Score",
                color_continuous_scale="RdYlGn_r",
                range_x=[0, 1],
                title="Predicted Side Effects & Severity Profile"
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e2e8f0',
                height=max(300, len(effects) * 55)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # ── FR5: LLM-based Explanation Section ──
            st.divider()
            st.subheader("🤖 AI-Powered Side Effect Explanation")
            st.caption("Select a side effect and click 'Explain' to understand why this drug causes it, powered by GPT.")
            
            effect_list = df_plot["Side Effect"].tolist()
            selected_effect = st.selectbox(
                "Select a side effect to explain:", 
                effect_list,
                index=0,
            )
            
            explain_btn = st.button("🧠 Explain Mechanism")
            
            if explain_btn and selected_effect:
                with st.spinner(f"Generating AI explanation for '{selected_effect}'..."):
                    explanation = explain_side_effect(
                        selected_drug, 
                        selected_effect, 
                        condition_val if condition_val else None
                    )
                    
                    # Store explanation in session state too
                    st.session_state["last_explanation"] = explanation
                    st.session_state["last_explained_effect"] = selected_effect
            
            # Display explanation from session state
            if "last_explanation" in st.session_state and st.session_state.get("last_explained_effect"):
                explained_effect = st.session_state["last_explained_effect"]
                explanation = st.session_state["last_explanation"]
                
                st.markdown("---")
                st.markdown(f"### Why does **{selected_drug}** cause **{explained_effect}**?")
                
                if explanation.startswith("⚠️"):
                    st.warning(explanation)
                else:
                    st.info(explanation)
                
                # Show severity context
                sev_score = effects.get(explained_effect, 0)
                sev_val = float(sev_score) if isinstance(sev_score, (int, float)) else 0
                if sev_val >= 0.7:
                    st.error(f"⚠️ Severity: **Severe** ({sev_val:.0%}) — Consult your doctor immediately if this occurs.")
                elif sev_val >= 0.4:
                    st.warning(f"⚡ Severity: **Moderate** ({sev_val:.0%}) — Monitor symptoms and report to your healthcare provider.")
                else:
                    st.success(f"✅ Severity: **Mild** ({sev_val:.0%}) — Generally manageable. Inform your doctor if persistent.")
        else:
            st.warning("No side effect data available for this drug.")

elif menu == "📚 Drug Database":
    st.title("📚 Drug Knowledge Base")
    current_db = load_db()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"Currently tracking **{len(current_db.drugs)}** pharmaceutical compounds across multiple drug classes.")
    with col2:
        if st.button("🔄 Refresh"):
            st.rerun()
    
    # Search/filter
    search = st.text_input("🔍 Filter drugs:", placeholder="Type to search...")
    
    # Build table data
    table_data = []
    for name, data in current_db.drugs.items():
        if search and search.lower() not in name.lower():
            continue
        eff = data.get("side_effects", {})
        top_effects = ", ".join(list(eff.keys())[:3]) if isinstance(eff, dict) else str(eff)
        table_data.append({
            "Drug Name": name,
            "Category": data.get("category", "Unknown"),
            "Mol. Weight": data.get("molecular_weight", 0),
            "Top Side Effects": top_effects
        })
    
    if table_data:
        df_db = pd.DataFrame(table_data)
        st.dataframe(df_db, use_container_width=True, hide_index=True)
    else:
        st.info("No drugs match your search.")

elif menu == "ℹ️ About":
    st.title("About PharmaAI")
    st.markdown("""
    **PharmaAI** is an advanced drug safety analysis tool powered by Deep Learning and Large Language Models.
    
    ### Features
    - **Deep Learning Model**: PyTorch neural network with attention mechanism for side effect prediction.
    - **Dynamic Knowledge Base**: Automatically discovers new drugs using GPT when not in the database.
    - **AI Explanations**: GPT-powered explanations of why drugs cause specific side effects.
    - **Anti-Hallucination**: Strict validation protocols to ensure medical accuracy.
    - **Interactive Visualizations**: Plotly-powered severity charts.
    
    ### Architecture
    ```
    User Input → Database Lookup → [If not found] → GPT Fallback → Add to DB
                                   → Side Effect Display
                                   → GPT Explanation (FR5)
    ```
    
    ### Tech Stack
    - Python 3.9+ | PyTorch | Streamlit | OpenAI GPT API | Plotly
    """)
