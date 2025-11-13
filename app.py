import streamlit as st
import pandas as pd
import json
import os
import time
from datetime import datetime
from typing import TypedDict, Annotated
import operator

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.tools import SerperDevTool
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field, ValidationError

# --- Configuration & Environment Setup ---
# You MUST set GROQ_API_KEY and SERPER_API_KEY in Streamlit secrets
SERPER_API_KEY = st.secrets.get("SERPER_API_KEY")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY or not SERPER_API_KEY:
    st.error("❌ ERROR: Both GROQ_API_KEY and SERPER_API_KEY must be set in Streamlit secrets.")
    st.stop()

# --- LLM and Tool Initialization ---
try:
    # Use a powerful, fast Groq model
    llm_groq = ChatGroq(
        model="llama3-70b-8192", 
        groq_api_key=GROQ_API_KEY,
        temperature=0
    )
    search_tool = SerperDevTool(api_key=SERPER_API_KEY)
    st.info("Using Groq (Llama 3 70B) for high-speed processing.")
except Exception as e:
    st.error(f"Failed to initialize Groq or Serper tools: {e}")
    st.stop()


# --- Pydantic Output Schema (CompanyData) ---
class CompanyData(BaseModel):
    # Basic Company Info
    linkedin_url: str = Field(description="LinkedIn URL and source/confidence.")
    company_website_url: str = Field(description="Official company website URL and source/confidence.")
    industry_category: str = Field(description="Industry category and source.")
    employee_count_linkedin: str = Field(description="Employee count range and source.")
    headquarters_location: str = Field(description="Headquarters city, country, and source.")
    revenue_source: str = Field(description="Revenue data point and specific source (ZoomInfo/Owler/Apollo/News).")
    
    # Core Research Fields
    branch_network_count: str = Field(description="Number of branches/facilities mentioned online and source.")
    expansion_news_12mo: str = Field(description="Summary of expansion news in the last 12 months and source link.")
    digital_transformation_initiatives: str = Field(description="Details on smart infra or digital programs and source link.")
    it_leadership_change: str = Field(description="Name and title of new CIO/CTO/Head of Infra if changed recently and source link.")
    existing_network_vendors: str = Field(description="Mentioned network vendors or tech stack and source.")
    wifi_lan_tender_found: str = Field(description="Yes/No and source link if a tender was found.")
    iot_automation_edge_integration: str = Field(description="Details on IoT/Automation/Edge mentions and source link.")
    cloud_adoption_gcc_setup: str = Field(description="Details on Cloud Adoption or Global Capability Centers (GCC) and source link.")
    physical_infrastructure_signals: str = Field(description="Any physical infra signals (new office, factory etc) and source link.")
    it_infra_budget_capex: str = Field(description="IT Infra Budget or Capex allocation details and source.")
    
    # Analysis Fields
    why_relevant_to_syntel: str = Field(description="Why this company is a relevant lead for Syntel (based on all data).")
    # Must be an integer for Pydantic validation. The LLM will be instructed to output 0 if no score can be calculated.
    intent_scoring: int = Field(description="Intent score 1-10 based on buying signals detected.") 


# --- LangGraph State Definition ---
class AgentState(TypedDict):
    """Represents the shared context/state of the graph's execution."""
    company_name: str
    raw_research: str
    validated_data_text: str # Holds text output from the validation node
    final_json_data: dict # Holds the final Pydantic object as a dictionary
    messages: Annotated[list, operator.add] 


# --- Graph Nodes (The Agents) ---

def research_node(state: AgentState) -> AgentState:
    """Node 1: Executes deep search and generates raw notes."""
    st.session_state.status_text.info(f"Phase 1/3: Conducting deep search for {state['company_name']}...")
    st.session_state.progress_bar.progress(33)
    
    company = state["company_name"]
    
    # Compile a comprehensive search query
    search_query = (
        f"'{company}' revenue 'LinkedIn' employee count headquarters AND ('expansion' OR 'IT budget' OR 'CIO change' OR 'digital transformation')"
    )
    
    # Execute the search tool
    search_results = search_tool.run(search_query)
    
    # Prompt for LLM to process raw search results
    research_prompt = f"""
    You are an expert Business Intelligence Researcher. Your goal is to find specific, citable data points for {company}.
    
    Search Results:
    ---
    {search_results}
    ---
    
    Based ONLY on the search results, generate comprehensive research notes for the fields listed in the final Pydantic schema (website, LinkedIn, revenue, expansion, IT leadership, vendors, etc.).
    CRITICAL: For every data point, provide the data AND a source reference. If a data point is not found, state: 'Not Found (No Source)'. 
    
    Output the raw research notes as a single block of text.
    """
    
    raw_research = llm_groq.invoke([
        SystemMessage(content=research_prompt),
        HumanMessage(content=f"Generate the raw research notes for {company}.")
    ]).content

    return {"raw_research": raw_research}


def validation_node(state: AgentState) -> AgentState:
    """Node 2: Validates the raw notes and calculates the Intent Score."""
    st.session_state.status_text.info(f"Phase 2/3: Validating data and calculating Intent Score...")
    st.session_state.progress_bar.progress(66)
    
    raw_research = state["raw_research"]
    company = state["company_name"]
    
    validation_prompt = f"""
    You are a Data Quality Specialist. Review the raw research notes and prepare them for final JSON formatting.
    
    1.  Ensure all required data fields are clearly separated and assigned a value (either the found data + source, or 'Not Found (No Source)').
    2.  Calculate the 'Intent Scoring' (1-10) based on buying signals detected (expansion, IT budget, IT leadership changes, IoT/Cloud adoption). If no signals are found, set the score to **0**.
    3.  Generate the 'Why Relevent to Syntel' analysis based on the strongest signals.

    Raw Research Notes:
    ---
    {raw_research}
    ---
    
    Output the processed data in a clear, key-value format ready for JSON conversion. DO NOT output JSON yet.
    """
    
    validated_output = llm_groq.invoke([
        SystemMessage(content=validation_prompt),
        HumanMessage(content=f"Validate and enrich the data for {company} now.")
    ]).content

    return {"validated_data_text": validated_output}


def formatter_node(state: AgentState) -> AgentState:
    """Node 3: Formats the validated data into the strict Pydantic JSON schema."""
    st.session_state.status_text.info(f"Phase 3/3: Converting to final Pydantic JSON...")
    st.session_state.progress_bar.progress(90)
    
    validated_data_text = state["validated_data_text"]
    
    # Use Groq's structured output capability to force compliance with the schema
    formatting_prompt = f"""
    You are a JSON Schema Specialist. Your task is to convert the following validated data into the exact JSON format defined by the CompanyData Pydantic schema.
    
    - **CRITICAL**: Every single field in the Pydantic schema must be present.
    - If a value is 'Not Found (No Source)', use that string for string fields.
    - **intent_scoring MUST be an integer (1-10 or 0).**
    
    Validated Data:
    ---
    {validated_data_text}
    ---
    
    Output ONLY the JSON object.
    """

    # Use .with_structured_output(CompanyData) to enforce the schema
    final_pydantic_object = llm_groq.with_structured_output(CompanyData).invoke([
        SystemMessage(content=formatting_prompt),
        HumanMessage(content="Generate the final JSON for CompanyData.")
    ])

    return {"final_json_data": final_pydantic_object.dict()}


# --- Graph Construction ---
def build_graph():
    """Builds and compiles the sequential LangGraph workflow."""
    workflow = StateGraph(AgentState)

    workflow.add_node("research", research_node)
    workflow.add_node("validate", validation_node)
    workflow.add_node("format", formatter_node)

    # Define the sequential flow
    workflow.add_edge(START, "research")
    workflow.add_edge("research", "validate")
    workflow.add_edge("validate", "format")
    workflow.add_edge("format", END)

    return workflow.compile()

# Build the graph once
app = build_graph()


# --- Helper Function for Custom Table Formatting ---
def format_data_for_display(company_input: str, validated_data: CompanyData) -> pd.DataFrame:
    """Transforms the Pydantic model into the specific 1-row table format requested."""
    data_dict = validated_data.dict()
    
    mapping = {
        "Company Name": "company_name_placeholder",
        "LinkedIn URL": "linkedin_url",
        "Company Website URL": "company_website_url",
        "Industry Category": "industry_category",
        "Employee Count (LinkedIn)": "employee_count_linkedin",
        "Headquarters (Location)": "headquarters_location",
        "Revenue (ZoomInfo / Owler / Apollo)": "revenue_source",
        "Branch Network / Facilities Count": "branch_network_count",
        "Expansion News (Last 12 Months)": "expansion_news_12mo",
        "Digital Transformation Initiatives / Smart Infra Programs": "digital_transformation_initiatives",
        "IT Infrastructure Leadership Change (CIO / CTO / Head Infra)": "it_leadership_change",
        "Existing Network Vendors / Tech Stack": "existing_network_vendors",
        "Recent Wi-Fi Upgrade or LAN Tender Found": "wifi_lan_tender_found",
        "IoT / Automation / Edge Integration Mentioned": "iot_automation_edge_integration",
        "Cloud Adoption / GCC Setup": "cloud_adoption_gcc_setup",
        "Physical Infrastructure Signals": "physical_infrastructure_signals",
        "IT Infra Budget / Capex Allocation": "it_infra_budget_capex",
        "Why Relevent to Syntel": "why_relevant_to_syntel",
        "Intent scoring": "intent_scoring",
    }
    
    row_data = {}
    for display_col, pydantic_field in mapping.items():
        if display_col == "Company Name":
            row_data[display_col] = company_input 
        else:
            # Ensure integer field is converted to string for display consistency
            if pydantic_field == "intent_scoring":
                 row_data[display_col] = str(data_dict.get(pydantic_field, "N/A"))
            else:
                 row_data[display_col] = data_dict.get(pydantic_field, "N/A")
            
    df = pd.DataFrame([row_data])
    df.index = ['']
    return df


# --- Streamlit UI ---
st.set_page_config(
    page_title="Syntel BI Agent (Groq/LangGraph)", 
    layout="wide"
)

st.title("Syntel Company Data AI Agent (LangGraph/Groq)")
st.markdown("### High-Speed Research Pipeline ")

# Initialize session state for UI components
if 'research_history' not in st.session_state:
    st.session_state.research_history = []
if 'company_input' not in st.session_state:
    st.session_state.company_input = "Snowman Logistics"
if 'status_text' not in st.session_state:
    st.session_state.status_text = st.empty()
if 'progress_bar' not in st.session_state:
    st.session_state.progress_bar = st.empty()

# Input section
col1, col2 = st.columns([2, 1])
with col1:
    company_input = st.text_input("Enter the company name to research:", st.session_state.company_input, key="company_input_widget")
with col2:
    with st.form("research_form"):
        submitted = st.form_submit_button("Start Deep Research", type="primary")

if submitted:
    st.session_state.company_input = company_input
    
    if not company_input:
        st.warning("Please enter a company name.")
        st.stop()

    # Initialize the progress bar and status text containers
    st.session_state.progress_bar = st.progress(0)
    st.session_state.status_text = st.empty()
    
    with st.spinner(f"AI Graph is running for **{company_input}**..."):
        try:
            # Initial State for LangGraph
            initial_state: AgentState = {
                "company_name": company_input,
                "raw_research": "",
                "validated_data_text": "",
                "final_json_data": {},
                "messages": []
            }

            # Invoke the LangGraph app
            final_state = app.invoke(initial_state)
            
            # --- Result Processing ---
            
            # The result is the final_json_data dictionary directly
            data_dict = final_state["final_json_data"]
            
            # Validate the final JSON dictionary against the Pydantic schema
            validated_data = CompanyData(**data_dict) 
            
            st.session_state.progress_bar.progress(100)
            st.session_state.status_text.success(f"Research Complete for {company_input}! (Groq Processing)")
            
            research_entry = {
                "company": company_input,
                "timestamp": datetime.now().isoformat(),
                "data": validated_data.dict()
            }
            st.session_state.research_history.append(research_entry)
            
            # --- Display Tabs ---
            tab1, tab2, tab3 = st.tabs(["📊 Final Report Table", "📋 Detailed View", "📈 Analysis Summary"])
            
            with tab1:
                st.subheader(f"Final Business Intelligence Report for {company_input}")
                final_df = format_data_for_display(company_input, validated_data)
                st.dataframe(final_df, use_container_width=True, height=200) 
                st.caption("The table can be scrolled horizontally to view all columns.")

            with tab2:
                st.subheader("Detailed Research Results")
                
                categories = {
                    # ... (categories list as before, using validated_data.dict())
                    "Basic Company Info": [
                        "linkedin_url", "company_website_url", "industry_category",
                        "employee_count_linkedin", "headquarters_location", "revenue_source"
                    ],
                    "Core Business Intelligence": [
                        "branch_network_count", "expansion_news_12mo", "digital_transformation_initiatives",
                        "it_leadership_change", "existing_network_vendors", "wifi_lan_tender_found",
                        "iot_automation_edge_integration", "cloud_adoption_gcc_setup", 
                        "physical_infrastructure_signals", "it_infra_budget_capex"
                    ],
                    "Analysis & Scoring": [
                        "why_relevant_to_syntel", "intent_scoring"
                    ]
                }

                for category, fields in categories.items():
                    with st.expander(category, expanded=True):
                        for field in fields:
                            if field in data_dict:
                                st.markdown(f"**{field.replace('_', ' ').title()}:** {data_dict[field]}")
                                st.divider()
            
            with tab3:
                st.subheader("Business Intelligence Analysis")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Intent Score", f"{validated_data.intent_scoring}/10")
                with col2:
                    filled_fields = sum(1 for value in validated_data.dict().values() if value and str(value).strip() and "not found" not in str(value).lower() and "mock:" not in str(value).lower())
                    total_fields = len(validated_data.dict())
                    completeness = (filled_fields / total_fields) * 100
                    st.metric("Data Completeness", f"{completeness:.1f}%")
                with col3:
                    st.metric("Research Date", datetime.now().strftime("%Y-%m-%d"))
                
                st.subheader("Relevance to Syntel")
                st.info(validated_data.why_relevant_to_syntel)
                
                st.subheader("Download Options")
                
                # Generate the final DataFrame for download consistency
                final_df_download = format_data_for_display(company_input, validated_data)

                # 1. Download JSON
                json_filename = f"{company_input.replace(' ', '_')}_data.json"
                st.download_button(
                    label="Download JSON Data",
                    data=json.dumps(validated_data.dict(), indent=2),
                    file_name=json_filename,
                    mime="application/json"
                )

                # 2. Download CSV
                csv_data = final_df_download.to_csv(index=False).encode('utf-8')
                csv_filename = f"{company_input.replace(' ', '_')}_data.csv"
                st.download_button(
                    label="Download CSV Data",
                    data=csv_data,
                    file_name=csv_filename,
                    mime="text/csv"
                )

                # 3. Download TSV (Tab Separated Values)
                tsv_data = final_df_download.to_csv(index=False, sep='\t').encode('utf-8')
                tsv_filename = f"{company_input.replace(' ', '_')}_data.tsv"
                st.download_button(
                    label="Download TSV Data",
                    data=tsv_data,
                    file_name=tsv_filename,
                    mime="text/tab-separated-values"
                )
                        
        except Exception as e:
            st.session_state.progress_bar.progress(100)
            st.error(f"Research failed: {type(e).__name__} - {str(e)}")
            st.markdown("""
            **Common Issues:**
            - Check your **`GROQ_API_KEY`** and **`SERPER_API_KEY`** in secrets.
            - Groq has very high speed but still has limits; try waiting 10-15 seconds before retrying.
            """)

# --- Research History (Sidebar) ---
if st.session_state.research_history:
    st.sidebar.header("Research History")
    for i, research in enumerate(reversed(st.session_state.research_history)):
        original_index = len(st.session_state.research_history) - 1 - i 
        
        with st.sidebar.expander(f"**{research['company']}** - {research['timestamp'][:10]}", expanded=False):
            st.write(f"Intent Score: {research['data'].get('intent_scoring', 'N/A')}/10")
            if st.button(f"Load {research['company']}", key=f"load_{original_index}"):
                st.session_state.company_input = research['company'] 
                st.rerun()

# --- Instructions (Sidebar) ---
with st.sidebar.expander("Setup Instructions ⚙️"):
    st.markdown("""
    This app uses **LangGraph** for workflow control and **Groq (Llama 3 70B)** for high-speed AI processing.

    **You MUST set both keys in your Streamlit Cloud secrets:**

    1.  **Search Tool:**
        - `SERPER_API_KEY`: Get from [serper.dev](https://serper.dev/).
    
    2.  **Language Model (LLM):**
        - **`GROQ_API_KEY`**: Get from [console.groq.com](https://console.groq.com/).

    **How the LangGraph Pipeline works:**
    1.  **Research Node:** Executes Serper search and generates raw notes.
    2.  **Validation Node:** Reviews and enriches data, calculates Intent Score (0-10).
    3.  **Formatter Node:** Uses Groq's structured output to guarantee perfect Pydantic JSON.
    """)
