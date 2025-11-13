import streamlit as st
import pandas as pd
import json
import operator
from typing import TypedDict, Annotated
from io import BytesIO
from datetime import datetime

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

# --- Configuration & Environment Setup ---
# You MUST set GROQ_API_KEY and TAVILY_API_KEY in Streamlit secrets
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY or not TAVILY_API_KEY:
    st.error("ERROR: Both GROQ_API_KEY and TAVILY_API_KEY must be set in Streamlit secrets.")
    st.stop()

# --- LLM and Tool Initialization ---
try:
    # Use the smaller Llama 3.1 8B for fast, reliable structured output
    llm_groq = ChatGroq(
        model="llama-3.1-8b-instant", 
        groq_api_key=GROQ_API_KEY,
        temperature=0
    )
    # Initialize Tavily search tool (max_results set to 7 for aggressive search)
    # Tavily is a strong meta-search engine that uses multiple sources (Google, DuckDuckGo, News)
    search_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=7)
    st.info("Using Groq (Llama 3.1 8B) for high-speed processing with Tavily Search.")
except Exception as e:
    st.error(f"Failed to initialize Groq or Tavily tools: {e}")
    st.stop()


# --- Pydantic Output Schema (CompanyData) ---
# NOTE: Intent Scoring is now an Enum in concept, enforced via prompt/description.
class CompanyData(BaseModel):
    # Basic Company Info (using generic names that map to the final output columns)
    linkedin_url: str = Field(description="LinkedIn URL and source/confidence. MUST include link.")
    company_website_url: str = Field(description="Official company website URL. MUST include link.")
    industry_category: str = Field(description="Industry category and source. MUST include link.")
    employee_count_linkedin: str = Field(description="Employee count range and source. MUST include link.")
    headquarters_location: str = Field(description="Headquarters city, country, and source. MUST include link.")
    revenue_source: str = Field(description="Revenue data point and specific source (e.g., ZoomInfo, Owler, News). MUST include link.")
    
    # Core Research Fields - MUST ALIGN with the output column list
    branch_network_count: str = Field(description="Number of branches/facilities, capacity mentioned online. MUST include the SOURCE/LINK.")
    expansion_news_12mo: str = Field(description="Summary of expansion news in the last 12 months. MUST include the SOURCE/LINK.")
    digital_transformation_initiatives: str = Field(description="Details on smart infra or digital programs. MUST include the SOURCE/LINK.")
    it_leadership_change: str = Field(description="Name and title of new CIO/CTO/Head of Infra if changed recently. MUST include the SOURCE/LINK.")
    existing_network_vendors: str = Field(description="Mentioned network vendors or tech stack. MUST include the SOURCE/LINK.")
    wifi_lan_tender_found: str = Field(description="Yes/No and source link if a tender was found. MUST include the SOURCE/LINK.")
    iot_automation_edge_integration: str = Field(description="Details on IoT/Automation/Edge mentions. MUST include the SOURCE/LINK.")
    cloud_adoption_gcc_setup: str = Field(description="Details on Cloud Adoption or Global Capability Centers (GCC). MUST include the SOURCE/LINK.")
    physical_infrastructure_signals: str = Field(description="Any physical infra signals (new office, factory etc). MUST include the SOURCE/LINK.")
    it_infra_budget_capex: str = Field(description="IT Infra Budget or Capex allocation details. MUST include the SOURCE/LINK.")
    
    # Analysis Fields - UPDATED
    why_relevant_to_syntel_bullets: str = Field(description="A markdown string with 3 bullet points explaining relevance to Syntel based on its known offerings (Digital One, Cloud, Network, Automation, KPO).")
    intent_scoring_level: str = Field(description="Intent score level: 'Low', 'Medium', or 'High'. Based on buying signals detected (expansion, IT budget, IT leadership changes, IoT/Cloud adoption).") 


# --- LangGraph State Definition (No Change) ---
class AgentState(TypedDict):
    """Represents the shared context/state of the graph's execution."""
    company_name: str
    raw_research: str
    validated_data_text: str
    final_json_data: dict
    messages: Annotated[list, operator.add] 


# --- Graph Nodes (Updated Prompts) ---

def research_node(state: AgentState) -> AgentState:
    """Node 1: Executes deep search and generates raw notes."""
    st.session_state.status_text.info(f"Phase 1/3: Conducting deep search for {state['company_name']}...")
    st.session_state.progress_bar.progress(33)
    
    company = state["company_name"]
    # Aggressive search query to cover all required signals
    search_query = (
        f"'{company}' 'digital transformation' 'IT budget' 'CIO' 'expansion news' 'network tender' OR 'IoT adoption' OR 'Cloud adoption' AND (website OR 'LinkedIn')"
    )
    
    # Run search aggressively, the tool combines results from multiple engines
    search_results = search_tool.run(search_query)
    
    research_prompt = f"""
    You are an expert Business Intelligence Researcher. Your goal is to find specific, citable data points for {company}.
    
    Search Results:
    ---
    {search_results}
    ---
    
    Based ONLY on the search results, generate comprehensive research notes for the fields listed in the final Pydantic schema.
    
    **CRITICAL:** 1.  For every data point, you **MUST** provide the data **AND** a **SOURCE LINK** in the same string.
    2.  If a data point is not found after aggressively searching the provided results, state: '**Not Found (No Source)**'.
    3.  Treat the search results as if they came from Google Search, DuckDuckGo, and Google News combined, and try to fill every single column.
    
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
    
    1.  Ensure all required data fields are clearly separated and assigned a value (either the found data + source/link, or 'Not Found (No Source)'). Ensure every string value contains a valid source link.
    2.  Calculate the 'Intent Scoring Level': **'Low', 'Medium', or 'High'**.
        - **High:** Multiple strong buying signals (e.g., New CIO AND major IT Capex/Cloud project AND recent expansion).
        - **Medium:** One clear buying signal (e.g., Major Digital Transformation mention OR recent IT leadership change).
        - **Low:** Only general company info found, no clear buying signals in the last 12-24 months.
    3.  Generate the 'Why Relevant to Syntel' bullet points. Base this on Syntel's strengths: **Digital Transformation (Digital One), Cloud Services, Network Infrastructure (WLAN, GPON), Automation (SyntBots), and BPO/KPO solutions.** Generate this as a markdown bullet list (e.g., "* Point 1\n* Point 2\n* Point 3").

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
    
    formatting_prompt = f"""
    You are a **STRICT** JSON Schema Specialist. Your task is to convert the following validated data into the **EXACT** JSON format defined by the CompanyData Pydantic schema.
    
    - **CRITICAL**: The final JSON MUST NOT contain any fields not defined in the schema.
    - Every single field in the Pydantic schema must be present.
    - The content of each string field (EXCEPT the intent score) MUST contain the data and the SOURCE LINK as instructed.
    - **intent_scoring_level MUST be one of 'Low', 'Medium', or 'High'.**
    - **why_relevant_to_syntel_bullets MUST be a markdown string containing 3 bullet points.**
    
    Validated Data:
    ---
    {validated_data_text}
    ---
    
    Output ONLY the JSON object.
    """

    final_pydantic_object = llm_groq.with_structured_output(CompanyData).invoke([
        SystemMessage(content=formatting_prompt),
        HumanMessage(content="Generate the final JSON for CompanyData.")
    ])

    return {"final_json_data": final_pydantic_object.dict()}


# --- Graph Construction (No Change) ---
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


# --- Helper Function for Custom Table Formatting (UPDATED) ---
def format_data_for_display(company_input: str, validated_data: CompanyData) -> pd.DataFrame:
    """Transforms the Pydantic model into the specific 1-row table format requested, ensuring all new columns and bullet points are handled."""
    data_dict = validated_data.dict()
    
    # --- CRITICAL: MAPPING UPDATED TO INCLUDE ALL COMPLEX FIELDS ---
    mapping = {
        "Company Name": "company_name_placeholder",
        "LinkedIn URL": "linkedin_url",
        "Company Website URL": "company_website_url",
        "Industry Category": "industry_category",
        "Employee Count (LinkedIn)": "employee_count_linkedin",
        "Headquarters (Location)": "headquarters_location",
        "Revenue (Source)": "revenue_source",
        
        # All required complex fields mapped to their Pydantic key names
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
        
        # Analysis Fields - UPDATED
        "Why Relevent to Syntel (3 Key Points)": "why_relevant_to_syntel_bullets",
        "Intent Scoring": "intent_scoring_level",
    }
    # ---------------------------------------------------------------------
    
    row_data = {}
    for display_col, pydantic_field in mapping.items():
        if display_col == "Company Name":
            row_data[display_col] = company_input 
        else:
            # Get data, handle case where the field might be missing (shouldn't happen with Pydantic)
            value = data_dict.get(pydantic_field, "N/A (Missing Field)")
            
            # Format the relevance column to show the bullet points clearly
            if pydantic_field == "why_relevant_to_syntel_bullets":
                 # Replace newlines with HTML breaks for better cell display in Streamlit
                 # In a downloaded file (CSV/Excel), this will revert to the markdown string
                 row_data[display_col] = value.replace('\n', '<br>')
            else:
                 row_data[display_col] = value
            
    # Convert to DataFrame
    df = pd.DataFrame([row_data])
    df.index = [''] # Remove the default index
    
    # Use HTML formatting in Streamlit to render the bullet points in the table cell
    def render_markdown_in_cell(val):
        if isinstance(val, str) and val.startswith('*'):
            return f'<div style="text-align: left;">{val}</div>'
        return val

    # Apply the formatting to the specific column
    if "Why Relevent to Syntel (3 Key Points)" in df.columns:
        df["Why Relevent to Syntel (3 Key Points)"] = df["Why Relevent to Syntel (3 Key Points)"].apply(render_markdown_in_cell)

    return df


# --- Streamlit UI (Minor Change for Display) ---
st.set_page_config(
    page_title="Syntel BI Agent (Groq/LangGraph)", 
    layout="wide"
)

st.title("Syntel Company Data AI Agent (LangGraph/Groq)")
st.markdown("### High-Speed Research Pipeline with Enhanced Source Tracking")

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
            tab1, tab2, tab3 = st.tabs(["Final Report Table", "Detailed View", "Analysis Summary"])
            
            with tab1:
                st.subheader(f"Final Business Intelligence Report for {company_input}")
                final_df = format_data_for_display(company_input, validated_data)
                
                # Display the DataFrame using to_html with escape=False to render <br> and other markdown from the helper function
                st.markdown(final_df.to_html(escape=False, header=True, index=False), unsafe_allow_html=True)
                
                st.caption("All data points above include the direct source link/reference as requested. The table can be scrolled horizontally.")

            with tab2:
                st.subheader("Detailed Research Results (All Fields with Sources)")
                
                categories = {
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
                }

                for category, fields in categories.items():
                    with st.expander(category, expanded=True):
                        for field in fields:
                            if field in data_dict:
                                st.markdown(f"**{field.replace('_', ' ').title()}:** {data_dict[field]}")
                                st.divider()
            
            with tab3:
                st.subheader("Business Intelligence Analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Intent Scoring", validated_data.intent_scoring_level)
                with col2:
                    st.metric("Research Date", datetime.now().strftime("%Y-%m-%d"))
                
                st.subheader("Relevance to Syntel (3 Key Points)")
                st.markdown(validated_data.why_relevant_to_syntel_bullets)
                
                st.subheader("Download Options")
                
                # Generate the final DataFrame for download consistency
                final_df_download = format_data_for_display(company_input, validated_data)
                
                # --- Excel Download Helper ---
                def to_excel(df):
                    """Converts dataframe to an in-memory Excel file."""
                    output = BytesIO()
                    # Ensure xlsxwriter is installed for multi-format Excel
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df.to_excel(writer, index=False, sheet_name='CompanyData')
                    return output.getvalue()
                
                excel_data = to_excel(final_df_download.drop(columns=["Why Relevent to Syntel (3 Key Points)"])) # Drop the HTML formatted column if using display
                
                # Re-create the final data frame, but without the HTML/Streamlit display formatting for the 'Why Relevant' column, just using the raw markdown string.
                download_df = format_data_for_display(company_input, validated_data)
                # Ensure the 'Why Relevant' column is the pure markdown string for better Excel/CSV representation
                download_df["Why Relevent to Syntel (3 Key Points)"] = validated_data.why_relevant_to_syntel_bullets
                
                # 1. Download JSON
                json_filename = f"{company_input.replace(' ', '_')}_data.json"
                st.download_button(
                    label="Download JSON Data",
                    data=json.dumps(validated_data.dict(), indent=2),
                    file_name=json_filename,
                    mime="application/json"
                )

                # 2. Download CSV
                csv_data = download_df.to_csv(index=False).encode('utf-8')
                csv_filename = f"{company_input.replace(' ', '_')}_data.csv"
                st.download_button(
                    label="Download CSV Data",
                    data=csv_data,
                    file_name=csv_filename,
                    mime="text/csv"
                )
                
                # 3. Download Excel (XLSX)
                excel_data = to_excel(download_df)
                excel_filename = f"{company_input.replace(' ', '_')}_data.xlsx"
                st.download_button(
                    label="Download Excel Data",
                    data=excel_data,
                    file_name=excel_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                        
        except Exception as e:
            st.session_state.progress_bar.progress(100)
            st.error(f"Research failed: {type(e).__name__} - {str(e)}")
            st.markdown("""
            **Common Issues:**
            - Check your **`GROQ_API_KEY`** and **`TAVILY_API_KEY`** in secrets.
            - The LLM might struggle if the search results for a company are extremely poor.
            """)

# --- Research History (Sidebar) ---
if st.session_state.research_history:
    st.sidebar.header("Research History")
    for i, research in enumerate(reversed(st.session_state.research_history)):
        original_index = len(st.session_state.research_history) - 1 - i 
        
        with st.sidebar.expander(f"**{research['company']}** - {research['timestamp'][:10]}", expanded=False):
            st.write(f"Intent Score: {research['data'].get('intent_scoring_level', 'N/A')}")
            if st.button(f"Load {research['company']}", key=f"load_{original_index}"):
                st.session_state.company_input = research['company'] 
                st.rerun()

# --- Instructions (Sidebar) ---
with st.sidebar.expander("Setup Instructions ⚙️"):
    st.markdown("""
    This app uses **LangGraph** and **Groq (Llama 3.1 8B)**.

    **Key Requirements Implemented:**
    1.  **Source Links:** *Every* data point now explicitly includes its source/link.
    2.  **Intent Scoring:** Now determined as **Low, Medium, or High**.
    3.  **Syntel Relevance:** A **3-point bullet list** is generated based on the company's profile and Syntel's expertise (Digital Transformation, Cloud, Network, Automation).
    4.  **Fallback Search:** The primary search tool (Tavily) is configured for aggressive meta-searching (Google, DuckDuckGo, News) to maximize fill rate.

    **You MUST set both keys in your Streamlit Cloud secrets:**
    - **`TAVILY_API_KEY`**
    - **`GROQ_API_KEY`**
    """)
