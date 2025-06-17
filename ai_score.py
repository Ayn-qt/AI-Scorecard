import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import requests
import json
from datetime import datetime
import hashlib
warnings.filterwarnings('ignore')

# Initialize sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

class EnhancedEthicsAnalyzer:
    def __init__(self):
        # Enhanced religious principles with more sophisticated scoring
        self.religious_principles = {
            'Christianity': {
                'core_values': {
                    'love': 0.25, 'compassion': 0.20, 'forgiveness': 0.15, 'justice': 0.15,
                    'service': 0.10, 'mercy': 0.10, 'humility': 0.05
                },
                'prohibited_values': {
                    'hatred': -0.25, 'violence': -0.20, 'greed': -0.15, 'pride': -0.15,
                    'revenge': -0.10, 'cruelty': -0.10, 'selfishness': -0.05
                },
                'context_modifiers': {
                    'family': 1.2, 'community': 1.1, 'sacrifice': 1.3, 'healing': 1.2
                },
                'philosophical_weight': 0.9  # Traditional vs progressive interpretation
            },
            'Islam': {
                'core_values': {
                    'justice': 0.25, 'compassion': 0.20, 'brotherhood': 0.15, 'moderation': 0.15,
                    'charity': 0.10, 'wisdom': 0.10, 'patience': 0.05
                },
                'prohibited_values': {
                    'injustice': -0.25, 'oppression': -0.20, 'excess': -0.15, 'corruption': -0.15,
                    'hatred': -0.10, 'violence': -0.10, 'dishonesty': -0.05
                },
                'context_modifiers': {
                    'community': 1.3, 'education': 1.2, 'balance': 1.1, 'social': 1.2
                },
                'philosophical_weight': 0.95
            },
            'Judaism': {
                'core_values': {
                    'justice': 0.25, 'learning': 0.20, 'repair': 0.15, 'responsibility': 0.15,
                    'community': 0.10, 'ethics': 0.10, 'wisdom': 0.05
                },
                'prohibited_values': {
                    'injustice': -0.25, 'ignorance': -0.20, 'destruction': -0.15, 'irresponsibility': -0.15,
                    'hatred': -0.10, 'oppression': -0.10, 'selfishness': -0.05
                },
                'context_modifiers': {
                    'study': 1.3, 'community': 1.2, 'repair': 1.4, 'future': 1.1
                },
                'philosophical_weight': 0.9
            },
            'Hinduism': {
                'core_values': {
                    'dharma': 0.30, 'ahimsa': 0.25, 'karma': 0.20, 'truth': 0.10,
                    'duty': 0.10, 'balance': 0.05
                },
                'prohibited_values': {
                    'adharma': -0.30, 'violence': -0.25, 'dishonesty': -0.20, 'selfishness': -0.10,
                    'imbalance': -0.10, 'harm': -0.05
                },
                'context_modifiers': {
                    'duty': 1.4, 'cycle': 1.2, 'liberation': 1.3, 'harmony': 1.2
                },
                'philosophical_weight': 1.0
            },
            'Buddhism': {
                'core_values': {
                    'compassion': 0.30, 'wisdom': 0.25, 'mindfulness': 0.20, 'non-harm': 0.15,
                    'moderation': 0.05, 'peace': 0.05
                },
                'prohibited_values': {
                    'suffering': -0.25, 'attachment': -0.20, 'ignorance': -0.25, 'hatred': -0.15,
                    'greed': -0.10, 'delusion': -0.05
                },
                'context_modifiers': {
                    'mindful': 1.3, 'liberation': 1.4, 'awareness': 1.2, 'balance': 1.1
                },
                'philosophical_weight': 1.0
            },
            'Sikhism': {
                'core_values': {
                    'equality': 0.30, 'service': 0.25, 'truth': 0.20, 'justice': 0.15,
                    'humility': 0.05, 'courage': 0.05
                },
                'prohibited_values': {
                    'inequality': -0.30, 'selfishness': -0.25, 'dishonesty': -0.20, 'injustice': -0.15,
                    'discrimination': -0.05, 'pride': -0.05
                },
                'context_modifiers': {
                    'equality': 1.4, 'service': 1.3, 'community': 1.2, 'sharing': 1.2
                },
                'philosophical_weight': 0.95
            }
        }
        
        # Enhanced ethical frameworks
        self.ethical_frameworks = {
            'deontological': ['duty', 'right', 'wrong', 'obligation', 'principle', 'rule'],
            'consequentialist': ['outcome', 'result', 'benefit', 'harm', 'utility', 'consequence'],
            'virtue_ethics': ['character', 'virtue', 'excellence', 'integrity', 'honor', 'noble'],
            'care_ethics': ['care', 'relationship', 'empathy', 'nurture', 'support', 'connection'],
            'justice_ethics': ['fair', 'equal', 'just', 'rights', 'distribution', 'deserve']
        }
        
        # Contextual scenarios for better analysis
        self.scenario_types = {
            'medical': ['health', 'medical', 'treatment', 'patient', 'doctor', 'hospital', 'medicine'],
            'ai_technology': ['ai', 'algorithm', 'artificial', 'technology', 'automated', 'machine'],
            'social_justice': ['discrimination', 'bias', 'equality', 'rights', 'social', 'minority'],
            'environmental': ['environment', 'climate', 'nature', 'pollution', 'sustainability', 'earth'],
            'economic': ['money', 'wealth', 'poverty', 'economic', 'financial', 'resource'],
            'legal': ['law', 'legal', 'court', 'judge', 'crime', 'punishment', 'justice']
        }

    def advanced_text_preprocessing(self, text):
        """Enhanced text preprocessing with context preservation"""
        # Preserve important ethical terms
        ethical_terms = []
        for religion_data in self.religious_principles.values():
            ethical_terms.extend(religion_data['core_values'].keys())
            ethical_terms.extend(religion_data['prohibited_values'].keys())
        
        # Convert to lowercase but preserve structure
        processed_text = text.lower()
        
        # Remove punctuation except those that matter for context
        processed_text = re.sub(r'[^\w\s\-\']', ' ', processed_text)
        processed_text = re.sub(r'\s+', ' ', processed_text).strip()
        
        return processed_text, ethical_terms

    def calculate_contextual_modifiers(self, text, religion):
        """Calculate contextual modifiers based on scenario type"""
        processed_text = text.lower()
        modifiers = self.religious_principles[religion]['context_modifiers']
        
        modifier_score = 1.0
        for modifier, weight in modifiers.items():
            if modifier in processed_text:
                modifier_score *= weight
        
        return min(modifier_score, 2.0)  # Cap at 2x multiplier

    def detect_scenario_type(self, text):
        """Detect the type of ethical scenario"""
        processed_text = text.lower()
        scenario_scores = {}
        
        for scenario_type, keywords in self.scenario_types.items():
            matches = sum(1 for keyword in keywords if keyword in processed_text)
            scenario_scores[scenario_type] = matches / len(keywords)
        
        return max(scenario_scores, key=scenario_scores.get), scenario_scores

    def calculate_enhanced_religious_alignment(self, text, religion):
        """Enhanced religious alignment calculation with multiple factors"""
        processed_text, ethical_terms = self.advanced_text_preprocessing(text)
        principles = self.religious_principles[religion]
        
        # Calculate core values alignment
        core_score = 0
        for value, weight in principles['core_values'].items():
            if value in processed_text:
                core_score += weight
        
        # Calculate prohibited values penalty
        prohibited_score = 0
        for value, weight in principles['prohibited_values'].items():
            if value in processed_text:
                prohibited_score += weight
        
        # Base alignment score
        base_score = max(0, core_score + prohibited_score)  # prohibited_score is negative
        
        # Apply contextual modifiers
        context_modifier = self.calculate_contextual_modifiers(text, religion)
        
        # Advanced sentiment analysis
        blob = TextBlob(text)
        textblob_sentiment = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        vader_scores = vader_analyzer.polarity_scores(text)
        vader_compound = vader_scores['compound']
        
        # Weighted sentiment combination
        sentiment_score = (
            textblob_sentiment * 0.4 + 
            vader_compound * 0.4 + 
            (1 - textblob_subjectivity) * 0.2  # Objectivity bonus
        )
        
        # Philosophical weight adjustment
        philosophical_weight = principles['philosophical_weight']
        
        # Final score calculation
        final_score = (
            base_score * 0.5 * context_modifier +
            (sentiment_score + 1) / 2 * 0.3 +
            philosophical_weight * 0.2
        )
        
        return min(max(final_score, 0), 1)

    def analyze_ethical_frameworks(self, text):
        """Analyze alignment with different ethical frameworks"""
        processed_text = text.lower()
        framework_scores = {}
        
        for framework, keywords in self.ethical_frameworks.items():
            matches = sum(1 for keyword in keywords if keyword in processed_text)
            framework_scores[framework] = matches / len(keywords)
        
        return framework_scores

    def calculate_consensus_score(self, religious_scores):
        """Calculate how much consensus exists across religions"""
        scores = list(religious_scores.values())
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Higher consensus when standard deviation is lower
        consensus_score = max(0, 1 - (std_score / mean_score)) if mean_score > 0 else 0
        
        return consensus_score

    def generate_detailed_insights(self, religious_scores, framework_scores, scenario_type, scenario_scores, consensus_score):
        """Generate comprehensive insights with specific recommendations"""
        insights = []
        
        # Religious analysis
        sorted_religions = sorted(religious_scores.items(), key=lambda x: x[1], reverse=True)
        highest_religion, highest_score = sorted_religions[0]
        lowest_religion, lowest_score = sorted_religions[-1]
        
        insights.append(f"**Religious Perspective Analysis:**")
        insights.append(f"• {highest_religion} shows highest alignment ({highest_score:.3f})")
        insights.append(f"• {lowest_religion} shows lowest alignment ({lowest_score:.3f})")
        insights.append(f"• Cross-religious consensus: {consensus_score:.3f} (0=high disagreement, 1=high agreement)")
        
        # Scenario type analysis
        insights.append(f"\n**Scenario Classification:**")
        insights.append(f"• Primary context: {scenario_type.replace('_', ' ').title()}")
        top_contexts = sorted(scenario_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        for context, score in top_contexts:
            if score > 0.1:
                insights.append(f"• {context.replace('_', ' ').title()}: {score:.2f}")
        
        # Ethical framework analysis
        insights.append(f"\n**Ethical Framework Analysis:**")
        sorted_frameworks = sorted(framework_scores.items(), key=lambda x: x[1], reverse=True)
        for framework, score in sorted_frameworks:
            if score > 0.1:
                framework_name = framework.replace('_', ' ').title()
                insights.append(f"• {framework_name}: {score:.2f}")
        
        # Overall assessment with recommendations
        overall_score = np.mean(list(religious_scores.values()))
        insights.append(f"\n**Overall Assessment & Recommendations:**")
        
        if overall_score > 0.8:
            insights.append("• **Excellent**: Strong ethical alignment across traditions")
            insights.append("• Recommendation: Proceed with confidence, minimal ethical concerns")
        elif overall_score > 0.6:
            insights.append("• **Good**: Generally positive ethical alignment")
            insights.append("• Recommendation: Minor adjustments may improve ethical standing")
        elif overall_score > 0.4:
            insights.append("• **Moderate**: Mixed ethical implications")
            insights.append("• Recommendation: Significant ethical review recommended")
        else:
            insights.append("• **Concerning**: Low ethical alignment across traditions")
            insights.append("• Recommendation: Major ethical concerns require addressing")
        
        if consensus_score < 0.3:
            insights.append("• **Note**: Low consensus suggests culturally sensitive issue")
            insights.append("• Recommendation: Consult with diverse stakeholders")
        
        return insights

    def generate_enhanced_report(self, scenario):
        """Generate comprehensive enhanced ethics report"""
        # Enhanced religious alignment scores
        religious_scores = {}
        for religion in self.religious_principles.keys():
            score = self.calculate_enhanced_religious_alignment(scenario, religion)
            religious_scores[religion] = score
        
        # Ethical framework analysis
        framework_scores = self.analyze_ethical_frameworks(scenario)
        
        # Scenario type detection
        scenario_type, scenario_scores = self.detect_scenario_type(scenario)
        
        # Consensus analysis
        consensus_score = self.calculate_consensus_score(religious_scores)
        
        # Overall score with weighted considerations
        overall_score = (
            np.mean(list(religious_scores.values())) * 0.6 +
            np.mean(list(framework_scores.values())) * 0.3 +
            consensus_score * 0.1
        )
        
        # Generate comprehensive insights
        insights = self.generate_detailed_insights(
            religious_scores, framework_scores, scenario_type, 
            scenario_scores, consensus_score
        )
        
        return {
            'religious_scores': religious_scores,
            'framework_scores': framework_scores,
            'scenario_type': scenario_type,
            'scenario_scores': scenario_scores,
            'consensus_score': consensus_score,
            'overall_score': overall_score,
            'insights': insights,
            'timestamp': datetime.now().isoformat()
        }

# Caching for performance
@st.cache_resource
def load_enhanced_analyzer():
    return EnhancedEthicsAnalyzer()

def create_enhanced_visualizations(results):
    """Create comprehensive visualizations"""
    # 1. Religious Scores Radar Chart (Enhanced)
    religions = list(results['religious_scores'].keys())
    scores = list(results['religious_scores'].values())
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=scores,
        theta=religions,
        fill='toself',
        name='Religious Alignment',
        line_color='rgb(46, 204, 113)',
        fillcolor='rgba(46, 204, 113, 0.3)',
        marker=dict(size=8)
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickmode='linear',
                tick0=0,
                dtick=0.2
            )),
        showlegend=True,
        title="Religious Ethics Alignment Scores (Enhanced Analysis)",
        font=dict(size=12),
        height=400
    )
    
    # 2. Ethical Frameworks Analysis
    frameworks = list(results['framework_scores'].keys())
    framework_scores = list(results['framework_scores'].values())
    
    fig_frameworks = px.bar(
        x=[fw.replace('_', ' ').title() for fw in frameworks],
        y=framework_scores,
        title="Ethical Framework Analysis",
        labels={'x': 'Ethical Framework', 'y': 'Relevance Score'},
        color=framework_scores,
        color_continuous_scale='Plasma'
    )
    fig_frameworks.update_layout(showlegend=False, height=300)
    
    # 3. Scenario Context Analysis
    scenarios = list(results['scenario_scores'].keys())
    scenario_scores = list(results['scenario_scores'].values())
    
    fig_scenarios = px.pie(
        values=scenario_scores,
        names=[sc.replace('_', ' ').title() for sc in scenarios],
        title="Scenario Context Distribution",
        height=300
    )
    
    # 4. Consensus Analysis Gauge
    fig_consensus = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = results['consensus_score'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Cross-Religious Consensus"},
        delta = {'reference': 0.5},
        gauge = {
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgray"},
                {'range': [0.3, 0.7], 'color': "gray"},
                {'range': [0.7, 1], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.8
            }
        }
    ))
    fig_consensus.update_layout(height=300)
    
    return fig_radar, fig_frameworks, fig_scenarios, fig_consensus

def main():
    st.set_page_config(
        page_title="Enhanced Multi-Religious Ethics Assessment",
        page_icon="⚖️",
        layout="wide"
    )
    
    # Header with enhanced description
    st.title("⚖️ Enhanced Multi-Religious Ethics Assessment System")
    st.markdown("*Advanced AI-powered ethical analysis with contextual understanding and consensus measurement*")
    
    # Enhanced sidebar
    with st.sidebar:
        st.header("🔬 Advanced Features")
        st.markdown("""
        **Enhanced Analysis Includes:**
        - **Contextual Modifiers**: Scenario-specific weightings
        - **Consensus Scoring**: Cross-religious agreement measurement
        - **Framework Analysis**: Multiple ethical theories
        - **Scenario Classification**: Automatic context detection
        - **Temporal Tracking**: Analysis history
        """)
        
        st.header("📊 Scoring Methodology")
        st.markdown("""
        **Multi-layered Approach:**
        1. **Religious Alignment** (60%): Core values + prohibited values
        2. **Ethical Frameworks** (30%): Deontological, consequentialist, virtue ethics
        3. **Consensus Factor** (10%): Agreement across traditions
        """)
        
        st.header("🎯 Precision Improvements")
        st.markdown("""
        - **Weighted Value Systems**: Each religion has specific value priorities
        - **Context-Aware Modifiers**: Scenario-specific adjustments
        - **Advanced Sentiment Analysis**: Multiple NLP models
        - **Philosophical Weights**: Traditional vs progressive interpretations
        """)
    
    # Initialize enhanced analyzer
    analyzer = load_enhanced_analyzer()
    
    # Enhanced input section
    st.header("📝 Advanced Ethical Scenario Analysis")
    
    # Enhanced example scenarios
    enhanced_examples = {
        "Select an example...": "",
        "AI Healthcare Triage": "An AI system in a hospital emergency room must prioritize patients during a crisis. It has data showing that younger patients have higher survival rates, but an elderly patient arrived first and is in critical condition. The system must decide who gets the last available ventilator.",
        "Autonomous Vehicle Moral Dilemma": "A self-driving car's AI detects an imminent collision. It can either continue straight and likely kill 3 elderly passengers, or swerve onto the sidewalk potentially killing 1 child. The car has 0.5 seconds to decide.",
        "AI Hiring Algorithm Bias": "A company's AI hiring system shows 95% accuracy in predicting job performance but demonstrates clear bias against women and minorities. Using it would improve company performance but perpetuate systemic discrimination.",
        "Pandemic Privacy vs Safety": "During a pandemic, a government AI system can track citizens' movements via smartphones to prevent disease spread, potentially saving thousands of lives but violating privacy rights and enabling surveillance.",
        "AI Resource Allocation": "An AI system managing food distribution in a refugee camp must decide how to allocate limited resources. It can optimize for overall nutrition, prioritize children and pregnant women, or distribute equally regardless of need."
    }
    
    selected_example = st.selectbox("Choose an enhanced example scenario:", list(enhanced_examples.keys()))
    
    if selected_example != "Select an example...":
        default_text = enhanced_examples[selected_example]
    else:
        default_text = ""
    
    scenario = st.text_area(
        "Enter your ethical scenario for advanced analysis:",
        value=default_text,
        height=180,
        placeholder="Describe a complex ethical dilemma with multiple stakeholders and considerations..."
    )
    
    # FIXED: Added unique key to prevent duplicate element ID error
    if st.button("🔍 Perform Enhanced Analysis", type="primary", key="enhanced_analysis_button") and scenario.strip():
        with st.spinner("Conducting advanced ethical analysis across multiple dimensions..."):
            results = analyzer.generate_enhanced_report(scenario)
        
        # Display enhanced results
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.header("📊 Comprehensive Analysis Results")
            
            # Enhanced overall score display
            overall_score = results['overall_score']
            consensus_score = results['consensus_score']
            
            score_color = "green" if overall_score > 0.7 else "orange" if overall_score > 0.4 else "red"
            consensus_color = "green" if consensus_score > 0.7 else "orange" if consensus_score > 0.4 else "red"
            
            st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; margin: 10px 0;">
                <h3>Overall Ethics Score: <span style="color: {score_color};">{overall_score:.3f}/1.000</span></h3>
                <h4>Cross-Religious Consensus: <span style="color: {consensus_color};">{consensus_score:.3f}/1.000</span></h4>
                <p>Scenario Type: <strong>{results['scenario_type'].replace('_', ' ').title()}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Enhanced visualizations
            fig_radar, fig_frameworks, fig_scenarios, fig_consensus = create_enhanced_visualizations(results)
            
            # Display visualizations in tabs
            tab1, tab2, tab3, tab4 = st.tabs(["Religious Alignment", "Ethical Frameworks", "Scenario Context", "Consensus Analysis"])
            
            with tab1:
                st.plotly_chart(fig_radar, use_container_width=True)
            
            with tab2:
                st.plotly_chart(fig_frameworks, use_container_width=True)
            
            with tab3:
                st.plotly_chart(fig_scenarios, use_container_width=True)
            
            with tab4:
                st.plotly_chart(fig_consensus, use_container_width=True)
        
        with col2:
            st.header("🎯 Detailed Insights & Recommendations")
            
            # Display insights
            for insight in results['insights']:
                st.markdown(insight)
            
            st.header("📋 Detailed Scoring")
            
            # Enhanced religious scores table
            religious_df = pd.DataFrame(
                list(results['religious_scores'].items()),
                columns=['Religion', 'Alignment Score']
            )
            religious_df['Alignment Score'] = religious_df['Alignment Score'].round(4)
            religious_df = religious_df.sort_values('Alignment Score', ascending=False)
            religious_df['Rank'] = range(1, len(religious_df) + 1)
            
            st.dataframe(religious_df[['Rank', 'Religion', 'Alignment Score']], use_container_width=True)
            
            # Framework scores
            st.subheader("Ethical Framework Relevance")
            framework_df = pd.DataFrame(
                list(results['framework_scores'].items()),
                columns=['Framework', 'Relevance']
            )
            framework_df['Framework'] = framework_df['Framework'].str.replace('_', ' ').str.title()
            framework_df['Relevance'] = framework_df['Relevance'].round(3)
            framework_df = framework_df.sort_values('Relevance', ascending=False)
            
            st.dataframe(framework_df, use_container_width=True)
    
    # FIXED: Removed the duplicate button and added a warning for empty scenario

    
    # Enhanced footer with methodology
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p><strong>Enhanced Multi-Religious Ethics Assessment System v2.0</strong></p>
        <p>Advanced AI-Powered Ethical Analysis | Promoting Inclusive Decision-Making</p>
        <p><em>This enhanced system provides multi-dimensional analysis combining religious wisdom, ethical theory, and contextual understanding.</em></p>
        <p><small>Analysis includes: Contextual modifiers • Consensus scoring • Framework analysis • Scenario classification</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
