import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import dotenv
from openai import OpenAI
import utils as ut

dotenv.load_dotenv()
# print(os.environ['GROQ_API_KEY'])

# Groq lpu(better than nvidia) makes its inference faster 
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ['GROQ_API_KEY']
    )

    
    
def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

xgboost_model = load_model("models/xgb_model.pkl")

naive_bayes_model = load_model("models/nb_model.pkl")

random_forest_model = load_model("models/rf_model.pkl")

decision_tree_model = load_model("models/dt_model.pkl")

svm_model = load_model("models/svm_model.pkl")

knn_model = load_model("models/knn_model.pkl")

voting_classifier_model = load_model("models/voting_classifier.pkl")

xgboost_SMOTE_model = load_model("models/xgboost_SMOTE.pkl")

xgboost_featureEngineered_model = load_model("models/xgboost_featureEngineered.pkl")


def prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary):
    input_dict = {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": int(has_credit_card),
        "IsActiveMember": int(is_active_member),
        "EstimatedSalary": estimated_salary,
        "Geography_France": 1 if location == "France" else 0,
        "Geography_Germany": 1 if location == "Germany" else 0,
        "Geography_Spain": 1 if location == "Spain" else 0,
        "Gender_Female": 1 if gender == "Female" else 0,
        "Gender_Male": 1 if gender == "Male" else 0
    }
    
    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict

def make_predictions(input_df, input_dict):
    
    probabilities = {
        'XGBoost': xgboost_model.predict_proba(input_df)[0][1],
        'Random Forest': random_forest_model.predict_proba(input_df)[0][1],
        'K-Nearest Neighbors': knn_model.predict_proba(input_df)[0][1],
        # 'SVM': svm_model.predict_proba(input_df)[0][1],
        # 'Decision Tree': decision_tree_model.predict_proba(input_df)[0][1],
        # 'Naive Bayes': naive_bayes_model.predict_proba(input_df)[0][1],
        # 'Voting Classifier': voting_classifier_model.predict_proba(input_df)[0][1],
        # 'XGBoost SMOTE': xgboost_SMOTE_model.predict_proba(input_df)[0][1],
        # 'XGBoost Feature Engineered': xgboost_featureEngineered_model.predict_proba(input_df)[0][1],

    }
    
    avg_probability = np.mean(list(probabilities.values()))
    col1, col2 = st.columns(2)

    with col1:
        fig = ut.create_gauge_chart(avg_probability)
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"The customer has a {avg_probability:.2%} probability of churning.")
    
    with col2:
        fig_probs = ut.create_model_probability_chart(probabilities)
        st.plotly_chart(fig_probs, use_container_width=True)
    # st.markdown("### Model Probabilities")
    # for model, prob in probabilities.items():
    #     st.write(f"{model}: {prob:.2f}")
    # st.write(f"Average Probability: {avg_probability:.2f}")
    return avg_probability

    # if avg_probability > 0.5:
    #     st.write("### Churn Prediction")
    #     st.write("The customer is likely to churn.")
    # else:
    #     st.write("### Churn Prediction")
    #     st.write("The customer is likely to not churn.")

# ... (previous imports and setup remain the same)

def explain_prediction(probability, input_dict, surname):
    prompt = f"""Task: Analyze a bank customer's risk of leaving the bank (churning) and provide a clear explanation.

Context: You are a senior data analyst at a major bank, specializing in customer retention analysis. You need to explain why {surname} might leave or stay with the bank.

Customer Profile:
- Name: {surname}
- Credit Score: {input_dict['CreditScore']}
- Age: {input_dict['Age']}
- Balance: ${input_dict['Balance']:,.2f}
- Products Held: {input_dict['NumOfProducts']}
- Active Member: {'Yes' if input_dict['IsActiveMember'] else 'No'}
- Location: {'France' if input_dict['Geography_France'] else 'Germany' if input_dict['Geography_Germany'] else 'Spain'}
- Tenure: {input_dict['Tenure']} years

Key Risk Factors (in order of importance):
1. Number of Products (32.4% impact)
2. Active Member Status (16.4% impact)
3. Age (11.0% impact)
4. Geographic Location (9.1% impact)
5. Account Balance (5.3% impact)

Typical Churning Customer Profile:
{df[df['Exited'] == 1].describe()}

Instructions:
1. If risk is above 40%: Focus on the specific factors putting this customer at high risk, using their actual numbers compared to typical churners.
2. If risk is below 40%: Highlight the positive factors keeping this customer stable, backed by their specific data points.

Requirements:
- Provide exactly three clear, data-backed sentences
- Use concrete numbers and comparisons
- Focus on the top 3 most relevant factors for this specific customer
- Avoid mentioning predictions, models, or probabilities
- Keep the tone professional but conversational

Write your analysis as if you're explaining to another banker:
"""
    
    raw_response = client.chat.completions.create(
        model="llama-3.2-3b-preview",
        messages=[{
            "role": "user", 
            "content": prompt
        }]
    )
    
    return raw_response.choices[0].message.content

def generate_email(probability, input_dict, surname, explanation):
    prompt = f"""Task: Write a personalized retention email to a bank customer.

Context:
You are Jennifer Martinez, Senior Relationship Manager at Equity Bank.
Customer: {surname}
Profile:
- Balance: ${input_dict['Balance']:,.2f}
- Products: {input_dict['NumOfProducts']} banking products
- Member for: {input_dict['Tenure']} years
- Active Status: {'Active' if input_dict['IsActiveMember'] else 'Inactive'}
- Credit Card: {'Yes' if input_dict['HasCrCard'] else 'No'}
- Age: {input_dict['Age']}
- Location: {'France' if input_dict['Geography_France'] else 'Germany' if input_dict['Geography_Germany'] else 'Spain'}

Analysis of their situation:
{explanation}

Instructions:
1. Write a personalized email that addresses the specific points mentioned in the analysis
2. Structure the email as follows:
   - Warm greeting
   - Appreciation of their relationship with the bank
   - Value proposition based on their specific situation
   - 3-4 bullet points with personalized offers/solutions
   - Clear call to action
   - Professional closing

Requirements:
- Keep the tone warm but professional
- Be specific to their situation and usage patterns
- Make offers that match their profile and needs
- Include specific numbers in offers where relevant
- Never mention risk scores or churn predictions
- Maximum 250 words

Personalization Guide:
- High-balance customers ($100k+): Focus on premium services and investment opportunities
- Long-term customers (5+ years): Emphasize loyalty rewards
- Inactive customers: Focus on digital banking features and convenience
- Single-product customers: Highlight complementary product benefits
- Young customers (<35): Emphasize digital features and future planning
- Older customers (>50): Focus on stability and personalized service

Write the email:"""

    raw_response = client.chat.completions.create(
        model='llama-3.2-3b-preview',
        messages=[{
            "role": "user", 
            "content": prompt
        }]
    )
    
    return raw_response.choices[0].message.content

 
st.title("Customer Churn Prediction")

df = pd.read_csv("churn.csv")

customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]

selected_customer_option = st.selectbox(
    "Select a customer",
    customers
)

if selected_customer_option:
    selected_customer_id = int(selected_customer_option.split(" - ")[0])
    
    print("Selected Customer ID:", selected_customer_id)
    # st.dataframe(df[df["CustomerId"] == selected_customer_id])
    
    selected_surname = selected_customer_option.split(" - ")[1]
    print("Selected Surname:", selected_surname)
    
    selected_customer = df.loc[df["CustomerId"] == selected_customer_id].iloc[0]
    print("Selected Customer:", selected_customer)
    
    
    col1, col2 = st.columns(2)
    
    with col1:
        credit_score = st.number_input(
            "Credit Score", 
            min_value=300, 
            max_value=850, 
            value=int(selected_customer["CreditScore"]))
        
        location = st.selectbox(
            "location", ["Spain", "France", "Germany"],
            index=["Spain", "France", "Germany"].index(selected_customer["Geography"])
        )
        
        gender = st.radio(
            "Gender",
            ["Male", "Female"],
            index = 0 if selected_customer["Gender"] == "Male" else 1
        )
            
        
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=int(selected_customer["Age"])
        )
        
        tenure = st.number_input(
            "Tenure (years)",
            min_value=0,
            max_value=50,
            value=int(selected_customer["Tenure"])
        )
    
    with col2:
        balance = st.number_input(
            "Balance",
            min_value=0.0,
            value=float(selected_customer["Balance"])
        )
        
        num_products = st.number_input(
            "Number of Products",
            min_value=1,
            max_value=10,
            value=int(selected_customer["NumOfProducts"])
        )
        
        has_credit_card = st.checkbox(
            "Has Credit Card?",
            value=bool(selected_customer["HasCrCard"])
        ) 
        
        is_active_member = st.checkbox(
            "Is Active Member?",
            value=bool(selected_customer["IsActiveMember"])
        )
        
        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=float(selected_customer["EstimatedSalary"])
        )
        
    input_df, input_dict = prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary)
    
    st.markdown("### Customer Input")
    st.dataframe(input_df)
    
    # if st.button("Get Predictions"):
    #     make_predictions(input_df, input_dict)
    avg_probability = make_predictions(input_df, input_dict)
    
    explanation = explain_prediction(avg_probability, input_dict, selected_customer['Surname'])
    
    st.markdown("-----")
    
    st.subheader("Explanation of Prediction")
    
    st.write(explanation)
    
    email = generate_email(avg_probability, input_dict, selected_customer['Surname'], explanation)
    
    st.markdown("-----")
    
    st.subheader("Personalized Email")
    
    st.write(email)