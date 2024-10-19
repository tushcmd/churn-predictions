import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import dotenv
from openai import OpenAI

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

    
    st.markdown("### Model Probabilities")
    for model, prob in probabilities.items():
        st.write(f"{model}: {prob:.2f}")
    st.write(f"Average Probability: {avg_probability:.2f}")
    return avg_probability

    # if avg_probability > 0.5:
    #     st.write("### Churn Prediction")
    #     st.write("The customer is likely to churn.")
    # else:
    #     st.write("### Churn Prediction")
    #     st.write("The customer is likely to not churn.")

def explain_prediction(probability, input_dict, surname):
    
    prompt = f"""
    You are an expert data scientist at a bank, where you specialize in 
    interpreting and explaining of machine learning models.
    
    Your machine learning models has predicted that a customer named {surname}
    has a {round(probability * 100, 1)}% of churning, based on the information provided below.
    
    Here is the customer information:
    {input_dict}
    
    Here are the machine learning model's top 100 most important features for predicting churn:
    
        Feature              |  Importance
    ------------------------------------
    NumOfProducts        |  0.323888
    IsActiveMember       |  0.164146
    Age                  |  0.109550
    Geography_Germany    |  0.091373
    Balance              |  0.052786
    Geography_France     |  0.046463
    Gender_Female        |  0.045283
    Geography_Spain      |  0.036855
    CreditScore          |  0.035005
    EstimatedSalary      |  0.032655
    HasCrCard            |  0.031940
    Tenure               |  0.030054
    Gender_Male          |  0.000000
    
    {pd.set_option('display.max_colwidth', None)}
    
    Here are summary statististics for churned customers:
    {df[df['Exited'] == 1].describe()}
    
    
    - If the customer has over  a 40% risk of churning, generate a 3 sentence explanation of 
    why they are at risk of churning.
    - If the customer has less than  a 40% risk of churning, generate a 3 sentence explanation of 
    why they might not be at risk of churning.
    - Your explanation should be based on the customer's information, the summary statistics for churned customers,
    of churned and non-churned customers, and the feature importance provided.
    
    
    - Do not mention the probability of churning, or the machine learning model, or anything like 
    "Based on the machine learning model's top 10 most important features for predicting churn",
    just explain the prdiction.(except name)
    
    """
    
    print("EXPLANATION PROMPT", prompt)
    
    raw_response = client.chat.completions.create(
        # meta-llama/llama-3.2-3b-instruct:free
        model="llama-3.2-3b-preview",
        messages=[{
            "role": "user", 
             "content": prompt
             }],
        # max_tokens=1000
    )
    
    return raw_response.choices[0].message.content

def generate_email(probability, input_dict, surname, explanation):
    prompt = f"""You are a banker at Equity Bank. You are responsible 
    for ensuring customers stay with the bank and are incentivized with various offers.
    
    You noticed a customer named {surname} has a {round(probability * 100, 1)}% probability of churning: {explanation}
    
    
    Generate an email to the customer based on their information, asking them to stay if they are at risk of churning, or offering them incentives so that they become more loyalto the bank.
    
    Make suyre to list out a set of incentives to stay based on the customer's information, in bullet point format. 
    Don't ever mention the probability of churning, or the machine learning model to the customer.
    """
    raw_response = client.chat.completions.create(
        model = 'llama-3.2-3b-preview',
        messages=[{
            "role": "user", 
             "content": prompt
        }]
    )
    
    print("\n\nEMAIL PROMPT", prompt)
    
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