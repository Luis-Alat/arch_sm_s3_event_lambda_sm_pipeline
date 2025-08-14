import json
import pytest
import requests


@pytest.mark.integration
def test_api_gateway_prediction(api_url):
    
    payload = {
        "LoanID": "I38PQUQS96",
        "Age": 56,
        "Income": 85994,
        "LoanAmount": 50587,
        "CreditScore": 520,
        "MonthsEmployed": 80,
        "NumCreditLines": 4,
        "InterestRate": 15.23,
        "LoanTerm": 36,
        "DTIRatio": 0.44,
        "Education": "Bachelor's",
        "EmploymentType": "Full-time",
        "MaritalStatus": "Divorced",
        "HasMortgage": "Yes",
        "HasDependents": "Yes",
        "LoanPurpose": "Other",
        "HasCoSigner": "Yes",
        "Default": 0
    }

    response = requests.post(api_url, json=payload)

    assert response.status_code == 200, f"Status response error: {response.status_code}"