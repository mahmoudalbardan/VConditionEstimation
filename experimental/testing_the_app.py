import requests
data = [2.014315292308302, 19.608, 0.0, 1.3772973818708674, 2.137002652204368, 2.1518784755693425,
        2.163466693500739, 4.699886703246037, 5.12455904041457, 0.0,
        3.878515099949295, 120.31, 4.8724454486622095, 4.876189197543333, 212.0, 212.0, 1083.0, 1083.0]

if __name__ == "__main__":
    prediction=  requests.post('http://localhost:8080/predict', json=data).json()
    print(prediction)
    if prediction ==0:
        print("the valve condition is not optimal: less than 100")
    else:
        print("the valve condition is optimal: equal than 100")
